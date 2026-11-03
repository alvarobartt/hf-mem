import warnings
from collections import Counter
from typing import Any, Dict, List

from hf_mem.safetensors.metadata import SafetensorsMetadata
from hf_mem.safetensors.types import get_safetensors_dtype_bytes, torch_dtype_to_safetensors_dtype

KV_CACHE_DTYPE_CHOICES = ["auto", "bfloat16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"]

_ALIASES = {
    "num_hidden_layers": ("num_hidden_layers", "n_layers", "num_layers", "n_layer"),
    "num_attention_heads": ("num_attention_heads", "n_heads", "n_head", "num_heads"),
    "num_key_value_heads": ("num_key_value_heads", "n_kv_heads", "num_kv_heads", "multi_query_group_num"),
    "hidden_size": ("hidden_size", "d_model", "n_embd", "dim"),
    "max_position_embeddings": ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"),
}
_TEXT_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "lfm", "thinker_config", "talker_config")
_NON_TEXT_HINTS = ("vision", "audio", "encoder", "codec", "depth", "speech", "image", "video", "visual", "detr")
_ENCODER_ARCH_HINTS = (
    "ForMaskedLM",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForImageClassification",
    "ForMaskGeneration",
    "ForTextEncoding",
    "Embedding",
)
_STATEFUL_LAYER_HINTS = (
    "linear",
    "conv",
    "mamba",
    "ssm",
    "recurrent",
    "delta",
    "rwkv",
    "lightning",
    "mlp",
    "moe",
)
_SLIDING_LAYER_HINTS = ("sliding", "local", "chunked", "swa")


def _normalize(config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    for key, aliases in _ALIASES.items():
        if config.get(key) is None:
            config[key] = next((config[a] for a in aliases if isinstance(config.get(a), int)), None)
    if config["num_hidden_layers"] is None:
        layers = (
            config.get("layer_types")
            or config.get("layers_block_type")
            or config.get("hybrid_override_pattern")
        )
        config["num_hidden_layers"] = len(layers) if layers else None
    return config


def _is_text_config(config: Any) -> bool:
    return (
        isinstance(config, dict)
        and isinstance(_normalize(config)["num_hidden_layers"], int)
        and isinstance(_normalize(config)["num_attention_heads"], int)
    )


def _nested_text_configs(config: Dict[str, Any], path: str = "") -> List[tuple[str, Dict[str, Any]]]:
    found = []
    for key, value in config.items():
        if isinstance(value, dict):
            child = f"{path}.{key}"
            if _is_text_config(value):
                found.append((child, value))
            found.extend(_nested_text_configs(value, child))
    return found


def resolve_text_config(config: Dict[str, Any]) -> Dict[str, Any] | None:
    if any(hint in arch for arch in config.get("architectures") or [] for hint in _ENCODER_ARCH_HINTS):
        return None

    text_config = None
    for key in _TEXT_CONFIG_KEYS:
        if isinstance(config.get(key), dict):
            text_config = resolve_text_config(config[key]) or (
                config[key] if _is_text_config(config[key]) else None
            )
            if text_config is not None:
                break

    if text_config is None and _is_text_config(config):
        text_config = config

    if text_config is None:
        candidates = [
            c for path, c in _nested_text_configs(config) if not any(h in path.lower() for h in _NON_TEXT_HINTS)
        ]
        if candidates:
            text_config = max(candidates, key=lambda c: _normalize(c)["hidden_size"] or 0)

    if text_config is None:
        return None

    text_config = _normalize(text_config)
    for key in ("dtype", "torch_dtype", "quantization_config"):
        if text_config.get(key) is None and config.get(key) is not None:
            text_config[key] = config[key]
    return text_config


def _dominant_float_dtype(metadata: SafetensorsMetadata) -> str:
    counts: Counter = Counter()
    for component in metadata.components.values():
        for dtype, dtype_metadata in component.dtypes.items():
            if dtype in {"BF16", "F16", "F32"}:
                counts[dtype] += dtype_metadata.param_count
    return counts.most_common(1)[0][0] if counts else "BF16"


def resolve_kv_cache_dtype(
    config: Dict[str, Any],
    kv_cache_dtype: str | None,
    metadata: SafetensorsMetadata,
    model_id: str,
) -> str:
    if kv_cache_dtype and kv_cache_dtype not in KV_CACHE_DTYPE_CHOICES:
        raise RuntimeError(
            f"`--kv-cache-dtype={kv_cache_dtype}` is not valid for Safetensors models, valid options are: {KV_CACHE_DTYPE_CHOICES}."
        )
    if kv_cache_dtype in {"fp8_e5m2", "fp8_e4m3"}:
        return kv_cache_dtype.upper().replace("FP8", "F8")  # type: ignore[union-attr]
    if kv_cache_dtype in {"fp8", "fp8_ds_mla", "fp8_inc"}:
        return "F8_E4M3"
    if kv_cache_dtype == "bfloat16":
        return "BF16"

    quantization_config = config.get("quantization_config") or {}
    kv_cache_scheme = quantization_config.get("kv_cache_scheme") or {}
    if kv_cache_scheme.get("num_bits") == 8 or quantization_config.get("kv_cache_quant_algo") == "FP8":
        return "F8_E4M3"

    if dtype := config.get("torch_dtype") or config.get("dtype"):
        if isinstance(dtype, str):
            return torch_dtype_to_safetensors_dtype(dtype)

    dtype = _dominant_float_dtype(metadata)
    warnings.warn(
        f"The KV cache dtype for `{model_id}` could not be inferred from `config.json`, so it defaults to {dtype}, set `--kv-cache-dtype` to override it."
    )
    return dtype


def _layer_kinds(config: Dict[str, Any]) -> List[str]:
    num_hidden_layers: int = config["num_hidden_layers"]

    if isinstance(config.get("sliding_window_pattern"), int) and config["sliding_window_pattern"] > 0:
        pattern = config["sliding_window_pattern"]
        return ["full" if (i + 1) % pattern == 0 else "sliding" for i in range(num_hidden_layers)]

    layer_types = config.get("layer_types") or config.get("layers_block_type")
    if isinstance(layer_types, list) and layer_types:
        kinds = []
        for layer_type in layer_types:
            layer_type = str(layer_type).lower()
            if any(h in layer_type for h in _SLIDING_LAYER_HINTS):
                kinds.append("sliding")
            elif any(h in layer_type for h in _STATEFUL_LAYER_HINTS):
                kinds.append("none")
            else:
                kinds.append("full")
        return kinds

    if isinstance(pattern := config.get("hybrid_override_pattern"), str):
        return ["full" if c == "*" else "none" for c in pattern]

    return ["full"] * num_hidden_layers


def compute_safetensors_kv_cache_size(
    config: Dict[str, Any],
    cache_dtype: str,
    max_model_len: int,
    batch_size: int = 1,
) -> int:
    config = _normalize(config)
    hidden_size = config["hidden_size"] or 0
    num_attention_heads: int = config["num_attention_heads"]
    num_key_value_heads: int = config["num_key_value_heads"] or num_attention_heads
    head_dim: int = config.get("head_dim") or hidden_size // num_attention_heads

    if kv_lora_rank := config.get("kv_lora_rank"):
        width = kv_lora_rank + (config.get("qk_rope_head_dim") or 0) + (config.get("index_head_dim") or 0)
    else:
        width = 2 * num_key_value_heads * head_dim

    kinds = _layer_kinds(config)
    num_full, num_sliding = kinds.count("full"), kinds.count("sliding")
    sliding_window = config.get("sliding_window") or max_model_len
    sliding_tokens = max_model_len if num_full and num_sliding else min(sliding_window, max_model_len)
    total_kv_tokens = num_full * max_model_len + num_sliding * sliding_tokens

    return int(total_kv_tokens * width * get_safetensors_dtype_bytes(cache_dtype) * batch_size)
