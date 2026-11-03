from typing import Any

from hf_mem._weights import Weights

KV_CACHE_DTYPES = ["auto", "bfloat16", "float16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"]
MAX_NUM_BATCHED_TOKENS = 8192
BLOCK_SIZE = 16

_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1}
_TORCH = {
    "float32": "F32",
    "float": "F32",
    "float16": "F16",
    "half": "F16",
    "bfloat16": "BF16",
    "float8_e4m3": "F8_E4M3",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
}
_ALIASES = {
    "num_hidden_layers": ("num_hidden_layers", "n_layers", "num_layers", "n_layer"),
    "num_attention_heads": ("num_attention_heads", "n_heads", "n_head", "num_heads"),
    "num_key_value_heads": ("num_key_value_heads", "n_kv_heads", "num_kv_heads", "multi_query_group_num"),
    "hidden_size": ("hidden_size", "d_model", "n_embd", "dim"),
    "max_position_embeddings": ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"),
}
_DEFAULTS = {
    "gemma2": {
        "head_dim": 256,
        "sliding_window": 4096,
        "sliding_window_pattern": 2,
        "max_position_embeddings": 8192,
        "vocab_size": 256000,
    },
    "gemma3_text": {
        "head_dim": 256,
        "sliding_window": 4096,
        "sliding_window_pattern": 6,
        "max_position_embeddings": 131072,
        "num_key_value_heads": 4,
        "vocab_size": 262208,
    },
}
_TEXT_KEYS = ("text_config", "language_config", "llm_config", "lfm", "thinker_config", "talker_config")
_NON_TEXT = ("vision", "audio", "encoder", "codec", "depth", "speech", "image", "video", "visual", "detr")
_ENCODER_ARCHS = (
    "ForMaskedLM",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForImageClassification",
    "ForMaskGeneration",
    "ForTextEncoding",
    "Embedding",
)
_GENERATIVE_ARCHS = ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")
_STATE = ("linear", "conv", "mamba", "ssm", "recurrent", "delta", "rwkv", "lightning")
_NO_CACHE = ("mlp", "moe")
_SLIDING = ("sliding", "local", "chunked", "swa")


def normalize(config: dict[str, Any]) -> dict[str, Any]:
    config = {
        **_DEFAULTS.get(config.get("model_type"), {}),
        **{k: v for k, v in config.items() if v is not None},
    }
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


def _is_text(config: Any) -> bool:
    if not isinstance(config, dict):
        return False
    config = normalize(config)
    return isinstance(config["num_hidden_layers"], int) and isinstance(config["num_attention_heads"], int)


def _nested(config: dict[str, Any], path: str = "") -> list[tuple[str, dict[str, Any]]]:
    found = []
    for key, value in config.items():
        if isinstance(value, dict):
            child = f"{path}.{key}"
            if _is_text(value):
                found.append((child, value))
            found.extend(_nested(value, child))
    return found


def is_generative(config: dict[str, Any], files: dict[str, int]) -> bool:
    architectures = config.get("architectures") or []
    return "generation_config.json" in files or any(a.endswith(_GENERATIVE_ARCHS) for a in architectures)


def text_config(config: dict[str, Any]) -> dict[str, Any] | None:
    if any(hint in arch for arch in config.get("architectures") or [] for hint in _ENCODER_ARCHS):
        return None

    found = None
    for key in _TEXT_KEYS:
        if isinstance(config.get(key), dict):
            found = text_config(config[key]) or (config[key] if _is_text(config[key]) else None)
            if found is not None:
                break
    if found is None and _is_text(config):
        found = config
    if found is None:
        candidates = [c for p, c in _nested(config) if not any(h in p.lower() for h in _NON_TEXT)]
        if candidates:
            found = max(candidates, key=lambda c: normalize(c)["hidden_size"] or 0)
    if found is None:
        return None

    found = normalize(found)
    for key in ("dtype", "torch_dtype", "quantization_config"):
        if found.get(key) is None and config.get(key) is not None:
            found[key] = config[key]
    return found


def kv_dtype(config: dict[str, Any], requested: str, weights: Weights, warnings: list[str]) -> str:
    if requested in {"fp8_e5m2", "fp8_e4m3"}:
        return requested.upper().replace("FP8", "F8")
    if requested in {"fp8", "fp8_ds_mla", "fp8_inc"}:
        return "F8_E4M3"
    if requested in {"bfloat16", "float16"}:
        return _TORCH[requested]
    if requested != "auto":
        raise RuntimeError(
            f"`--kv-cache-dtype={requested}` is not valid for Safetensors models, use any of {KV_CACHE_DTYPES}."
        )

    quantization = config.get("quantization_config") or {}
    scheme = quantization.get("kv_cache_scheme") or {}
    if scheme.get("num_bits") == 8 or quantization.get("kv_cache_quant_algo") == "FP8":
        return "F8_E4M3"
    dtype = config.get("torch_dtype") or config.get("dtype")
    if isinstance(dtype, str):
        return _TORCH.get(dtype.removeprefix("torch."), "F16")

    counts = {d: w.params for d, w in weights.dtypes.items() if d in {"BF16", "F16", "F32"}}
    dtype = max(counts, key=counts.__getitem__) if counts else "BF16"
    warnings.append(
        f"The KV cache dtype could not be inferred from `config.json`, so it defaults to {dtype}, set "
        "`--kv-cache-dtype` to override it."
    )
    return dtype


def layer_kinds(config: dict[str, Any]) -> list[str]:
    layers: int = config["num_hidden_layers"]
    types = config.get("layer_types") or config.get("layers_block_type")
    if isinstance(types, list) and types:
        kinds = []
        for layer in map(str.lower, map(str, types)):
            if any(h in layer for h in _SLIDING):
                kinds.append("sliding")
            elif any(h in layer for h in _STATE):
                kinds.append("state")
            elif any(h in layer for h in _NO_CACHE):
                kinds.append("none")
            else:
                kinds.append("full")
        return kinds

    if isinstance(pattern := config.get("sliding_window_pattern"), int) and pattern > 0:
        return ["full" if (i + 1) % pattern == 0 else "sliding" for i in range(layers)]
    if isinstance(pattern := config.get("hybrid_override_pattern"), str):
        return ["full" if c == "*" else ("state" if c == "M" else "none") for c in pattern]
    return ["full"] * layers


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _state(config: dict[str, Any], dtype_bytes: float) -> tuple[int, int]:
    hidden = config.get("hidden_size") or 0
    ssm_bytes = (
        4
        if config.get("mamba_ssm_dtype") == "float32" or config.get("model_type") == "nemotron_h"
        else dtype_bytes
    )
    if heads := config.get("linear_num_value_heads"):
        key_heads = config.get("linear_num_key_heads") or heads
        key_dim, value_dim = config.get("linear_key_head_dim") or 0, config.get("linear_value_head_dim") or 0
        conv = (2 * key_heads * key_dim + heads * value_dim) * ((config.get("linear_conv_kernel_dim") or 4) - 1)
        return int(conv * dtype_bytes + heads * key_dim * value_dim * ssm_bytes), 2
    if state := config.get("mamba_d_state") or config.get("ssm_state_size") or config.get("state_size"):
        heads = config.get("mamba_num_heads") or config.get("mamba_n_heads") or 0
        head_dim = config.get("mamba_head_dim") or config.get("mamba_d_head") or 0
        inner = heads * head_dim or (config.get("mamba_expand") or config.get("expand") or 2) * hidden
        groups = config.get("n_groups") or config.get("mamba_n_groups") or 1
        kernel = config.get("conv_kernel") or config.get("mamba_d_conv") or config.get("d_conv") or 4
        return int((inner + 2 * groups * state) * (kernel - 1) * dtype_bytes + inner * state * ssm_bytes), 3
    if cache := config.get("conv_L_cache"):
        return int(hidden * (cache - 1) * dtype_bytes), 2
    return 0, 2


def kv_heads(config: dict[str, Any]) -> int:
    config = normalize(config)
    if config.get("kv_lora_rank"):
        return 1
    return config["num_key_value_heads"] or config["num_attention_heads"]


def kv_cache(
    config: dict[str, Any],
    dtype: str,
    max_model_len: int,
    batch_size: int,
    batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
) -> tuple[int, dict[str, int]]:
    config = normalize(config)
    heads: int = config["num_attention_heads"]
    kv_heads: int = config["num_key_value_heads"] or heads
    head_dim: int = config.get("head_dim") or (config["hidden_size"] or 0) // heads
    if rank := config.get("kv_lora_rank"):
        width = full_width = rank + (config.get("qk_rope_head_dim") or 0) + (config.get("index_head_dim") or 0)
    else:
        width = 2 * kv_heads * head_dim
        full_width = (
            2
            * (config.get("num_global_key_value_heads") or kv_heads)
            * (config.get("global_head_dim") or head_dim)
        )

    kinds = layer_kinds(config)
    full, sliding, state = kinds.count("full"), kinds.count("sliding"), kinds.count("state")
    layers = {"full": full, "sliding": sliding, "state": state, "none": len(kinds) - full - sliding - state}
    window = config.get("sliding_window") or config.get("attention_chunk_size") or max_model_len
    sliding_tokens = min(window - 1 + batched_tokens, max_model_len)
    sliding_pages = _cdiv(sliding_tokens, BLOCK_SIZE) + (sliding_tokens < max_model_len)
    dtype_bytes = _BYTES.get(dtype, 2)
    token, full_token = int(width * dtype_bytes), int(full_width * dtype_bytes)
    if not state:
        size = full * _cdiv(max_model_len, BLOCK_SIZE) * full_token + sliding * sliding_pages * token
        return size * BLOCK_SIZE * batch_size, layers

    state_bytes, state_pages = _state(config, dtype_bytes)
    if not full + sliding:
        return state * state_pages * state_bytes * batch_size, layers
    block = max(BLOCK_SIZE, _cdiv(_cdiv(state_bytes, token), BLOCK_SIZE) * BLOCK_SIZE)
    group = min(n for n in (full, sliding, state) if n)
    size = _cdiv(full, group) * _cdiv(max_model_len, block) * full_token
    size += _cdiv(sliding, group) * (_cdiv(sliding_tokens, block) + (sliding_tokens < max_model_len)) * token
    size += _cdiv(state, group) * state_pages * token
    return group * size * block * batch_size, layers
