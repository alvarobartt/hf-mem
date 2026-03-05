import warnings
from typing import Any, Dict

from hf_mem.safetensors.metadata import SafetensorsMetadata
from hf_mem.safetensors.types import TorchDtypes, get_safetensors_dtype_bytes, torch_dtype_to_safetensors_dtype


def _resolve_num_attention_layers(config: Dict[str, Any]) -> int:
    """Return the number of full-attention (global) layers that grow the KV cache with max_model_len.

    Models with hybrid attention mix full attention and sliding window attention layers. Only the
    full attention layers contribute to a KV cache that scales with the context length; sliding
    window layers reuse a fixed-size buffer and are therefore excluded from the estimate.
    """
    num_hidden_layers: int = config["num_hidden_layers"]

    # NOTE: Gemma3-style hybrid attention: every N-th layer (0-indexed where `i % N == N-1`)
    # is a global attention layer; the rest are sliding window attention layers.
    # For N=6 and 46 total layers: 46 // 6 = 7 global attention layers.
    # For N=4 and 60 total layers: 60 // 4 = 15 global attention layers (e.g. Qwen3.5-397B-A17B).
    if "sliding_window_pattern" in config:
        return num_hidden_layers // config["sliding_window_pattern"]

    # NOTE: Some models provide an explicit list of layer types — count the non-sliding-window ones.
    # Known string values for full attention vary by architecture.
    if "layer_types" in config:
        return sum(1 for t in config["layer_types"] if t in {"attention", "full_attention", "global_attention"})

    # NOTE: Default — assume all layers use full attention (standard MHA / GQA without SWA).
    return num_hidden_layers


def resolve_kv_cache_dtype(
    config: Dict[str, Any],
    kv_cache_dtype: str | None,
    metadata: SafetensorsMetadata,
    model_id: str,
) -> str:
    """Resolve the effective KV cache dtype string (a `SafetensorsDtypes` value) from the CLI flag and config."""
    if kv_cache_dtype in {"fp8_e5m2", "fp8_e4m3"}:
        return kv_cache_dtype.upper().replace("FP8", "F8")  # type: ignore[union-attr]

    if kv_cache_dtype in {"fp8", "fp8_ds_mla", "fp8_inc"}:
        # NOTE: Default to `F8_E4M3` for the calculations, given that all those take 1 byte, but only F8_E5M2
        # or `F8_E4M3` are supported in Safetensors, whilst `FP8_DS_MLA` (DeepSeek MLA) and `FP8_INC` (Intel HPUs)
        # are not; and `F8_E4M3` is supported on both CUDA and AMD, hence seems a reasonable default
        warnings.warn(
            f"--kv-cache-dtype={kv_cache_dtype}` has been provided, but given that none of those matches an actual Safetensors dtype since it should be any of `F8_E5M2` or `F8_E4M3`, the `--kv-cache-dtype` will default to `F8_E4M3` instead, which implies that the calculations are the same given that both dtypes take 1 byte despite the quantization scheme of it, or the hardware compatibility; so the estimations should be accurate enough."
        )
        return "F8_E4M3"

    if kv_cache_dtype == "bfloat16":
        return "BF16"

    if "quantization_config" in config and "quant_method" in config["quantization_config"]:
        _quantization_config = config["quantization_config"]
        _quant_method = _quantization_config["quant_method"]

        if _quant_method != "fp8":  # NOTE: e.g., compressed-tensors for `moonshotai/Kimi-K2.5`
            raise RuntimeError(
                f"Provided `--kv-cache-dtype=auto` (or unset) and given that `config.json` contains the following `quantization_config={_quantization_config}` with a `quant_method` different than `fp8` i.e., `{_quant_method}`, which is not supported; you should enforce the `--kv-cache-dtype` value to whatever quantization precision it's using, if applicable.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
            )

        _fmt = _quantization_config.get("fmt", _quantization_config.get("format", None))
        if _fmt:
            if not _fmt.startswith("float8_"):
                _fmt = f"float8_{_fmt}"

            if _fmt not in TorchDtypes.__args__:
                raise RuntimeError(
                    f"Provided `--kv-cache-dtype=auto` (or unset) and given that `config.json` contains the following `quantization_config={_quantization_config}` with a `fmt` (or `format`) value of `{_fmt}` that's not supported (should be any of {TorchDtypes.__args__}), you might need to set `--kv-cache-dtype=fp8` to enforce the dtype instead of pulling it from the `config.json`.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
                )

            return torch_dtype_to_safetensors_dtype(_fmt)

        # NOTE: If `quant_method` in `quantization_config` is set to `fp8` and `fmt` is not set, then
        # we get the most used `F8_*` Safetensors dtype to map the `quant_method=fp8` to an actual Safetensors
        # dtype, as `F8` is not a valid dtype neither on PyTorch nor on Safetensors, as we need to append
        # the scheme / format.
        # SAFETY: As per the snippets above, if `_fmt` is None we assume that `_quant_method=fp8`
        cache_dtype = max(
            (
                l := [
                    d
                    for c in metadata.components.values()
                    for d in c.dtypes.keys()
                    if d in {"F8_E5M2", "F8_E4M3"}
                ]
            ),
            key=l.count,
            default=None,
        )

        # TODO: Not sure if we should default to `F8_E4M3` as a reasonable default as when `FP8`,
        # `FP8_DS_MLA` or `FP8_INC` are provided... to prevent raising an exception
        if not cache_dtype:
            raise RuntimeError(
                f"The `config.json` file for `--model-id={model_id}` contains `quantization_config={_quantization_config}` but the `quant_method=fp8` whereas any tensor in the model weights is set to any of `F8_E4M3` nor `F8_E5M2`, which means that the `F8_` format for the Safetensors dtype cannot be inferred; so you might need to set `--kv-cache-dtype=fp8` to enforce the dtype instead of pulling it from the `config.json`.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
            )
        return cache_dtype

    if _cache_dtype := config.get("torch_dtype", None):
        return torch_dtype_to_safetensors_dtype(_cache_dtype)

    if _cache_dtype := config.get("dtype", None):
        return torch_dtype_to_safetensors_dtype(_cache_dtype)

    raise RuntimeError(
        f"Provided `--kv-cache-dtype={kv_cache_dtype}` but it needs to be any of `auto`, `bfloat16`, `fp8`, `fp8_ds_mla`, `fp8_e4m3`, `fp8_e5m2` or `fp8_inc`. If `--kv-cache-dtype=auto` (or unset), then the `config.json` should either contain the `torch_dtype` or `dtype` fields set; or if quantized, then `quantization_config` needs to be set and contain the key `quant_method` with value `fp8` (as none of `fp32`, `fp16` or `bf16` is considered within the `quantization_config`), and optionally also contain `fmt` set to any valid FP8 format as `float8_e4m3` or `float8_e4m3fn`."
    )


def compute_safetensors_kv_cache_size(
    config: Dict[str, Any],
    cache_dtype: str,
    max_model_len: int,
    batch_size: int = 1,
) -> int:
    """Compute the KV cache memory requirement in bytes for a Safetensors model.

    Reference: https://gist.github.com/alvarobartt/1097ca1b07c66fd71470937d599c2072
    """
    hidden_size: int = config["hidden_size"]
    num_attention_heads: int = config["num_attention_heads"]

    # NOTE: `num_key_value_heads` defaults to `num_attention_heads` in MHA, and is explicitly
    # set to a smaller value in GQA / MQA
    num_key_value_heads: int = config.get("num_key_value_heads", num_attention_heads)

    # NOTE: Use head_dim directly if specified in the config; some models (e.g. Qwen3) set
    # hidden_size and num_attention_heads independently from the actual per-head size,
    # making the fallback `hidden_size // num_attention_heads` incorrect for those models
    head_dim: int = config.get("head_dim", hidden_size // num_attention_heads)

    # NOTE: For hybrid attention models only full-attention layers grow with max_model_len;
    # sliding-window layers reuse a fixed-size buffer and are excluded from the estimate
    num_attention_layers = _resolve_num_attention_layers(config)

    return (
        # NOTE: 2 because it applies to both key and value projections
        2
        * num_attention_layers
        * num_key_value_heads
        * head_dim
        * max_model_len
        * get_safetensors_dtype_bytes(cache_dtype)
        * batch_size
    )
