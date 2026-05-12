import warnings
from typing import Any, Dict, List, Literal

from hf_mem._types import WarmupPeak
from hf_mem.safetensors.metadata import _get_config_int
from hf_mem.safetensors.types import get_safetensors_dtype_bytes, torch_dtype_to_safetensors_dtype

ModelClass = Literal[
    "causal_lm",
    "conditional_gen",
    "masked_lm",
    "seq_classification",
    "token_classification",
    "base_encoder",
]

# Architecture-suffix → ModelClass. Matching is via `__contains__` (substring),
# mirroring how run.py already detects ForCausalLM / ForConditionalGeneration.
_ARCH_SUFFIX_MAP: list[tuple[str, ModelClass]] = [
    # Order matters: more specific suffixes must come before less specific ones
    # so that e.g. `BertForSequenceClassification` matches before `BertModel` would
    # (it never does here since we only match the listed suffixes, but keep the
    # discipline for future additions).
    ("ForCausalLM", "causal_lm"),
    ("ForConditionalGeneration", "conditional_gen"),
    ("ForMaskedLM", "masked_lm"),
    ("ForPreTraining", "masked_lm"),
    ("ForSequenceClassification", "seq_classification"),
    ("ForQuestionAnswering", "seq_classification"),
    ("ForMultipleChoice", "seq_classification"),
    ("ForTokenClassification", "token_classification"),
]


def classify_architecture(architectures: List[str], file_paths: List[str]) -> ModelClass | None:
    """Map HF `config.architectures` to a ModelClass used by the warmup peak formula.

    Returns None when no known suffix is found AND the repo is not a Sentence Transformers
    model — in that case the caller should warn and skip the warmup peak estimate.
    """
    for arch in architectures or []:
        for suffix, model_class in _ARCH_SUFFIX_MAP:
            if suffix in arch:
                return model_class

    # Sentence Transformers and bare `…Model` (e.g. BertModel used as an encoder)
    # both fall under "base_encoder". Sentence Transformers is identifiable via
    # the presence of `config_sentence_transformers.json` in the repo.
    if "config_sentence_transformers.json" in file_paths:
        return "base_encoder"

    # Bare `…Model` architectures (e.g. BertModel, RobertaModel, XLMRobertaModel) —
    # only when the architecture string ends in "Model" without a head suffix.
    for arch in architectures or []:
        if arch.endswith("Model") and "ForCausalLM" not in arch and "ForConditionalGeneration" not in arch:
            return "base_encoder"

    return None


def resolve_activation_dtype(config: Dict[str, Any]) -> str:
    """Resolve the activation (compute) dtype for the warmup forward pass.

    Returns a Safetensors dtype string (e.g. "BF16", "F16", "F32"). Never returns
    None — falls back to F32 (HuggingFace's default) when no dtype info is found.

    Resolution order:
      1. torch_dtype / dtype, if it is a non-quantized dtype.
      2. If torch_dtype / dtype resolves to a quantized dtype (float8_*), assume bf16
         as the dequantized compute dtype.
      3. If quantization_config is present but no torch_dtype/dtype, assume bf16.
      4. Otherwise F32 — HuggingFace Transformers' default compute dtype when
         torch_dtype / dtype is absent (common for older BERT-family models).
    """
    _QUANTIZED_TORCH_DTYPES = {"float8_e4m3", "float8_e4m3fn", "float8_e5m2", "int8"}

    raw = config.get("torch_dtype") or config.get("dtype")
    if isinstance(raw, str):
        normalized = raw.replace("torch.", "")
        if normalized in _QUANTIZED_TORCH_DTYPES:
            return "BF16"  # dequantized compute dtype
        return torch_dtype_to_safetensors_dtype(normalized)

    if "quantization_config" in config:
        return "BF16"

    # HuggingFace Transformers' default when torch_dtype / dtype absent is float32.
    # Many older BERT-family / Sentence Transformers models don't set the field at all.
    return "F32"


def _resolve_num_labels(config: Dict[str, Any]) -> int | None:
    value = config.get("num_labels")
    if isinstance(value, int) and value > 0:
        return value
    id2label = config.get("id2label")
    if isinstance(id2label, dict) and len(id2label) > 0:
        return len(id2label)
    return None


def _resolve_vocab_size(config: Dict[str, Any], model_id: str, label: str) -> int | None:
    vocab_size = config.get("vocab_size")
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    warnings.warn(
        f"Warmup peak estimate skipped for `--model-id={model_id}`: "
        f"`config.json` missing `vocab_size` for a {label}."
    )
    return None


def compute_safetensors_warmup_peak(
    config: Dict[str, Any],
    max_num_batched_tokens: int,
    batch_size: int,
    file_paths: List[str],
    model_id: str,
) -> WarmupPeak | None:
    """Approximate the peak activation memory during one warmup forward pass.

    Models what PyTorch's caching allocator holds: a persistent residual stream
    of (T × H × d_act), plus the largest transient at any one point in the forward.

    Returns None (with a warning) on any of:
      - architecture not in the supported taxonomy
      - required config keys missing
      - activation dtype unresolvable
      - classifier architecture without num_labels / id2label
    """
    architectures = config.get("architectures", []) or []
    model_class = classify_architecture(architectures, file_paths)
    if model_class is None:
        warnings.warn(
            f"Warmup peak estimate skipped for `--model-id={model_id}`: architecture(s) "
            f"{architectures or '<missing>'} not in the supported taxonomy (causal LM, "
            f"VLM, masked LM, classifier, base encoder / Sentence Transformers)."
        )
        return None

    activation_dtype = resolve_activation_dtype(config)

    required = ("hidden_size", "num_attention_heads")
    missing = [k for k in required if k not in config]
    if missing:
        warnings.warn(
            f"Warmup peak estimate skipped for `--model-id={model_id}`: `config.json` "
            f"missing required keys: {missing}."
        )
        return None

    intermediate_size = _get_config_int(config, "intermediate_size", "ffn_dim", "n_inner")
    if intermediate_size is None:
        warnings.warn(
            f"Warmup peak estimate skipped for `--model-id={model_id}`: could not resolve "
            f"the FFN intermediate size from `config.json` (looked for `intermediate_size`, "
            f"`ffn_dim`, `n_inner`)."
        )
        return None

    hidden_size: int = config["hidden_size"]
    num_attention_heads: int = config["num_attention_heads"]
    num_key_value_heads: int = config.get("num_key_value_heads", num_attention_heads)
    head_dim: int = config.get("head_dim", hidden_size // num_attention_heads)

    # MoE adjustment: when num_experts_per_tok > 1, the FFN transient grows linearly
    # with the active expert count.
    num_experts_per_tok = _get_config_int(config, "num_experts_per_tok", "num_experts_per_token", "top_k")
    moe_intermediate_size = config.get("moe_intermediate_size")
    if num_experts_per_tok is not None:
        ffn_width = moe_intermediate_size if isinstance(moe_intermediate_size, int) else intermediate_size
        ffn_multiplier = 2 * num_experts_per_tok
    else:
        ffn_width = intermediate_size
        ffn_multiplier = 2

    d = get_safetensors_dtype_bytes(activation_dtype)
    T = max_num_batched_tokens
    S = min(T, batch_size)

    residual_stream = T * hidden_size * d
    attn_transient = T * (num_attention_heads + 2 * num_key_value_heads) * head_dim * d
    ffn_transient = ffn_multiplier * T * ffn_width * d

    # Terminal "head" spike — model-class-dependent.
    lm_head_spike: int
    match model_class:
        case "causal_lm" | "conditional_gen":
            vocab_size = _resolve_vocab_size(config, model_id, "causal LM / VLM")
            if vocab_size is None:
                return None
            lm_head_spike = S * vocab_size * d
        case "masked_lm":
            vocab_size = _resolve_vocab_size(config, model_id, "masked LM")
            if vocab_size is None:
                return None
            lm_head_spike = T * vocab_size * d  # LM head applied to every token
        case "seq_classification":
            num_labels = _resolve_num_labels(config)
            if num_labels is None:
                warnings.warn(
                    f"Warmup peak estimate skipped for `--model-id={model_id}`: classifier "
                    f"architecture without `num_labels` or `id2label` in `config.json`."
                )
                return None
            lm_head_spike = S * num_labels * d
        case "token_classification":
            num_labels = _resolve_num_labels(config)
            if num_labels is None:
                warnings.warn(
                    f"Warmup peak estimate skipped for `--model-id={model_id}`: classifier "
                    f"architecture without `num_labels` or `id2label` in `config.json`."
                )
                return None
            lm_head_spike = T * num_labels * d
        case "base_encoder":
            lm_head_spike = S * hidden_size * d

    peak_bytes = residual_stream + max(attn_transient, ffn_transient, lm_head_spike)

    return WarmupPeak(
        max_num_batched_tokens=T,
        max_num_seqs=S,
        peak_bytes=peak_bytes,
        activation_dtype=activation_dtype,
    )
