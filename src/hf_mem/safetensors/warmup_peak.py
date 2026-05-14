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
    "question_answering",
    "base_encoder",
]

# NOTE: Architecture-suffix → ModelClass; matched via substring, mirroring how
# run.py already detects ForCausalLM / ForConditionalGeneration
_ARCH_SUFFIX_MAP: list[tuple[str, ModelClass]] = [
    ("ForCausalLM", "causal_lm"),
    ("ForConditionalGeneration", "conditional_gen"),
    ("ForMaskedLM", "masked_lm"),
    ("ForPreTraining", "masked_lm"),
    ("ForSequenceClassification", "seq_classification"),
    ("ForMultipleChoice", "seq_classification"),
    # NOTE: QA heads produce per-token start/end logits (T × 2), not a pooled
    # (S × num_labels) spike; structurally fixed at 2 outputs (start, end)
    ("ForQuestionAnswering", "question_answering"),
    ("ForTokenClassification", "token_classification"),
]

# NOTE: Safetensors dtypes that are quantized weight storage formats; for the warmup
# forward pass the activations are dequantized to a higher-precision compute dtype
_QUANTIZED_SAFETENSORS_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E8M0", "I8"}


def classify_architecture(architectures: List[str], file_paths: List[str]) -> ModelClass | None:
    for arch in architectures or []:
        for suffix, model_class in _ARCH_SUFFIX_MAP:
            if suffix in arch:
                return model_class

    # NOTE: Sentence Transformers repos are identified by `config_sentence_transformers.json`
    if "config_sentence_transformers.json" in file_paths:
        return "base_encoder"

    # NOTE: Bare `…Model` architectures (e.g. BertModel) are encoder-only
    for arch in architectures or []:
        if arch.endswith("Model") and "ForCausalLM" not in arch and "ForConditionalGeneration" not in arch:
            return "base_encoder"

    return None


def resolve_activation_dtype(config: Dict[str, Any]) -> str:
    # NOTE: Falls back to F32 (HuggingFace Transformers' default when `torch_dtype` / `dtype`
    # is absent) for older BERT-family / Sentence Transformers models. For quantized weights,
    # assume BF16 as the dequantized compute dtype.
    if raw := (config.get("torch_dtype") or config.get("dtype")):
        if isinstance(raw, str):
            resolved = torch_dtype_to_safetensors_dtype(raw)
            if resolved in _QUANTIZED_SAFETENSORS_DTYPES:
                return "BF16"
            return resolved

    if "quantization_config" in config:
        return "BF16"

    return "F32"


def _resolve_num_labels(config: Dict[str, Any], model_id: str) -> int | None:
    value = config.get("num_labels")
    if isinstance(value, int) and value > 0:
        return value
    id2label = config.get("id2label")
    if isinstance(id2label, dict) and len(id2label) > 0:
        return len(id2label)
    warnings.warn(
        f"Warmup peak estimate skipped for `--model-id={model_id}`: classifier "
        f"architecture without `num_labels` or `id2label` in `config.json`."
    )
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
    # NOTE: Models PyTorch's caching-allocator behavior: a persistent residual stream
    # of (T × H × d_act), plus the largest transient (attention / FFN / head) at any
    # one point in the forward. Returns None (with a warning) when the architecture
    # is unsupported or required config keys are missing.
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

    # NOTE: Terminal head spike — applied to every token for masked LM / token
    # classification, only the last token of each sequence (S) for the rest.
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
            lm_head_spike = T * vocab_size * d
        case "seq_classification":
            num_labels = _resolve_num_labels(config, model_id)
            if num_labels is None:
                return None
            lm_head_spike = S * num_labels * d
        case "token_classification":
            num_labels = _resolve_num_labels(config, model_id)
            if num_labels is None:
                return None
            lm_head_spike = T * num_labels * d
        case "question_answering":
            lm_head_spike = T * 2 * d
        case "base_encoder":
            lm_head_spike = S * hidden_size * d

    peak_bytes = residual_stream + max(attn_transient, ffn_transient, lm_head_spike)

    return WarmupPeak(
        max_num_batched_tokens=T,
        max_num_seqs=S,
        peak_bytes=peak_bytes,
        activation_dtype=activation_dtype,
    )
