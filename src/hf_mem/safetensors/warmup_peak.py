import warnings
from typing import Any, Dict, List, Literal

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


def resolve_activation_dtype(config: Dict[str, Any]) -> str | None:
    """Resolve the activation (compute) dtype for the warmup forward pass.

    Returns a Safetensors dtype string (e.g. "BF16", "F16"), or None when it cannot
    be determined. Caller should warn and skip warmup peak when None is returned.

    Resolution order:
      1. torch_dtype / dtype, if it is a non-quantized dtype.
      2. If torch_dtype / dtype resolves to a quantized dtype (float8_*), assume bf16
         as the dequantized compute dtype.
      3. If quantization_config is present but no torch_dtype/dtype, assume bf16.
      4. Otherwise None.
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

    return None


def _activation_dtype_bytes(dtype: str) -> int:
    return get_safetensors_dtype_bytes(dtype)


def _resolve_intermediate_size(config: Dict[str, Any]) -> int | None:
    for key in ("intermediate_size", "ffn_dim", "n_inner"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _resolve_num_labels(config: Dict[str, Any]) -> int | None:
    value = config.get("num_labels")
    if isinstance(value, int) and value > 0:
        return value
    id2label = config.get("id2label")
    if isinstance(id2label, dict) and len(id2label) > 0:
        return len(id2label)
    return None
