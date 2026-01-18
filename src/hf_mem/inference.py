"""KV cache and inference memory estimation."""

from typing import Optional

from hf_mem.metadata import InferenceEstimate, KVCacheEstimate, ModelConfig

# Mapping of torch dtype strings to bytes per element
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
}


def get_dtype_bytes(torch_dtype: Optional[str]) -> int:
    """Get bytes per element for a torch dtype string."""
    if torch_dtype is None:
        return 2  # Default to float16
    # Handle both "torch.float16" and "float16" formats
    dtype = torch_dtype.replace("torch.", "")
    return DTYPE_BYTES.get(dtype, 2)


def calculate_kv_cache(
    model_config: ModelConfig,
    batch_size: int,
    context_length: int,
    concurrent_requests: int,
) -> KVCacheEstimate:
    """Calculate KV cache memory requirements.

    KV Cache Formula:
    kv_bytes = 2 x num_layers x head_dim x num_kv_heads x seq_len x batch x concurrent x dtype_bytes

    Where:
    - 2 = Key + Value tensors
    - num_kv_heads accounts for GQA/MQA (< num_attention_heads)
    """
    dtype_bytes = get_dtype_bytes(model_config.torch_dtype)

    # Per-token memory: 2 (K+V) x layers x head_dim x kv_heads x dtype_bytes
    per_token_bytes = (
        2
        * model_config.num_hidden_layers
        * model_config.head_dim
        * model_config.num_key_value_heads
        * dtype_bytes
    )

    # Per-request memory: per_token x context_length x batch_size
    per_request_bytes = per_token_bytes * context_length * batch_size

    # Total memory: per_request x concurrent_requests
    total_bytes = per_request_bytes * concurrent_requests

    return KVCacheEstimate(
        context_length=context_length,
        batch_size=batch_size,
        concurrent_requests=concurrent_requests,
        per_token_bytes=per_token_bytes,
        per_request_bytes=per_request_bytes,
        total_bytes=total_bytes,
    )


def calculate_inference_estimate(
    weights_bytes: int,
    model_config: Optional[ModelConfig],
    batch_size: int,
    context_length: int,
    concurrent_requests: int,
) -> InferenceEstimate:
    """Combine weights + KV cache for total serving memory.

    If model_config is None (non-transformer model), KV cache is skipped.
    """
    kv_cache = None
    total_bytes = weights_bytes

    if model_config is not None:
        kv_cache = calculate_kv_cache(
            model_config=model_config,
            batch_size=batch_size,
            context_length=context_length,
            concurrent_requests=concurrent_requests,
        )
        total_bytes = weights_bytes + kv_cache.total_bytes

    return InferenceEstimate(
        weights_bytes=weights_bytes,
        kv_cache=kv_cache,
        total_bytes=total_bytes,
        model_config=model_config,
    )
