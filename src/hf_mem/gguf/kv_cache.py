from typing import Any, Dict

from hf_mem.gguf.types import GGUFDtype, GGUFDtypeBitsPerWeight

# NOTE: KV metadata field suffixes used to extract the fields needed for KV cache estimation;
# matched by suffix so they work across all model families (e.g. "llama.block_count" -> "block_count")
KV_CACHE_FIELD_ENDINGS = ["block_count", "head_count_kv", "head_count", "embedding_length", "context_length"]


def compute_gguf_kv_cache_size(
    kv_metadata: Dict[str, Any], kv_cache_dtype: str = "F16", batch_size: int = 1
) -> int:
    block_count = kv_metadata["block_count"]
    head_count_kv = kv_metadata["head_count_kv"]
    head_count = kv_metadata["head_count"]
    embedding_length = kv_metadata["embedding_length"]
    context_length = kv_metadata["context_length"]

    if not all(
        isinstance(v, int) for v in [block_count, head_count_kv, head_count, embedding_length, context_length]
    ):
        raise RuntimeError("KV cache metadata fields must be integers for GGUF KV cache size estimation.")

    if kv_cache_dtype not in GGUFDtype.__members__:
        raise RuntimeError(
            f"--kv-cache-dtype={kv_cache_dtype} not recognized for GGUF KV cache size estimation. Valid options: {list(GGUFDtype.__members__.keys())}."
        )

    # NOTE: 2 because it applies to both key and value projections; head_dim = embedding_length // head_count
    size_per_token = 2 * head_count_kv * (embedding_length // head_count)
    return int(
        block_count
        * size_per_token
        * context_length
        * batch_size
        * GGUFDtypeBitsPerWeight[GGUFDtype[kv_cache_dtype]]
        / 8.0
    )
