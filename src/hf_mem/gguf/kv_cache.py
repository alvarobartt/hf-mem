from typing import Any, Dict

from hf_mem.gguf.types import GGUFDtype, GGUFDtypeBitsPerWeight


def compute_gguf_kv_cache_size(
    kv_metadata: Dict[str, Any],
    max_model_len: int,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
) -> int | None:
    arch = kv_metadata.get("general.architecture")
    block_count = kv_metadata.get(f"{arch}.block_count")
    head_count = kv_metadata.get(f"{arch}.attention.head_count")
    if not isinstance(block_count, int) or not head_count:
        return None

    if kv_cache_dtype not in GGUFDtype.__members__:
        raise RuntimeError(
            f"--kv-cache-dtype={kv_cache_dtype} not recognized for GGUF KV cache size estimation. Valid options: {list(GGUFDtype.__members__.keys())}."
        )

    if isinstance(head_count, list):
        head_count = max(head_count)
    head_count_kv = kv_metadata.get(f"{arch}.attention.head_count_kv", head_count)
    if not isinstance(head_count_kv, list):
        head_count_kv = [head_count_kv] * block_count
        if interval := kv_metadata.get(f"{arch}.full_attention_interval"):
            nextn = kv_metadata.get(f"{arch}.nextn_predict_layers", 0)
            head_count_kv = [
                h if i >= block_count - nextn or (i + 1) % interval == 0 else 0
                for i, h in enumerate(head_count_kv)
            ]

    if kv_lora_rank := kv_metadata.get(f"{arch}.attention.kv_lora_rank"):
        rope_dim = kv_metadata.get(f"{arch}.rope.dimension_count", 0)
        width_per_layer = [(kv_lora_rank + rope_dim) if h else 0 for h in head_count_kv]
    else:
        embedding_length = kv_metadata.get(f"{arch}.embedding_length", 0)
        key_length = kv_metadata.get(f"{arch}.attention.key_length", embedding_length // head_count)
        value_length = kv_metadata.get(f"{arch}.attention.value_length", key_length)
        width_per_layer = [h * (key_length + value_length) for h in head_count_kv]

    return int(
        sum(width_per_layer)
        * max_model_len
        * batch_size
        * GGUFDtypeBitsPerWeight[GGUFDtype[kv_cache_dtype]]
        / 8.0
    )
