from dataclasses import dataclass


@dataclass
class KvCache:
    max_model_len: int
    cache_size: int
    batch_size: int
    cache_dtype: str | None = None
