from dataclasses import dataclass


@dataclass
class KvCache:
    max_model_len: int
    cache_size: int
    batch_size: int
    cache_dtype: str | None = None


@dataclass
class WarmupPeak:
    max_num_batched_tokens: int
    max_num_seqs: int
    peak_bytes: int
    activation_dtype: str | None = None
