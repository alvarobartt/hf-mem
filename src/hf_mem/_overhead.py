import math
from dataclasses import dataclass
from typing import Any

GIB = 1 << 30
SAMPLER_ROWS = 1024
LARGE_GPU = 70 * GIB


@dataclass
class Overhead:
    activations: int
    cuda_graphs: int
    runtime: int

    @property
    def bytes(self) -> int:
        return self.activations + self.cuda_graphs + self.runtime

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "activations": self.activations,
            "cuda_graphs": self.cuda_graphs,
            "runtime": self.runtime,
        }


def batched_tokens(gpu_memory: str | None) -> int:
    return 2048 if gpu_memory is not None and parse_gpu_memory(gpu_memory)[1] < LARGE_GPU else 8192


def vllm(config: dict[str, Any], moe: bool, tokens: int) -> Overhead:
    hidden = config.get("hidden_size") or 0
    vocab = config.get("vocab_size") or 32000
    intermediate = config.get("intermediate_size") or config.get("ffn_hidden_size") or 4 * hidden
    if moe and not config.get("intermediate_size"):
        intermediate = (config.get("moe_intermediate_size") or 0) * (config.get("num_experts_per_tok") or 1)
    activations = SAMPLER_ROWS * vocab * 8 + tokens * intermediate * 16
    return Overhead(activations, int((0.36 if moe else 0.25) * GIB), int(0.75 * GIB))


def llama_cpp() -> Overhead:
    return Overhead(int(0.15 * GIB), 0, int(0.47 * GIB))


@dataclass
class Fit:
    gpu_memory: int
    gpus: int | None
    needed: int | None
    max_model_len: int

    @property
    def fits(self) -> bool:
        return self.needed is not None and self.needed <= (self.gpus or 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_memory": self.gpu_memory,
            "gpus": self.gpus,
            "needed": self.needed,
            "fits": self.fits,
            "max_model_len": self.max_model_len,
        }


def parse_gpu_memory(value: str) -> tuple[int | None, int]:
    count, sep, size = value.lower().removesuffix("gib").removesuffix("gb").rpartition("x")
    try:
        gpus, memory = (int(count) if sep else None), int(float(size) * GIB)
    except ValueError:
        gpus = memory = 0
    if gpus == 0 or memory <= 0:
        raise RuntimeError(f"`--gpu-memory={value}` must look like `80` or `8x80` (GiB per GPU).")
    return gpus, memory


def fit(
    weights: int,
    kv_fixed: int,
    kv_per_token: float,
    max_model_len: int,
    context_length: int,
    overhead: int,
    gpu_memory: str,
    utilization: float,
    kv_heads: int | None = None,
) -> Fit:
    gpus, size = parse_gpu_memory(gpu_memory)
    usable = size * utilization - overhead

    def per_token(n: int) -> float:
        return kv_per_token * (max(1.0, n / kv_heads) if kv_heads else 1.0)

    needed = next(
        (
            n
            for n in (1, 2, 4, 8, 16, 32, 64)
            if weights + kv_fixed + per_token(n) * max_model_len <= n * usable
        ),
        None,
    )
    n = gpus or 1
    budget = n * usable - weights - kv_fixed
    longest = math.floor(budget / per_token(n)) if kv_per_token > 0 and budget > 0 else 0
    return Fit(size, gpus, needed, min(longest, context_length))
