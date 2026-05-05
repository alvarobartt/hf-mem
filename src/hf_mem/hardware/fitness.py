import math
from typing import List

from hf_mem.gguf.types import GGUFDtype, GGUFDtypeBitsPerWeight
from hf_mem.hardware.types import FitnessResult, HardwareProfile, QuantizationEstimate
from hf_mem.run import Result

_QUANT_LEVELS: List[tuple[str, float]] = [
    ("F16", GGUFDtypeBitsPerWeight[GGUFDtype.F16]),
    ("BF16", GGUFDtypeBitsPerWeight[GGUFDtype.BF16]),
    ("Q8_K", GGUFDtypeBitsPerWeight[GGUFDtype.Q8_K]),
    ("Q6_K", GGUFDtypeBitsPerWeight[GGUFDtype.Q6_K]),
    ("Q5_K", GGUFDtypeBitsPerWeight[GGUFDtype.Q5_K]),
    ("Q4_K", GGUFDtypeBitsPerWeight[GGUFDtype.Q4_K]),
    ("Q3_K", GGUFDtypeBitsPerWeight[GGUFDtype.Q3_K]),
    ("Q2_K", GGUFDtypeBitsPerWeight[GGUFDtype.Q2_K]),
    ("IQ4_XS", GGUFDtypeBitsPerWeight[GGUFDtype.IQ4_XS]),
    ("IQ3_S", GGUFDtypeBitsPerWeight[GGUFDtype.IQ3_S]),
    ("IQ2_XS", GGUFDtypeBitsPerWeight[GGUFDtype.IQ2_XS]),
]


def check_fitness(result: Result, profile: HardwareProfile) -> FitnessResult:
    """Check whether a model estimation fits on the given hardware profile."""
    if result.total_memory is None:
        raise RuntimeError(
            "Fitness check requires a single-file memory estimate (`total_memory` must not be None). "
            "Use `--gguf-file` to select a specific GGUF file."
        )

    total_vram = profile.total_vram_bytes
    total_needed = result.total_memory
    model_memory = result.memory if isinstance(result.memory, int) else 0

    fits = total_needed <= total_vram
    headroom = total_vram - total_needed
    headroom_pct = headroom / total_vram if total_vram > 0 else 0.0

    # How many GPUs of this type would be needed?
    per_gpu_vram = min(g.vram_bytes for g in profile.gpus) if profile.gpus else 0
    gpu_count_needed = math.ceil(total_needed / per_gpu_vram) if per_gpu_vram > 0 else 0

    param_count: int | None = result.param_count if isinstance(result.param_count, int) else None

    kv_cache_bytes = 0
    if isinstance(result.kv_cache, int):
        kv_cache_bytes = result.kv_cache

    quant_options: List[QuantizationEstimate] = []
    if param_count is not None:
        for name, bits in _QUANT_LEVELS:
            est_bytes = int(param_count * bits / 8.0)
            est_total = est_bytes + kv_cache_bytes
            q_fits = est_total <= total_vram
            q_gpus = math.ceil(est_total / per_gpu_vram) if per_gpu_vram > 0 else 0
            quant_options.append(
                QuantizationEstimate(
                    name=name,
                    bits_per_weight=bits,
                    estimated_bytes=est_bytes,
                    estimated_total_bytes=est_total,
                    fits=q_fits,
                    gpu_count_needed=q_gpus,
                )
            )

    notes: List[str] = []
    if gpu_count_needed > len(profile.gpus):
        if quant_options:
            # First quant that fits (highest quality, list is sorted bits desc)
            first_fit = next((q for q in quant_options if q.gpu_count_needed <= len(profile.gpus)), None)
            if first_fit is not None:
                notes.append(
                    f"Does not fit at full precision ({gpu_count_needed} GPUs). {first_fit.name} or below fits."
                )
            else:
                # Even most aggressive quant doesn't fit
                most_aggressive = min(quant_options, key=lambda q: q.estimated_total_bytes)
                notes.append(
                    f"Needs {gpu_count_needed} GPUs at full precision, "
                    f"{most_aggressive.gpu_count_needed} at {most_aggressive.name}. "
                    f"Profile has {len(profile.gpus)}."
                )
        else:
            notes.append(f"Needs {gpu_count_needed} GPUs but profile has {len(profile.gpus)}.")
    elif gpu_count_needed > 1:
        notes.append(f"Requires tensor parallelism across {gpu_count_needed} GPUs.")

    if fits and 0 <= headroom_pct < 0.10:
        notes.append("Tight fit (<10% headroom). May OOM under load.")

    return FitnessResult(
        profile=profile,
        model_memory=model_memory,
        total_memory=total_needed,
        fits=fits,
        headroom_bytes=headroom,
        headroom_pct=headroom_pct,
        gpu_count_needed=gpu_count_needed,
        param_count=param_count,
        quantization_options=quant_options,
        notes=notes,
    )
