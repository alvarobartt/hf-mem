from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GPU:
    name: str
    vram_bytes: int


@dataclass
class HardwareProfile:
    name: str
    gpus: List[GPU]
    source: str
    total_vram_bytes: int = 0
    system_ram_bytes: int | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.total_vram_bytes == 0:
            self.total_vram_bytes = sum(g.vram_bytes for g in self.gpus)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "total_vram_bytes": self.total_vram_bytes,
            "gpu_count": len(self.gpus),
            "gpus": [{"name": g.name, "vram_bytes": g.vram_bytes} for g in self.gpus],
        }
        if self.system_ram_bytes is not None:
            out["system_ram_bytes"] = self.system_ram_bytes
        if self.notes is not None:
            out["notes"] = self.notes
        return out


@dataclass
class QuantizationEstimate:
    name: str
    bits_per_weight: float
    estimated_bytes: int
    estimated_total_bytes: int
    fits: bool
    gpu_count_needed: int


@dataclass
class FitnessResult:
    profile: HardwareProfile
    model_memory: int
    total_memory: int
    fits: bool
    headroom_bytes: int
    headroom_pct: float
    gpu_count_needed: int
    param_count: int | None
    quantization_options: List[QuantizationEstimate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_json(),
            "model_memory": self.model_memory,
            "total_memory": self.total_memory,
            "fits": self.fits,
            "headroom_bytes": self.headroom_bytes,
            "headroom_pct": round(self.headroom_pct, 4),
            "gpu_count_needed": self.gpu_count_needed,
            "param_count": self.param_count,
            "quantization_options": [
                {
                    "name": q.name,
                    "bits_per_weight": q.bits_per_weight,
                    "estimated_bytes": q.estimated_bytes,
                    "estimated_total_bytes": q.estimated_total_bytes,
                    "fits": q.fits,
                    "gpu_count_needed": q.gpu_count_needed,
                }
                for q in self.quantization_options
            ],
            "notes": self.notes,
        }
