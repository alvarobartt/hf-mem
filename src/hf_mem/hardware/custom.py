import json
from typing import Any, Dict

from hf_mem.hardware.types import GPU, HardwareProfile

_GiB = 1024**3


def load_custom_profile(path: str) -> HardwareProfile:
    """Load a hardware profile from a JSON file.

    Expected schema:
        {
            "name": "Our Slurm node",
            "gpus": [
                {"name": "A100-SXM4", "vram_gb": 80},
                {"name": "A100-SXM4", "vram_gb": 80}
            ],
            "system_ram_gb": 512   // optional
        }
    """
    with open(path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    if "gpus" not in data or not isinstance(data["gpus"], list) or len(data["gpus"]) == 0:
        raise RuntimeError(f"Hardware config at '{path}' must have a non-empty 'gpus' list.")

    gpus = []
    for i, g in enumerate(data["gpus"]):
        if "name" not in g or "vram_gb" not in g:
            raise RuntimeError(f"GPU entry {i} in '{path}' must have 'name' and 'vram_gb' fields.")
        gpus.append(GPU(name=g["name"], vram_bytes=int(g["vram_gb"] * _GiB)))

    return HardwareProfile(
        name=data.get("name", path),
        gpus=gpus,
        source="custom",
        system_ram_bytes=int(data["system_ram_gb"] * _GiB) if "system_ram_gb" in data else None,
    )
