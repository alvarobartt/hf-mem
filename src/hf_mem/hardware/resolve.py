import re

from hf_mem.hardware.catalog import get_profile
from hf_mem.hardware.custom import load_custom_profile
from hf_mem.hardware.detect import detect_local_gpus
from hf_mem.hardware.types import GPU, HardwareProfile

_GiB = 1024**3

# Matches "4x a100-80gb" or "4xa100-80gb"
_MULTI_GPU_RE = re.compile(r"^(\d+)\s*x\s*(.+)$", re.IGNORECASE)

# Matches bare VRAM like "24gb", "24gib", "24g", or just "24"
_VRAM_RE = re.compile(r"^(\d+)\s*(gib|gb|g)?$", re.IGNORECASE)


def resolve_hardware(
    hardware: str | None = None,
    hardware_file: str | None = None,
) -> HardwareProfile | None:
    """Resolve a --hardware or --hardware-file argument into a HardwareProfile."""
    if hardware_file is not None:
        return load_custom_profile(hardware_file)

    if hardware is None:
        return None

    value = hardware.strip()
    lower = value.lower()

    # 1. "local" or "auto" -> detect GPUs
    if lower in ("local", "auto"):
        return detect_local_gpus()

    # 2. "4x a100-80gb" -> multiply a single-GPU catalog entry
    if m := _MULTI_GPU_RE.match(lower):
        count = int(m.group(1))
        base_key = m.group(2).strip()
        base = get_profile(base_key)
        if base is None:
            raise RuntimeError(
                f"Unknown hardware profile: '{base_key}'. "
                "Run `hf-mem --list-hardware` to see available profiles."
            )
        if len(base.gpus) != 1:
            raise RuntimeError(
                f"Multiplier syntax only works with single-GPU profiles, "
                f"but '{base_key}' has {len(base.gpus)} GPUs."
            )
        return HardwareProfile(
            name=f"{count}x {base.name}",
            gpus=base.gpus * count,
            source=base.source,
            notes=base.notes,
        )

    # 3. Direct catalog lookup (exact, prefix, substring)
    profile = get_profile(lower)
    if profile is not None:
        return profile

    # 4. Bare VRAM: "24", "24gb", "24gib", "24g"
    if m := _VRAM_RE.match(lower):
        vram_val = int(m.group(1))
        unit = (m.group(2) or "gib").lower()
        if unit == "gb":
            vram_bytes = vram_val * 1000**3
        else:  # gib, g, or bare number
            vram_bytes = vram_val * _GiB
        return HardwareProfile(
            name=f"Custom GPU ({vram_val} {'GB' if unit == 'gb' else 'GiB'})",
            gpus=[GPU(name="Custom", vram_bytes=vram_bytes)],
            source="custom",
        )

    raise RuntimeError(
        f"Unknown hardware profile: '{hardware}'. "
        "Use '--hardware local' for auto-detection, a catalog name like 'a100-80gb', "
        "a cloud instance like 'aws:p4d.24xlarge', or '--hardware-file config.json'. "
        "Run `hf-mem --list-hardware` to see all profiles."
    )
