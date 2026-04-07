import json
from importlib import resources
from typing import Any, Dict

from hf_mem.hardware.types import GPU, HardwareProfile

_GiB = 1024**3


def _load_catalog() -> Dict[str, HardwareProfile]:
    catalog_path = resources.files("hf_mem.hardware").joinpath("catalog.json")
    raw: Dict[str, Any] = json.loads(catalog_path.read_text(encoding="utf-8"))

    catalog: Dict[str, HardwareProfile] = {}
    for key, entry in raw.items():
        gpus = [GPU(name=g["name"], vram_bytes=int(g["vram_gb"] * _GiB)) for g in entry["gpus"]]
        catalog[key] = HardwareProfile(
            name=entry["name"],
            gpus=gpus,
            source=entry.get("source", "catalog"),
            notes=entry.get("notes"),
        )
    return catalog


CATALOG: Dict[str, HardwareProfile] = _load_catalog()


def get_profile(key: str) -> HardwareProfile | None:
    """Look up a hardware profile by exact key, then prefix, then substring."""
    normalized = key.strip().lower()

    # Exact match
    if normalized in CATALOG:
        return CATALOG[normalized]

    # Prefix match
    for catalog_key, profile in CATALOG.items():
        if catalog_key.startswith(normalized):
            return profile

    # Segment match: input must match a full segment (split by - or space)
    # e.g. "h100" matches "h100-80gb" but "8gb" does NOT match "a40-48gb"
    for catalog_key, profile in CATALOG.items():
        key_segments = catalog_key.split("-")
        name_segments = profile.name.lower().split()
        if normalized in key_segments or normalized in name_segments:
            return profile

    return None


def list_profiles() -> Dict[str, HardwareProfile]:
    return CATALOG
