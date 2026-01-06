import math
from dataclasses import dataclass
from typing import Any, Dict

from typing_extensions import Self

from hf_mem.types import get_safetensors_dtype_bytes


@dataclass
class SafetensorsMetadata:
    components: Dict[str, Any]
    param_count: int
    bytes_count: int

    def to_dict(self: Self) -> Dict[str, Any]:
        return {
            "components": self.components,
            "param_count": self.param_count,
            "bytes_count": self.bytes_count,
        }


def parse_safetensors_metadata(raw_metadata: Dict[str, Any]) -> SafetensorsMetadata:
    # NOTE: This is a small "hack" to prevent from having dedicated parsing functions for Transformers, Sentence
    # Transformers and Diffusers, and rather unify within the same function
    if "__metadata__" in raw_metadata:
        raw_metadata = {"transformer": raw_metadata}

    components = {}
    param_count, bytes_count = 0, 0

    for component_name, component_metadata in raw_metadata.items():
        component = {"param_count": 0, "bytes_count": 0, "dtypes": {}}
        for key, value in component_metadata.items():
            if key in {"__metadata__"}:
                continue

            dtype = value["dtype"]
            if dtype not in component["dtypes"]:
                component["dtypes"][dtype] = {"param_count": 0, "bytes_count": 0}

            dtype_bytes = get_safetensors_dtype_bytes(dtype)
            current_shape = math.prod(value["shape"])
            current_shape_bytes = current_shape * dtype_bytes

            component["dtypes"][dtype]["param_count"] += current_shape
            component["param_count"] += current_shape
            param_count += current_shape

            component["dtypes"][dtype]["bytes_count"] += current_shape_bytes
            component["bytes_count"] += current_shape
            bytes_count += current_shape_bytes

        if component:
            components[component_name] = component

    return SafetensorsMetadata(
        components=components,
        param_count=param_count,
        bytes_count=bytes_count,
    )
