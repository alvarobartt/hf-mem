import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from hf_mem.types import SafetensorsDtypes, get_safetensors_dtype_bytes


@dataclass
class ModelConfig:
    """Transformer architecture from config.json."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int  # For GQA; defaults to num_attention_heads
    max_position_embeddings: int
    head_dim: int  # hidden_size // num_attention_heads
    torch_dtype: Optional[str] = None


@dataclass
class KVCacheEstimate:
    """KV cache memory for specific parameters."""

    context_length: int
    batch_size: int
    concurrent_requests: int
    per_token_bytes: int
    per_request_bytes: int
    total_bytes: int


@dataclass
class InferenceEstimate:
    """Complete inference memory breakdown."""

    weights_bytes: int
    kv_cache: Optional[KVCacheEstimate]
    total_bytes: int
    model_config: Optional[ModelConfig]


@dataclass
class DtypeMetadata:
    param_count: int
    bytes_count: int


@dataclass
class ComponentMetadata:
    dtypes: Dict[SafetensorsDtypes, DtypeMetadata]
    param_count: int
    bytes_count: int


@dataclass
class SafetensorsMetadata:
    components: Dict[str, ComponentMetadata]
    param_count: int
    bytes_count: int


def parse_safetensors_metadata(
    raw_metadata: Dict[str, Dict[str, Any]],
) -> SafetensorsMetadata:
    components = {}
    total_param_count, total_bytes_count = 0, 0

    for name, metadata in raw_metadata.items():
        component = ComponentMetadata(dtypes={}, param_count=0, bytes_count=0)
        for key, value in metadata.items():
            if key in {"__metadata__"}:
                continue

            dtype = value["dtype"]
            if dtype not in component.dtypes:
                component.dtypes[dtype] = DtypeMetadata(param_count=0, bytes_count=0)

            dtype_bytes = get_safetensors_dtype_bytes(dtype)
            current_shape = math.prod(value["shape"])
            current_shape_bytes = current_shape * dtype_bytes

            component.dtypes[dtype].param_count += current_shape
            component.dtypes[dtype].bytes_count += current_shape_bytes
            component.param_count += current_shape
            component.bytes_count += current_shape_bytes
            total_param_count += current_shape
            total_bytes_count += current_shape_bytes

        components[name] = component

    return SafetensorsMetadata(
        components=components,
        param_count=total_param_count,
        bytes_count=total_bytes_count,
    )
