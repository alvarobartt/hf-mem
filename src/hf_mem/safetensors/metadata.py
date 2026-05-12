import math
from dataclasses import dataclass
from typing import Any, Dict

from hf_mem.safetensors.types import SafetensorsDtypes, get_safetensors_dtype_bytes


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


@dataclass
class MoEMetadata:
    base_model: ComponentMetadata
    experts: Dict[int, ComponentMetadata]
    expert_count: int
    active_expert_count: int | None
    expert_param_count: int
    expert_bytes_count: int
    expert_template: ComponentMetadata
    expert_display_template: ComponentMetadata


def _accumulate_tensor(
    component: ComponentMetadata,
    *,
    dtype: SafetensorsDtypes | str,
    shape: Any,
) -> tuple[int, int]:
    if dtype not in component.dtypes:
        component.dtypes[dtype] = DtypeMetadata(param_count=0, bytes_count=0)

    dtype_bytes = get_safetensors_dtype_bytes(dtype)
    current_shape = math.prod(shape)
    current_shape_bytes = current_shape * dtype_bytes

    component.dtypes[dtype].param_count += current_shape
    component.dtypes[dtype].bytes_count += current_shape_bytes
    component.param_count += current_shape
    component.bytes_count += current_shape_bytes
    return current_shape, current_shape_bytes


def _extract_expert_id(tensor_name: str) -> int | None:
    expert_tokens = {
        "expert",
        "experts",
        "local_expert",
        "local_experts",
        "routed_expert",
        "routed_experts",
    }
    parts = tensor_name.split(".")
    for idx, part in enumerate(parts):
        if part in expert_tokens and idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])

    return None


def _get_config_int(config: Dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 1:
            return value
    return None


def _components_match(component1: ComponentMetadata, component2: ComponentMetadata) -> bool:
    if component1.param_count != component2.param_count or component1.bytes_count != component2.bytes_count:
        return False
    if component1.dtypes.keys() != component2.dtypes.keys():
        return False
    return all(
        component1.dtypes[dtype].param_count == component2.dtypes[dtype].param_count
        and component1.dtypes[dtype].bytes_count == component2.dtypes[dtype].bytes_count
        for dtype in component1.dtypes
    )


def _is_expert_weight_tensor(tensor_name: str) -> bool:
    return tensor_name.endswith(".weight")


def parse_moe_metadata(
    raw_metadata: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> MoEMetadata | None:
    configured_expert_count = _get_config_int(
        config,
        "num_local_experts",
        "n_routed_experts",
        "num_experts",
        "moe_num_experts",
    )
    active_expert_count = _get_config_int(
        config,
        "num_experts_per_tok",
        "num_experts_per_token",
        "top_k",
    )

    base_model = ComponentMetadata(dtypes={}, param_count=0, bytes_count=0)
    experts: Dict[int, ComponentMetadata] = {}
    expert_display_components: Dict[int, ComponentMetadata] = {}

    for metadata in raw_metadata.values():
        for tensor_name, value in metadata.items():
            if tensor_name == "__metadata__":
                continue

            expert_id = _extract_expert_id(tensor_name)
            target = base_model
            if expert_id is not None:
                if expert_id not in experts:
                    experts[expert_id] = ComponentMetadata(dtypes={}, param_count=0, bytes_count=0)
                    expert_display_components[expert_id] = ComponentMetadata(
                        dtypes={}, param_count=0, bytes_count=0
                    )
                target = experts[expert_id]

            _accumulate_tensor(
                target,
                dtype=value["dtype"],
                shape=value["shape"],
            )
            if expert_id is not None and _is_expert_weight_tensor(tensor_name):
                _accumulate_tensor(
                    expert_display_components[expert_id],
                    dtype=value["dtype"],
                    shape=value["shape"],
                )

    if not experts:
        return None

    observed_ids = sorted(experts)
    observed_expert_count = len(observed_ids)
    if configured_expert_count is not None:
        expected_ids = list(range(configured_expert_count))
        if observed_ids != expected_ids:
            raise RuntimeError(
                "MoE expert indices inferred from the Safetensors metadata do not match the "
                f"`config.json` expert count. Expected IDs {expected_ids[:3]}...{expected_ids[-3:]} "
                f"(count={configured_expert_count}), found IDs {observed_ids[:3]}...{observed_ids[-3:]} "
                f"(count={observed_expert_count})."
            )
        expert_count = configured_expert_count
    else:
        expert_count = observed_expert_count

    expert_param_count = sum(component.param_count for component in experts.values())
    expert_bytes_count = sum(component.bytes_count for component in experts.values())
    expert_template = experts[observed_ids[0]]
    expert_display_template = expert_display_components[observed_ids[0]]
    for expert_id in observed_ids[1:]:
        if not _components_match(expert_template, experts[expert_id]):
            raise RuntimeError(
                "MoE experts inferred from the Safetensors metadata are not uniform, so they "
                "cannot be summarized as `N x EXPERTS` safely."
            )
        if not _components_match(expert_display_template, expert_display_components[expert_id]):
            raise RuntimeError(
                "MoE expert weight tensors inferred from the Safetensors metadata are not uniform, "
                "so their dtype breakdown cannot be summarized as `N x EXPERTS` safely."
            )
    return MoEMetadata(
        base_model=base_model,
        experts=dict(sorted(experts.items())),
        expert_count=expert_count,
        active_expert_count=active_expert_count,
        expert_param_count=expert_param_count,
        expert_bytes_count=expert_bytes_count,
        expert_template=expert_template,
        expert_display_template=expert_display_template,
    )


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

            current_shape, current_shape_bytes = _accumulate_tensor(
                component,
                dtype=value["dtype"],
                shape=value["shape"],
            )
            total_param_count += current_shape
            total_bytes_count += current_shape_bytes

        components[name] = component

    return SafetensorsMetadata(
        components=components,
        param_count=total_param_count,
        bytes_count=total_bytes_count,
    )
