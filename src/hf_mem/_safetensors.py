import asyncio
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Any

from hf_mem._weights import Weights

PROBE_SIZE = 100_000
SHARD = re.compile(r"-\d+-of-\d+(?=\.safetensors$)")
WEIGHT_EXTENSIONS = (".bin", ".pt", ".pth", ".ckpt", ".nemo")
_EXPERT_TOKENS = {"expert", "experts", "local_expert", "local_experts", "routed_expert", "routed_experts"}


@dataclass
class MoE:
    experts: int
    active: int | None
    base: Weights
    expert: Weights
    experts_total: Weights

    def to_dict(self) -> dict[str, Any]:
        return {
            "experts": self.experts,
            "active": self.active,
            "base": self.base.to_dict(),
            "expert": self.expert.to_dict(),
            "experts_total": {"bytes": self.experts_total.bytes, "params": self.experts_total.params},
        }


async def read_header(source: Any, path: str, probe: int = PROBE_SIZE) -> dict[str, Any]:
    head = await source.read(path, 0, probe)
    if len(head) < 8:
        raise RuntimeError(f"`{path}` is not a valid Safetensors file.")
    size = struct.unpack("<Q", head[:8])[0]
    if size + 8 > len(head):
        head += await source.read(path, len(head), size + 8)
    try:
        header = json.loads(head[8 : size + 8])
        header["__size__"] = size + 8
        return header
    except ValueError as e:
        raise RuntimeError(f"`{path}` is not a valid Safetensors file.") from e


def _groups(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in sorted(paths):
        groups.setdefault(SHARD.sub("", path).removesuffix(".safetensors"), []).append(path)
    return groups


def _select(groups: dict[str, list[str]], file: str) -> dict[str, list[str]]:
    target = file.strip("/").removesuffix(".safetensors")
    target = SHARD.sub("", target + ".safetensors").removesuffix(".safetensors")
    return {
        name: paths
        for name, paths in groups.items()
        if name == target or name.startswith(target + "/") or os.path.basename(name) == target
    }


async def discover(source: Any, file: str | None, warnings: list[str]) -> dict[str, list[str]]:
    files = source.files
    groups = _groups([p for p in files if p.endswith(".safetensors")])
    if not groups:
        return {}

    if file is not None:
        selected = _select(groups, file)
        if not selected:
            raise RuntimeError(f"No Safetensors file matching `{file}` found.")
        return selected

    if "model_index.json" in files:
        index = await source.json("model_index.json")
        components: dict[str, list[str]] = {}
        for name, value in index.items():
            if name.startswith("_") or not isinstance(value, list):
                continue
            candidates = [g for g in groups if os.path.dirname(g) == name]
            preferred = [g for g in (f"{name}/diffusion_pytorch_model", f"{name}/model") if g in groups]
            if chosen := (preferred or candidates):
                components[name] = groups[chosen[0]]
        if components:
            return components

    if "model" in groups or "model.safetensors.index.json" in files:
        if "model.safetensors.index.json" in files:
            index = await source.json("model.safetensors.index.json")
            paths = sorted(set(index["weight_map"].values()))
        else:
            paths = groups["model"]
        if "config_sentence_transformers.json" not in files:
            return {"Transformer": paths}
        components = {"0_Transformer": paths}
        if "modules.json" in files:
            for module in await source.json("modules.json"):
                path = (module.get("path") or "").strip("/")
                if path and (
                    module_paths := [p for g, ps in groups.items() if os.path.dirname(g) == path for p in ps]
                ):
                    components[path] = module_paths
        return components

    if len(groups) > 1:
        warnings.append(
            "No `model.safetensors`, `model.safetensors.index.json` or `model_index.json` found, so every "
            "`*.safetensors` file is counted as a separate component, assuming all of them are loaded; use "
            "`--file` to pick one."
        )
    return groups


async def read_components(
    source: Any, components: dict[str, list[str]]
) -> tuple[dict[str, Weights], dict[str, dict[int, Weights]]]:
    paths = sorted({p for ps in components.values() for p in ps})
    semaphore, probe = asyncio.Semaphore(32), [PROBE_SIZE]

    async def read(path: str) -> dict[str, Any]:
        async with semaphore:
            header = await read_header(source, path, probe[0])
            probe[0] = max(probe[0], header.pop("__size__"))
            return header

    headers = dict(zip(paths, await asyncio.gather(*(read(p) for p in paths))))
    weights: dict[str, Weights] = {}
    experts: dict[str, dict[int, Weights]] = {}
    for name, component_paths in components.items():
        totals: dict[str, list[int]] = {}
        by_expert: dict[int, Weights] = {}
        for path in component_paths:
            for tensor, info in headers[path].items():
                if tensor == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                params, dtype = math.prod(info["shape"]), info["dtype"]
                entry = totals.setdefault(dtype, [0, 0])
                entry[0] += params
                entry[1] += end - start
                if "expert" in tensor and (expert := _expert_id(tensor)) is not None:
                    by_expert.setdefault(expert, Weights()).add(dtype, params, end - start)
        component = Weights()
        for dtype, (params, nbytes) in totals.items():
            component.add(dtype, params, nbytes)
        weights[name], experts[name] = component, by_expert
    return weights, experts


def from_file_sizes(files: dict[str, int], file: str | None) -> dict[str, Weights]:
    paths = [p for p in files if p.endswith(WEIGHT_EXTENSIONS) and "optimizer" not in p and "training" not in p]
    if file is not None:
        target = file.strip("/")
        paths = [p for p in paths if p == target or p.startswith(target + "/") or os.path.basename(p) == target]
    root = [p for p in paths if "/" not in p and os.path.basename(p).startswith("pytorch_model")]
    weights: dict[str, Weights] = {}
    for path in root or paths:
        name = "Transformer" if root else re.sub(r"-\d+-of-\d+(?=\.\w+$)", "", path).rsplit(".", 1)[0]
        weights.setdefault(name, Weights()).bytes += files[path]
    return weights


def _int(config: dict[str, Any], *keys: str) -> int | None:
    return next((v for k in keys if isinstance(v := config.get(k), int) and v > 1), None)


def _expert_id(tensor: str) -> int | None:
    parts = tensor.split(".")
    for i, part in enumerate(parts[:-1]):
        if part in _EXPERT_TOKENS and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def moe_breakdown(
    weights: Weights, experts: dict[int, Weights], config: dict[str, Any], warnings: list[str]
) -> MoE | None:
    if not experts:
        return None
    ids = sorted(experts)
    count = _int(config, "num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts")
    if count is not None and ids != list(range(count)):
        warnings.append(
            f"MoE breakdown skipped, as the expert indices in the weights ({len(ids)}) do not match the "
            f"`config.json` expert count ({count})."
        )
        return None
    template = experts[ids[0]]
    if any(experts[i] != template for i in ids[1:]):
        warnings.append("MoE breakdown skipped, as the experts in the weights are not uniform.")
        return None
    total = Weights()
    for expert in experts.values():
        total.merge(expert)
    base = Weights()
    for dtype, entry in weights.dtypes.items():
        removed = total.dtypes.get(dtype, Weights())
        if entry.params - removed.params or entry.bytes - removed.bytes:
            base.add(dtype, entry.params - removed.params, entry.bytes - removed.bytes)
    return MoE(
        experts=count or len(ids),
        active=_int(config, "num_experts_per_tok", "num_experts_per_token", "top_k"),
        base=base,
        expert=template,
        experts_total=total,
    )
