import asyncio
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Any

from hf_mem._source import HubError
from hf_mem._weights import Weights, choices

PROBE_SIZE = 100_000
MAX_HEADER_SIZE = 100_000_000
SHARD = re.compile(r"-\d+-of-\d+(?=\.safetensors$)")
_MODEL_SHARD = re.compile(r"model-\d+-of-(\d+)\.safetensors")
# Embedding tables vLLM keeps in pinned host memory (`EngramConfig.cpu_offload`, on by default): the DeepSeek-V4.1
# Engram tables and the Qwen3.8 / Qwen4Exp per-layer n-gram embeddings
OFFLOADED = re.compile(
    r"(?:\.engram\.embed\.(?:weight|scale)|engram_tables\.\d+\.(?:weight|weight_scale_inv)"
    r"|\.ngram_embedding\.shard_\d+\.weight)$"
)
_EXPERT = re.compile(r"(?:^|\.)(?:local_|routed_)?experts?\.(\d+)(?:\.|$)")
POOL_MIN_FILES: int | None = None
# Sub-models loaded next to the main model, e.g. the audio tokenizer that decodes the generated speech
AUXILIARY = ("audio_tokenizer", "speech_tokenizer", "audio_decoder", "vocoder", "codec")


def language_model_prefixes(config: dict[str, Any]) -> tuple[str, ...]:
    """Recognized encoder/projector namespaces; never match decoder substrings."""
    root = config.get("thinker_config") or config
    names = []
    if isinstance(root.get("vision_config"), dict):
        names.extend(
            (
                "visual",
                "vision_tower",
                "vision_model",
                "vision_encoder",
                "multi_modal_projector",
                "mm_projector",
                "vision_projector",
            )
        )
    if isinstance(root.get("audio_config"), dict):
        names.extend(("audio_tower", "audio_model", "audio_encoder", "audio_projector", "audio_projection"))
    return tuple(
        prefix + name + "." for prefix in ("", "model.", "thinker.", "model.thinker.") for name in names
    )


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


async def read_header(source: Any, path: str, probe: int = PROBE_SIZE) -> bytes:
    head = await source.read(path, 0, probe)
    if len(head) < 8:
        raise RuntimeError(f"`{path}` is not a valid Safetensors file.")
    size = struct.unpack("<Q", head[:8])[0]
    if size > MAX_HEADER_SIZE:
        raise RuntimeError(f"`{path}` is not a valid Safetensors file.")
    if size + 8 > len(head):
        head += await source.read(path, len(head), size + 8)
    return head[8 : size + 8]


Summary = tuple[dict[str, list[int]], dict[tuple[int, str], list[int]], dict[str, list[int]]]


def summarize(raw: bytes, quantization: dict[str, Any] | None = None) -> Summary:
    dtypes: dict[str, list[int]] = {}
    experts: dict[tuple[int, str], list[int]] = {}
    offloaded: dict[str, list[int]] = {}
    quantization = quantization or {}
    packed, scales = None, None
    if quantization.get("format") == "mxfp4-pack-quantized":
        packed, scales = "weight_packed", "weight_scale"
    elif quantization.get("quant_method") == "mxfp4":
        packed, scales = "_blocks", "_scales"
    for tensor, info in json.loads(raw).items():
        if tensor == "__metadata__":
            continue
        start, end = info["data_offsets"]
        params, dtype = math.prod(info["shape"]), info["dtype"]
        expert = _EXPERT.search(tensor) if "expert" in tensor else None
        if packed and scales and dtype == "U8":
            if tensor.endswith(packed):
                params, dtype = params * 2, "MXFP4"
            elif tensor.endswith(scales):
                params, dtype = 0, "MXFP4"
        elif expert and quantization.get("expert_dtype") == "fp4":
            if dtype == "I8" and tensor.endswith(".weight"):
                params, dtype = params * 2, "MXFP4"
            elif dtype == "F8_E8M0" and tensor.endswith(".scale"):
                params, dtype = 0, "MXFP4"
        if quantization.get("quant_method") == "fp8" and (
            dtype == "F8_E8M0"
            and tensor.endswith(".scale")
            or dtype == "F32"
            and tensor.endswith(".weight_scale_inv")
        ):
            params = 0
        entry = (offloaded if OFFLOADED.search(tensor) else dtypes).setdefault(dtype, [0, 0])
        entry[0] += params
        entry[1] += end - start
        if expert:
            entry = experts.setdefault((int(expert.group(1)), dtype), [0, 0])
            entry[0] += params
            entry[1] += end - start
    return dtypes, experts, offloaded


def _groups(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in sorted(paths):
        groups.setdefault(SHARD.sub("", path).removesuffix(".safetensors"), []).append(path)
    return groups


def _select(groups: dict[str, list[str]], file: str) -> dict[str, list[str]]:
    target = file.strip("/").removesuffix(".safetensors")
    target = SHARD.sub("", target + ".safetensors").removesuffix(".safetensors")
    return {
        name: paths for name, paths in groups.items() if name == target or name.startswith(target + "/")
    } or {name: paths for name, paths in groups.items() if os.path.basename(name) == target}


def _complete(paths: list[str]) -> bool:
    totals = {m.group(1) if (m := _MODEL_SHARD.fullmatch(p)) else "" for p in paths}
    return len(totals) == 1 and "" not in totals and int(next(iter(totals))) == len(paths)


async def discover(source: Any, file: str | None, warnings: list[str]) -> dict[str, list[str]]:
    files = source.files
    groups = _groups([p for p in files if p.endswith(".safetensors")])
    if not groups:
        return {}

    if file is not None:
        selected = _select(groups, file)
        if not selected:
            raise RuntimeError(
                f"No Safetensors file matching `{file}` found, use one of {choices(list(groups))}."
            )
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
        paths = groups.get("model", [])
        if "model.safetensors.index.json" in files and not _complete(paths):
            index = await source.json("model.safetensors.index.json")
            indexed = sorted(set((index.get("weight_map") or {}).values())) or paths
            paths = [p for p in indexed if p in files]
            if len(paths) < len(indexed):
                warnings.append(
                    f"`model.safetensors.index.json` references {len(indexed) - len(paths)} files that are not in "
                    "the repository, so the estimate only covers the uploaded `*.safetensors` files."
                )
                paths = paths or sorted(p for ps in groups.values() for p in ps)
        if "config_sentence_transformers.json" not in files:
            auxiliary = {
                name: ps
                for name in AUXILIARY
                if (ps := [p for g, ps in groups.items() if os.path.dirname(g) == name for p in ps])
            }
            return {"Transformer": paths, **auxiliary}
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
    source: Any,
    components: dict[str, list[str]],
    warnings: list[str] | None = None,
    quantization: dict[str, Any] | None = None,
    tie_word_embeddings: bool = False,
    skip_mtp: bool = False,
    skip_prefixes: tuple[str, ...] = (),
    tensor_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Weights], dict[str, dict[int, Weights]], Weights]:
    paths = sorted({p for ps in components.values() for p in ps})
    semaphore, probe = asyncio.Semaphore(32), [PROBE_SIZE]
    summaries: dict[str, Summary] = {}
    embedding_tables: dict[str, dict[str, Any]] = {}
    skipped_mtp: dict[str, int] = {}
    skipped_encoder: dict[str, int] = {}
    headers: dict[str, dict[str, Any]] = {}
    pool = _pool() if POOL_MIN_FILES and len(paths) >= POOL_MIN_FILES else None
    loop = asyncio.get_running_loop()

    async def read(path: str) -> None:
        async with semaphore:
            raw = await read_header(source, path, probe[0])
        try:
            if skip_prefixes:
                tensors = json.loads(raw)
                excluded = {name: info for name, info in tensors.items() if name.startswith(skip_prefixes)}
                skipped_encoder[path] = sum(
                    info["data_offsets"][1] - info["data_offsets"][0] for info in excluded.values()
                )
                raw = json.dumps(
                    {name: info for name, info in tensors.items() if name not in excluded}
                ).encode()
            if skip_mtp and not os.path.dirname(path):
                tensors = json.loads(raw)
                excluded = {name: info for name, info in tensors.items() if name.startswith("mtp.")}
                if excluded:
                    skipped_mtp[path] = sum(
                        info["data_offsets"][1] - info["data_offsets"][0] for info in excluded.values()
                    )
                    raw = json.dumps(
                        {name: info for name, info in tensors.items() if name not in excluded}
                    ).encode()
            if tie_word_embeddings:
                embedding_tables[path] = {
                    name: info
                    for name, info in json.loads(raw).items()
                    if name.endswith(("lm_head.weight", "model.embed_tokens.weight"))
                }
            summaries[path] = (
                await loop.run_in_executor(pool, summarize, raw, quantization)
                if pool
                else summarize(raw, quantization)
            )
            if tensor_metadata is not None:
                headers[path] = {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise RuntimeError(f"`{path}` is not a valid Safetensors file.") from e
        probe[0] = max(probe[0], len(raw) + 8)

    try:
        results = await asyncio.gather(*(read(p) for p in paths), return_exceptions=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    failed = {p: r for p, r in zip(paths, results) if isinstance(r, BaseException)}
    if failed:
        error = next(iter(failed.values()))
        broken = [name for name, ps in components.items() if any(p in failed for p in ps)]
        if not isinstance(error, RuntimeError) or isinstance(error, HubError) or len(broken) == len(components):
            raise error
        components = {name: ps for name, ps in components.items() if name not in broken}
        if warnings is not None:
            warnings.append(f"Skipping {len(broken)} invalid `*.safetensors` files, e.g. {error}")
    weights: dict[str, Weights] = {}
    experts: dict[str, dict[int, Weights]] = {}
    offloaded = Weights()
    for name, component_paths in components.items():
        component, merged = Weights(), dict[int, Weights]()
        for path in component_paths:
            if tensor_metadata is not None:
                tensor_metadata.update(headers.get(path, {}))
            dtypes, by_expert, cpu = summaries[path]
            for dtype, (params, nbytes) in dtypes.items():
                component.add(dtype, params, nbytes)
            for (expert, dtype), (params, nbytes) in by_expert.items():
                merged.setdefault(expert, Weights()).add(dtype, params, nbytes)
            for dtype, (params, nbytes) in cpu.items():
                offloaded.add(dtype, params, nbytes)
        if tie_word_embeddings and all(not os.path.dirname(path) for path in component_paths):
            # Configured tying is a runtime alias even when both copies are serialized.
            # Match actual dense tables across shards; do not guess for packed heads,
            # missing inputs, incompatible geometry or separately configured submodels.
            tables: dict[str, Any] = {}
            duplicated = set()
            for path in component_paths:
                for tensor, info in embedding_tables.get(path, {}).items():
                    if tensor in tables:
                        duplicated.add(tensor)
                    tables[tensor] = info
            for tensor, output in tables.items():
                if not tensor.endswith("lm_head.weight"):
                    continue
                input_name = tensor.removesuffix("lm_head.weight") + "model.embed_tokens.weight"
                embedding = tables.get(input_name)
                dtype = output["dtype"]
                shape = output["shape"]
                if (
                    embedding is None
                    or tensor in duplicated
                    or input_name in duplicated
                    or len(shape) != 2
                    or embedding["shape"] != shape
                    or embedding["dtype"] != dtype
                    or dtype not in {"F16", "BF16", "F32"}
                ):
                    continue
                params = math.prod(shape)
                nbytes = params * (4 if dtype == "F32" else 2)
                if any(
                    info["data_offsets"][1] - info["data_offsets"][0] != nbytes for info in [output, embedding]
                ):
                    continue
                component.add(dtype, -params, -nbytes)
                if warnings is not None:
                    warnings.append(
                        f"`tie_word_embeddings=true` shares `{tensor}` with `{input_name}` at runtime; "
                        f"the duplicate {nbytes / (1 << 30):.2f} GiB checkpoint table is counted once."
                    )
        excluded_bytes = sum(skipped_mtp.get(path, 0) for path in component_paths)
        if excluded_bytes and warnings is not None:
            warnings.append(
                f"Qwen3.5 ordinary decoding does not load the root `mtp.*` prediction head; "
                f"{excluded_bytes / (1 << 20):.2f} MiB of checkpoint weights are excluded. "
                "Speculative decoding with an MTP draft requires additional memory."
            )
        weights[name], experts[name] = component, merged
    if skip_prefixes and warnings is not None:
        skipped = sum(skipped_encoder.values())
        if skipped:
            warnings.append(
                f"Language-model-only mode excludes {skipped / (1 << 30):.2f} GiB of identified encoder/projector weights."
            )
        else:
            warnings.append(
                "Language-model-only mode could not identify encoder/projector tensor names; "
                "unrecognized weights remain included."
            )
    return weights, experts, offloaded


def inkling_weight_adjustment(
    tensors: dict[str, Any], ranks: int, encoder_prefixes: tuple[str, ...], marlin: bool = True
) -> int:
    """Additional storage in the pinned SGLang SM90 Inkling loader.

    Marlin keeps MXFP8 scales in BF16, but its final NVFP4 scales remain FP8.
    Encoder and router tensors are replicated rather than expert/TP sharded.
    Only charge a scale conversion when its corresponding packed weight proves
    the MXFP8 layout; tensor headers can span different checkpoint shards.
    """
    extra, replicated = 0, 0
    for name, info in tensors.items():
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if name.startswith(encoder_prefixes) or name.endswith(".mlp.gate.weight"):
            replicated += size
        if not name.endswith((".experts.w13_weight_scale", ".experts.w2_weight_scale")):
            continue
        weight = tensors.get(name.removesuffix("_scale"), {})
        shape, scale = weight.get("shape", []), info["shape"]
        if (
            marlin
            and info["dtype"] == "U8"
            and weight.get("dtype") == "F8_E4M3"
            and len(shape) == len(scale) == 3
            and shape[:2] == scale[:2]
            and shape[2] == 32 * scale[2]
        ):
            extra += size
    return extra + (ranks - 1) * replicated


def _pool() -> Any:
    try:
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(min(os.cpu_count() or 1, 8))
    except (ImportError, NotImplementedError, OSError):
        return None


def _int(config: dict[str, Any], *keys: str) -> int | None:
    return next((v for k in keys if isinstance(v := config.get(k), int) and v > 1), None)


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
