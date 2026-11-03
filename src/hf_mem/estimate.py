import asyncio
from dataclasses import dataclass, field
from typing import Any

from hf_mem import _config, _gguf, _overhead, _safetensors
from hf_mem._source import HubError, Progress, open_source
from hf_mem._version import __version__
from hf_mem._weights import Weights


@dataclass
class KVCache:
    bytes: int
    dtype: str
    max_model_len: int
    batch_size: int
    layers: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "batch_size": self.batch_size,
            "layers": {k: v for k, v in self.layers.items() if v},
        }


@dataclass
class Estimate:
    model: str
    revision: str | None
    url: str
    format: str
    weights: Weights
    components: dict[str, Weights] = field(default_factory=dict)
    kv_cache: KVCache | None = None
    moe: _safetensors.MoE | None = None
    overhead: _overhead.Overhead | None = None
    fit: _overhead.Fit | None = None
    file: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        kv = self.kv_cache.bytes if self.kv_cache else 0
        return self.weights.bytes + kv + (self.overhead.bytes if self.overhead else 0)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"hf_mem": __version__, "model": self.model, "revision": self.revision}
        if self.file is not None:
            out["file"] = self.file
        out["format"] = self.format
        out["total"] = self.total
        out["weights"] = self.weights.to_dict()
        if len(self.components) > 1:
            out["components"] = {name: w.to_dict() for name, w in self.components.items()}
        out["kv_cache"] = self.kv_cache.to_dict() if self.kv_cache else None
        out["overhead"] = self.overhead.to_dict() if self.overhead else None
        if self.moe is not None:
            out["moe"] = self.moe.to_dict()
        if self.fit is not None:
            out["fit"] = self.fit.to_dict()
        out["warnings"] = self.warnings
        return out


@dataclass
class Listing:
    model: str
    revision: str | None
    url: str
    files: list[Estimate]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hf_mem": __version__,
            "model": self.model,
            "revision": self.revision,
            "format": "gguf",
            "files": [
                {
                    "file": f.file,
                    "total": f.total,
                    "weights": {"bytes": f.weights.bytes, "params": f.weights.params},
                    "kv_cache": f.kv_cache.to_dict() if f.kv_cache else None,
                    "overhead": f.overhead.to_dict() if f.overhead else None,
                }
                for f in self.files
            ],
            "warnings": self.warnings,
        }


def _gguf_kv(
    metadata: dict[str, Any], max_model_len: int | None, batch_size: int, dtype: str
) -> tuple[KVCache, int, float] | None:
    arch = metadata.get("general.architecture")
    context_length = metadata.get(f"{arch}.context_length")
    length = max_model_len or context_length
    if not isinstance(length, int):
        return None
    dtype = "F16" if dtype == "auto" else dtype
    if dtype not in _gguf.KV_DTYPES:
        raise RuntimeError(
            f"`--kv-cache-dtype={dtype}` is not valid for GGUF files, use any of {sorted(_gguf.KV_DTYPES)}."
        )
    result = _gguf.kv_cache(metadata, length, batch_size, dtype)
    if result is None:
        return None
    half = _gguf.kv_cache(metadata, max(1, length // 2), batch_size, dtype)[0]  # type: ignore[index]
    slope = (result[0] - half) / (length - max(1, length // 2)) if length > 1 else 0.0
    return KVCache(result[0], dtype, length, batch_size, result[1]), context_length or length, slope


def _fit(
    estimate: Estimate, context_length: int, slope: float, gpu_memory: str | None, utilization: float
) -> None:
    if gpu_memory is None:
        return
    kv = estimate.kv_cache
    estimate.fit = _overhead.fit(
        weights=estimate.weights.bytes,
        kv_fixed=(kv.bytes - slope * kv.max_model_len) if kv else 0,
        kv_per_token=slope,
        max_model_len=kv.max_model_len if kv else 0,
        context_length=context_length,
        overhead=estimate.overhead.bytes if estimate.overhead else 0,
        gpu_memory=gpu_memory,
        utilization=utilization,
    )


async def _estimate_gguf(
    source: Any,
    file: str | None,
    max_model_len: int | None,
    batch_size: int,
    kv_dtype: str,
    warnings: list[str],
    gpu_memory: str | None,
    utilization: float,
) -> Estimate | Listing:
    groups = _gguf.group_files(_gguf.gguf_paths(source.files))
    if not groups:
        raise RuntimeError(f"No GGUF files found for {source.name}.")

    if file is not None or len(groups) == 1:
        name, paths = _gguf.select_file(groups, file) if file is not None else next(iter(groups.items()))
        header = await _gguf.read_group(source, paths)
        kv = _gguf_kv(header.metadata, max_model_len, batch_size, kv_dtype)
        estimate = Estimate(
            model=source.name,
            revision=source.revision,
            url=source.url,
            format="gguf",
            file=name,
            weights=header.weights,
            kv_cache=kv[0] if kv else None,
            overhead=_overhead.llama_cpp() if kv else None,
            warnings=warnings,
        )
        _fit(estimate, kv[1] if kv else 0, kv[2] if kv else 0.0, gpu_memory, utilization)
        return estimate

    names = list(groups)
    first, *results = await asyncio.gather(
        _gguf.read_header(source, groups[names[0]][0]),
        *(_gguf.read_metadata(source, groups[name][0]) for name in names),
        return_exceptions=True,
    )
    files = []
    for name, result in zip(names, results):
        if isinstance(result, HubError):
            raise result
        if isinstance(result, Exception):
            warnings.append(f"Skipping `{name}`: {result}")
            continue
        metadata, tokenizer = result
        header_size = first.data_start if tokenizer and isinstance(first, _gguf.Header) else 0
        paths = groups[name]
        kv = _gguf_kv(metadata, max_model_len, batch_size, kv_dtype)
        files.append(
            Estimate(
                model=source.name,
                revision=source.revision,
                url=source.url,
                format="gguf",
                file=name,
                weights=Weights(bytes=sum(source.files[p] for p in paths) - header_size * len(paths)),
                kv_cache=kv[0] if kv else None,
                overhead=_overhead.llama_cpp() if kv else None,
            )
        )
    if not files:
        raise RuntimeError(f"No valid GGUF files found for {source.name}.")
    return Listing(source.name, source.revision, source.url, files, warnings)


async def _estimate_safetensors(
    source: Any,
    file: str | None,
    max_model_len: int | None,
    batch_size: int,
    kv_dtype: str,
    warnings: list[str],
    token: str | None,
    gpu_memory: str | None,
    utilization: float,
) -> Estimate | None:
    files = source.files
    config_task = asyncio.ensure_future(source.json("config.json")) if "config.json" in files else None
    try:
        selected = await _safetensors.discover(source, file, warnings)
        if selected:
            components, experts = await _safetensors.read_components(source, selected)
            fmt = "safetensors"
        elif components := _safetensors.from_file_sizes(files, file):
            experts = {}
            fmt = "file-sizes"
            warnings.append(
                "No Safetensors or GGUF weights found, so the estimate is based on the weight file sizes."
            )
        else:
            return None
        try:
            config = await config_task if config_task else {}
        except (RuntimeError, ValueError) as e:
            warnings.append(f"KV cache estimation skipped, as `config.json` could not be read ({e}).")
            config = {}
    finally:
        if config_task and not config_task.done():
            config_task.cancel()

    weights = Weights()
    for component in components.values():
        weights.merge(component)
    estimate = Estimate(
        model=source.name,
        revision=source.revision,
        url=source.url,
        format=fmt,
        weights=weights,
        components=components,
        file=file,
        warnings=warnings,
    )

    if not config or "config_sentence_transformers.json" in files or not _config.is_generative(config, files):
        _fit(estimate, 0, 0.0, gpu_memory, utilization)
        return estimate
    text = _config.text_config(config)
    reference = (config.get("text_config") or {}).get("_name_or_path") or ""
    if text is None and "/" in reference and not reference.startswith(("/", ".")):
        try:
            referenced, _ = await open_source(reference, token=token)
            try:
                text = _config.text_config({**await referenced.json("config.json"), **config["text_config"]})
            finally:
                await referenced.close()
        except RuntimeError:
            pass
    if text is None:
        warnings.append("KV cache estimation skipped, as no text decoder config was found in `config.json`.")
        _fit(estimate, 0, 0.0, gpu_memory, utilization)
        return estimate

    if len(components) == 1:
        name = next(iter(components))
        estimate.moe = _safetensors.moe_breakdown(components[name], experts.get(name, {}), text, warnings)
    length = max_model_len or text.get("max_position_embeddings")
    if not isinstance(length, int):
        warnings.append(
            "KV cache estimation skipped, as the context length is not in `config.json`, set `--max-model-len`."
        )
        _fit(estimate, 0, 0.0, gpu_memory, utilization)
        return estimate
    dtype = _config.kv_dtype(text, kv_dtype, weights, warnings)
    size, layers = _config.kv_cache(text, dtype, length, batch_size)
    half = _config.kv_cache(text, dtype, max(1, length // 2), batch_size)[0]
    slope = (size - half) / (length - max(1, length // 2)) if length > 1 else 0.0
    estimate.kv_cache = KVCache(size, dtype, length, batch_size, layers)
    estimate.overhead = _overhead.vllm(text, moe=estimate.moe is not None)
    _fit(estimate, text.get("max_position_embeddings") or length, slope, gpu_memory, utilization)
    return estimate


async def arun(
    model_id: str,
    revision: str | None = None,
    file: str | None = None,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str = "auto",
    hf_token: str | None = None,
    gpu_memory: str | None = None,
    gpu_memory_utilization: float = 0.9,
    progress: Progress | None = None,
) -> Estimate | Listing:
    source, url_file = await open_source(model_id, revision, hf_token, progress)
    file = file or url_file
    warnings: list[str] = []
    try:
        gguf = _gguf.gguf_paths(source.files)
        wants_gguf = file is not None and (
            file.endswith(".gguf") or any(p.startswith(file.strip("/") + "/") for p in gguf)
        )
        has_safetensors = "model_index.json" in source.files or any(
            _safetensors.SHARD.sub("", p) in {"model.safetensors", "model.safetensors.index.json"}
            for p in source.files
        )
        if gguf and (wants_gguf or not has_safetensors):
            return await _estimate_gguf(
                source,
                file,
                max_model_len,
                batch_size,
                kv_cache_dtype,
                warnings,
                gpu_memory,
                gpu_memory_utilization,
            )
        if gguf and file is None:
            warnings.append(
                "Both Safetensors and GGUF files were found, estimating the Safetensors weights; use `--file` "
                "to estimate a GGUF file."
            )
        if estimate := await _estimate_safetensors(
            source,
            file,
            max_model_len,
            batch_size,
            kv_cache_dtype,
            warnings,
            hf_token,
            gpu_memory,
            gpu_memory_utilization,
        ):
            return estimate
        raise RuntimeError(f"No Safetensors or GGUF weights found for {source.name}.")
    finally:
        await source.close()


def run(
    model_id: str,
    revision: str | None = None,
    file: str | None = None,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str = "auto",
    hf_token: str | None = None,
    gpu_memory: str | None = None,
    gpu_memory_utilization: float = 0.9,
) -> Estimate | Listing:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            arun(
                model_id,
                revision,
                file,
                max_model_len,
                batch_size,
                kv_cache_dtype,
                hf_token,
                gpu_memory,
                gpu_memory_utilization,
            )
        )
    raise RuntimeError(
        "`hf_mem.run(...)` cannot be called from a running event loop, use `await hf_mem.arun(...)`."
    )
