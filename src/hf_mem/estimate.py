import asyncio
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hf_mem import _config, _gguf, _onnx, _overhead, _safetensors
from hf_mem._source import HubError, Progress, gather, open_source
from hf_mem._version import __version__
from hf_mem._weights import DTYPES, FLOAT_BYTES, Weights

FORMATS = ("safetensors", "gguf", "onnx")


@dataclass
class KVCache:
    bytes: int
    dtype: str
    max_model_len: int
    batch_size: int
    layers: dict[str, int]
    full_sliding_window_cache: bool = False
    state_capacity: int | None = None
    swa_full_tokens_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bytes": self.bytes,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "batch_size": self.batch_size,
            "layers": {k: v for k, v in self.layers.items() if v},
        }
        if self.full_sliding_window_cache:
            out["full_sliding_window_cache"] = True
        if self.state_capacity is not None:
            out["state_capacity"] = self.state_capacity
        if self.swa_full_tokens_ratio is not None:
            out["swa_full_tokens_ratio"] = self.swa_full_tokens_ratio
        return out


@dataclass
class Estimate:
    model: str
    revision: str | None
    url: str
    format: str
    weights: Weights
    components: dict[str, Weights] = field(default_factory=dict)
    offloaded: Weights | None = None
    kv_cache: KVCache | None = None
    moe: _safetensors.MoE | None = None
    overhead: _overhead.Overhead | None = None
    fit: _overhead.Fit | None = None
    file: str | None = None
    warnings: list[str] = field(default_factory=list)
    text_only: bool = False
    language_model_only: bool = False
    gpu_memory_utilization: float = 0.90
    startup_reserve_per_gpu: int = 0
    mem_fraction_static: float | None = None

    @property
    def total(self) -> int:
        kv = self.kv_cache.bytes if self.kv_cache else 0
        return self.weights.bytes + kv + (self.overhead.bytes if self.overhead else 0)

    @property
    def recommended_gpu_memory(self) -> int | None:
        if self.overhead is None:
            return None
        allocation = self.fit.required if self.fit else self.total
        allocation += self.startup_reserve_per_gpu * ((self.fit.gpus or 1) if self.fit else 1)
        recommended = _overhead.recommended_memory(allocation, self.gpu_memory_utilization)
        if self.fit is not None and self.fit.capacity_required is not None:
            return max(recommended, self.fit.capacity_required)
        if self.mem_fraction_static is not None:
            static = self.weights.bytes + (self.kv_cache.bytes if self.kv_cache else 0)
            static += self.overhead.buffers + self.overhead.weight_adjustment
            recommended = max(recommended, math.ceil(static / self.mem_fraction_static) + self.overhead.context)
        return recommended

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": f"hf-mem@{__version__}",
            "model": self.model,
            "revision": self.revision,
        }
        if self.language_model_only:
            out["workload"] = "language-model-only"
        elif self.text_only:
            out["workload"] = "text-only"
        out["file"] = self.file
        out["format"] = self.format
        out["total"] = self.total
        out["recommended_gpu_memory"] = self.recommended_gpu_memory
        out["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.mem_fraction_static is not None:
            out["mem_fraction_static"] = self.mem_fraction_static
        if self.startup_reserve_per_gpu:
            out["startup_reserve_per_gpu"] = self.startup_reserve_per_gpu
        out["weights"] = self.weights.to_dict()
        out["offloaded"] = self.offloaded.to_dict() if self.offloaded else None
        out["components"] = (
            {name: w.to_dict() for name, w in self.components.items()} if len(self.components) > 1 else None
        )
        out["kv_cache"] = self.kv_cache.to_dict() if self.kv_cache else None
        out["overhead"] = self.overhead.to_dict() if self.overhead else None
        out["moe"] = self.moe.to_dict() if self.moe else None
        out["fit"] = self.fit.to_dict() if self.fit else None
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
            "version": f"hf-mem@{__version__}",
            "model": self.model,
            "revision": self.revision,
            "format": self.files[0].format,
            "files": [
                {
                    "file": f.file,
                    **({"workload": "language-model-only"} if f.language_model_only else {}),
                    "total": f.total,
                    "recommended_gpu_memory": f.recommended_gpu_memory,
                    "gpu_memory_utilization": f.gpu_memory_utilization,
                    "weights": {"bytes": f.weights.bytes, "params": f.weights.params or None},
                    "kv_cache": f.kv_cache.to_dict() if f.kv_cache else None,
                    "overhead": f.overhead.to_dict() if f.overhead else None,
                }
                for f in self.files
            ],
            "warnings": self.warnings,
        }


def _gguf_kv(
    metadata: dict[str, Any],
    max_model_len: int | None,
    batch_size: int,
    dtype: str,
    full_sliding_window_cache: bool = False,
) -> tuple[KVCache, int, float] | None:
    arch = metadata.get("general.architecture")
    context_length = metadata.get(f"{arch}.context_length")
    length = max_model_len or context_length
    if not isinstance(length, int):
        return None
    requested, dtype = dtype, "F16" if dtype == "auto" else dtype.upper()
    if dtype not in _gguf.CACHE_TYPES:
        raise RuntimeError(
            f"`--kv-cache-dtype={requested}` is not a llama.cpp KV cache type, use one of "
            f"{', '.join(t.lower() for t in _gguf.CACHE_TYPES)}."
        )
    result = _gguf.kv_cache(metadata, length, batch_size, dtype, full_sliding_window_cache)
    if result is None:
        return None
    half = _gguf.kv_cache(metadata, max(1, length // 2), batch_size, dtype, full_sliding_window_cache)
    assert half is not None
    slope = (result[0] - half[0]) / (length - max(1, length // 2)) if length > 1 else 0.0
    return (
        KVCache(result[0], dtype, length, batch_size, result[1], full_sliding_window_cache),
        context_length or length,
        slope,
    )


def _fit(
    estimate: Estimate,
    context_length: int,
    slope: float,
    gpu_memory: str | None,
    utilization: float,
    kv_heads: int | None = None,
    padding: Callable[[int], int] | None = None,
    kv_bytes: Callable[[int, int], int] | None = None,
    rank_overhead: Callable[[int], _overhead.Overhead] | None = None,
    overhead_at_context: Callable[[int, int], int] | None = None,
    max_gpus: int | None = None,
    startup_reserve: Callable[[int], int] | None = None,
    capacity_at_context: Callable[[int, int], int] | None = None,
) -> None:
    estimate.gpu_memory_utilization = utilization
    if startup_reserve is not None:
        ranks = (_overhead.parse_gpu_memory(gpu_memory)[0] or 1) if gpu_memory else 1
        estimate.startup_reserve_per_gpu = startup_reserve(ranks)
    if gpu_memory is None:
        return
    kv = estimate.kv_cache
    estimate.fit = _overhead.fit(
        weights=estimate.weights.bytes,
        kv_fixed=int(kv.bytes - slope * kv.max_model_len) if kv else 0,
        kv_per_token=slope,
        max_model_len=kv.max_model_len if kv else 0,
        context_length=max(context_length, kv.max_model_len if kv else 0),
        overhead=(lambda n: rank_overhead(n).bytes)
        if rank_overhead is not None
        else estimate.overhead.bytes
        if estimate.overhead
        else 0,
        gpu_memory=gpu_memory,
        utilization=utilization,
        kv_heads=kv_heads,
        padding=padding,
        kv_bytes=kv_bytes,
        overhead_at_context=overhead_at_context,
        startup_reserve=startup_reserve or estimate.startup_reserve_per_gpu,
        max_gpus=max_gpus,
        capacity_at_context=capacity_at_context,
    )
    ranks = estimate.fit.gpus or 1
    if estimate.overhead is not None:
        selected = rank_overhead(ranks) if rank_overhead else estimate.overhead
        estimate.overhead = selected.scaled(ranks)
    if kv is not None and estimate.fit.cache_bytes is not None:
        kv.bytes = estimate.fit.cache_bytes
    if estimate.overhead is not None:
        estimate.overhead.weight_adjustment += estimate.fit.required - estimate.total


def _mxfp4_padding(
    config: dict[str, Any], text: dict[str, Any], weights: Weights
) -> Callable[[int], int] | None:
    quantization = config.get("quantization_config") or text.get("quantization_config") or {}
    if quantization.get("quant_method") != "mxfp4" and quantization.get("expert_dtype") != "fp4":
        return None
    experts = sum(weights.dtypes[d].bytes for d in ("U8", "I8", "F8_E8M0", "MXFP4") if d in weights.dtypes)
    intermediate = text.get("moe_intermediate_size") or text.get("intermediate_size") or 0
    if not experts or not intermediate or not text.get("hidden_size"):
        return None
    return _overhead.mxfp4_padding(experts, intermediate, text["hidden_size"])


async def _estimate_gguf(
    source: Any,
    file: str | None,
    max_model_len: int | None,
    batch_size: int,
    kv_dtype: str,
    warnings: list[str],
    gpu_memory: str | None,
    utilization: float,
    full_sliding_window_cache: bool,
    language_model_only: bool,
) -> Estimate | Listing:
    groups = _gguf.group_files(_gguf.gguf_paths(source.files))
    if not groups:
        raise RuntimeError(_no_weights(source, "gguf"))

    if file is not None or len(groups) == 1:
        name, paths = _gguf.select_file(groups, file) if file is not None else next(iter(groups.items()))
        projector = None if language_model_only else _gguf.mmproj(source.files, name)
        header, *extra = await gather(
            _gguf.read_group(source, paths), *([_gguf.read_group(source, [projector])] if projector else [])
        )
        kv = _gguf_kv(header.metadata, max_model_len, batch_size, kv_dtype, full_sliding_window_cache)
        components = {"Model": header.weights, "mmproj": extra[0].weights} if extra else {}
        estimate = Estimate(
            model=source.name,
            revision=source.revision,
            url=source.url,
            format="gguf",
            file=name,
            weights=Weights().merge(header.weights).merge(extra[0].weights) if extra else header.weights,
            components=components,
            offloaded=header.offloaded if header.offloaded.bytes else None,
            kv_cache=kv[0] if kv else None,
            warnings=warnings,
            language_model_only=language_model_only,
        )

        def cache_bytes(length: int, n: int) -> int:
            assert kv is not None
            result = _gguf.kv_cache(header.metadata, length, batch_size, kv[0].dtype, full_sliding_window_cache)
            assert result is not None
            return result[0]

        def overhead(length: int) -> _overhead.Overhead:
            return _overhead.llama_cpp(
                extra[0].metadata if extra else None,
                gpu_memory,
                header.metadata,
                length,
                estimate.weights.bytes + cache_bytes(length, 1),
                utilization,
            )

        if kv:
            estimate.overhead = overhead(kv[0].max_model_len)
        requested_length = kv[0].max_model_len if kv else 0

        _fit(
            estimate,
            kv[1] if kv else 0,
            kv[2] if kv else 0.0,
            gpu_memory,
            utilization,
            kv_bytes=cache_bytes if kv else None,
            rank_overhead=(lambda n: overhead(requested_length)) if kv else None,
            overhead_at_context=(lambda n, length: overhead(length).bytes) if kv else None,
        )
        return estimate

    names = list(groups)
    projectors = {name: None if language_model_only else _gguf.mmproj(source.files, name) for name in names}
    unique = sorted({p for p in projectors.values() if p})
    first, *results = await gather(
        _gguf.read_header(source, groups[names[0]][0]),
        *(_gguf.read_metadata(source, groups[name][0]) for name in names),
        *(_gguf.read_metadata(source, p) for p in unique),
        return_exceptions=True,
    )
    results, loaded = results[: len(names)], results[len(names) :]
    projected = {p: r[0] for p, r in zip(unique, loaded) if isinstance(r, tuple)}
    files = []
    for name, result in zip(names, results):
        if isinstance(result, HubError):
            raise result
        if isinstance(result, Exception):
            warnings.append(f"Skipping `{name}`: {result}")
            continue
        assert isinstance(result, tuple)
        metadata, tokenizer = result
        header_size = first.data_start if tokenizer and isinstance(first, _gguf.Header) else 0
        paths = groups[name]
        kv = _gguf_kv(metadata, max_model_len, batch_size, kv_dtype, full_sliding_window_cache)
        files.append(
            Estimate(
                model=source.name,
                revision=source.revision,
                url=source.url,
                format="gguf",
                file=name,
                language_model_only=language_model_only,
                gpu_memory_utilization=utilization,
                weights=Weights(
                    bytes=sum(source.files[p] for p in paths)
                    - header_size
                    + (source.files[projector] if (projector := projectors[name]) else 0)
                ),
                kv_cache=kv[0] if kv else None,
            )
        )
        if kv:
            files[-1].overhead = _overhead.llama_cpp(
                projected.get(projector) if projector else None,
                gpu_memory,
                metadata,
                kv[0].max_model_len,
                files[-1].weights.bytes + kv[0].bytes,
                utilization,
            )
    if not files:
        raise RuntimeError(f"No valid GGUF files found for {source.name}.")
    if gpu_memory is not None:
        warnings.append("`--gpu-memory` is only checked for a single GGUF file, use `--file` to pick one.")
    return Listing(source.name, source.revision, source.url, files, warnings)


async def _onnx_lengths(source: Any, labels: list[str]) -> dict[str, tuple[int | None, int | None]]:
    def config(label: str) -> str | None:
        directory = os.path.dirname(label)
        return next((p for p in (f"{directory}/config.json", "config.json") if p in source.files), None)

    paths = {label: config(label) for label in labels}
    unique = sorted({p for p in paths.values() if p})
    loaded = dict(zip(unique, await gather(*(source.json(p) for p in unique), return_exceptions=True)))
    lengths: dict[str, tuple[int | None, int | None]] = {}
    for label, path in paths.items():
        found = loaded[path] if path else None
        text = (_config.text_config(found) or {}) if isinstance(found, dict) else {}
        encoder = found.get("max_source_positions") if isinstance(found, dict) else None
        length = text.get("max_position_embeddings")
        lengths[label] = (
            length if isinstance(length, int) else None,
            encoder if isinstance(encoder, int) else None,
        )
    return lengths


async def _estimate_onnx(
    source: Any,
    file: str | None,
    max_model_len: int | None,
    batch_size: int,
    kv_dtype: str,
    warnings: list[str],
    gpu_memory: str | None,
    utilization: float,
) -> Estimate | Listing:
    groups = _onnx.group_files(source.files)
    if not groups:
        raise RuntimeError(_no_weights(source, "onnx"))
    if kv_dtype != "auto":
        raise RuntimeError("`--kv-cache-dtype` is set by the ONNX graph inputs, so only `auto` is valid.")
    if file is not None:
        groups = dict([_onnx.select_file(groups, file)])

    labels = list(groups)
    lengths, *results = await gather(
        _onnx_lengths(source, labels),
        *(_onnx.read_group(source, groups[label]) for label in labels),
        return_exceptions=True,
    )
    if isinstance(lengths, BaseException):
        raise lengths
    files = []
    for label, result in zip(labels, results):
        if isinstance(result, HubError) or (isinstance(result, BaseException) and len(labels) == 1):
            raise result
        if isinstance(result, BaseException):
            warnings.append(f"Skipping `{label}`: {result}")
            continue
        context, encoder_length = lengths[label]
        length = max_model_len or context
        kv = _onnx.kv_cache(result, length, batch_size, encoder_length) if length else None
        if result.cache and not length:
            warnings.append(
                f"KV cache estimation skipped for `{label}`, as no context length was found in `config.json`; "
                "set `--max-model-len`."
            )
        estimate = Estimate(
            model=source.name,
            revision=source.revision,
            url=source.url,
            format="onnx",
            file=label,
            weights=result.weights,
            kv_cache=KVCache(kv[0], kv[1], length, batch_size, {}) if kv and length else None,
        )
        if len(labels) == 1:

            def cache_bytes(length: int, n: int) -> int:
                cache = _onnx.kv_cache(result, length, batch_size, encoder_length)
                assert cache is not None
                return cache[0]

            _fit(
                estimate,
                context or 0,
                kv[2] if kv else 0.0,
                gpu_memory,
                utilization,
                kv_bytes=cache_bytes if kv else None,
            )
        files.append(estimate)
    if not files:
        raise RuntimeError(f"No valid ONNX files found for {source.name}.")
    if any(not f.weights.dtypes for f in files):
        warnings.append(
            f"ONNX files over {_onnx.SMALL_FILE >> 20} MiB with the weights inline are estimated from their file sizes, without the "
            "parameter count and dtypes."
        )
    warnings.append("The ONNX Runtime overhead (activations and workspaces) is not estimated.")
    if len(labels) == 1:
        files[0].warnings = warnings
        return files[0]
    if gpu_memory is not None:
        warnings.append("`--gpu-memory` is only checked for a single ONNX file, use `--file` to pick one.")
    return Listing(source.name, source.revision, source.url, files, warnings)


_ENCODER_DTYPE_ONLY = (
    "`--dtype` only supports unquantized Safetensors text encoders; other models keep their stored dtypes."
)

_SENTENCE_TRANSFORMERS_CONFIGS = (
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
)


async def _encoder_length(source: Any, config: dict[str, Any], override: int | None) -> int | None:
    if override is not None:
        return override
    for name in _SENTENCE_TRANSFORMERS_CONFIGS:
        if name in source.files:
            try:
                if isinstance(length := (await source.json(name)).get("max_seq_length"), int) and length > 0:
                    return length
            except (RuntimeError, ValueError) as error:
                if isinstance(error, HubError) and error.status in {401, 403}:
                    raise
    positions = config.get("max_position_embeddings")
    if not isinstance(positions, int) or positions <= 0:
        return None
    roberta = config.get("model_type") in ("roberta", "xlm-roberta", "camembert", "mpnet")
    length = positions - ((config.get("pad_token_id") or 0) + 1 if roberta else 0)
    return length if length > 0 else None


def _loading_transient(components: dict[str, Weights], dtype: str) -> int:
    target, _ = DTYPES[dtype]
    return max(
        (
            entry.largest * FLOAT_BYTES[name]
            for component in components.values()
            for name, entry in component.dtypes.items()
            if name in FLOAT_BYTES and name != target
        ),
        default=0,
    )


def _float_size(config: dict[str, Any], weights: Weights) -> int:
    dtype = config.get("dtype") or config.get("torch_dtype")
    sizes = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}
    if isinstance(dtype, str) and dtype.removeprefix("torch.") in sizes:
        return sizes[dtype.removeprefix("torch.")]
    floats = {d: w.params for d, w in weights.dtypes.items() if d in ("F64", "F32", "F16", "BF16") and w.params}
    largest = max(floats, key=floats.__getitem__) if floats else "BF16"
    return {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}[largest]


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
    max_num_batched_tokens: int | None,
    max_num_seqs: int | None,
    full_sliding_window_cache: bool,
    engine: str | None,
    gpu_type: str | None,
    text_only: bool,
    language_model_only: bool,
    dtype: str | None,
    swa_full_tokens_ratio: float | None,
    mem_fraction_static: float | None,
) -> Estimate | None:
    files = source.files
    config_task = asyncio.ensure_future(source.json("config.json")) if "config.json" in files else None
    processor_tasks = [
        asyncio.ensure_future(source.json(name))
        for name in ("preprocessor_config.json", "processor_config.json")
        if name in files
    ]
    try:
        selected = await _safetensors.discover(source, file, warnings)
        if not selected:
            return None
        try:
            config = await config_task if config_task else {}
        except (RuntimeError, ValueError) as e:
            if isinstance(e, HubError) and e.status in {401, 403}:
                raise
            warnings.append(f"KV cache estimation skipped, as `config.json` could not be read ({e}).")
            config = {}
        inkling = config.get("model_type") == "inkling2"
        if engine is None:
            engine = "sglang" if inkling else "vllm"
        elif inkling and engine == "vllm":
            warnings.append(
                "Inkling2 runtime storage is calibrated with the pinned SGLang image; "
                "the vLLM layout is not validated. Use `--engine sglang` for that reference."
            )
        quantization = (
            config.get("quantization_config")
            or (config.get("text_config") or {}).get("quantization_config")
            or {}
        )
        prefixes = _safetensors.language_model_prefixes(config) if language_model_only else ()
        if language_model_only and prefixes:
            selected = {name: paths for name, paths in selected.items() if name not in _safetensors.AUXILIARY}
        tensor_metadata: dict[str, Any] | None = {} if engine == "sglang" and inkling else None
        skipped_weights: dict[str, int] = {}
        components, experts, offloaded = await _safetensors.read_components(
            source,
            selected,
            warnings,
            quantization,
            bool((_config.text_config(config) or config).get("tie_word_embeddings")),
            (_config.text_config(config) or config).get("model_type")
            in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"},
            prefixes,
            tensor_metadata,
            skipped_weights,
            _safetensors.expert_count(_config.text_config(config) or config),
        )
        processor: dict[str, Any] = {}
        for result in await gather(*processor_tasks, return_exceptions=True):
            if isinstance(result, dict):
                processor.update(result, **(result.get("image_processor") or {}))
    finally:
        for task in (config_task, *processor_tasks):
            if task and not task.done():
                task.cancel()
            elif task and not task.cancelled():
                task.exception()

    weights = Weights()
    for component in components.values():
        weights.merge(component)
    estimate = Estimate(
        model=source.name,
        revision=source.revision,
        url=source.url,
        format="safetensors",
        weights=weights,
        components=components,
        offloaded=offloaded if offloaded.bytes else None,
        file=file,
        text_only=text_only,
        language_model_only=language_model_only,
        warnings=warnings,
        startup_reserve_per_gpu=skipped_weights.get("encoder", 0) if language_model_only else 0,
    )
    if offloaded.bytes:
        warnings.append(
            f"vLLM keeps the {offloaded.bytes / _overhead.GIB:.2f} GiB of embedding tables (Engram or n-gram embeddings) "
            "in pinned host memory, so they are not in the GPU total but need that much free CPU memory."
        )

    if not config and "params.json" in files:
        warnings.append(
            "KV cache estimation skipped, as Mistral `params.json` configs are not supported, estimate a version of "
            "the model with a Transformers `config.json` for the KV cache."
        )
    text_encoder = bool(config) and _config.is_text_encoder(config, files)
    if dtype is not None:
        if (
            not text_encoder
            or quantization
            or any(d not in ("F64", "F32", "F16", "BF16", "I64", "I32", "I16", "BOOL") for d in weights.dtypes)
        ):
            raise RuntimeError(_ENCODER_DTYPE_ONLY)
    weight_dtype = dtype
    if (
        not text_encoder
        and config
        and _config.is_generative(config, files)
        and not quantization
        and _float_size(_config.text_config(config) or config, weights) == 4
    ):
        weight_dtype = "float16"
        warnings.append(
            "FP32 decoder weights use a 16-bit CUDA serving reference; engines may select F16 or BF16."
        )
    if weight_dtype is not None:
        estimate.components = {name: component.cast(weight_dtype) for name, component in components.items()}
        estimate.weights = weights = Weights()
        for component in estimate.components.values():
            weights.merge(component)
        experts = {
            name: {index: expert.cast(weight_dtype) for index, expert in group.items()}
            for name, group in experts.items()
        }
    if text_encoder:
        encoder_config = _config.normalize(config)
        length = await _encoder_length(source, encoder_config, max_model_len)
        tokens = _overhead.batched_tokens(gpu_memory, max_num_batched_tokens)
        if not encoder_config.get("hidden_size") or not encoder_config.get("num_attention_heads"):
            warnings.append(
                "Encoder warmup estimation skipped, as the model dimensions are missing from `config.json`."
            )
        elif (
            any(experts.values())
            or any(
                encoder_config.get(k)
                for k in (
                    "num_local_experts",
                    "n_routed_experts",
                    "num_experts",
                    "moe_num_experts",
                    "kv_lora_rank",
                )
            )
            or "state" in _config.layer_kinds(encoder_config)
        ):
            warnings.append("Encoder warmup estimation skipped for MoE, recurrent or latent-attention models.")
        elif length is None:
            warnings.append(
                "Encoder warmup estimation skipped, as the sequence length is unknown, set `--max-model-len`."
            )
        else:
            size = DTYPES[dtype][1] if dtype else _float_size(encoder_config, weights)
            estimate.overhead = overhead = _overhead.encoder_warmup(encoder_config, files, tokens, length, size)
            if dtype:
                transient = _loading_transient(components, dtype)
                overhead.loading = max(0, transient - overhead.activations - overhead.buffers)
            full, remaining = divmod(tokens, min(tokens, length))
            if _overhead.encoder_eager(encoder_config, size):
                tensors = _overhead.encoder_attention_tensors(encoder_config, full + bool(remaining))
                attention = f"eager attention, keeping {tensors} tensors of attention scores per sequence" + (
                    ", as memory-efficient attention kernels need half precision" if size >= 4 else ""
                )
            else:
                attention = "memory-efficient attention"
            warnings.append(
                f"Encoder warmup assumes {tokens} tokens in sequences of up to {length} tokens with {attention}. "
                + (
                    f"Floating-point parameters and activations use {dtype}. "
                    if dtype
                    else "Weights are estimated as stored. "
                )
                + "Engine kernels, padding and workspaces can change memory use."
            )
            if overhead.buffers:
                positions = encoder_config.get("max_position_embeddings")
                warnings.append(
                    f"Eager ALiBi attention keeps the biases of {positions} x {positions} positions per head, "
                    f"{overhead.buffers / _overhead.GIB:.2f} GiB for the model's lifetime."
                )
            if overhead.loading:
                warnings.append(
                    f"Casting to {dtype} while loading reserves the largest tensor in its stored precision, "
                    f"peaking at {(overhead.bytes + weights.bytes) / _overhead.GIB:.2f} GiB while loading, "
                    "assuming tensors are cast on the device."
                )
        if gpu_memory and (_overhead.parse_gpu_memory(gpu_memory)[0] or 1) > 1:
            warnings.append("Encoder fit assumes one replica per GPU, without tensor parallelism.")
        _fit(estimate, 0, 0.0, gpu_memory, utilization, max_gpus=1)
        return estimate
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
        except (RuntimeError, ValueError) as error:
            if isinstance(error, HubError) and error.status in {401, 403}:
                raise
    if text is None:
        warnings.append("KV cache estimation skipped, as no text decoder config was found in `config.json`.")
        _fit(estimate, 0, 0.0, gpu_memory, utilization)
        return estimate

    if len(components) == 1:
        name = next(iter(components))
        estimate.moe = _safetensors.moe_breakdown(
            estimate.components[name], experts.get(name, {}), text, warnings
        )
    length = max_model_len or text.get("max_position_embeddings")
    if not isinstance(length, int):
        warnings.append(
            "KV cache estimation skipped, as the context length is not in `config.json`, set `--max-model-len`."
        )
        _fit(estimate, 0, 0.0, gpu_memory, utilization)
        return estimate
    dtype = _config.kv_dtype(text, kv_dtype, weights, warnings)
    profile_memory = gpu_memory
    estimate.mem_fraction_static = mem_fraction_static
    tokens = _overhead.batched_tokens(profile_memory, max_num_batched_tokens, gpu_type, engine)
    if (
        max_num_batched_tokens is None
        and engine == "vllm"
        and gpu_type is None
        and gpu_memory
        and 70 <= _overhead.parse_gpu_memory(gpu_memory)[1] / _overhead.GIB < 90
    ):
        warnings.append(
            "vLLM batches 8192 tokens on GPUs with 70 GiB or more except on A100s, which use 2048, so set "
            "`--max-num-batched-tokens 2048` for an A100."
        )

    full_sliding_window_cache = full_sliding_window_cache or (
        engine == "sglang" and text.get("model_type") == "gemma3_text"
    )

    state_capacity = max_num_seqs or batch_size
    dedicated_state = engine == "sglang" and bool(text.get("linear_num_value_heads"))
    cache_text = text
    kinds = _config.layer_kinds(text, include_shared=True)
    if engine == "sglang" and "full" in kinds and "sliding" in kinds and "state" not in kinds:
        cache_text = {
            key: value
            for key, value in text.items()
            if key not in {"num_kv_shared_layers", "kv_sharing_modes", "kv_source_layers"}
        }

    def cache(length: int, n: int = 1) -> tuple[int, dict[str, int]]:
        if dedicated_state:
            return _config.sglang_kv_cache(
                text,
                dtype,
                length,
                batch_size,
                tokens,
                n,
                full_sliding_window_cache,
                state_capacity,
            )
        if swa_full_tokens_ratio is not None and not full_sliding_window_cache:
            return _config.partitioned_kv_cache(
                cache_text, dtype, length, batch_size, tokens, n, swa_full_tokens_ratio
            )
        return _config.kv_cache(cache_text, dtype, length, batch_size, tokens, n, full_sliding_window_cache)

    def weight_adjustment(n: int) -> int:
        return _safetensors.inkling_weight_adjustment(
            tensor_metadata or {},
            n,
            _safetensors.language_model_prefixes(config),
            marlin=_overhead.hardware(profile_memory, gpu_type)[2],
        )

    size, layers = cache(length)
    if (
        engine == "sglang"
        and layers["full"]
        and layers["sliding"]
        and not layers["state"]
        and not full_sliding_window_cache
    ):
        missing = [
            f"`--{name}`"
            for name, value in (
                ("swa-full-tokens-ratio", swa_full_tokens_ratio),
                ("mem-fraction-static", mem_fraction_static),
            )
            if value is None
        ]
        if missing:
            warnings.append(
                "To include SGLang's cache allocation limits, set "
                + " and ".join(missing)
                + " to match the server."
            )
    half = cache(max(1, length // 2))[0]
    slope = (size - half) / (length - max(1, length // 2)) if length > 1 else 0.0
    estimate.kv_cache = KVCache(
        size,
        dtype,
        length,
        batch_size,
        layers,
        full_sliding_window_cache,
        state_capacity if dedicated_state else None,
        swa_full_tokens_ratio
        if layers["full"] and layers["sliding"] and not layers["state"] and not full_sliding_window_cache
        else None,
    )
    if dedicated_state and max_num_seqs is None:
        warnings.append(
            "SGLang automatically reserves recurrent-state cache from available VRAM; "
            "this estimate includes the minimum state for the requested batch. "
            "`--max-num-seqs` models a dedicated state pool with radix caching disabled."
        )
    if not (text_only or language_model_only) and engine == "vllm" and _overhead.supports_video(config, files):
        warnings.append(
            "The encoder is estimated for the largest image, but vLLM profiles the largest video when it has more "
            "tokens, which can reserve a few GiB more unless videos are disabled with "
            "`--limit-mm-per-prompt '{\"video\": 0}'`."
        )

    is_moe = estimate.moe is not None or any(
        isinstance(text.get(key), int) and text[key] > 1
        for key in ("num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts")
    )

    def overhead(n: int = 1, context_length: int = length) -> _overhead.Overhead:
        capacity = max_num_seqs
        if capacity is None and (profile_memory is not None or gpu_type is not None or engine == "sglang"):
            capacity = max(batch_size, _overhead.default_sequences(profile_memory, gpu_type, engine))
        result = _overhead.decoder(
            text,
            moe=is_moe,
            hybrid=layers["state"] > 0,
            tokens=tokens,
            encoder_bytes=0
            if text_only or language_model_only
            else _overhead.encoder(config, processor, context_length, engine),
            max_num_seqs=capacity,
            tensor_parallel_size=n,
            gpu_memory=profile_memory,
            gpu_type=gpu_type,
            engine=engine,
            batch_size=batch_size,
            context_length=context_length,
        )
        if tensor_metadata is not None:
            result.weight_adjustment = math.ceil(weight_adjustment(n) / n)
            if _overhead.hardware(profile_memory, gpu_type)[2]:
                blocks = math.ceil(text["num_hidden_layers"] / text["n_layers_per_attn_res_block"])
                routed = text.get("num_experts_per_tok") or 1
                prefill_pool = 0.30 * _overhead.GIB + tokens * text["hidden_size"] * (
                    12 * blocks + 24 * routed / n
                )
                result.graphs += int(max(0, prefill_pool - result.activations))
        return result

    encoder_reserve = estimate.startup_reserve_per_gpu

    def startup(n: int = 1) -> int:
        return max(encoder_reserve, _overhead.GIB - overhead(n).activations, 0)

    def static_capacity(n: int, context_length: int) -> int:
        reserved = overhead(n, context_length)
        padding = _mxfp4_padding(config, text, weights)
        static = weights.bytes + cache(context_length, n)[0] + (padding(n) if padding else 0)
        static += n * (reserved.buffers + reserved.weight_adjustment)
        return math.ceil(static / (mem_fraction_static or 1)) + n * reserved.context

    if gpu_memory is None and gpu_type is None:
        for capacity in (16, 24, 48, 80, 141):
            profile_memory = str(capacity)
            tokens = _overhead.batched_tokens(profile_memory, max_num_batched_tokens, None, engine)
            size, layers = cache(length)
            candidate = overhead()
            allocation = weights.bytes + size + candidate.bytes + startup()
            recommended = _overhead.recommended_memory(allocation, utilization)
            if mem_fraction_static is not None:
                recommended = max(recommended, static_capacity(1, length))
            if recommended <= capacity * _overhead.GIB * (0.9 if capacity <= 48 else 0.98):
                break
        estimate.kv_cache.bytes = size
        half = cache(max(1, length // 2))[0]
        slope = (size - half) / (length - max(1, length // 2)) if length > 1 else 0.0
    estimate.overhead = overhead()
    _fit(
        estimate,
        text.get("max_position_embeddings") or length,
        slope,
        gpu_memory,
        utilization,
        _config.kv_heads(text),
        _mxfp4_padding(config, text, weights),
        lambda length, n: cache(length, n)[0],
        overhead,
        lambda n, length: overhead(n, length).bytes,
        startup_reserve=startup,
        capacity_at_context=static_capacity if mem_fraction_static is not None else None,
    )
    return estimate


_PICKLES = (".bin", ".pt", ".pth", ".ckpt", ".nemo")


def _no_weights(source: Any, format: str | None) -> str:
    files = list(source.files)
    found = [
        name
        for name, present in (
            ("Safetensors", any(p.endswith(".safetensors") for p in files)),
            ("GGUF", any(p.endswith(".gguf") for p in files)),
            ("ONNX", any(p.endswith(".onnx") for p in files)),
        )
        if present
    ]
    if format and found:
        name = {"safetensors": "Safetensors", "gguf": "GGUF", "onnx": "ONNX"}[format]
        return (
            f"No {name} weights found for `{source.name}`, which has {' and '.join(found)} weights, use "
            f"`--format {found[0].lower()}` or drop `--format`."
        )
    if pickles := sorted((p for p in files if p.endswith(_PICKLES)), key=lambda p: -source.files[p]):
        return (
            f"`{source.name}` only has PyTorch pickle weights (e.g. `{pickles[0]}`), which hf-mem cannot read without "
            "loading them; estimate a version of the model in Safetensors, GGUF or ONNX instead."
        )
    return f"No Safetensors, GGUF or ONNX weights found for `{source.name}`."


async def arun(
    model_id: str,
    revision: str | None = None,
    file: str | None = None,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str = "auto",
    hf_token: str | None = None,
    gpu_memory: str | None = None,
    gpu_memory_utilization: float = 0.90,
    max_num_batched_tokens: int | None = None,
    progress: Progress | None = None,
    format: str | None = None,
    dtype: str | None = None,
    max_num_seqs: int | None = None,
    full_sliding_window_cache: bool = False,
    engine: str | None = None,
    gpu_type: str | None = None,
    text_only: bool = False,
    language_model_only: bool = False,
    swa_full_tokens_ratio: float | None = None,
    mem_fraction_static: float | None = None,
) -> Estimate | Listing:
    engine = engine.lower() if engine is not None else None
    gpu_type = _overhead.normalize_gpu_type(gpu_type) if gpu_type is not None else None
    if engine not in {None, "vllm", "sglang"}:
        raise RuntimeError("`--engine` must be vllm or sglang for Safetensors overhead.")
    for name, value in (
        ("swa-full-tokens-ratio", swa_full_tokens_ratio),
        ("mem-fraction-static", mem_fraction_static),
    ):
        if value is not None and not 0 < value <= 1:
            raise RuntimeError(f"`--{name}` must be in (0, 1].")
    if swa_full_tokens_ratio is not None or mem_fraction_static is not None:
        if engine not in {None, "sglang"} or format not in {None, "safetensors"}:
            raise RuntimeError(
                "`--swa-full-tokens-ratio` and `--mem-fraction-static` require SGLang and Safetensors."
            )
        engine = "sglang"
        format = "safetensors"
    if batch_size < 1 or any(
        v is not None and v < 1 for v in (max_model_len, max_num_batched_tokens, max_num_seqs)
    ):
        raise RuntimeError(
            "`--batch-size`, `--max-model-len`, `--max-num-batched-tokens` and `--max-num-seqs` must be positive."
        )
    if max_num_seqs is not None and max_num_seqs < batch_size:
        raise RuntimeError("`--max-num-seqs` must be at least `--batch-size`.")
    if not 0 < gpu_memory_utilization <= 1:
        raise RuntimeError("`--gpu-memory-utilization` must be in (0, 1].")
    if gpu_memory is not None:
        _overhead.parse_gpu_memory(gpu_memory)
        if gpu_type is not None:
            _overhead.validate_gpu_memory(gpu_memory, gpu_type)
    if format not in (None, *FORMATS):
        raise RuntimeError(f"`--format={format}` is not valid, use one of {', '.join(FORMATS)}.")
    if dtype is not None and dtype not in DTYPES:
        raise RuntimeError(f"`--dtype={dtype}` is not valid, use one of {', '.join(DTYPES)}.")
    source, url_file = await open_source(model_id, revision, hf_token, progress)
    file = file or url_file
    warnings: list[str] = []
    try:
        gguf = _gguf.gguf_paths(source.files)
        onnx = [p for p in source.files if p.endswith(".onnx")]
        target = file.strip("/") if file is not None else None
        wants_gguf = target is not None and (
            target.endswith(".gguf")
            or any(p.startswith(target + "/") for p in gguf)
            or bool(_gguf.tagged(_gguf.group_files(gguf), target))
        )
        wants_onnx = target is not None and (
            target.endswith(".onnx")
            or any(p.startswith(target + "/") for p in onnx)
            or target in _onnx.group_files(source.files)
        )
        has_safetensors = "model_index.json" in source.files or any(
            _safetensors.SHARD.sub("", p) in {"model.safetensors", "model.safetensors.index.json"}
            for p in source.files
        )
        only_onnx = onnx and not any(p.endswith(".safetensors") for p in source.files) and not gguf
        if format == "onnx" or (format is None and ((wants_onnx and not wants_gguf) or only_onnx)):
            if dtype is not None:
                raise RuntimeError(_ENCODER_DTYPE_ONLY)
            return await _estimate_onnx(
                source,
                file,
                max_model_len,
                batch_size,
                kv_cache_dtype,
                warnings,
                gpu_memory,
                gpu_memory_utilization,
            )
        if format == "gguf" or (format is None and gguf and (wants_gguf or not has_safetensors)):
            if dtype is not None:
                raise RuntimeError(_ENCODER_DTYPE_ONLY)
            return await _estimate_gguf(
                source,
                file,
                max_model_len,
                batch_size,
                kv_cache_dtype,
                warnings,
                gpu_memory,
                gpu_memory_utilization,
                full_sliding_window_cache,
                language_model_only,
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
            max_num_batched_tokens,
            max_num_seqs,
            full_sliding_window_cache,
            engine,
            gpu_type,
            text_only,
            language_model_only,
            dtype,
            swa_full_tokens_ratio,
            mem_fraction_static,
        ):
            for other, found in (("GGUF", gguf), ("ONNX", onnx)):
                if found and format is None and file is None:
                    warnings.append(
                        f"{other} files were also found and are not estimated; use `--format {other.lower()}` "
                        "to estimate them."
                    )
            return estimate
        if onnx and format is None:
            if dtype is not None:
                raise RuntimeError(_ENCODER_DTYPE_ONLY)
            return await _estimate_onnx(
                source,
                file,
                max_model_len,
                batch_size,
                kv_cache_dtype,
                warnings,
                gpu_memory,
                gpu_memory_utilization,
            )
        raise RuntimeError(_no_weights(source, format))
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
    gpu_memory_utilization: float = 0.90,
    max_num_batched_tokens: int | None = None,
    format: str | None = None,
    dtype: str | None = None,
    max_num_seqs: int | None = None,
    full_sliding_window_cache: bool = False,
    engine: str | None = None,
    gpu_type: str | None = None,
    text_only: bool = False,
    language_model_only: bool = False,
    swa_full_tokens_ratio: float | None = None,
    mem_fraction_static: float | None = None,
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
                max_num_batched_tokens,
                format=format,
                max_num_seqs=max_num_seqs,
                full_sliding_window_cache=full_sliding_window_cache,
                engine=engine,
                gpu_type=gpu_type,
                text_only=text_only,
                language_model_only=language_model_only,
                dtype=dtype,
                swa_full_tokens_ratio=swa_full_tokens_ratio,
                mem_fraction_static=mem_fraction_static,
            )
        )
    raise RuntimeError(
        "`hf_mem.run(...)` cannot be called from a running event loop, use `await hf_mem.arun(...)`."
    )
