import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hf_mem import _config, _gguf, _onnx, _overhead, _safetensors
from hf_mem._source import HubError, Progress, open_source
from hf_mem._version import __version__
from hf_mem._weights import Weights

FORMATS = ("safetensors", "gguf", "onnx")


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
    offloaded: Weights | None = None
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
        out: dict[str, Any] = {
            "version": f"hf-mem@{__version__}",
            "model": self.model,
            "revision": self.revision,
        }
        out["file"] = self.file
        out["format"] = self.format
        out["total"] = self.total
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
                    "total": f.total,
                    "weights": {"bytes": f.weights.bytes, "params": f.weights.params or None},
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
    requested, dtype = dtype, "F16" if dtype == "auto" else dtype.upper()
    if dtype not in _gguf.CACHE_TYPES:
        raise RuntimeError(
            f"`--kv-cache-dtype={requested}` is not a llama.cpp KV cache type, use one of "
            f"{', '.join(t.lower() for t in _gguf.CACHE_TYPES)}."
        )
    result = _gguf.kv_cache(metadata, length, batch_size, dtype)
    if result is None:
        return None
    half = _gguf.kv_cache(metadata, max(1, length // 2), batch_size, dtype)
    assert half is not None
    slope = (result[0] - half[0]) / (length - max(1, length // 2)) if length > 1 else 0.0
    return KVCache(result[0], dtype, length, batch_size, result[1]), context_length or length, slope


def _fit(
    estimate: Estimate,
    context_length: int,
    slope: float,
    gpu_memory: str | None,
    utilization: float,
    kv_heads: int | None = None,
    padding: Callable[[int], int] | None = None,
    kv_bytes: Callable[[int, int], int] | None = None,
    max_gpus: int | None = None,
) -> None:
    if gpu_memory is None:
        return
    kv = estimate.kv_cache
    estimate.fit = _overhead.fit(
        weights=estimate.weights.bytes,
        kv_fixed=int(kv.bytes - slope * kv.max_model_len) if kv else 0,
        kv_per_token=slope,
        max_model_len=kv.max_model_len if kv else 0,
        context_length=max(context_length, kv.max_model_len if kv else 0),
        overhead=estimate.overhead.bytes if estimate.overhead else 0,
        gpu_memory=gpu_memory,
        utilization=utilization,
        kv_heads=kv_heads,
        context=estimate.overhead.context if estimate.overhead else 0,
        padding=padding,
        kv_bytes=kv_bytes,
        max_gpus=max_gpus,
    )


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
) -> Estimate | Listing:
    groups = _gguf.group_files(_gguf.gguf_paths(source.files))
    if not groups:
        raise RuntimeError(_no_weights(source, "gguf"))

    if file is not None or len(groups) == 1:
        name, paths = _gguf.select_file(groups, file) if file is not None else next(iter(groups.items()))
        projector = _gguf.mmproj(source.files, name)
        header, *extra = await asyncio.gather(
            _gguf.read_group(source, paths), *([_gguf.read_group(source, [projector])] if projector else [])
        )
        kv = _gguf_kv(header.metadata, max_model_len, batch_size, kv_dtype)
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
            overhead=_overhead.llama_cpp(extra[0].metadata if extra else None) if kv else None,
            warnings=warnings,
        )

        def cache_bytes(length: int, n: int) -> int:
            assert kv is not None
            result = _gguf.kv_cache(header.metadata, length, batch_size, kv[0].dtype)
            assert result is not None
            return result[0]

        _fit(
            estimate,
            kv[1] if kv else 0,
            kv[2] if kv else 0.0,
            gpu_memory,
            utilization,
            kv_bytes=cache_bytes if kv else None,
        )
        return estimate

    names = list(groups)
    projectors = {name: _gguf.mmproj(source.files, name) for name in names}
    unique = sorted({p for p in projectors.values() if p})
    first, *results = await asyncio.gather(
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
        kv = _gguf_kv(metadata, max_model_len, batch_size, kv_dtype)
        files.append(
            Estimate(
                model=source.name,
                revision=source.revision,
                url=source.url,
                format="gguf",
                file=name,
                weights=Weights(
                    bytes=sum(source.files[p] for p in paths)
                    - header_size
                    + (source.files[projector] if (projector := projectors[name]) else 0)
                ),
                kv_cache=kv[0] if kv else None,
                overhead=_overhead.llama_cpp(projected.get(projector) if projector else None) if kv else None,
            )
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
    loaded = dict(zip(unique, await asyncio.gather(*(source.json(p) for p in unique), return_exceptions=True)))
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
    lengths_task = asyncio.ensure_future(_onnx_lengths(source, labels))
    results = await asyncio.gather(
        *(_onnx.read_group(source, groups[label]) for label in labels), return_exceptions=True
    )
    lengths = await lengths_task
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


_IGNORED_DTYPE = (
    "`--dtype` only applies to encoder models served with Text Embeddings Inference, so the weights are "
    "estimated as stored."
)
# As Text Embeddings Inference reads the maximum input length, in order of priority
_SENTENCE_TRANSFORMERS_CONFIGS = (
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
)


async def _max_input_length(source: Any, config: dict[str, Any], dtype: str | None) -> int | None:
    # Only the padded models need it, as they warm up with a single input of the maximum input length: the
    # Sentence Transformers `max_seq_length`, or else the positions minus the RoBERTa padding offset
    if not _overhead.text_embeddings_inference_padded(config, dtype):
        return None
    for name in _SENTENCE_TRANSFORMERS_CONFIGS:
        if name in source.files:
            try:
                if isinstance(length := (await source.json(name)).get("max_seq_length"), int):
                    return length
            except (RuntimeError, ValueError):
                pass
            break
    positions = config.get("max_position_embeddings")
    if not isinstance(positions, int):
        return None
    roberta = config.get("model_type") in ("roberta", "xlm-roberta", "camembert")
    return positions - ((config.get("pad_token_id") or 0) + 1 if roberta else 0)


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
    dtype: str | None = None,
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
            warnings.append(f"KV cache estimation skipped, as `config.json` could not be read ({e}).")
            config = {}
        quantization = (
            config.get("quantization_config")
            or (config.get("text_config") or {}).get("quantization_config")
            or {}
        )
        components, experts, offloaded = await _safetensors.read_components(
            source, selected, warnings, quantization
        )
        processor: dict[str, Any] = {}
        for result in await asyncio.gather(*processor_tasks, return_exceptions=True):
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
        warnings=warnings,
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
    if config and _config.is_text_encoder(config, files):
        default, size = _overhead.text_embeddings_inference_dtype(config)
        loaded = sum(component.cast(default, size).bytes for component in components.values())
        if dtype is None and abs(loaded - weights.bytes) > 0.01 * weights.bytes:
            warnings.append(
                f"Text Embeddings Inference loads the weights in {default} on GPUs by default, i.e. "
                f"{loaded / _overhead.GIB:.2f} GiB instead of {weights.bytes / _overhead.GIB:.2f} GiB, so set "
                f"`--dtype {'float32' if default == 'F32' else 'float16'}` to estimate them as loaded."
            )
        if dtype is not None:
            # As Text Embeddings Inference casts the floating point weights to `--dtype`
            served = _overhead.text_embeddings_inference_dtype(config, dtype)
            estimate.components = {name: component.cast(*served) for name, component in components.items()}
            estimate.weights = Weights()
            for component in estimate.components.values():
                estimate.weights.merge(component)
        if dtype == "float16" and _overhead.text_embeddings_inference_dtype(config)[0] == "F32":
            warnings.append(
                "Text Embeddings Inference only supports Gemma 3 in F32 and fails to start with "
                "`--dtype float16`, so use `--dtype float32` or leave it unset."
            )
        estimate.overhead = _overhead.text_embeddings_inference(
            _config.normalize(config),
            files,
            max_num_batched_tokens or _overhead.TEXT_EMBEDDINGS_INFERENCE_BATCHED_TOKENS,
            await _max_input_length(source, config, dtype),
            dtype,
        )
        if gpu_memory and (_overhead.parse_gpu_memory(gpu_memory)[0] or 1) > 1:
            warnings.append(
                "Text Embeddings Inference runs each model on a single GPU, without tensor parallelism, so the "
                "fit is for one of the given GPUs, serving one replica per GPU."
            )
        _fit(estimate, 0, 0.0, gpu_memory, utilization, max_gpus=1)
        return estimate
    if dtype is not None:
        warnings.append(_IGNORED_DTYPE)
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
        except (RuntimeError, ValueError):
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
    tokens = _overhead.batched_tokens(gpu_memory, max_num_batched_tokens)
    if (
        max_num_batched_tokens is None
        and gpu_memory
        and 70 <= _overhead.parse_gpu_memory(gpu_memory)[1] / _overhead.GIB < 90
    ):
        warnings.append(
            "vLLM batches 8192 tokens on GPUs with 70 GiB or more except on A100s, which use 2048, so set "
            "`--max-num-batched-tokens 2048` for an A100."
        )
    size, layers = _config.kv_cache(text, dtype, length, batch_size, tokens)
    half = _config.kv_cache(text, dtype, max(1, length // 2), batch_size, tokens)[0]
    slope = (size - half) / (length - max(1, length // 2)) if length > 1 else 0.0
    estimate.kv_cache = KVCache(size, dtype, length, batch_size, layers)
    if _overhead.supports_video(config, files):
        warnings.append(
            "The encoder is estimated for the largest image, but vLLM profiles the largest video when it has more "
            "tokens, which can reserve a few GiB more unless videos are disabled with "
            "`--limit-mm-per-prompt '{\"video\": 0}'`."
        )
    estimate.overhead = _overhead.vllm(
        text,
        moe=estimate.moe is not None,
        hybrid=layers["state"] > 0,
        tokens=tokens,
        encoder_bytes=_overhead.encoder(config, processor),
    )
    _fit(
        estimate,
        text.get("max_position_embeddings") or length,
        slope,
        gpu_memory,
        utilization,
        _config.kv_heads(text),
        _mxfp4_padding(config, text, weights),
        lambda length, n: _config.kv_cache(text, dtype, length, batch_size, tokens, n)[0],
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
    gpu_memory_utilization: float = 0.92,
    max_num_batched_tokens: int | None = None,
    progress: Progress | None = None,
    format: str | None = None,
    dtype: str | None = None,
) -> Estimate | Listing:
    if batch_size < 1 or any(v is not None and v < 1 for v in (max_model_len, max_num_batched_tokens)):
        raise RuntimeError("`--batch-size`, `--max-model-len` and `--max-num-batched-tokens` must be positive.")
    if not 0 < gpu_memory_utilization <= 1:
        raise RuntimeError("`--gpu-memory-utilization` must be in (0, 1].")
    if gpu_memory is not None:
        _overhead.parse_gpu_memory(gpu_memory)
    if format not in (None, *FORMATS):
        raise RuntimeError(f"`--format={format}` is not valid, use one of {', '.join(FORMATS)}.")
    if dtype not in (None, *_overhead.TEXT_EMBEDDINGS_INFERENCE_DTYPES):
        choices = ", ".join(_overhead.TEXT_EMBEDDINGS_INFERENCE_DTYPES)
        raise RuntimeError(f"`--dtype={dtype}` is not valid, use one of {choices}.")
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
        if dtype is not None and (format in ("gguf", "onnx") or (format is None and not has_safetensors)):
            warnings.append(_IGNORED_DTYPE)
        if format == "onnx" or (format is None and ((wants_onnx and not wants_gguf) or only_onnx)):
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
            dtype,
        ):
            for other, found in (("GGUF", gguf), ("ONNX", onnx)):
                if found and format is None and file is None:
                    warnings.append(
                        f"{other} files were also found and are not estimated; use `--format {other.lower()}` "
                        "to estimate them."
                    )
            return estimate
        if onnx and format is None:
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
    gpu_memory_utilization: float = 0.92,
    max_num_batched_tokens: int | None = None,
    format: str | None = None,
    dtype: str | None = None,
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
                dtype=dtype,
            )
        )
    raise RuntimeError(
        "`hf_mem.run(...)` cannot be called from a running event loop, use `await hf_mem.arun(...)`."
    )
