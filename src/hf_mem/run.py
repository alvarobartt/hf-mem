import asyncio
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

import httpx

from hf_mem._fetch import get_json_file
from hf_mem._types import KvCache
from hf_mem._version import __version__
from hf_mem.gguf.fetch import fetch_gguf_with_semaphore
from hf_mem.gguf.metadata import GGUFDtype, GGUFMetadata, merge_shards
from hf_mem.safetensors.fetch import fetch_modules_and_dense_metadata, fetch_safetensors_metadata
from hf_mem.safetensors.kv_cache import compute_safetensors_kv_cache_size, resolve_kv_cache_dtype
from hf_mem.safetensors.metadata import SafetensorsMetadata, parse_safetensors_metadata

MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))

# NOTE: Shard filename pattern, e.g. "Kimi-K2.5-BF16-00001-of-00046.gguf"
_SHARD_PATTERN = re.compile(r"(.+)-(\d+)-of-(\d+)\.gguf$")


@dataclass
class Result:
    model_id: str
    revision: str

    # `filename` is set only for a single-GGUF request; None for Safetensors and multi-GGUF
    filename: str | None
    # `memory` is the model weights in bytes, a scalar for Safetensors or a single GGUF file,
    # a {filename: bytes} dict when multiple GGUF files are present in the repo
    memory: Union[int, Dict[str, int]]
    # `kv_cache` mirrors the shape of `memory` but for the KV cache estimation; None when
    # experimental=False or the model architecture does not support KV cache estimation
    kv_cache: Union[int, Dict[str, int], None]
    # `total_memory` is memory + kv_cache for single-file results; None for multi-GGUF
    total_memory: int | None

    # NOTE: When True, to_json() enriches `memory` and `kv_cache` with per-component /
    # per-dtype breakdowns rather than bare byte counts
    details: bool = False

    safetensors: SafetensorsMetadata | None = field(default=None, repr=False)
    gguf_files: Dict[str, GGUFMetadata] | None = field(default=None, repr=False)
    kv_cache_metadata: KvCache | None = field(default=None, repr=False)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"model_id": self.model_id}
        if self.filename is not None:
            out["filename"] = self.filename

        if not self.details:
            out["memory"] = self.memory
            out["kv_cache"] = self.kv_cache
            out["total_memory"] = self.total_memory
            return out

        if self.safetensors is not None:
            out["memory"] = {
                "bytes": self.safetensors.bytes_count,
                "components": {
                    name: {
                        "bytes": comp.bytes_count,
                        "param_count": comp.param_count,
                        "dtypes": {
                            dtype: {"bytes": dm.bytes_count, "param_count": dm.param_count}
                            for dtype, dm in comp.dtypes.items()
                        },
                    }
                    for name, comp in self.safetensors.components.items()
                },
            }
            out["kv_cache"] = (
                {
                    "bytes": self.kv_cache_metadata.cache_size,
                    "dtype": self.kv_cache_metadata.cache_dtype,
                    "max_model_len": self.kv_cache_metadata.max_model_len,
                    "batch_size": self.kv_cache_metadata.batch_size,
                }
                if self.kv_cache_metadata is not None
                else None
            )

        elif self.gguf_files is not None:
            if self.filename is not None:
                gguf_meta = self.gguf_files[self.filename]
                out["memory"] = {
                    "bytes": gguf_meta.bytes_count,
                    "components": {
                        name: {
                            "bytes": comp.bytes_count,
                            "param_count": comp.param_count,
                            "dtypes": {
                                k.name: {"bytes": dm.bytes_count, "param_count": dm.param_count}
                                for k, dm in comp.dtypes.items()
                            },
                        }
                        for name, comp in gguf_meta.components.items()
                    },
                }
                out["kv_cache"] = (
                    {
                        "bytes": gguf_meta.kv_cache.cache_size,
                        "dtype": gguf_meta.kv_cache.cache_dtype,
                        "max_model_len": gguf_meta.kv_cache.max_model_len,
                        "batch_size": gguf_meta.kv_cache.batch_size,
                    }
                    if gguf_meta.kv_cache is not None
                    else None
                )
            else:
                out["memory"] = {
                    fn: {
                        "bytes": m.bytes_count,
                        "components": {
                            name: {
                                "bytes": comp.bytes_count,
                                "param_count": comp.param_count,
                                "dtypes": {
                                    k.name: {"bytes": dm.bytes_count, "param_count": dm.param_count}
                                    for k, dm in comp.dtypes.items()
                                },
                            }
                            for name, comp in m.components.items()
                        },
                    }
                    for fn, m in self.gguf_files.items()
                }
                has_kv = any(m.kv_cache is not None for m in self.gguf_files.values())
                out["kv_cache"] = (
                    {
                        fn: {
                            "bytes": m.kv_cache.cache_size,
                            "dtype": m.kv_cache.cache_dtype,
                            "max_model_len": m.kv_cache.max_model_len,
                            "batch_size": m.kv_cache.batch_size,
                        }
                        for fn, m in self.gguf_files.items()
                        if m.kv_cache is not None
                    }
                    if has_kv
                    else None
                )

        out["total_memory"] = self.total_memory
        return out


def _collect_gguf_results(
    results: List[Tuple[str, GGUFMetadata, re.Match | None]],
) -> Dict[str, GGUFMetadata]:
    gguf_files: Dict[str, GGUFMetadata] = {}
    for path, metadata, shard_pattern in results:
        if shard_pattern:
            # NOTE: e.g., `base_name` is `Kimi-K2.5-BF16.gguf`
            base_name = shard_pattern.group(1) + ".gguf"
            if base_name in gguf_files:
                gguf_files[base_name] = merge_shards(gguf_files[base_name], metadata)
            else:
                gguf_files[base_name] = metadata
        else:
            gguf_files[path] = metadata
    return gguf_files


async def arun(
    model_id: str,
    revision: str = "main",
    hf_token: str | None = None,
    experimental: bool = False,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str = "auto",
    gguf_file: str | None = None,
    details: bool = False,
) -> Result:
    headers: Dict[str, str] = {
        "User-Agent": f"hf-mem/{__version__}; id={uuid4()}; model_id={model_id}; revision={revision}"
    }

    # NOTE: The Hugging Face Hub token is not only required to read the files for gated / private models, but also
    # to benefit from more generous request tiers to the Hub
    if hf_token is not None:
        headers["Authorization"] = f"Bearer {hf_token}"
    # NOTE: Read from `HF_TOKEN` if provided
    elif token := os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    # NOTE: If neither the `--hf-token` is provided nor the `HF_TOKEN` is set, then fallback to reading from
    # `$HF_HOME/token`
    elif "Authorization" not in headers:
        path = os.getenv("HF_HOME", ".cache/huggingface")
        filename = (
            os.path.join(os.path.expanduser("~"), path, "token")
            if not os.path.isabs(path)
            else os.path.join(path, "token")
        )
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                headers["Authorization"] = f"Bearer {f.read().strip()}"

    client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=MAX_CONCURRENCY,
            max_connections=MAX_CONCURRENCY,
        ),
        timeout=httpx.Timeout(30.0),
        # NOTE: HTTP/2 for header-compression and connection multiplexing
        http2=True,
        follow_redirects=True,
    )
    url = f"https://huggingface.co/api/models/{model_id}/tree/{revision}?recursive=true"
    files = await get_json_file(client=client, url=url, headers=headers)
    file_paths = [f["path"] for f in files if f.get("path") and f.get("type") == "file"]

    # NOTE: Exclude the `mmproj-*` files as those are the multimodal projection and not the language model
    # weights per se, so excluding those from the estimation (especifically when `--experimental`)
    # See https://github.com/alvarobartt/hf-mem/issues/47
    gguf_paths = [f for f in file_paths if str(f).endswith(".gguf") and not f.__contains__("mmproj-")]
    has_safetensors = any(
        f in ["model.safetensors", "model.safetensors.index.json", "model_index.json"] for f in file_paths
    )
    gguf = gguf_file is not None or (gguf_paths and not has_safetensors)

    if not gguf and (has_safetensors and gguf_paths):
        warnings.warn(
            f"Both Safetensors and GGUF files have been found for {model_id} @ {revision}, if you want to estimate any of the GGUF file sizes, please use the `--gguf-file` flag with the path to the specific GGUF file. GGUF files found: {gguf_paths}."
        )

    if gguf:
        if kv_cache_dtype not in GGUFDtype.__members__ and kv_cache_dtype != "auto":
            raise RuntimeError(
                f"--kv-cache-dtype={kv_cache_dtype} not recognized for GGUF files. Valid options: {list(GGUFDtype.__members__.keys())} or `auto`."
            )

        if not gguf_paths:
            raise RuntimeError(f"No GGUF files found for {model_id} @ {revision}.")

        if gguf_file:
            # NOTE: Check if it's a sharded file (e.g. model-00001-of-00046.gguf)
            if prefix_match := re.match(r"(.+)-\d+-of-\d+\.gguf$", gguf_file):
                prefix = prefix_match.group(1)
                gguf_paths = [
                    path
                    for path in gguf_paths
                    if re.match(rf"{re.escape(prefix)}-\d+-of-\d+\.gguf$", str(path))
                ]
            else:
                gguf_paths = [path for path in gguf_paths if str(path).endswith(gguf_file)]
                if len(gguf_paths) > 1:
                    raise RuntimeError(
                        f"Multiple GGUF files named `{gguf_file}` found for {model_id} @ {revision}."
                    )

            if not gguf_paths:
                raise RuntimeError(f"No GGUF file matching `{gguf_file}` found for {model_id} @ {revision}.")

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        tasks = []
        for path in gguf_paths:
            shard_pattern = _SHARD_PATTERN.match(str(path))
            # NOTE: For sharded files, parsing KV cache data on shards > 1 may fail (missing fields)
            parse_kv_cache = experimental and (not shard_pattern or int(shard_pattern.group(2)) == 1)
            tasks.append(
                asyncio.create_task(
                    fetch_gguf_with_semaphore(
                        semaphore=semaphore,
                        client=client,
                        model_id=model_id,
                        revision=revision,
                        path=path,
                        parse_kv_cache=parse_kv_cache,
                        shard_pattern=shard_pattern,
                        max_model_len=max_model_len,
                        kv_cache_dtype=kv_cache_dtype or "auto",
                        batch_size=batch_size,
                        headers=headers,
                    )
                )
            )

        gguf_files_dict = _collect_gguf_results(await asyncio.gather(*tasks))
        await client.aclose()

        if gguf_file is not None:
            single_filename = next(iter(gguf_files_dict))
            gguf_meta = gguf_files_dict[single_filename]
            kv_bytes = gguf_meta.kv_cache.cache_size if gguf_meta.kv_cache is not None else None
            return Result(
                model_id=model_id,
                revision=revision,
                filename=single_filename,
                memory=gguf_meta.bytes_count,
                kv_cache=kv_bytes,
                total_memory=gguf_meta.bytes_count + (kv_bytes or 0),
                details=details,
                gguf_files=gguf_files_dict,
            )
        else:
            memory_dict: Dict[str, int] = {fn: m.bytes_count for fn, m in gguf_files_dict.items()}
            has_kv = any(m.kv_cache is not None for m in gguf_files_dict.values())
            kv_dict: Union[Dict[str, int], None] = (
                {fn: m.kv_cache.cache_size for fn, m in gguf_files_dict.items() if m.kv_cache is not None}
                if has_kv
                else None
            )
            first_file = next(iter(gguf_files_dict))
            warnings.warn(
                f"Multiple GGUF files found — `total_memory` is not set for multi-file results. "
                f"For a single-file estimate pass `gguf_file='{first_file}'` (library) or "
                f"`--gguf-file {first_file}` (CLI)."
            )
            return Result(
                model_id=model_id,
                revision=revision,
                filename=None,
                memory=memory_dict,
                kv_cache=kv_dict,
                total_memory=None,
                details=details,
                gguf_files=gguf_files_dict,
            )

    if "model.safetensors" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors"
        raw_metadata = await fetch_safetensors_metadata(client=client, url=url, headers=headers)

        if "config_sentence_transformers.json" in file_paths:
            dense_metadata = (
                {}
                if "modules.json" not in file_paths
                else await fetch_modules_and_dense_metadata(
                    client=client,
                    url=f"https://huggingface.co/{model_id}/resolve/{revision}",
                    headers=headers,
                )
            )
            raw_metadata = {"0_Transformer": raw_metadata, **dense_metadata}
        else:
            # NOTE: If the model is a transformers model, then we simply set the component name to `Transformer`, to
            # make sure that we provide the expected input to the `parse_safetensors_metadata`
            raw_metadata = {"Transformer": raw_metadata}

        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)

    elif "model.safetensors.index.json" in file_paths:
        # TODO: We could eventually skip this request in favour of a greedy approach on trying to pull all the
        # files following the formatting `model-00000-of-00000.safetensors`
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors.index.json"
        files_index = await get_json_file(client=client, url=url, headers=headers)

        urls = {
            f"https://huggingface.co/{model_id}/resolve/{revision}/{f}"
            for f in set(files_index["weight_map"].values())
        }

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def fetch_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await fetch_safetensors_metadata(client=client, url=url, headers=headers)

        metadata_list: List[Dict[str, Any]] = await asyncio.gather(
            *[asyncio.create_task(fetch_with_semaphore(url)) for url in urls]
        )
        raw_metadata = reduce(lambda acc, m: acc | m, metadata_list, {})

        if "config_sentence_transformers.json" in file_paths:
            dense_metadata = (
                {}
                if "modules.json" not in file_paths
                else await fetch_modules_and_dense_metadata(
                    client=client,
                    url=f"https://huggingface.co/{model_id}/resolve/{revision}",
                    headers=headers,
                )
            )
            raw_metadata = {"0_Transformer": raw_metadata, **dense_metadata}
        else:
            raw_metadata = {"Transformer": raw_metadata}

        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)

    elif "model_index.json" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model_index.json"
        files_index = await get_json_file(client=client, url=url, headers=headers)
        paths = {k for k, _ in files_index.items() if not k.startswith("_")}

        path_urls: Dict[str, List[str]] = {}
        for path in paths:
            if f"{path}/diffusion_pytorch_model.safetensors" in file_paths:
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/diffusion_pytorch_model.safetensors"
                ]
            elif f"{path}/model.safetensors" in file_paths:
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/model.safetensors"
                ]
            elif f"{path}/diffusion_pytorch_model.safetensors.index.json" in file_paths:
                url = f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/diffusion_pytorch_model.safetensors.index.json"
                files_index = await get_json_file(client=client, url=url, headers=headers)
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/{f}"
                    for f in set(files_index["weight_map"].values())
                ]
            elif f"{path}/model.safetensors.index.json" in file_paths:
                url = (
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/model.safetensors.index.json"
                )
                files_index = await get_json_file(client=client, url=url, headers=headers)
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/{f}"
                    for f in set(files_index["weight_map"].values())
                ]

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def fetch_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await fetch_safetensors_metadata(client=client, url=url, headers=headers)

        # NOTE: Given that we need to fetch the Safetensors metadata for multiple components on Diffusers models,
        # to speed the download up and not block (await) the for-loop, we instead create all the tasks within a
        # for-loop then we await for those outside
        _tasks: Dict[str, List] = {
            path: [asyncio.create_task(fetch_with_semaphore(url)) for url in urls]
            for path, urls in path_urls.items()
        }
        await asyncio.gather(*[task for tasks in _tasks.values() for task in tasks])

        raw_metadata = {
            path: reduce(lambda acc, m: acc | m, [task.result() for task in tasks], {})
            for path, tasks in _tasks.items()
        }
        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)

    else:
        await client.aclose()
        raise RuntimeError(
            "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` FILES HAVE BEEN FOUND"
        )

    kv_cache_cls: KvCache | None = None
    if experimental and "config.json" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
        config: Dict[str, Any] = await get_json_file(client, url, headers)

        if not any(
            arch.__contains__("ForCausalLM") or arch.__contains__("ForConditionalGeneration")
            for arch in config.get("architectures", [])
        ):
            warnings.warn(
                "`experimental=True` was set, but either `config.json` doesn't have the `architectures` key meaning that the model architecture cannot be inferred, or rather that it's neither `...ForCausalLM` nor `...ForConditionalGeneration`, meaning that the KV Cache estimation might not apply. If that's the case, then set `experimental=False` to suppress this warning."
            )
        else:
            if (
                any(arch.__contains__("ForConditionalGeneration") for arch in config["architectures"])
                and "text_config" in config
            ):
                warnings.warn(
                    f"Given that `model_id={model_id}` is a `...ForConditionalGeneration` model, then the configuration from `config.json` will be retrieved from the key `text_config` instead."
                )
                text_config = config["text_config"]

                if referenced_model := text_config.get("_name_or_path"):
                    referenced_url = f"https://huggingface.co/{referenced_model}/resolve/{revision}/config.json"
                    warnings.warn(
                        f"The `text_config` contains `_name_or_path={referenced_model}`, so fetching the config from `{referenced_model}` to retrieve the required fields for KV cache estimation."
                    )
                    referenced_config = await get_json_file(client, referenced_url, headers)
                    referenced_config.update(text_config)
                    text_config = referenced_config

                for key in ("dtype", "torch_dtype", "quantization_config"):
                    if key in config and key not in text_config:
                        text_config[key] = config[key]

                config = text_config

            if max_model_len is None:
                max_model_len = config.get(
                    "max_position_embeddings",
                    config.get("n_positions", config.get("max_seq_len", max_model_len)),
                )

            if max_model_len is None:
                warnings.warn(
                    "Either `max_model_len` was not set, is not available in `config.json` under any of `max_position_embeddings`, `n_positions`, or `max_seq_len` (in that order of priority), or both; so the memory required to fit the context length cannot be estimated."
                )
            elif not all(k in config for k in {"hidden_size", "num_hidden_layers", "num_attention_heads"}):
                warnings.warn(
                    f"`config.json` doesn't contain all the required keys `hidden_size`, `num_hidden_layers`, and `num_attention_heads`, but only {list(config.keys())}."
                )
            else:
                cache_dtype = resolve_kv_cache_dtype(
                    config=config,
                    kv_cache_dtype=kv_cache_dtype,
                    metadata=metadata,
                    model_id=model_id,
                )
                kv_cache_cls = KvCache(
                    max_model_len=max_model_len,
                    cache_size=compute_safetensors_kv_cache_size(
                        config=config,
                        cache_dtype=cache_dtype,
                        max_model_len=max_model_len,
                        batch_size=batch_size,
                    ),
                    batch_size=batch_size,
                    cache_dtype=cache_dtype,
                )

    await client.aclose()
    kv_bytes = kv_cache_cls.cache_size if kv_cache_cls is not None else None
    return Result(
        model_id=model_id,
        revision=revision,
        filename=None,
        memory=metadata.bytes_count,
        kv_cache=kv_bytes,
        total_memory=metadata.bytes_count + (kv_bytes or 0),
        details=details,
        safetensors=metadata,
        kv_cache_metadata=kv_cache_cls,
    )


def run(
    model_id: str,
    revision: str = "main",
    hf_token: str | None = None,
    experimental: bool = False,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str = "auto",
    gguf_file: str | None = None,
    details: bool = False,
) -> Result:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            arun(
                model_id=model_id,
                revision=revision,
                hf_token=hf_token,
                experimental=experimental,
                max_model_len=max_model_len,
                batch_size=batch_size,
                kv_cache_dtype=kv_cache_dtype,
                gguf_file=gguf_file,
                details=details,
            )
        )

    raise RuntimeError(
        "`hf_mem.run(...)` is synchronous and cannot be used from an active event loop. "
        "Use `await hf_mem.run.arun(...)` instead."
    )
