import asyncio
import json
import os
import re
import warnings
from dataclasses import asdict
from functools import reduce
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import httpx

from hf_mem._fetch import get_json_file
from hf_mem._version import __version__
from hf_mem.gguf.fetch import fetch_gguf_with_semaphore
from hf_mem.gguf.metadata import GGUFDtype, GGUFMetadata, gguf_metadata_to_json, merge_shards
from hf_mem.gguf.print import print_gguf_files_report, print_gguf_report
from hf_mem.safetensors.fetch import fetch_modules_and_dense_metadata, fetch_safetensors_metadata
from hf_mem.safetensors.kv_cache import compute_safetensors_kv_cache_size, resolve_kv_cache_dtype
from hf_mem.safetensors.metadata import parse_safetensors_metadata
from hf_mem.safetensors.print import print_safetensors_report

MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))

# NOTE: Shard filename pattern, e.g. "Kimi-K2.5-BF16-00001-of-00046.gguf"
_SHARD_PATTERN = re.compile(r"(.+)-(\d+)-of-(\d+)\.gguf$")


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


async def run(
    model_id: str,
    revision: str,
    hf_token: str | None = None,
    # START_KV_CACHE_ARGS
    experimental: bool = False,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str | None = None,
    # END_KV_CACHE_ARGS
    json_output: bool = False,
    ignore_table_width: bool = False,
    gguf_file: str | None = None,
) -> Dict[str, Any] | None:
    headers = {
        "User-Agent": f"hf-mem/{__version__}; id={uuid4()}; model_id={model_id}; revision={revision}"  # type: ignore
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

    gguf_paths = [f for f in file_paths if str(f).endswith(".gguf")]
    has_safetensors = any(
        f in ["model.safetensors", "model.safetensors.index.json", "model_index.json"] for f in file_paths
    )
    gguf = gguf_file is not None or (gguf_paths and not has_safetensors)

    if not gguf and (has_safetensors and gguf_paths):
        warnings.warn(
            f"Both Safetensors and GGUF files have been found for {model_id} @ {revision}, if you want to estimate any of the GGUF file sizes, please use the `--gguf-file` flag with the path to the specific GGUF file. GGUF files found: {gguf_paths}."
        )

    # START_GGUF
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

        gguf_files = _collect_gguf_results(await asyncio.gather(*tasks))

        if json_output:
            records = [
                gguf_metadata_to_json(model_id=filename, revision=revision, metadata=gguf_metadata)
                for filename, gguf_metadata in gguf_files.items()
            ]
            print(json.dumps(records[0] if gguf_file else records))
        else:
            if gguf_file:
                gguf_metadata = list(gguf_files.values())[0]
                cache = (
                    {
                        "max_model_len": gguf_metadata.kv_cache_info.max_model_len,
                        "cache_size": gguf_metadata.kv_cache_info.cache_size,
                        "batch_size": gguf_metadata.kv_cache_info.batch_size,
                        "cache_dtype": gguf_metadata.kv_cache_info.cache_dtype,
                    }
                    if experimental and gguf_metadata.kv_cache_info is not None
                    else None
                )
                print_gguf_report(
                    model_id=list(gguf_files.keys())[0],
                    revision=revision,
                    metadata=gguf_metadata,
                    cache=cache,
                    ignore_table_width=ignore_table_width,
                )
            else:
                print_gguf_files_report(
                    model_id=model_id,
                    revision=revision,
                    gguf_files=gguf_files,
                    ignore_table_width=ignore_table_width,
                )
        return
    # END_GGUF
    # START_SAFETENSORS
    elif "model.safetensors" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors"
        raw_metadata = await fetch_safetensors_metadata(client=client, url=url, headers=headers)

        if "config_sentence_transformers.json" in file_paths:
            dense_metadata = (
                {}
                if "modules.json" not in file_paths
                else await fetch_modules_and_dense_metadata(
                    client=client, url=f"https://huggingface.co/{model_id}/resolve/{revision}", headers=headers
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
                    client=client, url=f"https://huggingface.co/{model_id}/resolve/{revision}", headers=headers
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
        raise RuntimeError(
            "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` FILES HAVE BEEN FOUND"
        )

    cache_size = None
    if experimental and "config.json" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
        config: Dict[str, Any] = await get_json_file(client, url, headers)

        if not any(
            arch.__contains__("ForCausalLM") or arch.__contains__("ForConditionalGeneration")
            for arch in config.get("architectures", [])
        ):
            warnings.warn(
                "`--experimental` was provided, but either `config.json` doesn't have the `architectures` key meaning that the model architecture cannot be inferred, or rather that it's neither `...ForCausalLM` not `...ForConditionalGeneration`, meaning that the KV Cache estimation might not apply. If that's the case, then remove the `--experimental` flag from the command to suppress this warning."
            )
        else:
            if (
                any(arch.__contains__("ForConditionalGeneration") for arch in config["architectures"])
                and "text_config" in config
            ):
                warnings.warn(
                    f"Given that `--model-id={model_id}` is a `...ForConditionalGeneration` model, then the configuration from `config.json` will be retrieved from the key `text_config` instead."
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

                config = text_config

            if max_model_len is None:
                max_model_len = config.get(
                    "max_position_embeddings",
                    config.get("n_positions", config.get("max_seq_len", max_model_len)),
                )

            if max_model_len is None:
                warnings.warn(
                    "Either the `--max-model-len` was not set, is not available in `config.json` with the any of the keys: `max_position_embeddings`, `n_positions`, or `max_seq_len` (in that order of priority), or both; so the memory required to fit the context length cannot be estimated."
                )

            if not all(k in config for k in {"hidden_size", "num_hidden_layers", "num_attention_heads"}):  # type: ignore
                warnings.warn(
                    f"`config.json` doesn't contain all the keys `hidden_size`, `num_hidden_layers`, and `num_attention_heads`, but only {config.keys()}."  # type: ignore
                )

            cache_dtype = resolve_kv_cache_dtype(
                config=config,
                kv_cache_dtype=kv_cache_dtype,
                metadata=metadata,
                model_id=model_id,
            )
            cache_size = compute_safetensors_kv_cache_size(
                config=config,
                cache_dtype=cache_dtype,
                max_model_len=max_model_len,  # type: ignore[arg-type]
                batch_size=batch_size,
            )

    if json_output:
        out = {"version": __version__, "model_id": model_id, "revision": revision, **asdict(metadata)}
        if experimental and cache_size:
            out["max_model_len"] = max_model_len
            out["batch_size"] = batch_size
            out["cache_size"] = cache_size
            out["cache_dtype"] = cache_dtype  # type: ignore
        print(json.dumps(out))
    else:
        # TODO: Use a `KvCache` dataclass instead and make sure that the JSON output is aligned
        if experimental and cache_size:
            print_safetensors_report(
                model_id=model_id,
                revision=revision,
                metadata=metadata,
                cache={
                    "max_model_len": max_model_len,
                    "cache_size": cache_size,
                    "batch_size": batch_size,
                    "cache_dtype": cache_dtype,  # type: ignore
                },
                ignore_table_width=ignore_table_width,
            )
        else:
            print_safetensors_report(
                model_id=model_id,
                revision=revision,
                metadata=metadata,
                ignore_table_width=ignore_table_width,
            )
    # END_SAFETENSORS
