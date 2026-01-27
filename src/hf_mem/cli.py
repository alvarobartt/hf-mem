import argparse
import asyncio
import json
import os
import struct
from dataclasses import asdict
from functools import reduce
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

from hf_mem.config import fetch_model_config, get_json_file
from hf_mem.inference import calculate_inference_estimate
from hf_mem.metadata import InferenceEstimate, ModelConfig, parse_safetensors_metadata
from hf_mem.print import print_report

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 10.0))
MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))


async def fetch_safetensors_metadata(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    headers = {"Range": f"bytes=0-{MAX_METADATA_SIZE}", **(headers or {})}
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata = response.read()
    # NOTE: Parse the first 8 bytes as a little-endian uint64 (size of the metadata)
    metadata_size = struct.unpack("<Q", metadata[:8])[0]

    if metadata_size < MAX_METADATA_SIZE:
        metadata = metadata[8 : metadata_size + 8]
        return json.loads(metadata)

    # NOTE: Given that by default we just fetch the first 100_000 bytes, if the content is larger
    # then we simply fetch the remainder again
    metadata = metadata[8 : MAX_METADATA_SIZE + 8]
    headers["Range"] = f"bytes={MAX_METADATA_SIZE + 1}-{metadata_size + 7}"

    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata += response.read()
    return json.loads(metadata)


async def fetch_modules_and_dense_metadata(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]]
) -> Dict[str, Any]:
    dense_metadata = {}

    modules = await get_json_file(client=client, url=f"{url}/modules.json", headers=headers)
    paths = [
        module.get("path")
        for module in modules
        if "type" in module and module.get("type") == "sentence_transformers.models.Dense" and "path" in module
    ]

    for path in paths:
        # NOTE: It's "safe" to assume that if there's a `Dense` module defined in `modules.json`, it contains
        # Safetensors weights and if so, it's a single `model.safetensors` file as the sharding has a default on
        # ~5Gb per file, and usually the extra `Dense` layers are not larger than that (usually not even close).
        dense_metadata[path] = await fetch_safetensors_metadata(
            client=client, url=f"{url}/{path}/model.safetensors", headers=headers
        )

    return dense_metadata


async def run(
    model_id: str,
    revision: str,
    json_output: bool = False,
    ignore_table_width: bool = False,
    context_length: Optional[int] = None,
    batch_size: int = 1,
    concurrent_requests: int = 1,
    no_kv_cache: bool = False,
) -> Dict[str, Any] | None:
    headers = {"User-Agent": f"hf-mem/0.3; id={uuid4()}; model_id={model_id}; revision={revision}"}
    # NOTE: Read from `HF_TOKEN` if provided, then fallback to reading from `$HF_HOME/token`
    if token := os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
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
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        # NOTE: HTTP/2 for header-compression and connection multiplexing
        http2=True,
        follow_redirects=True,
    )

    # TODO: `recursive=true` shouldn't really be required unless it's a Diffusers
    # models... I don't think this adds extra latency anyway
    url = f"https://huggingface.co/api/models/{model_id}/tree/{revision}?recursive=true"
    files = await get_json_file(client=client, url=url, headers=headers)
    file_paths = [f["path"] for f in files if f.get("path") and f.get("type") == "file"]

    # Fetch model config and generation config (if KV cache estimation is enabled)
    model_config: Optional[ModelConfig] = None
    generation_config: Optional[Dict[str, Any]] = None
    effective_context_length = context_length

    if not no_kv_cache and "config.json" in file_paths:
        model_config = await fetch_model_config(client, model_id, revision, headers, REQUEST_TIMEOUT)

        # Fetch generation_config only if file exists
        if "generation_config.json" in file_paths:
            url = f"https://huggingface.co/{model_id}/resolve/{revision}/generation_config.json"
            try:
                generation_config = await get_json_file(client, url, headers, REQUEST_TIMEOUT)
            except httpx.HTTPStatusError:
                generation_config = None

        # Determine context length: CLI arg > max_position_embeddings > generation_config.max_length > 2048
        if effective_context_length is None:
            if model_config:
                effective_context_length = model_config.max_position_embeddings
            elif generation_config and "max_length" in generation_config:
                effective_context_length = generation_config["max_length"]
            else:
                effective_context_length = 2048

    if "model.safetensors" in file_paths:
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

        tasks = [asyncio.create_task(fetch_with_semaphore(url)) for url in urls]
        metadata_list: List[Dict[str, Any]] = await asyncio.gather(*tasks, return_exceptions=False)

        raw_metadata = reduce(lambda acc, metadata: acc | metadata, metadata_list, {})

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
        _tasks = {}
        for path, urls in path_urls.items():
            _tasks[path] = [asyncio.create_task(fetch_with_semaphore(url)) for url in urls]
        await asyncio.gather(*[task for tasks in _tasks.values() for task in tasks], return_exceptions=False)

        raw_metadata = {}
        for path, tasks in _tasks.items():
            metadata_list = [task.result() for task in tasks]
            raw_metadata[path] = reduce(lambda acc, metadata: acc | metadata, metadata_list, {})

        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    else:
        raise RuntimeError(
            "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` HAS BEEN FOUND"
        )

    # Calculate inference estimate (weights + KV cache)
    inference_estimate: Optional[InferenceEstimate] = None
    if effective_context_length is not None:
        inference_estimate = calculate_inference_estimate(
            weights_bytes=metadata.bytes_count,
            model_config=model_config,
            batch_size=batch_size,
            context_length=effective_context_length,
            concurrent_requests=concurrent_requests,
        )

    if json_output:
        out = {"model_id": model_id, "revision": revision, **asdict(metadata)}
        if inference_estimate:
            out["inference"] = asdict(inference_estimate)
        print(json.dumps(out))
    else:
        print_report(
            model_id=model_id,
            revision=revision,
            metadata=metadata,
            ignore_table_width=ignore_table_width,
            inference_estimate=inference_estimate,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="Model ID on the Hugging Face Hub")
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision on the Hugging Face Hub",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Whether to provide the output as a JSON instead of printed as table.",
    )
    parser.add_argument(
        "--ignore-table-width",
        action="store_true",
        help="Whether to ignore the maximum recommended table width, in case the `--model-id` and/or `--revision` cause a row overflow when printing those.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length for KV cache estimation (default: from generation_config or max_position_embeddings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for KV cache estimation (default: 1)",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests for KV cache estimation (default: 1)",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Skip KV cache estimation",
    )

    args = parser.parse_args()

    asyncio.run(
        run(
            model_id=args.model_id,
            revision=args.revision,
            # NOTE: Below are the arguments that affect the output format
            json_output=args.json_output,
            ignore_table_width=args.ignore_table_width,
            # NOTE: Below are the arguments that affect the KV cache estimation
            context_length=args.context_length,
            batch_size=args.batch_size,
            concurrent_requests=args.concurrent_requests,
            no_kv_cache=args.no_kv_cache,
        )
    )
