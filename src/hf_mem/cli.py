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

from hf_mem.metadata import parse_safetensors_metadata
from hf_mem.print import print_report, print_report_for_gguf

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 10.0))
MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))


# NOTE: Return type-hint set to `Any`, but it will only be a JSON-compatible object
async def get_json_file(client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


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
        # Check for GGUF files
        gguf_files = {
            f["path"]: f["size"]
            for f in files
            if f.get("path", "").endswith(".gguf") and f.get("size") is not None
        }

        if gguf_files:
            print_report_for_gguf(
                model_id=model_id,
                revision=revision,
                gguf_files=gguf_files,
                ignore_table_width=ignore_table_width,
            )
            return
        else:
            raise RuntimeError(
                "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json`, OR `.gguf` FILES HAVE BEEN FOUND"
            )

    if json_output:
        out = {"model_id": model_id, "revision": revision, **asdict(metadata)}
        print(json.dumps(out))
    else:
        print_report(
            model_id=model_id,
            revision=revision,
            metadata=metadata,
            ignore_table_width=ignore_table_width,
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

    args = parser.parse_args()

    asyncio.run(
        run(
            model_id=args.model_id,
            revision=args.revision,
            # NOTE: Below are the arguments that affect the output format
            json_output=args.json_output,
            ignore_table_width=args.ignore_table_width,
        )
    )
