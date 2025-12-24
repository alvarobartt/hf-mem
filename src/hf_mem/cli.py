import argparse
import asyncio
import json
import math
import os
import struct
import sys
from functools import reduce
from typing import Any, Dict, List, Optional

import httpx

from hf_mem.print import print_report

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 10.0))
MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))


# NOTE: Return type-hint set to `Any`, but it will only be a JSON-compatible object
async def get_json_file(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None
) -> Any:
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
    headers["Range"] = f"bytes={MAX_METADATA_SIZE}-{metadata_size + 7}"

    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata += response.read()
    return json.loads(metadata)


async def run(model_id: str, revision: str) -> None:
    headers = {}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=MAX_CONCURRENCY,
            max_connections=MAX_CONCURRENCY,
        ),
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        http2=True,
        follow_redirects=True,
    )

    # TODO: `recursive=true` shouldn't really be required unless it's a Diffusers
    # models... I don't think this adds extra latency anyway
    url = f"https://huggingface.co/api/models/{model_id}/tree/{revision}?recursive=true"

    files = await get_json_file(client=client, url=url, headers=headers)

    file_paths = [
        f["path"]
        for f in files
        if f.get("path", None) is not None and f.get("type", None) == "file"
    ]

    if "model.safetensors" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors"
        metadata = await fetch_safetensors_metadata(
            client=client, url=url, headers=headers
        )
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

        async def fetch_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await fetch_safetensors_metadata(
                    client=client, url=url, headers=headers
                )

        tasks = [asyncio.create_task(fetch_semaphore(url)) for url in urls]
        metadata_list: List[Dict[str, Any]] = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        metadata = reduce(lambda acc, metadata: acc | metadata, metadata_list, {})
    else:
        raise RuntimeError(
            "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` HAS BEEN FOUND"
        )

    ppdt = {}
    for key, value in metadata.items():
        if key in {"__metadata__"}:
            continue
        if value["dtype"] not in ppdt:
            ppdt[value["dtype"]] = (0, 0)

        match value["dtype"]:
            case "F64" | "I64" | "U64":
                dtype_b = 8
            case "F32" | "I32" | "U32":
                dtype_b = 4
            case "F16" | "BF16" | "I16" | "U16":
                dtype_b = 2
            case "F8_E5M2" | "F8_E4M3" | "I8" | "U8":
                dtype_b = 1
            case _:
                raise RuntimeError(f"DTYPE={value['dtype']} NOT HANDLED")

        current_shape = math.prod(value["shape"])
        current_shape_bytes = current_shape * dtype_b

        ppdt[value["dtype"]] = (
            ppdt[value["dtype"]][0] + current_shape,
            ppdt[value["dtype"]][1] + current_shape_bytes,
        )

    print_report(model_id, ppdt)
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", required=True, help="Model ID on the Hugging Face Hub"
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision on the Hugging Face Hub",
    )

    args = parser.parse_args()
    model_id = args.model_id
    revision = args.revision

    asyncio.run(run(model_id, revision))
