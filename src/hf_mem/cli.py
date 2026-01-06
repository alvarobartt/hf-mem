import argparse
import asyncio
import json
import os
import struct
from functools import reduce
from typing import Any, Dict, List, Optional

import httpx

from hf_mem.metadata import parse_safetensors_metadata
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
    headers["Range"] = f"bytes={MAX_METADATA_SIZE + 1}-{metadata_size + 7}"

    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata += response.read()
    return json.loads(metadata)


async def run(
    model_id: str,
    revision: str,
    json_output: bool = False,
    ignore_table_width: bool = False,
) -> Dict[str, Any] | None:
    headers = {}
    # NOTE: Read from `HF_TOKEN` if provided, then fallback to reading from `$HF_HOME/token`
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"
    if "Authorization" not in headers:
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
        raw_metadata = await fetch_safetensors_metadata(
            client=client, url=url, headers=headers
        )

        # NOTE: Small "hack" so that the default component for Sentence Transformers models is not "transformer"
        # but rather "0_Transformer" following the current Sentence Transformers convention as defined in modules.json
        if "config_sentence_transformers.json" in file_paths:
            raw_metadata = {"0_Transformer": raw_metadata}

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

        async def fetch_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await fetch_safetensors_metadata(
                    client=client, url=url, headers=headers
                )

        tasks = [asyncio.create_task(fetch_semaphore(url)) for url in urls]
        metadata_list: List[Dict[str, Any]] = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        raw_metadata = reduce(lambda acc, metadata: acc | metadata, metadata_list, {})

        # NOTE: Small "hack" so that the default component for Sentence Transformers models is not "transformer"
        # but rather "0_Transformer" following the current Sentence Transformers convention as defined in modules.json
        if "config_sentence_transformers.json" in file_paths:
            raw_metadata = {"0_Transformer": raw_metadata}

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
                files_index = await get_json_file(
                    client=client, url=url, headers=headers
                )
                path_urls[path] = list({
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/{f}"
                    for _, f in files_index["weight_map"].items()  # type: ignore
                })
            elif f"{path}/model.safetensors.index.json" in file_paths:
                url = f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/model.safetensors.index.json"
                files_index = await get_json_file(
                    client=client, url=url, headers=headers
                )
                path_urls[path] = list({
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/{f}"
                    for _, f in files_index["weight_map"].items()  # type: ignore
                })

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def fetch_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await fetch_safetensors_metadata(
                    client=client, url=url, headers=headers
                )

        raw_metadata = {}
        for path, urls in path_urls.items():
            tasks = [asyncio.create_task(fetch_semaphore(url)) for url in urls]
            metadata_list: List[Dict[str, Any]] = await asyncio.gather(
                *tasks, return_exceptions=False
            )
            raw_metadata[path] = reduce(
                lambda acc, metadata: acc | metadata, metadata_list, {}
            )

        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    else:
        raise RuntimeError(
            "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` HAS BEEN FOUND"
        )

    if json_output:
        out = {"model_id": model_id, "revision": revision}
        out.update(metadata.to_dict())
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
    parser.add_argument(
        "--model-id", required=True, help="Model ID on the Hugging Face Hub"
    )
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
