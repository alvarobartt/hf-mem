import json
import os
import struct
from typing import Any, Dict, List, Optional

import httpx

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))


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
