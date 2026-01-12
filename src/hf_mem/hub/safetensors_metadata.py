import json
import struct
from typing import Any, Dict, Optional

import httpx

from ..utils.constants import MAX_METADATA_SIZE, REQUEST_TIMEOUT


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