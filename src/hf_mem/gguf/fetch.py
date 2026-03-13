import asyncio
import re
import struct
from typing import Dict, Tuple

import httpx

from hf_mem._fetch import REQUEST_TIMEOUT
from hf_mem.gguf.metadata import GGUFMetadata, parse_gguf_metadata

# NOTE: Start with 1 MB; this is sufficient for the metadata of most GGUF models.
# If the metadata section is larger, the fetch size doubles on each retry up to the cap.
_INITIAL_FETCH_SIZE = 1_000_000  # 1 MB
_MAX_FETCH_SIZE = 100_000_000  # 100 MB


async def fetch_gguf_metadata(
    client: httpx.AsyncClient,
    url: str,
    experimental: bool = False,
    max_model_len: int | None = None,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
    headers: Dict[str, str] | None = None,
) -> GGUFMetadata:
    # NOTE: `auto` falls back to F16 for GGUF files as there's no `config.json` to infer dtype from
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "F16"

    size = _INITIAL_FETCH_SIZE
    while True:
        try:
            request_headers = {"Range": f"bytes=0-{size}", **(headers or {})}
            response = await client.get(url, headers=request_headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            return parse_gguf_metadata(
                raw_metadata=response.read(),
                experimental=experimental,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
                batch_size=batch_size,
            )

        except (struct.error, UnicodeDecodeError):
            if size >= _MAX_FETCH_SIZE:
                raise RuntimeError(
                    f"Failed to parse GGUF metadata from {url}, the metadata section exceeds {_MAX_FETCH_SIZE // 1_000_000} MB."
                )
            # NOTE: Double on each retry until the full metadata section fits within the fetched range
            size = min(size * 2, _MAX_FETCH_SIZE)


async def fetch_gguf_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    model_id: str,
    revision: str,
    path: str,
    parse_kv_cache: bool,
    shard_pattern: re.Match | None,
    max_model_len: int | None = None,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
    headers: Dict[str, str] | None = None,
) -> Tuple[str, GGUFMetadata, re.Match | None]:
    async with semaphore:
        metadata = await fetch_gguf_metadata(
            client=client,
            url=f"https://huggingface.co/{model_id}/resolve/{revision}/{path}",
            experimental=parse_kv_cache,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            batch_size=batch_size,
            headers=headers,
        )
        return (str(path), metadata, shard_pattern)
