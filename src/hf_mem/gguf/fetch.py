import asyncio
import os
import re
import struct
from typing import Dict, Optional, Tuple

import httpx

from hf_mem.gguf.metadata import GGUFMetadata, parse_gguf_metadata

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))

# NOTE: Progressive fetch sizes — we try each in order until the full metadata fits within the fetched range
SIZES = [1_000_000, 10_000_000, 50_000_000, 100_000_000]


async def fetch_gguf_metadata(
    client: httpx.AsyncClient,
    url: str,
    experimental: bool = False,
    max_model_len: Optional[int] = None,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
    headers: Optional[Dict[str, str]] = None,
) -> GGUFMetadata:
    # NOTE: `auto` falls back to F16 for GGUF files as there's no config.json to infer dtype from
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "F16"

    for size in SIZES:
        try:
            request_headers = {"Range": f"bytes=0-{size}", **(headers or {})}
            response = await client.get(url, headers=request_headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            raw_metadata = response.read()
            return parse_gguf_metadata(
                raw_metadata=raw_metadata,
                experimental=experimental,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
                batch_size=batch_size,
            )

        except (struct.error, UnicodeDecodeError):
            if size == SIZES[-1]:
                raise RuntimeError(
                    f"Failed to parse GGUF metadata from {url}, metadata larger than {size / 1_000_000:.2f} MB."
                )


async def fetch_gguf_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    model_id: str,
    revision: str,
    path: str,
    parse_kv_cache: bool,
    shard_pattern: Optional[re.Match],
    max_model_len: Optional[int] = None,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, GGUFMetadata, Optional[re.Match]]:
    async with semaphore:
        f_url = f"https://huggingface.co/{model_id}/resolve/{revision}/{str(path)}"
        metadata = await fetch_gguf_metadata(
            client=client,
            url=f_url,
            experimental=parse_kv_cache,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            batch_size=batch_size,
            headers=headers,
        )
        return (str(path), metadata, shard_pattern)
