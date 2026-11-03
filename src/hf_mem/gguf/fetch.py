import asyncio
import re
import struct
import warnings
from typing import Dict, Tuple

import httpx2

from hf_mem._fetch import REQUEST_TIMEOUT, HubError, raise_for_status
from hf_mem.gguf.metadata import GGUFMetadata, parse_gguf_metadata

_INITIAL_FETCH_SIZE = 1_000_000
_MAX_FETCH_SIZE = 100_000_000


async def fetch_gguf_metadata(
    client: httpx2.AsyncClient,
    url: str,
    experimental: bool = False,
    max_model_len: int | None = None,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
    headers: Dict[str, str] | None = None,
) -> GGUFMetadata:
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "F16"

    raw_metadata, file_size, size = b"", None, _INITIAL_FETCH_SIZE
    while True:
        request_headers = {"Range": f"bytes={len(raw_metadata)}-{size - 1}", **(headers or {})}
        response = await client.get(url, headers=request_headers, timeout=REQUEST_TIMEOUT)
        raise_for_status(response)
        raw_metadata += response.read()
        content_range = response.headers.get("content-range", "")
        file_size = int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else len(raw_metadata)

        try:
            return parse_gguf_metadata(
                raw_metadata=raw_metadata,
                experimental=experimental,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
                batch_size=batch_size,
                file_size=file_size,
            )
        except (struct.error, UnicodeDecodeError):
            if size >= _MAX_FETCH_SIZE or len(raw_metadata) >= file_size:
                raise RuntimeError(
                    f"Failed to parse GGUF metadata from {url}, the metadata section exceeds {_MAX_FETCH_SIZE // 1_000_000} MB."
                )
            size = min(size * 2, _MAX_FETCH_SIZE)


async def fetch_gguf_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: httpx2.AsyncClient,
    model_id: str,
    revision_encoded: str,
    path: str,
    parse_kv_cache: bool,
    shard_pattern: re.Match | None,
    max_model_len: int | None = None,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
    headers: Dict[str, str] | None = None,
    skip_invalid: bool = False,
) -> Tuple[str, GGUFMetadata | None, re.Match | None]:
    async with semaphore:
        try:
            metadata = await fetch_gguf_metadata(
                client=client,
                url=f"https://huggingface.co/{model_id}/resolve/{revision_encoded}/{path}",
                experimental=parse_kv_cache,
                max_model_len=max_model_len,
                kv_cache_dtype=kv_cache_dtype,
                batch_size=batch_size,
                headers=headers,
            )
        except RuntimeError as e:
            if not skip_invalid or isinstance(e, HubError):
                raise
            warnings.warn(f"Skipping `{path}`: {e}")
            metadata = None
        return (str(path), metadata, shard_pattern)
