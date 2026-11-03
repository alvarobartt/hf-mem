import os
from typing import Any, Dict

import httpx2

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))


def raise_for_status(response: httpx2.Response) -> None:
    if response.is_error:
        message = response.headers.get("x-error-message") or f"HTTP {response.status_code} for {response.url}"
        raise RuntimeError(message)


async def get_json_file(client: httpx2.AsyncClient, url: str, headers: Dict[str, str] | None = None) -> Any:
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    raise_for_status(response)
    return response.json()
