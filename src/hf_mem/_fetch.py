import os
from typing import Any, Dict

import httpx

# NOTE: Shared across all HTTP fetch operations; overridable via the `REQUEST_TIMEOUT` environment variable
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))


# NOTE: Return type-hint set to `Any`, but it will only be a JSON-compatible object
async def get_json_file(client: httpx.AsyncClient, url: str, headers: Dict[str, str] | None = None) -> Any:
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()
