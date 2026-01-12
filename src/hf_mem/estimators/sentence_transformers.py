from typing import Any, Dict, Optional

import httpx

from .safetensors_metadata import get_json_file, fetch_safetensors_metadata


async def fetch_modules_and_dense_metadata(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None
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