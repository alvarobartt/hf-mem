"""Fetch and parse model configuration from HuggingFace Hub."""

from typing import Any, Dict, Optional

import httpx

from hf_mem.metadata import ModelConfig

# Default values for optional config fields
DEFAULT_MAX_POSITION_EMBEDDINGS = 2048
DEFAULT_TORCH_DTYPE = "float16"


async def fetch_json_file(
    client: httpx.AsyncClient,
    model_id: str,
    revision: str,
    filename: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Fetch a JSON file from HuggingFace Hub, returning None on 404."""
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
    try:
        response = await client.get(url, headers=headers, timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError:
        return None


async def fetch_model_config(
    client: httpx.AsyncClient,
    model_id: str,
    revision: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Optional[ModelConfig]:
    """Fetch config.json and parse required fields for KV cache calculation.

    Returns None if config.json is missing or lacks required fields.
    """
    config = await fetch_json_file(client, model_id, revision, "config.json", headers, timeout)
    if config is None:
        return None

    # Required fields - return None if any are missing
    hidden_size = config.get("hidden_size")
    num_hidden_layers = config.get("num_hidden_layers")
    num_attention_heads = config.get("num_attention_heads")

    if hidden_size is None or num_hidden_layers is None or num_attention_heads is None:
        return None

    # Optional fields with fallbacks
    # num_key_value_heads defaults to num_attention_heads (MHA)
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)

    # max_position_embeddings has multiple possible field names
    max_position_embeddings = (
        config.get("max_position_embeddings")
        or config.get("n_positions")
        or config.get("max_seq_len")
        or DEFAULT_MAX_POSITION_EMBEDDINGS
    )

    # head_dim can be explicit or derived
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)

    # torch_dtype for byte calculation
    torch_dtype = config.get("torch_dtype", DEFAULT_TORCH_DTYPE)

    return ModelConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        head_dim=head_dim,
        torch_dtype=torch_dtype,
    )


async def fetch_generation_config(
    client: httpx.AsyncClient,
    model_id: str,
    revision: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Fetch generation_config.json for max_length defaults.

    Returns None if the file doesn't exist.
    """
    return await fetch_json_file(client, model_id, revision, "generation_config.json", headers, timeout)
