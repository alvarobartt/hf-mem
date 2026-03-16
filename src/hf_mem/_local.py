import json
import os
import struct
from typing import Any, Dict, List

# Cap for GGUF header reads — metadata sections rarely exceed this
_MAX_GGUF_READ_SIZE = 100_000_000  # 100 MB


def list_local_files(directory: str) -> List[str]:
    """Walk directory and return relative file paths, following symlinks."""
    file_paths = []
    for root, _dirs, files in os.walk(directory, followlinks=True):
        for f in files:
            full_path = os.path.join(root, f)
            # Skip broken symlinks
            if not os.path.exists(full_path):
                continue
            rel_path = os.path.relpath(full_path, directory)
            file_paths.append(rel_path)
    return file_paths


def read_safetensors_header(filepath: str) -> Dict[str, Any]:
    """Read safetensors metadata header from a local file.

    Returns the same dict shape as fetch_safetensors_metadata() returns from HTTP.
    """
    with open(filepath, "rb") as f:
        size_bytes = f.read(8)
        if len(size_bytes) < 8:
            raise RuntimeError(f"File too small to be a valid safetensors file: {filepath}")
        metadata_size = struct.unpack("<Q", size_bytes)[0]
        metadata_bytes = f.read(metadata_size)
        if len(metadata_bytes) < metadata_size:
            raise RuntimeError(
                f"Safetensors header truncated in {filepath}: expected {metadata_size} bytes, got {len(metadata_bytes)}"
            )
    raw = json.loads(metadata_bytes)
    # Remove __metadata__ key to match the shape returned by HTTP fetch
    # (the HTTP path returns the full dict including __metadata__, and
    # parse_safetensors_metadata skips it via the `if key in {"__metadata__"}` check)
    return raw


def read_local_json(filepath: str) -> Any:
    """Read and parse a local JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def read_gguf_bytes(filepath: str) -> bytes:
    """Read GGUF file header bytes (up to 100MB) for metadata parsing."""
    file_size = os.path.getsize(filepath)
    read_size = min(file_size, _MAX_GGUF_READ_SIZE)
    with open(filepath, "rb") as f:
        return f.read(read_size)
