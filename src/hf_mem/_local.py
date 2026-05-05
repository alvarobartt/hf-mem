import json
import os
import struct
from typing import Any, Dict, List

_MAX_GGUF_READ_SIZE = 100_000_000


def list_local_files(directory: str) -> List[str]:
    file_paths = []
    for root, _dirs, files in os.walk(directory, followlinks=True):
        _dirs[:] = [d for d in _dirs if not d.startswith(".")]
        for f in files:
            full_path = os.path.join(root, f)
            if not os.path.exists(full_path):
                continue
            rel_path = os.path.relpath(full_path, directory)
            file_paths.append(rel_path)
    return file_paths


def read_safetensors_header(filepath: str) -> Dict[str, Any]:
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
    return json.loads(metadata_bytes)


def read_local_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def read_gguf_bytes(filepath: str) -> bytes:
    file_size = os.path.getsize(filepath)
    read_size = min(file_size, _MAX_GGUF_READ_SIZE)
    with open(filepath, "rb") as f:
        return f.read(read_size)
