import json
import math
import os
import struct
from typing import Any

from hf_mem._gguf import TYPES

_BITS = {"F64": 64, "F32": 32, "F16": 16, "BF16": 16, "I64": 64, "I32": 32, "F8_E4M3": 8, "U8": 8, "I8": 8}


def write_safetensors(path: str, tensors: dict[str, tuple[str, list[int]]]) -> int:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * _BITS[dtype] // 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(8 + len(raw) + offset)
    return offset


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(value: Any) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return struct.pack("<Iq", 11, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _string(value)
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_string(v) for v in value)
        if all(isinstance(v, bool) for v in value):
            return struct.pack("<IIQ", 9, 7, len(value)) + struct.pack(f"<{len(value)}?", *value)
        return struct.pack("<IIQ", 9, 5, len(value)) + struct.pack(f"<{len(value)}i", *value)
    raise TypeError(value)


def write_gguf(path: str, metadata: dict[str, Any], tensors: list[tuple[str, int, list[int]]]) -> int:
    kv = b"".join(_string(k) + _value(v) for k, v in metadata.items())
    infos, offset, sizes = b"", 0, []
    for name, kind, shape in tensors:
        _, block, size = TYPES.get(kind, ("", 1, 1))
        nbytes = -(-math.prod(shape) // block) * size
        infos += _string(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        infos += struct.pack("<IQ", kind, offset)
        offset += -(-nbytes // 32) * 32
        sizes.append(nbytes)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)) + kv + infos
    head += b"\0" * (-len(head) % 32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + offset)
    return offset


def llama_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 1024,
        "torch_dtype": "bfloat16",
    }
    config.update(overrides)
    return config


def llama_tensors(layers: int = 4) -> dict[str, tuple[str, list[int]]]:
    tensors = {"model.embed_tokens.weight": ("BF16", [1000, 256])}
    for i in range(layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = ("BF16", [256, 256])
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = ("BF16", [512, 256])
        tensors[f"model.layers.{i}.input_layernorm.weight"] = ("F32", [256])
    return tensors
