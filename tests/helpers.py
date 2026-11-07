import json
import math
import os
import struct
from typing import Any, TypeVar

from hf_mem import Estimate, Listing, run
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


def kimi_config(full: int = 3, state: int = 9, sparse: bool = True) -> dict[str, Any]:
    config = llama_config(
        model_type="kimi_linear",
        architectures=["KimiLinearForCausalLM"],
        hidden_size=7168,
        num_hidden_layers=full + state,
        num_attention_heads=96,
        num_key_value_heads=96,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_position_embeddings=1048576,
        linear_attn_config={
            "full_attn_layers": list(range(1, full + 1)),
            "kda_layers": list(range(full + 1, full + state + 1)),
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
    )
    if sparse:
        config["dsa_config"] = {"dsa_layers": list(range(1, full + 1)), "index_head_dim": 128}
    return config


def llama_tensors(layers: int = 4) -> dict[str, tuple[str, list[int]]]:
    tensors = {"model.embed_tokens.weight": ("BF16", [1000, 256])}
    for i in range(layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = ("BF16", [256, 256])
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = ("BF16", [512, 256])
        tensors[f"model.layers.{i}.input_layernorm.weight"] = ("F32", [256])
    return tensors


T = TypeVar("T")


def some(value: T | None) -> T:
    assert value is not None
    return value


def estimate(*args: Any, **kwargs: Any) -> Estimate:
    result = run(*args, **kwargs)
    assert isinstance(result, Estimate)
    return result


def listing(*args: Any, **kwargs: Any) -> Listing:
    result = run(*args, **kwargs)
    assert isinstance(result, Listing)
    return result


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        out.append((value & 0x7F) | (0x80 if value > 0x7F else 0))
        value >>= 7
        if not value:
            return bytes(out)


def _field(number: int, value: int | bytes) -> bytes:
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    return _varint(number << 3 | 2) + _varint(len(value)) + value


_ONNX_BITS = {1: 32, 2: 8, 7: 64, 10: 16, 16: 16, 22: 4}


def write_onnx(
    path: str,
    initializers: list[tuple[str, int, list[int]]],
    inputs: list[tuple[str, int, list[int | str]]] = [],
    external: str | None = None,
    nodes: list[tuple[str, list[str], int]] = [],
) -> int:
    graph, data = b"", 0
    for op, names, bits in nodes:
        attribute = _field(1, b"bits") + _field(3, bits)
        graph += _field(
            1, b"".join(_field(1, n.encode()) for n in names) + _field(4, op.encode()) + _field(5, attribute)
        )
    for name, dtype, dims in initializers:
        size = -(-math.prod(dims) * _ONNX_BITS[dtype] // 8)
        tensor = b"".join(_field(1, d) for d in dims) + _field(2, dtype) + _field(8, name.encode())
        if external:
            entries = [("location", external), ("offset", str(data)), ("length", str(size))]
            tensor += b"".join(_field(13, _field(1, k.encode()) + _field(2, v.encode())) for k, v in entries)
            tensor += _field(14, 1)
        else:
            tensor += _field(9, b"\0" * size)
        graph += _field(5, tensor)
        data += size
    for name, dtype, shape_dims in inputs:
        shape = b"".join(
            _field(1, _field(1, d) if isinstance(d, int) else _field(2, d.encode())) for d in shape_dims
        )
        graph += _field(
            11, _field(1, name.encode()) + _field(2, _field(1, _field(1, dtype) + _field(2, shape)))
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(_field(1, 8) + _field(7, graph))
    if external:
        with open(os.path.join(os.path.dirname(path), external), "wb") as f:
            f.truncate(data)
    return data
