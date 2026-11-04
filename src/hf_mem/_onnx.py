import asyncio
import math
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from hf_mem._weights import Weights

CHUNK = 1 << 20
SMALL_FILE = 16 << 20
TAIL_SIZE = 1 << 20
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto (TensorProto.DataType)
TYPES = {
    1: ("F32", 32),
    2: ("U8", 8),
    3: ("I8", 8),
    4: ("U16", 16),
    5: ("I16", 16),
    6: ("I32", 32),
    7: ("I64", 64),
    9: ("BOOL", 8),
    10: ("F16", 16),
    11: ("F64", 64),
    12: ("U32", 32),
    13: ("U64", 64),
    16: ("BF16", 16),
    17: ("F8_E4M3", 8),
    18: ("F8_E4M3", 8),
    19: ("F8_E5M2", 8),
    20: ("F8_E5M2", 8),
    21: ("U4", 4),
    22: ("I4", 4),
    23: ("F4", 4),
}
_VARIANT = re.compile(r"_(fp16|q8|int8|uint8|q4|q4f16|bnb4|quantized)$")
_DATA = re.compile(r"(?<=\.onnx)[._]data(?:_\d+)?$")
_PAST_INPUT = re.compile(rb"\x5a[\x80-\xff]{0,3}[\x00-\x7f]\x0a[\x00-\x7f]past_")
_PARTS = ("embed_tokens", "vision_encoder", "audio_encoder", "speech_encoder", "encoder_model")
_DECODERS = ("decoder_model_merged", "decoder_model", "decoder_with_past_model")
# Weight input of the ONNX Runtime ops that pack sub-byte weights into UINT8 blobs
_QUANTIZED = {"MatMulNBits": 1, "GatherBlockQuantized": 0}


@dataclass
class Graph:
    weights: Weights = field(default_factory=Weights)
    cache: list[tuple[str, int, int, bool]] = field(default_factory=list)
    packed: dict[str, int] = field(default_factory=dict)


def group_files(files: dict[str, int]) -> dict[str, list[str]]:
    variants: dict[tuple[str, str], dict[str, str]] = {}
    data: dict[str, list[str]] = {}
    for path in sorted(files):
        if path.endswith(".onnx"):
            directory, name = os.path.split(path[:-5])
            match = _VARIANT.search(name)
            variant, part = (match[1], name[: match.start()]) if match else ("", name)
            variants.setdefault((directory, variant), {})[part] = path
        elif match := _DATA.search(path):
            data.setdefault(path[: match.start()], []).append(path)
    groups: dict[str, list[str]] = {}
    for (directory, variant), parts in variants.items():
        for part, path in parts.items():
            if part not in _PARTS and part not in _DECODERS:
                groups[path] = [path]
        decoders = ["decoder_model_merged"] if "decoder_model_merged" in parts else list(_DECODERS[1:])
        if loaded := [parts[p] for p in (*_PARTS, *decoders) if p in parts]:
            names = [os.path.basename(p)[:-5].removesuffix(f"_{variant}") for p in loaded]
            label = (
                loaded[0]
                if len(loaded) == 1
                else os.path.join(directory, "+".join(names) + (f"_{variant}" if variant else ""))
            )
            groups[label] = loaded
    return {
        label: [*paths, *(f for p in paths for f in data.get(p, []))] for label, paths in sorted(groups.items())
    }


def select_file(groups: dict[str, list[str]], file: str) -> tuple[str, list[str]]:
    target = file.strip("/")
    found = {
        label: paths
        for label, paths in groups.items()
        if label == target or target in paths or label.startswith(target + "/")
    } or {label: paths for label, paths in groups.items() if os.path.basename(label) == target}
    if not found:
        raise RuntimeError(f"No ONNX file matching `{file}` found.")
    if len(found) == 1:
        return next(iter(found.items()))
    raise RuntimeError(f"Multiple ONNX files match `{file}`: {sorted(found)}.")


class _Reader:
    def __init__(self, source: Any, path: str) -> None:
        self.source, self.path, self.size = source, path, source.files[path]
        self.chunk = self.size if self.size <= SMALL_FILE else CHUNK
        self.start, self.buffer = 0, b""

    async def read(self, pos: int, n: int) -> bytes:
        if pos < self.start or pos + n > self.start + len(self.buffer):
            self.start = pos
            self.buffer = await self.source.read(self.path, pos, min(self.size, pos + max(n, self.chunk)))
        return self.buffer[pos - self.start : pos - self.start + n]

    async def varint(self, pos: int) -> tuple[int, int]:
        value = 0
        for i, byte in enumerate(await self.read(pos, min(10, self.size - pos))):
            value |= (byte & 0x7F) << (7 * i)
            if byte < 0x80:
                return value, pos + i + 1
        raise RuntimeError(f"`{self.path}` is not a valid ONNX file.")

    async def fields(self, start: int, end: int) -> AsyncIterator[tuple[int, int, int, int]]:
        pos = start
        while pos < end:
            tag, pos = await self.varint(pos)
            number, wire = tag >> 3, tag & 7
            if wire == 0:
                value, pos = await self.varint(pos)
                yield number, wire, value, pos
            elif wire == 2:
                length, pos = await self.varint(pos)
                yield number, wire, pos, pos + length
                pos += length
            elif wire in (1, 5):
                pos += 8 if wire == 1 else 4
            else:
                raise RuntimeError(f"`{self.path}` is not a valid ONNX file.")
        if pos != end:
            raise RuntimeError(f"`{self.path}` is not a valid ONNX file.")


def _packed(data: bytes) -> list[int]:
    values, value, shift = [], 0, 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            values.append(value)
            value = shift = 0
    return values


async def _tensor(reader: _Reader, start: int, end: int, graph: Graph) -> None:
    dtype, dims, name = 0, list[int](), ""
    async for number, wire, a, b in reader.fields(start, end):
        if number == 1:
            dims.extend([a] if wire == 0 else _packed(await reader.read(a, b - a)))
        elif number == 2:
            dtype = a
        elif number == 8:
            name = (await reader.read(a, b - a)).decode(errors="replace")
    label, bits = TYPES.get(dtype, (f"TYPE{dtype}", 8))
    elements = math.prod(dims)
    graph.weights.add(label, elements * graph.packed.get(name, 1), math.ceil(elements * bits / 8))


async def _node(reader: _Reader, start: int, end: int, graph: Graph) -> None:
    inputs, op, bits = list[str](), "", 4
    async for number, _, a, b in reader.fields(start, end):
        if number == 1:
            inputs.append((await reader.read(a, b - a)).decode(errors="replace"))
        elif number == 4:
            op = (await reader.read(a, b - a)).decode(errors="replace")
            if op not in _QUANTIZED:
                return
        elif number == 5:
            attribute, value = "", None
            async for key, wire, c, d in reader.fields(a, b):
                if key == 1:
                    attribute = (await reader.read(c, d - c)).decode(errors="replace")
                elif key == 3 and wire == 0:
                    value = c
            if attribute == "bits" and value:
                bits = value
    if op in _QUANTIZED and len(inputs) > _QUANTIZED[op] and 0 < bits < 8:
        graph.packed[inputs[_QUANTIZED[op]]] = 8 // bits


async def _dims(reader: _Reader, start: int, end: int) -> list[int | None]:
    dims: list[int | None] = []
    async for number, _, a, b in reader.fields(start, end):
        if number == 1:
            value = None
            async for key, wire, v, _ in reader.fields(a, b):
                if key == 1 and wire == 0:
                    value = v
            dims.append(value)
    return dims


async def _input(reader: _Reader, start: int, end: int, graph: Graph) -> None:
    name, dtype, dims = "", 0, list[int | None]()
    async for number, _, a, b in reader.fields(start, end):
        if number == 1:
            name = (await reader.read(a, b - a)).decode(errors="replace")
        elif number == 2 and name.startswith("past_"):
            async for kind, _, c, d in reader.fields(a, b):
                if kind == 1:
                    async for key, _, e, f in reader.fields(c, d):
                        if key == 1:
                            dtype = e
                        elif key == 2:
                            dims = await _dims(reader, e, f)
    if name.startswith("past_"):
        dtype_name, bits = TYPES.get(dtype, ("F32", 32))
        static = math.prod(d for d in dims if d)
        graph.cache.append((dtype_name, static * bits // 8, sum(not d for d in dims), ".encoder." in name))


async def _walk(reader: _Reader, graph: Graph) -> None:
    async for number, wire, start, end in reader.fields(0, reader.size):
        if number != 7 or wire != 2:
            continue
        async for field_number, field_wire, a, b in reader.fields(start, end):
            if field_number == 1 and field_wire == 2:
                await _node(reader, a, b, graph)
            elif field_number == 5 and field_wire == 2:
                await _tensor(reader, a, b, graph)
            elif field_number == 11 and field_wire == 2:
                await _input(reader, a, b, graph)
        return
    raise RuntimeError(f"`{reader.path}` is not a valid ONNX file.")


async def _tail(reader: _Reader, graph: Graph) -> None:
    start = max(0, reader.size - TAIL_SIZE)
    tail = await reader.read(start, reader.size - start)
    seen = set()
    for match in _PAST_INPUT.finditer(tail):
        length, pos = await reader.varint(start + match.start() + 1)
        if pos + length <= reader.size and (key := tail[match.start() : pos - start + length]) not in seen:
            seen.add(key)
            await _input(reader, pos, pos + length, graph)


async def read_group(source: Any, paths: list[str]) -> Graph:
    async def read(path: str) -> Graph:
        reader, graph = _Reader(source, path), Graph()
        if reader.size <= SMALL_FILE:
            await _walk(reader, graph)
        else:
            data = [p for p in paths if p.startswith(path) and _DATA.search(p)]
            graph.weights.bytes = sum(source.files[p] for p in (path, *data))
            await _tail(reader, graph)
        return graph

    merged = Graph()
    for graph in await asyncio.gather(*(read(p) for p in paths if p.endswith(".onnx"))):
        merged.weights.merge(graph.weights)
        merged.cache.extend(graph.cache)
    return merged


def kv_cache(
    graph: Graph, max_model_len: int, batch_size: int, encoder_length: int | None = None
) -> tuple[int, str, float] | None:
    if not graph.cache:
        return None
    per_token = sum(b for _, b, symbolic, encoder in graph.cache if symbolic >= 2 and not encoder)
    fixed = sum(
        b * ((encoder_length or max_model_len) if symbolic >= 2 else 1)
        for _, b, symbolic, encoder in graph.cache
        if encoder or symbolic < 2
    )
    dtypes = [d for d, *_ in graph.cache]
    return (
        (per_token * max_model_len + fixed) * batch_size,
        max(set(dtypes), key=dtypes.count),
        float(per_token * batch_size),
    )
