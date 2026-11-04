import asyncio
import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Any

from hf_mem._weights import Weights, choices

FIRST_READ = 1 << 20
PREFIX_READ = 1 << 17
MAX_READ = 256 << 20
LISTING_READ = 16 << 20
SHARD = re.compile(r"-(\d+)-of-(\d+)\.gguf$")
_SKIP = ("mmproj", "imatrix")
_QUANT = re.compile(
    r"(?:UD-)?(?:I?Q\d\w*?|T?Q\d_\d|[BM]?F\d+|NVFP4|MXFP4(?:_MOE)?)(?=(?:-\d+-of-\d+)?\.gguf$)", re.I
)
_MMPROJ = ("Q8_0", "F16", "BF16", "F32")

# ggml type id -> (name, block size, bytes per block)
TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}
KV_DTYPES = {name: size / block for name, block, size in TYPES.values()}
# KV cache types accepted by llama.cpp's `--cache-type-k` / `--cache-type-v`
CACHE_TYPES = ("F16", "BF16", "F32", "Q8_0", "Q4_0", "Q4_1", "IQ4_NL", "Q5_0", "Q5_1")

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_SWA_PATTERN = {"gemma2": 2, "gemma3": 6, "gemma3n": 5, "cohere2": 4}


_U64 = struct.Struct("<Q").unpack_from


class _Truncated(Exception):
    pass


def _unpack(fmt: str, buf: bytes, offset: int) -> tuple[Any, ...]:
    if offset + struct.calcsize(fmt) > len(buf):
        raise _Truncated
    return struct.unpack_from(fmt, buf, offset)


def _skip_strings(buf: bytes, offset: int, count: int) -> int:
    try:
        for _ in range(count):
            offset += 8 + _U64(buf, offset)[0]
    except struct.error:
        raise _Truncated from None
    if offset > len(buf):
        raise _Truncated
    return offset


def _value(buf: bytes, offset: int, kind: int, keep: bool) -> tuple[Any, int]:
    if kind == 8:
        (length,) = _unpack("<Q", buf, offset)
        offset += 8
        if offset + length > len(buf):
            raise _Truncated
        return (buf[offset : offset + length].decode("utf-8", "replace") if keep else None), offset + length
    if kind == 9:
        item, count = _unpack("<IQ", buf, offset)
        offset += 12
        if item in _SCALAR:
            fmt = f"<{count}{_SCALAR[item]}"
            values = _unpack(fmt, buf, offset) if keep else None
            if offset + struct.calcsize(fmt) > len(buf):
                raise _Truncated
            return (list(values) if keep else None), offset + struct.calcsize(fmt)  # type: ignore[arg-type]
        if item == 8 and not keep:
            return None, _skip_strings(buf, offset, count)
        items: list[Any] = []
        for _ in range(count):
            value, offset = _value(buf, offset, item, keep)
            items.append(value)
        return (items if keep else None), offset
    fmt = "<" + _SCALAR[kind]
    return _unpack(fmt, buf, offset)[0], offset + struct.calcsize(fmt)


@dataclass
class Header:
    metadata: dict[str, Any]
    weights: Weights
    data_start: int


def _metadata(buf: bytes, prefix: bool = False, partial: bool = False) -> tuple[dict[str, Any], int, int, bool]:
    if buf[:4] != b"GGUF":
        raise RuntimeError("not a valid GGUF file (magic number mismatch)")
    _, tensor_count, kv_count = _unpack("<IQQ", buf, 4)
    offset, metadata = 24, dict[str, Any]()
    try:
        for _ in range(kv_count):
            start = offset
            key, offset = _value(buf, offset, 8, True)
            if prefix and key.startswith("tokenizer."):
                return metadata, offset, tensor_count, True
            (kind,) = _unpack("<I", buf, offset)
            value, offset = _value(buf, offset + 4, kind, not key.startswith("tokenizer."))
            metadata[key] = value
    except _Truncated:
        if not partial:
            raise
        return metadata, start, tensor_count, False
    return metadata, offset, tensor_count, False


def parse_header(buf: bytes, file_size: int) -> Header:
    metadata, offset, tensor_count, _ = _metadata(buf)
    tensors = []
    for _ in range(tensor_count):
        _, offset = _value(buf, offset, 8, False)
        (dims,) = _unpack("<I", buf, offset)
        shape = _unpack(f"<{dims}Q", buf, offset + 4)
        offset += 4 + 8 * dims
        kind, data_offset = _unpack("<IQ", buf, offset)
        offset += 12
        tensors.append((data_offset, kind, math.prod(shape)))

    alignment = metadata.get("general.alignment", 32)
    data_start = -(-offset // alignment) * alignment
    tensors.sort()
    ends = [t[0] for t in tensors[1:]] + [file_size - data_start]
    weights = Weights()
    for (start, kind, params), end in zip(tensors, ends):
        if kind in TYPES:
            name, block, size = TYPES[kind]
            weights.add(name, params, -(-params // block) * size)
        else:
            weights.add(f"TYPE_{kind}", params, end - start)
    return Header(metadata, weights, data_start)


async def read_header(source: Any, path: str) -> Header:
    size, buf = source.files[path], b""
    end = min(FIRST_READ, size)
    while True:
        buf += await source.read(path, len(buf), end)
        try:
            return parse_header(buf, size)
        except _Truncated:
            if end >= min(MAX_READ, size):
                raise RuntimeError(f"`{path}` has a metadata section larger than {MAX_READ >> 20} MiB.")
            end = min(end * 2, MAX_READ, size)
        except (struct.error, UnicodeDecodeError, KeyError) as e:
            raise RuntimeError(f"`{path}` is not a valid GGUF file ({e}).") from e


async def read_metadata(source: Any, path: str) -> tuple[dict[str, Any], bool]:
    size, buf = source.files[path], b""
    end, prefix = min(PREFIX_READ, size), True
    while True:
        if len(buf) < end:
            buf += await source.read(path, len(buf), end)
        try:
            metadata, _, _, tokenizer = _metadata(buf, prefix)
            if not prefix or not tokenizer or f"{metadata.get('general.architecture')}.block_count" in metadata:
                return metadata, tokenizer or not prefix
            prefix = False
            continue
        except _Truncated:
            metadata, offset, _, _ = _metadata(buf, prefix, partial=True)
            try:
                key = _value(buf, offset, 8, True)[0]
            except _Truncated:
                key = "tokenizer."
            tokenizer = key.startswith("tokenizer.")
            if end >= min(LISTING_READ, size) or (not tokenizer and len(buf) - offset > PREFIX_READ):
                return metadata, tokenizer
        except (struct.error, UnicodeDecodeError, KeyError) as e:
            raise RuntimeError(f"`{path}` is not a valid GGUF file ({e}).") from e
        end = min(max(end * 2, FIRST_READ), LISTING_READ, size)


def gguf_paths(files: dict[str, int]) -> list[str]:
    return sorted(
        p for p in files if p.endswith(".gguf") and not any(t in os.path.basename(p).lower() for t in _SKIP)
    )


def group_files(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        match = SHARD.search(path)
        groups.setdefault(path[: match.start()] + ".gguf" if match else path, []).append(path)
    return groups


def select_file(groups: dict[str, list[str]], file: str) -> tuple[str, list[str]]:
    target = file.strip("/")
    if match := SHARD.search(target):
        target = target[: match.start()] + ".gguf"
    found = {
        name: paths for name, paths in groups.items() if name == target or name.startswith(target + "/")
    } or {name: paths for name, paths in groups.items() if os.path.basename(name) == target}
    if not found and (matches := tagged(groups, target)):
        name = min(matches, key=lambda n: (len(os.path.basename(n)), n))
        return name, groups[name]
    if not found:
        raise RuntimeError(
            f"No GGUF file matching `{file}` found, use a quantization tag or one of {choices(list(groups))}."
        )
    if len(found) > 1:
        raise RuntimeError(f"Multiple GGUF files match `{file}`, use one of {choices(list(found))}.")
    return next(iter(found.items()))


def tagged(groups: dict[str, list[str]], tag: str) -> list[str]:
    pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(tag)}(?![a-z0-9_])", re.I)
    return [name for name in groups if pattern.search(os.path.basename(name))]


def mmproj(files: dict[str, int], name: str) -> str | None:
    directory = os.path.dirname(name)
    candidates = sorted(
        p
        for p in files
        if p.endswith(".gguf")
        and "mmproj" in os.path.basename(p).lower()
        and os.path.dirname(p) in (directory, "")
    )
    if not candidates:
        return None
    quant = _QUANT.search(os.path.basename(name))
    for tag in ([quant[0]] if quant else []) + list(_MMPROJ):
        if matches := tagged({p: [] for p in candidates}, tag):
            return matches[0]
    return candidates[0]


def _state_bytes(metadata: dict[str, Any], arch: str) -> int:
    if state := metadata.get(f"{arch}.ssm.state_size"):
        inner = metadata.get(f"{arch}.ssm.inner_size", 0)
        groups = metadata.get(f"{arch}.ssm.group_count") or 1
        kernel = metadata.get(f"{arch}.ssm.conv_kernel", 4)
        return 4 * ((kernel - 1) * (inner + 2 * groups * state) + inner * state)
    if cache := metadata.get(f"{arch}.shortconv.l_cache"):
        return 4 * metadata.get(f"{arch}.embedding_length", 0) * (cache - 1)
    return 0


def kv_cache(
    metadata: dict[str, Any], max_model_len: int, batch_size: int, dtype: str
) -> tuple[int, dict[str, int]] | None:
    arch = str(metadata.get("general.architecture"))
    blocks = metadata.get(f"{arch}.block_count")
    heads = metadata.get(f"{arch}.attention.head_count")
    state = _state_bytes(metadata, arch)
    if not isinstance(blocks, int) or not (heads or state):
        return None
    blocks -= metadata.get(f"{arch}.nextn_predict_layers", 0)
    heads = (max(heads) if isinstance(heads, list) else heads) or 0

    kv_heads = metadata.get(f"{arch}.attention.head_count_kv", heads)
    if isinstance(kv_heads, list):
        kv_heads = kv_heads[:blocks]
    else:
        interval = metadata.get(f"{arch}.full_attention_interval")
        kv_heads = [kv_heads if not interval or (i + 1) % interval == 0 else 0 for i in range(blocks)]

    if rank := metadata.get(f"{arch}.attention.kv_lora_rank"):
        rope = metadata.get(f"{arch}.rope.dimension_count", 0)
        widths = [(rank + rope) if h else 0 for h in kv_heads]
    else:
        embedding = metadata.get(f"{arch}.embedding_length", 0)
        key = metadata.get(f"{arch}.attention.key_length", embedding // heads if heads else 0)
        value = metadata.get(f"{arch}.attention.value_length", key)
        widths = [h * (key + value) for h in kv_heads]

    window = metadata.get(f"{arch}.attention.sliding_window")
    pattern = metadata.get(f"{arch}.attention.sliding_window_pattern")
    if isinstance(pattern, list) and len(pattern) >= blocks:
        sliding = [bool(p) for p in pattern[:blocks]]
    elif window and arch in _SWA_PATTERN:
        sliding = [(i + 1) % _SWA_PATTERN[arch] != 0 for i in range(blocks)]
    else:
        sliding = [False] * blocks
    swa_tokens = min(max_model_len, (window or max_model_len) + 512)

    tokens = sum(w * (swa_tokens if s else max_model_len) for w, s in zip(widths, sliding))
    recurrent = sum(1 for w in widths if not w) if state else 0
    layers = {
        "full": sum(1 for w, s in zip(widths, sliding) if w and not s),
        "sliding": sum(1 for w, s in zip(widths, sliding) if w and s),
        "state": recurrent,
        "none": sum(1 for w in widths if not w) - recurrent,
    }
    return int((tokens * KV_DTYPES[dtype] + recurrent * state) * batch_size), layers


async def read_group(source: Any, paths: list[str]) -> Header:
    headers = await asyncio.gather(*(read_header(source, p) for p in paths))
    weights = Weights()
    for header in headers:
        weights.merge(header.weights)
    return Header(headers[0].metadata, weights, headers[0].data_start)
