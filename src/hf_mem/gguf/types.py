import struct
from enum import IntEnum
from typing import Any, Callable, Dict


# --- GGUF weight dtype enum ---
class GGUFDtype(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # Q4_2 = 4
    # Q4_3 = 5
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    # Q4_0_4_4
    # Q4_0_4_8
    # Q4_0_8_8
    TQ1_0 = 34
    TQ2_0 = 35
    # IQ4_NL_4_4 = 36
    # IQ4_NL_4_8 = 37
    # IQ4_NL_8_8 = 38
    MXFP4 = 39
    COUNT = 40  # NOTE: For exhaustivity, not used as a real dtype


# NOTE: Bits-per-weight for each GGUF dtype, used to compute byte sizes without storing all tensor data
GGUFDtypeBitsPerWeight: Dict[GGUFDtype, float] = {
    GGUFDtype.F64: 64.0,
    GGUFDtype.I64: 64.0,
    GGUFDtype.F32: 32.0,
    GGUFDtype.I32: 32.0,
    GGUFDtype.F16: 16.0,
    GGUFDtype.BF16: 16.0,
    GGUFDtype.I16: 16.0,
    GGUFDtype.I8: 8.0,
    # NOTE: GGUF-specific quantized dtypes
    GGUFDtype.Q8_K: 8.03125,
    GGUFDtype.Q6_K: 6.5625,
    GGUFDtype.Q5_K: 5.5,
    GGUFDtype.Q4_K: 4.5,
    GGUFDtype.Q3_K: 3.4375,
    GGUFDtype.Q2_K: 2.625,
    GGUFDtype.IQ4_NL: 4.5,
    GGUFDtype.IQ4_XS: 4.25,
    GGUFDtype.IQ3_S: 3.44,
    GGUFDtype.IQ3_XXS: 3.06,
    GGUFDtype.IQ2_XXS: 2.06,
    GGUFDtype.IQ2_S: 2.5,
    GGUFDtype.IQ2_XS: 2.31,
    GGUFDtype.IQ1_S: 1.56,
    GGUFDtype.IQ1_M: 1.75,
    GGUFDtype.TQ1_0: 1.6875,
    GGUFDtype.TQ2_0: 2.0625,
    GGUFDtype.MXFP4: 4.25,
    # NOTE: Legacy GGUF dtypes, kept for exhaustivity
    GGUFDtype.Q8_0: 8.5,
    GGUFDtype.Q8_1: 9.0,
    GGUFDtype.Q5_0: 5.5,
    GGUFDtype.Q5_1: 6.0,
    GGUFDtype.Q4_0: 4.5,
    GGUFDtype.Q4_1: 5.0,
}


# --- GGUF metadata value type enum (used in KV metadata pairs) ---
class GGUFMetadataDtype(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# --- Low-level binary readers for each GGUF metadata value type ---


def _read_string(raw_metadata: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack("<Q", raw_metadata[offset : offset + 8])[0]
    offset += 8
    key = raw_metadata[offset : offset + length].decode()
    offset += length
    return key, offset


def _read_uint8(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<B", raw_metadata[offset : offset + 1])[0]
    offset += 1
    return value, offset


def _read_int8(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<b", raw_metadata[offset : offset + 1])[0]
    offset += 1
    return value, offset


def _read_uint16(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<H", raw_metadata[offset : offset + 2])[0]
    offset += 2
    return value, offset


def _read_int16(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<h", raw_metadata[offset : offset + 2])[0]
    offset += 2
    return value, offset


def _read_uint32(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<I", raw_metadata[offset : offset + 4])[0]
    offset += 4
    return value, offset


def _read_int32(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<i", raw_metadata[offset : offset + 4])[0]
    offset += 4
    return value, offset


def _read_float32(raw_metadata: bytes, offset: int) -> tuple[float, int]:
    value = struct.unpack("<f", raw_metadata[offset : offset + 4])[0]
    offset += 4
    return value, offset


def _read_bool(raw_metadata: bytes, offset: int) -> tuple[bool, int]:
    value = struct.unpack("<?", raw_metadata[offset : offset + 1])[0]
    offset += 1
    return value, offset


def _read_uint64(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<Q", raw_metadata[offset : offset + 8])[0]
    offset += 8
    return value, offset


def _read_int64(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<q", raw_metadata[offset : offset + 8])[0]
    offset += 8
    return value, offset


def _read_float64(raw_metadata: bytes, offset: int) -> tuple[float, int]:
    value = struct.unpack("<d", raw_metadata[offset : offset + 8])[0]
    offset += 8
    return value, offset


def _read_array(raw_metadata: bytes, offset: int) -> tuple[list[Any], int]:
    value_type = struct.unpack("<I", raw_metadata[offset : offset + 4])[0]
    offset += 4
    length = struct.unpack("<Q", raw_metadata[offset : offset + 8])[0]
    offset += 8
    value = []
    for _ in range(length):
        sub_value, offset = _PARSERS[value_type](raw_metadata, offset)
        value.append(sub_value)
    return value, offset


# NOTE: Dispatch table mapping GGUFMetadataDtype integer values to their reader functions
_PARSERS: Dict[GGUFMetadataDtype, Callable] = {
    GGUFMetadataDtype.UINT8: _read_uint8,
    GGUFMetadataDtype.INT8: _read_int8,
    GGUFMetadataDtype.UINT16: _read_uint16,
    GGUFMetadataDtype.INT16: _read_int16,
    GGUFMetadataDtype.UINT32: _read_uint32,
    GGUFMetadataDtype.INT32: _read_int32,
    GGUFMetadataDtype.FLOAT32: _read_float32,
    GGUFMetadataDtype.BOOL: _read_bool,
    GGUFMetadataDtype.STRING: _read_string,
    GGUFMetadataDtype.ARRAY: _read_array,
    GGUFMetadataDtype.UINT64: _read_uint64,
    GGUFMetadataDtype.INT64: _read_int64,
    GGUFMetadataDtype.FLOAT64: _read_float64,
}
