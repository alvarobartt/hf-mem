from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Callable
from enum import IntEnum
import httpx
import os
import struct
import math
from hf_mem.types import get_safetensors_dtype_bytes
from hf_mem.metadata import DtypeMetadata

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))
SIZES = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
KV_CACHE_FIELD_ENDINGS = [
    "block_count", 
    "head_count_kv", 
    "head_count",
    "embedding_length",
    "context_length",
]

# --- GGUF Dtypes (weights) ---
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
    COUNT = 40 # For exhaustivity, not used

# Conversion .gguf weight dtype -> bits-per-weight
GGUFDtypeBitsPerWeight: Dict[GGUFDtype, float] = {
    GGUFDtype.F64: 64.0,
    GGUFDtype.I64: 64.0,
    GGUFDtype.F32: 32.0,
    GGUFDtype.I32: 32.0,
    GGUFDtype.F16: 16.0,
    GGUFDtype.BF16: 16.0,
    GGUFDtype.I16: 16.0,
    # .GGUF specific dtypes
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
    # .GGUF specific dtypes (legacy type, added for exhaustivity)
    GGUFDtype.Q8_0: 8.5,
    GGUFDtype.Q8_1: 9.0,
    GGUFDtype.Q5_0: 5.5,
    GGUFDtype.Q5_1: 6.0,
    GGUFDtype.Q4_0: 4.5,
    GGUFDtype.Q4_1: 5.0,
}


# --- Auxiliary read functions ---
# Related to .gguf metadata dtypes, not weights dtypes
def _read_string(raw_metadata: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack("<Q", raw_metadata[offset:offset+8])[0]
    offset += 8
    key = raw_metadata[offset:offset+length].decode()
    offset += length
    return key, offset

def _read_uint8(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<B", raw_metadata[offset:offset+1])[0]
    offset += 1
    return value, offset

def _read_int8(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<b", raw_metadata[offset:offset+1])[0]
    offset += 1
    return value, offset

def _read_uint16(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<H", raw_metadata[offset:offset+2])[0]
    offset += 2
    return value, offset

def _read_int16(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<h", raw_metadata[offset:offset+2])[0]
    offset += 2
    return value, offset

def _read_uint32(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<I", raw_metadata[offset:offset+4])[0]
    offset += 4
    return value, offset

def _read_int32(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<i", raw_metadata[offset:offset+4])[0]
    offset += 4
    return value, offset

def _read_float32(raw_metadata: bytes, offset: int) -> tuple[float, int]:
    value = struct.unpack("<f", raw_metadata[offset:offset+4])[0]
    offset += 4
    return value, offset

def _read_bool(raw_metadata: bytes, offset: int) -> tuple[bool, int]:
    value = struct.unpack("<?", raw_metadata[offset:offset+1])[0]
    offset += 1
    return value, offset

def _read_array(raw_metadata: bytes, offset: int) -> tuple[list[Any], int]:
    value_type = struct.unpack("<I", raw_metadata[offset:offset+4])[0]
    offset += 4
    length = struct.unpack("<Q", raw_metadata[offset:offset+8])[0]
    offset += 8
    value = []
    for i in range(length):
        sub_value, offset = Parsers[value_type](raw_metadata, offset)
        value.append(sub_value)
    return value, offset

def _read_uint64(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<Q", raw_metadata[offset:offset+8])[0]
    offset += 8
    return value, offset

def _read_int64(raw_metadata: bytes, offset: int) -> tuple[int, int]:
    value = struct.unpack("<q", raw_metadata[offset:offset+8])[0]
    offset += 8
    return value, offset

def _read_float64(raw_metadata: bytes, offset: int) -> tuple[float, int]:
    value = struct.unpack("<d", raw_metadata[offset:offset+8])[0]
    offset += 8
    return value, offset


# --- GGUF Metadata Dtypes ---
# Related to .gguf metadata dtypes, not weights dtypes
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

Parsers: Dict[GGUFMetadataDtype, Callable] = {
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


@dataclass
class GGUFComponentMetadata:
    dtypes: Dict[GGUFDtype, DtypeMetadata]
    param_count: int
    bytes_count: int

@dataclass
class GGUFKVCacheInfo:
    max_model_len: int
    cache_size: int
    batch_size: int
    cache_dtype: Optional[str] = None

@dataclass
class GGUFMetadata:
    components: Dict[str, GGUFComponentMetadata]
    param_count: int
    bytes_count: int
    kv_cache_info: Optional[GGUFKVCacheInfo] = None


def merge_shards(shard1: GGUFMetadata, shard2: GGUFMetadata) -> GGUFMetadata:
    merged_components = {}
    all_component_names = set(shard1.components.keys()) | set(shard2.components.keys())
    
    for component_name in all_component_names:
        comp1 = shard1.components.get(component_name)
        comp2 = shard2.components.get(component_name)
        
        if comp1 and comp2:
            # Both have it -> merge
            merged_dtypes = {}
            all_dtypes = set(comp1.dtypes.keys()) | set(comp2.dtypes.keys())
            
            for dtype in all_dtypes:
                dtype_meta1 = comp1.dtypes.get(dtype)
                dtype_meta2 = comp2.dtypes.get(dtype)
                
                if dtype_meta1 and dtype_meta2:
                    # Both have it -> merge
                    merged_dtypes[dtype] = DtypeMetadata(
                        param_count=dtype_meta1.param_count + dtype_meta2.param_count,
                        bytes_count=dtype_meta1.bytes_count + dtype_meta2.bytes_count
                    )
                elif dtype_meta1:
                    merged_dtypes[dtype] = dtype_meta1
                else:
                    merged_dtypes[dtype] = dtype_meta2
            
            merged_components[component_name] = GGUFComponentMetadata(
                dtypes=merged_dtypes,
                param_count=comp1.param_count + comp2.param_count,
                bytes_count=comp1.bytes_count + comp2.bytes_count
            )
        elif comp1:
            merged_components[component_name] = comp1
        else:
            merged_components[component_name] = comp2
    
    # KV-cache info should be the same for all shards, we use earlier ones since sometimes
    # cache metadata gets dropped after first shard
    kv_cache_info = shard1.kv_cache_info or shard2.kv_cache_info
    
    return GGUFMetadata(
        components=merged_components,
        param_count=shard1.param_count + shard2.param_count,
        bytes_count=shard1.bytes_count + shard2.bytes_count,
        kv_cache_info=kv_cache_info
    )


def gguf_metadata_to_json(model_id: str, revision: str, metadata: GGUFMetadata) -> Dict[str, Any]:
    out = asdict(metadata)
    # Convert dtypes enum keys to string names
    out["components"]["Transformer"]["dtypes"] = {
        k.name: v for k, v in out["components"]["Transformer"]["dtypes"].items()
    }
    out = {"model_id": model_id, "revision": revision, **out}
    
    # If --experimental, move cache info out
    if out.get("kv_cache_info") is not None:
        out["max_model_len"] = out["kv_cache_info"]["max_model_len"]
        out["cache_size"] = out["kv_cache_info"]["cache_size"]
        out["batch_size"] = out["kv_cache_info"]["batch_size"]
        out["cache_dtype"] = out["kv_cache_info"]["cache_dtype"]
    out.pop("kv_cache_info", None)

    return out

def compute_gguf_kv_cache_size(
        kv_metadata: dict, 
        kv_cache_dtype: Optional[str] = "F16", 
        batch_size: Optional[int] = 1
        ) -> int:
    block_count = kv_metadata["block_count"]
    head_count_kv = kv_metadata["head_count_kv"]
    head_count = kv_metadata["head_count"]
    embedding_length = kv_metadata["embedding_length"]
    context_length = kv_metadata["context_length"]

    size_per_token = (2 * head_count_kv * (embedding_length // head_count))
    if kv_cache_dtype is not None:
        if kv_cache_dtype not in GGUFDtype.__members__:
            raise RuntimeError(f"--kv-cache-dtype={kv_cache_dtype} not recognized for GGUF KV cache size estimation. Valid options: {list(GGUFDtype.__members__.keys())}.")
        total_size = (
            block_count 
            * size_per_token 
            * context_length 
            * batch_size 
            * int(GGUFDtypeBitsPerWeight[GGUFDtype[kv_cache_dtype]] / 8.0)
        )
    else:
        # Default to F16 size
        total_size = (
            block_count 
            * size_per_token 
            * context_length 
            * batch_size 
            * 2
        )
    return total_size

def parse_gguf_metadata(
        raw_metadata: bytes,
        experimental: bool = False,
        max_model_len: Optional[int] = None,
        kv_cache_dtype: Optional[str] = "F16",
        batch_size: int = 1
        ) -> GGUFMetadata:
    # Header
    magic = raw_metadata[:4].decode("ascii")
    version = struct.unpack("<I", raw_metadata[4:8])[0]
    tensor_count = struct.unpack("<Q", raw_metadata[8:16])[0]
    metadata_kv_count = struct.unpack("<Q", raw_metadata[16:24])[0]
    offset = 24

    kv_metadata = dict()
    # KV Cache Metadata
    for i in range(metadata_kv_count):
        key, offset = _read_string(raw_metadata, offset)
        value_type = version = struct.unpack("<I", raw_metadata[offset:offset+4])[0]
        offset += 4
        value, offset = Parsers[value_type](raw_metadata, offset)
        kv_metadata[key] = value
    
    kv_cache_info = None
    if experimental:
        # Extract and rename KV only necessary cache fields
        kv_cache_dict = {
            ending: kv_metadata[key]
            for ending in KV_CACHE_FIELD_ENDINGS
            for key in kv_metadata
            if key.endswith(ending)
        }

        # Replace with optional fields
        if max_model_len is not None: 
            kv_cache_dict["context_length"] = max_model_len
        if kv_cache_dtype is not None:
            kv_cache_dict["kv_dtype"] = kv_cache_dtype
        if batch_size is not None:
            kv_cache_dict["batch_size"] = batch_size

        missing_fields = set(KV_CACHE_FIELD_ENDINGS) - set(kv_cache_dict.keys())
        if missing_fields:
            raise RuntimeError(
                f"Incomplete KV cache metadata for size estimation. Missing fields: {missing_fields}"
            )
        
        kv_size = compute_gguf_kv_cache_size(
            kv_cache_dict, 
            kv_cache_dtype,
            kv_cache_dict.get("batch_size", 1)
        )
        
        kv_cache_info = GGUFKVCacheInfo(
            max_model_len=kv_cache_dict["context_length"],
            cache_size=kv_size,
            batch_size=kv_cache_dict["batch_size"],
            cache_dtype=kv_cache_dtype if kv_cache_dtype is not None else "F16",
        )
        
    
    # Tensor info
    component = GGUFComponentMetadata(dtypes={}, param_count=0, bytes_count=0)
    for i in range(tensor_count):
        name, offset = _read_string(raw_metadata, offset)
        n_dimensions, offset = _read_uint32(raw_metadata, offset)
        dimensions = []
        for _ in range(n_dimensions):
            dim, offset = _read_uint64(raw_metadata, offset)
            dimensions.append(dim)
        tensor_type, offset = _read_uint32(raw_metadata, offset)
        tensor_offset, offset = _read_uint64(raw_metadata, offset)

        param_count = math.prod(dimensions)
        bytes_count = int(GGUFDtypeBitsPerWeight[GGUFDtype(tensor_type)] / 8.0 * param_count)
        if GGUFDtype(tensor_type) in component.dtypes:
            dtype = component.dtypes[GGUFDtype(tensor_type)]
            dtype.param_count += param_count
            dtype.bytes_count += bytes_count
        else:
            component.dtypes[GGUFDtype(tensor_type)] = DtypeMetadata(
                param_count=param_count,
                bytes_count=bytes_count,
            )
        component.param_count += param_count
        component.bytes_count += bytes_count

    return GGUFMetadata(
        components={"Transformer": component},
        param_count=component.param_count,
        bytes_count=component.bytes_count,
        kv_cache_info=kv_cache_info,
    )


async def fetch_gguf_metadata(
    client: httpx.AsyncClient, 
    url: str, 
    experimental: bool = False,
    max_model_len: Optional[int] = None,
    kv_cache_dtype: Optional[str] = "F16",
    batch_size: int = 1,
    headers: Optional[Dict[str, str]] = None
) -> GGUFMetadata:
    for size in SIZES:
        try: 
            if kv_cache_dtype is not None and kv_cache_dtype == "auto":
                kv_cache_dtype = "F16"
            request_headers = {"Range": f"bytes=0-{size}", **(headers or {})}
            response = await client.get(url, headers=request_headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            raw_metadata = response.read()
            processed_metadata = parse_gguf_metadata(
                raw_metadata = raw_metadata, 
                experimental = experimental, 
                max_model_len = max_model_len, 
                kv_cache_dtype = kv_cache_dtype, 
                batch_size = batch_size)
            
            return processed_metadata
        
        except (struct.error, UnicodeDecodeError):
            if size == SIZES[-1]:
                raise RuntimeError(f"Failed to parse GGUF metadata from {url}, metadata larger than {size/1_000_000:.2f} MB.")
            continue
