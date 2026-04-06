import math
import struct
from dataclasses import asdict, dataclass
from typing import Any, Dict

from hf_mem._types import KvCache
from hf_mem._version import __version__
from hf_mem.gguf.kv_cache import KV_CACHE_FIELD_ENDINGS, compute_gguf_kv_cache_size
from hf_mem.gguf.types import (
    _PARSERS,
    GGUFDtype,
    GGUFDtypeBitsPerWeight,
    _read_string,
    _read_uint32,
    _read_uint64,
)
from hf_mem.safetensors.metadata import DtypeMetadata


@dataclass
class GGUFComponentMetadata:
    dtypes: Dict[GGUFDtype, DtypeMetadata]
    param_count: int
    bytes_count: int


@dataclass
class GGUFMetadata:
    components: Dict[str, GGUFComponentMetadata]
    param_count: int
    bytes_count: int
    kv_cache: KvCache | None = None


def merge_shards(shard1: GGUFMetadata, shard2: GGUFMetadata) -> GGUFMetadata:
    merged_components: Dict[str, GGUFComponentMetadata] = {}
    all_component_names = set(shard1.components.keys()) | set(shard2.components.keys())

    for component_name in all_component_names:
        comp1 = shard1.components.get(component_name)
        comp2 = shard2.components.get(component_name)

        if comp1 and comp2:
            merged_dtypes: Dict[GGUFDtype, DtypeMetadata] = {}
            all_dtypes = set(comp1.dtypes.keys()) | set(comp2.dtypes.keys())

            for dtype in all_dtypes:
                dtype_meta1 = comp1.dtypes.get(dtype)
                dtype_meta2 = comp2.dtypes.get(dtype)

                if dtype_meta1 and dtype_meta2:
                    merged_dtypes[dtype] = DtypeMetadata(
                        param_count=dtype_meta1.param_count + dtype_meta2.param_count,
                        bytes_count=dtype_meta1.bytes_count + dtype_meta2.bytes_count,
                    )
                elif dtype_meta1:
                    merged_dtypes[dtype] = dtype_meta1
                else:
                    merged_dtypes[dtype] = dtype_meta2  # type: ignore[assignment]

            merged_components[component_name] = GGUFComponentMetadata(
                dtypes=merged_dtypes,
                param_count=comp1.param_count + comp2.param_count,
                bytes_count=comp1.bytes_count + comp2.bytes_count,
            )
        elif comp1:
            merged_components[component_name] = comp1
        else:
            merged_components[component_name] = comp2  # type: ignore[assignment]

    # NOTE: KV cache info is the same for all shards — prefer earlier shards since metadata can
    # get dropped after the first shard
    kv_cache = shard1.kv_cache or shard2.kv_cache

    return GGUFMetadata(
        components=merged_components,
        param_count=shard1.param_count + shard2.param_count,
        bytes_count=shard1.bytes_count + shard2.bytes_count,
        kv_cache=kv_cache,
    )


def gguf_metadata_to_json(model_id: str, revision: str, metadata: GGUFMetadata) -> Dict[str, Any]:
    out = asdict(metadata)

    # NOTE: Convert GGUFDtype enum keys to their string names so the dict is JSON-serializable
    for component in out["components"].values():
        component["dtypes"] = {k.name: v for k, v in component["dtypes"].items()}

    out = {"version": __version__, "model_id": model_id, "revision": revision, **out}

    # NOTE: If --experimental, flatten kv_cache fields to the top level to match the
    # safetensors JSON output shape
    if kv_cache := out.pop("kv_cache"):
        out["max_model_len"] = kv_cache["max_model_len"]
        out["cache_size"] = kv_cache["cache_size"]
        out["batch_size"] = kv_cache["batch_size"]
        out["cache_dtype"] = kv_cache["cache_dtype"]

    return out


def parse_gguf_metadata(
    raw_metadata: bytes,
    experimental: bool = False,
    max_model_len: int | None = None,
    kv_cache_dtype: str = "F16",
    batch_size: int = 1,
) -> "GGUFMetadata":
    # NOTE: Validate the GGUF magic number to fail fast on invalid or truncated files
    if raw_metadata[:4] != b"GGUF":
        raise RuntimeError("Not a valid GGUF file: magic number mismatch.")

    tensor_count = struct.unpack("<Q", raw_metadata[8:16])[0]
    metadata_kv_count = struct.unpack("<Q", raw_metadata[16:24])[0]
    offset = 24

    kv_metadata: Dict[str, Any] = {}
    for _ in range(metadata_kv_count):
        key, offset = _read_string(raw_metadata, offset)
        value_type = struct.unpack("<I", raw_metadata[offset : offset + 4])[0]
        offset += 4
        value, offset = _PARSERS[value_type](raw_metadata, offset)
        kv_metadata[key] = value

    kv_cache = None
    if experimental:
        # NOTE: Extract only the fields needed for KV cache estimation, matched by suffix to be
        # model-agnostic (e.g. "llama.block_count" -> "block_count")
        kv_cache_dict = {
            ending: kv_metadata[key]
            for ending in KV_CACHE_FIELD_ENDINGS
            for key in kv_metadata
            if key.endswith(ending)
        }
        if max_model_len is not None:
            kv_cache_dict["context_length"] = max_model_len

        missing_fields = set(KV_CACHE_FIELD_ENDINGS) - set(kv_cache_dict.keys())
        if missing_fields:
            raise RuntimeError(
                f"Incomplete KV cache metadata for size estimation. Missing fields: {missing_fields}"
            )

        kv_cache = KvCache(
            max_model_len=kv_cache_dict["context_length"],
            cache_size=compute_gguf_kv_cache_size(
                kv_metadata=kv_cache_dict,
                kv_cache_dtype=kv_cache_dtype,
                batch_size=batch_size,
            ),
            batch_size=batch_size,
            cache_dtype=kv_cache_dtype,
        )

    component = GGUFComponentMetadata(dtypes={}, param_count=0, bytes_count=0)
    for _ in range(tensor_count):
        _, offset = _read_string(raw_metadata, offset)
        n_dimensions, offset = _read_uint32(raw_metadata, offset)
        dimensions = []
        for _ in range(n_dimensions):
            dim, offset = _read_uint64(raw_metadata, offset)
            dimensions.append(dim)
        tensor_type, offset = _read_uint32(raw_metadata, offset)
        _, offset = _read_uint64(raw_metadata, offset)

        param_count = math.prod(dimensions)
        bytes_count = int(GGUFDtypeBitsPerWeight[GGUFDtype(tensor_type)] / 8.0 * param_count)

        if GGUFDtype(tensor_type) in component.dtypes:
            dtype_meta = component.dtypes[GGUFDtype(tensor_type)]
            dtype_meta.param_count += param_count
            dtype_meta.bytes_count += bytes_count
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
        kv_cache=kv_cache,
    )
