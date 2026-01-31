from dataclasses import dataclass
from typing import Literal, Dict, TypeGuard
from hf_mem.metadata import DtypeMetadata

GGUFDtypes = Literal[
    "F64", "I64",
    "F32", "I32",
    "F16", "BF16", "I16",
    # .GGUF specific dtypes
    "Q8_K", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K",
    "IQ4_NL", "IQ4_XS", 
    "IQ3_S", "IQ3_XXS", 
    "IQ2_XXS", "IQ2_S", "IQ2_XS",
    "IQ1_S", "IQ1_M",
    "TQ1_0", "TQ2_0",
    "MXFP4",
    # .GGUF specific dtypes (legacy type, added for exhaustivity)
    "Q8_0", "Q8_1", "Q5_0", "Q5_1", "Q4_0", "Q4_1",
]

GGUFDtypeBitsPerWeight: Dict[GGUFDtypes, float] = {
    "F64": 64.0,
    "I64": 64.0,
    "F32": 32.0,
    "I32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "I16": 16.0,
    # .GGUF specific dtypes
    "Q8_K": 8.03125,
    "Q6_K": 6.5625,
    "Q5_K": 5.5,
    "Q4_K": 4.5,
    "Q3_K": 3.4375,
    "Q2_K": 2.625,
    "IQ4_NL": 4.5,
    "IQ4_XS": 4.25,
    "IQ3_S": 3.44,
    "IQ3_XXS": 3.06,
    "IQ2_XXS": 2.06,
    "IQ2_S": 2.5,
    "IQ2_XS": 2.31,
    "IQ1_S": 1.56,
    "IQ1_M": 1.75,
    "TQ1_0": 1.6875,
    "TQ2_0": 2.0625,
    "MXFP4": 4.25,
    # .GGUF specific dtypes (legacy type, added for exhaustivity)
    "Q8_0": 8.5,
    "Q8_1": 9.0,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
}


# Very similar to ComponentMetadata, might want to generalize it / create father type of GGUFDtypes
@dataclass
class GGUFComponentMetadata:
    dtypes: Dict[GGUFDtypes, DtypeMetadata]
    param_count: int
    bytes_count: int

# Very similar to SafetensorsMetadata
@dataclass
class GGUFMetadata:
    components: Dict[str, GGUFComponentMetadata]
    param_count: int
    bytes_count: int


def is_gguf_dtype(value: str) -> TypeGuard[GGUFDtypes]:
    return value in GGUFDtypeBitsPerWeight


def get_gguf_dtype_bytes(dtype: str) -> float:
    if not is_gguf_dtype(dtype):
        raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")
    
    return GGUFDtypeBitsPerWeight[dtype] / 8.0 # bit to byte conversion