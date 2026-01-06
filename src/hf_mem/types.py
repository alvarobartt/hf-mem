from typing import Literal

SafetensorsDtypes = Literal[
    "F64",
    "I64",
    "U64",
    "F32",
    "I32",
    "U32",
    "F16",
    "BF16",
    "I16",
    "U16",
    "F8_E5M2",
    "F8_E4M3",
    "I8",
    "U8",
]


def get_safetensors_dtype_bytes(dtype: str) -> int:
    match dtype:
        case "F64" | "I64" | "U64":
            return 8
        case "F32" | "I32" | "U32":
            return 4
        case "F16" | "BF16" | "I16" | "U16":
            return 2
        case "F8_E5M2" | "F8_E4M3" | "I8" | "U8":
            return 1
        case _:
            raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")
