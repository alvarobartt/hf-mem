from typing import Literal

SafetensorsDtypes = Literal[
    "C64",
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
    "F8_E8M0",
    "F8_E5M2",
    "F8_E4M3",
    "I8",
    "U8",
    "BOOL",
    "F6_E2M3",
    "F6_E3M2",
    "F4",
]


def get_safetensors_dtype_bits(dtype: SafetensorsDtypes | str) -> int:
    match dtype:
        case "C64" | "F64" | "I64" | "U64":
            return 64
        case "F32" | "I32" | "U32":
            return 32
        case "F16" | "BF16" | "I16" | "U16":
            return 16
        case "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "I8" | "U8" | "BOOL":
            return 8
        case "F6_E2M3" | "F6_E3M2":
            return 6
        case "F4":
            return 4
        case _:
            raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")


def get_safetensors_dtype_bytes(dtype: SafetensorsDtypes | str) -> float:
    return get_safetensors_dtype_bits(dtype) / 8


TorchDtypes = Literal["float32", "float16", "bfloat16", "float8_e4m3", "float8_e4m3fn", "float8_e5m2", "int8"]


def torch_dtype_to_safetensors_dtype(dtype: TorchDtypes | str) -> SafetensorsDtypes:
    if dtype.startswith("torch."):
        dtype = dtype.replace("torch.", "")
    match dtype:
        case "float32":
            return "F32"
        case "float16":
            return "F16"
        case "bfloat16":
            return "BF16"
        case "float8_e4m3" | "float8_e4m3fn":
            return "F8_E4M3"
        case "float8_e5m2":
            return "F8_E5M2"
        # NOTE: `I8` is usually not used for quantizing i.e., the KV cache will never be of type `I8`, hence this
        # case might never be hit
        case "int8":
            return "I8"
        case _:
            return "F16"
