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
    "F8_E5M2",  # NOTE: Only CUDA +11.8
    "F8_E4M3",  # NOTE: CUDA +11.8 and AMD ROCm
    "I8",
    "U8",
]


def get_safetensors_dtype_bytes(dtype: SafetensorsDtypes | str) -> int:
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
        case "int8":
            return "I8"
        case _:
            return "F16"
