GGUFDtypeBitsPerWeight = {
    "F64": 64,
    "I64": 64,
    "F32": 32,
    "I32": 32,
    "F16": 16,
    "BF16": 16,
    "I16": 16,
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
    "Q8_1": 9,
    "Q5_0": 5.5,
    "Q5_1": 6,
    "Q4_0": 4.5,
    "Q4_1": 5,
}

def get_gguf_dtype_bytes(dtype: str) -> float:
    if dtype not in GGUFDtypeBitsPerWeight.keys():
        raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")
    
    return GGUFDtypeBitsPerWeight[dtype] / 8.0 # bit to byte conversion