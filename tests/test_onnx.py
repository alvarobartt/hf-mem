from pathlib import Path

import pytest
from helpers import (
    estimate,
    listing,
    llama_config,
    llama_tensors,
    some,
    write_gguf,
    write_json,
    write_onnx,
    write_safetensors,
)
from test_gguf import ARCH, TENSORS, TOKENIZER

from hf_mem import _onnx

WEIGHTS = [("embed", 10, [1000, 64]), ("proj", 22, [64, 64]), ("scale", 1, [64])]
PAST: list[tuple[str, int, list[int | str]]] = [
    (f"past_key_values.{i}.{kind}", 10, ["batch", 2, "past_sequence_length", 16])
    for i in range(2)
    for kind in ("key", "value")
]
PER_TOKEN = 4 * 2 * 16 * 2


def test_single_file_reads_dtypes_and_the_kv_cache_from_the_graph(tmp_path: Path) -> None:
    write_onnx(str(tmp_path / "model.onnx"), WEIGHTS, [("input_ids", 7, ["batch", "sequence"]), *PAST])
    write_json(str(tmp_path / "config.json"), llama_config(max_position_embeddings=1024))

    result = estimate(str(tmp_path), batch_size=2)

    assert result.format == "onnx" and result.weights.params == 64000 + 4096 + 64
    assert {d: w.bytes for d, w in result.weights.dtypes.items()} == {"F16": 128000, "I4": 2048, "F32": 256}
    kv = some(result.kv_cache)
    assert (kv.dtype, kv.max_model_len, kv.bytes) == ("F16", 1024, PER_TOKEN * 1024 * 2)
    assert any("ONNX Runtime overhead" in w for w in result.warnings)
    with pytest.raises(RuntimeError, match="only `auto`"):
        estimate(str(tmp_path), kv_cache_dtype="fp8")


def test_external_data_and_large_inline_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    size = write_onnx(str(tmp_path / "model.onnx"), WEIGHTS, PAST, external="model.onnx_data")
    write_onnx(str(tmp_path / "model_fp16.onnx"), WEIGHTS, PAST)
    write_onnx(str(tmp_path / "model_q4.onnx"), WEIGHTS, PAST, external="model_q4.onnx_data")
    (tmp_path / "model_q4.onnx").write_bytes((tmp_path / "model_q4.onnx").read_bytes() + b"\0" * 2000)
    monkeypatch.setattr(_onnx, "SMALL_FILE", 1500)

    result = listing(str(tmp_path), max_model_len=100)

    files = {some(f.file): f for f in result.files}
    assert files["model.onnx"].weights.bytes == size and files["model.onnx"].weights.dtypes
    large = files["model_fp16.onnx"]
    assert large.weights.bytes == (tmp_path / "model_fp16.onnx").stat().st_size and not large.weights.dtypes
    assert some(large.kv_cache).bytes == PER_TOKEN * 100
    q4 = files["model_q4.onnx"]
    assert q4.weights.bytes == sum(
        (tmp_path / f).stat().st_size for f in ("model_q4.onnx", "model_q4.onnx_data")
    )
    assert any("estimated from their file sizes" in w for w in result.warnings)


@pytest.mark.parametrize("prefix", ["past_key_values", "cache", "cache_params", "input.cache"])
@pytest.mark.parametrize("large", [False, True])
def test_transformers_cache_input_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str, large: bool
) -> None:
    inputs: list[tuple[str, int, list[int | str]]] = [
        (f"{prefix}.layers.{i}.{kind}", 10, ["batch", 2, "past_sequence_length", 16])
        for i in range(2)
        for kind in ("keys", "values")
    ]
    inputs += [("cache_position", 7, ["sequence"]), (f"{prefix}.layers.0.cumulative_length", 7, [])]
    write_onnx(str(tmp_path / "model.onnx"), WEIGHTS, inputs)
    write_json(str(tmp_path / "config.json"), llama_config(max_position_embeddings=None, model_max_length=2048))
    if large:
        monkeypatch.setattr(_onnx, "SMALL_FILE", 1)

    result = some(estimate(str(tmp_path), batch_size=2).kv_cache)

    assert (result.max_model_len, result.dtype, result.bytes) == (2048, "F16", PER_TOKEN * 2048 * 2)


@pytest.mark.parametrize("large", [False, True])
def test_cache_position_alone_is_not_a_kv_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, large: bool
) -> None:
    write_onnx(str(tmp_path / "model.onnx"), WEIGHTS, [("cache_position", 7, ["sequence"])])
    if large:
        monkeypatch.setattr(_onnx, "SMALL_FILE", 1)
    assert estimate(str(tmp_path), max_model_len=1024).kv_cache is None


def test_variants_and_encoder_decoder_parts_are_grouped() -> None:
    files = dict.fromkeys(
        [
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx",
            "onnx/decoder_model_merged.onnx",
            "onnx/decoder_with_past_model.onnx",
            "onnx/encoder_model_fp16.onnx",
            "onnx/decoder_model_merged_fp16.onnx",
            "onnx/decoder_model_merged_fp16.onnx_data",
            "cuda/model.onnx",
            "cuda/model.onnx.data",
            "model_O1.onnx",
            "model.onnx",
        ],
        1,
    )
    assert _onnx.group_files(files) == {
        "onnx/encoder_model+decoder_model_merged": [
            "onnx/encoder_model.onnx",
            "onnx/decoder_model_merged.onnx",
        ],
        "onnx/encoder_model+decoder_model_merged_fp16": [
            "onnx/encoder_model_fp16.onnx",
            "onnx/decoder_model_merged_fp16.onnx",
            "onnx/decoder_model_merged_fp16.onnx_data",
        ],
        "cuda/model.onnx": ["cuda/model.onnx", "cuda/model.onnx.data"],
        "model_O1.onnx": ["model_O1.onnx"],
        "model.onnx": ["model.onnx"],
    }
    groups = _onnx.group_files(files)
    assert _onnx.select_file(groups, "cuda")[0] == "cuda/model.onnx"
    assert _onnx.select_file(groups, "model.onnx")[0] == "model.onnx"
    with pytest.raises(RuntimeError, match="No ONNX file"):
        _onnx.select_file(groups, "missing.onnx")


def test_onnx_comes_after_safetensors_and_file_sizes(tmp_path: Path) -> None:
    write_onnx(str(tmp_path / "onnx" / "model.onnx"), WEIGHTS, PAST)
    assert estimate(str(tmp_path)).format == "onnx"

    (tmp_path / "pytorch_model.bin").write_bytes(b"\0" * 10)
    assert estimate(str(tmp_path)).format == "onnx"
    assert estimate(str(tmp_path), format="onnx").format == "onnx"
    assert estimate(str(tmp_path), file="onnx/model.onnx").format == "onnx"

    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    default = estimate(str(tmp_path))
    assert default.format == "safetensors" and any("--format onnx" in w for w in default.warnings)


def test_invalid_onnx_files(tmp_path: Path) -> None:
    write_onnx(str(tmp_path / "model.onnx"), WEIGHTS)
    (tmp_path / "model_fp16.onnx").write_bytes(b"\xff" * 16)

    result = listing(str(tmp_path))

    assert [f.file for f in result.files] == ["model.onnx"]
    assert any("Skipping `model_fp16.onnx`" in w for w in result.warnings)
    with pytest.raises(RuntimeError, match="not a valid ONNX file"):
        estimate(str(tmp_path), file="model_fp16.onnx")


def test_review_fixes(tmp_path: Path) -> None:
    write_onnx(
        str(tmp_path / "onnx" / "model_q4.onnx"),
        [("w_quant", 2, [64, 2, 16]), ("emb_quant", 2, [100, 32])],
        nodes=[
            ("MatMulNBits", ["x", "w_quant", "scales"], 4),
            ("GatherBlockQuantized", ["emb_quant", "ids"], 4),
        ],
    )
    write_onnx(str(tmp_path / "onnx" / "model.onnx"), WEIGHTS, PAST)
    write_json(str(tmp_path / "config.json"), llama_config(max_position_embeddings=64))

    q4 = estimate(str(tmp_path), file="onnx/model_q4.onnx")
    assert q4.weights.params == (64 * 2 * 16 + 100 * 32) * 2 and q4.weights.bytes == 64 * 2 * 16 + 100 * 32
    result = listing(str(tmp_path))
    assert result.to_dict()["format"] == "onnx"
    with pytest.raises(RuntimeError, match="Multiple ONNX files"):
        estimate(str(tmp_path), file="onnx")

    write_gguf(str(tmp_path / "model-Q4_K_M.gguf"), {**ARCH, **TOKENIZER}, TENSORS)
    assert estimate(str(tmp_path), file="onnx/model.onnx").format == "onnx"
    assert estimate(str(tmp_path)).format == "gguf"

    (tmp_path / "config.json").unlink()
    no_length = estimate(str(tmp_path), file="onnx/model.onnx")
    assert no_length.kv_cache is None and any("--max-model-len" in w for w in no_length.warnings)


@pytest.mark.parametrize("prefix", ["past_key_values", "cache"])
@pytest.mark.parametrize("large", [False, True])
def test_decoder_with_past_and_encoder_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str, large: bool
) -> None:
    cross: list[tuple[str, int, list[int | str]]] = [
        (
            f"past_key_values.0.{side}.{kind}"
            if prefix == "past_key_values"
            else f"input.cache.{'cross' if side == 'encoder' else 'self'}_attention_cache.layers.0.{kind}s",
            1,
            ["batch", 2, f"{side}_sequence_length", 16],
        )
        for side in ("decoder", "encoder")
        for kind in ("key", "value")
    ]
    write_onnx(str(tmp_path / "onnx" / "encoder_model.onnx"), WEIGHTS)
    write_onnx(str(tmp_path / "onnx" / "decoder_model.onnx"), WEIGHTS)
    write_onnx(str(tmp_path / "onnx" / "decoder_with_past_model.onnx"), WEIGHTS, cross)
    write_json(
        str(tmp_path / "config.json"), {**llama_config(max_position_embeddings=100), "max_source_positions": 10}
    )
    if large:
        monkeypatch.setattr(_onnx, "SMALL_FILE", 1)

    result = estimate(str(tmp_path))

    assert result.file == "onnx/encoder_model+decoder_model+decoder_with_past_model"
    per_token = 2 * 2 * 16 * 4
    assert some(result.kv_cache).bytes == per_token * 100 + per_token * 10
