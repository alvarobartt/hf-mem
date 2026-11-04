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
    write_safetensors,
)

from hf_mem import Listing, _gguf, run
from hf_mem._source import LocalSource

ARCH = {
    "general.architecture": "llama",
    "llama.block_count": 4,
    "llama.context_length": 2048,
    "llama.embedding_length": 256,
    "llama.attention.head_count": 8,
    "llama.attention.head_count_kv": 2,
}
TENSORS = [
    ("token_embd.weight", 12, [256, 1000]),
    ("output_norm.weight", 0, [256]),
    ("blk.0.attn_q.weight", 8, [256, 256]),
]
TOKENIZER = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": ["a", "b", "c"] * 1000}


def test_single_file(tmp_path: Path) -> None:
    write_gguf(str(tmp_path / "model-Q4_K_M.gguf"), {**ARCH, **TOKENIZER}, TENSORS)

    result = estimate(str(tmp_path))

    assert result.file == "model-Q4_K_M.gguf"
    assert result.weights.dtypes["Q4_K"].bytes == 256 * 1000 // 256 * 144
    assert result.weights.dtypes["F32"].bytes == 256 * 4
    assert result.weights.dtypes["Q8_0"].bytes == 256 * 256 // 32 * 34
    assert some(result.kv_cache).bytes == 4 * 2 * (32 + 32) * 2048 * 2
    assert some(result.kv_cache).dtype == "F16"


def test_listing_uses_file_sizes_and_per_file_metadata(tmp_path: Path) -> None:
    for quant in ("Q4_K_M", "Q8_0"):
        write_gguf(str(tmp_path / f"model-{quant}.gguf"), {**ARCH, **TOKENIZER}, TENSORS)
    write_gguf(str(tmp_path / "mtp-model-Q4_K_M.gguf"), {**ARCH, "llama.block_count": 5, **TOKENIZER}, TENSORS)
    for i in (1, 2):
        write_gguf(str(tmp_path / f"BF16/model-BF16-0000{i}-of-00002.gguf"), {**ARCH, **TOKENIZER}, TENSORS[:2])
    write_gguf(str(tmp_path / "mmproj-model-F16.gguf"), ARCH, TENSORS)

    result = listing(str(tmp_path), max_model_len=512)

    files = {some(f.file): f for f in result.files}
    assert sorted(files) == [
        "BF16/model-BF16.gguf",
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
        "mtp-model-Q4_K_M.gguf",
    ]
    single = estimate(str(tmp_path), file="model-Q4_K_M.gguf")
    assert abs(files["model-Q4_K_M.gguf"].weights.bytes - single.weights.bytes) < 1024
    assert (
        some(files["mtp-model-Q4_K_M.gguf"].kv_cache).bytes
        == 5 / 4 * some(files["model-Q4_K_M.gguf"].kv_cache).bytes
    )
    bf16 = estimate(str(tmp_path), file="BF16")
    assert bf16.components["Model"].bytes == 2 * (256 * 1000 // 256 * 144 + 256 * 4)
    assert bf16.weights.bytes == bf16.components["Model"].bytes + bf16.components["mmproj"].bytes


def test_kv_cache_variants() -> None:
    assert (
        some(_gguf.kv_cache({**ARCH, "llama.attention.head_count_kv": [2, 0, 2, 0]}, 100, 1, "F16"))[0]
        == 2 * 2 * 64 * 100 * 2
    )

    mla = {**ARCH, "llama.attention.kv_lora_rank": 512, "llama.rope.dimension_count": 64}
    assert some(_gguf.kv_cache(mla, 100, 1, "F16"))[0] == 4 * 576 * 100 * 2

    hybrid = {**ARCH, "llama.full_attention_interval": 2, "llama.nextn_predict_layers": 1}
    assert some(_gguf.kv_cache(hybrid, 100, 1, "Q8_0"))[1] == {"full": 1, "sliding": 0, "state": 0, "none": 2}

    ssm = {
        **ARCH,
        "llama.full_attention_interval": 4,
        "llama.ssm.state_size": 128,
        "llama.ssm.inner_size": 6144,
        "llama.ssm.group_count": 16,
        "llama.ssm.conv_kernel": 4,
    }
    size, layers = some(_gguf.kv_cache(ssm, 100, 2, "F16"))
    assert layers == {"full": 1, "sliding": 0, "state": 3, "none": 0}
    assert size == 2 * (2 * 64 * 100 * 2 + 3 * 4 * (3 * (6144 + 2 * 16 * 128) + 6144 * 128))

    swa = {
        **ARCH,
        "llama.attention.sliding_window": 64,
        "llama.attention.sliding_window_pattern": [True, True, True, False],
    }
    size, layers = some(_gguf.kv_cache(swa, 4096, 1, "F16"))
    assert layers == {"full": 1, "sliding": 3, "state": 0, "none": 0}
    assert size == (4096 + 3 * (64 + 512)) * 2 * 64 * 2

    assert _gguf.kv_cache({"general.architecture": "clip"}, 100, 1, "F16") is None


def test_unknown_tensor_types_use_data_offsets(tmp_path: Path) -> None:
    path = str(tmp_path / "model.gguf")
    write_gguf(path, ARCH, [("a", 0, [64]), ("b", 142, [64]), ("c", 0, [64])])
    result = estimate(path)
    assert result.weights.dtypes["TYPE_142"].bytes == 64


def test_large_headers_are_read_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_gguf, "FIRST_READ", 1024)
    monkeypatch.setattr(_gguf, "PREFIX_READ", 256)
    write_gguf(str(tmp_path / "a-Q4_0.gguf"), {**ARCH, **TOKENIZER}, TENSORS)
    write_gguf(str(tmp_path / "b-Q4_0.gguf"), {**TOKENIZER, **ARCH}, TENSORS)
    result = listing(str(tmp_path))
    assert [some(f.kv_cache).bytes for f in result.files] == [4 * 2 * 64 * 2048 * 2] * 2


def test_invalid_files(tmp_path: Path) -> None:
    (tmp_path / "broken-Q4_0.gguf").write_bytes(b"\0" * 64)
    write_gguf(str(tmp_path / "ok-Q4_0.gguf"), ARCH, TENSORS)
    result = listing(str(tmp_path))
    assert [f.file for f in result.files] == ["ok-Q4_0.gguf"]
    assert any("broken-Q4_0.gguf" in w for w in result.warnings)
    with pytest.raises(RuntimeError, match="not a valid GGUF"):
        run(str(tmp_path), file="broken-Q4_0.gguf")
    write_gguf(str(tmp_path / "Q8_0/a.gguf"), ARCH, TENSORS)
    write_gguf(str(tmp_path / "Q8_0/b.gguf"), ARCH, TENSORS)
    with pytest.raises(RuntimeError, match="Multiple GGUF files"):
        run(str(tmp_path), file="Q8_0")
    with pytest.raises(RuntimeError, match="not a llama.cpp KV cache type"):
        run(str(tmp_path), file="ok-Q4_0.gguf", kv_cache_dtype="bfloat16")


def test_listing_stops_at_large_embedded_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    embedded = {"general.architecture": "audio", "audio.embedded_files.data": list(range(1 << 20)), **ARCH}
    for quant in ("F16", "Q8_0"):
        write_gguf(str(tmp_path / f"audio-{quant}.gguf"), embedded, TENSORS)
    reads = []
    read = LocalSource.read

    async def counting(self: LocalSource, path: str, start: int, end: int) -> bytes:
        reads.append(end - start)
        return await read(self, path, start, end)

    monkeypatch.setattr(LocalSource, "read", counting)
    result = run(str(tmp_path))

    assert isinstance(result, Listing)
    assert len(result.files) == 2
    assert all(f.kv_cache is None for f in result.files)
    assert sum(reads) < 8 << 20


def test_listing_warns_that_the_gpu_fit_needs_a_file(tmp_path: Path) -> None:
    for quant in ("Q4_K_M", "Q8_0"):
        write_gguf(str(tmp_path / f"model-{quant}.gguf"), {**ARCH, **TOKENIZER}, TENSORS)
    result = listing(str(tmp_path), gpu_memory="24")
    assert any("`--gpu-memory` is only checked" in w for w in result.warnings)
    with pytest.raises(RuntimeError, match="--gpu-memory"):
        listing(str(tmp_path), gpu_memory="lots")


def test_format_picks_the_weights(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    write_gguf(str(tmp_path / "model-Q4_K_M.gguf"), {**ARCH, **TOKENIZER}, TENSORS)

    default = estimate(str(tmp_path))
    assert default.format == "safetensors" and any("--format gguf" in w for w in default.warnings)
    assert estimate(str(tmp_path), format="gguf").format == "gguf"
    assert not run(str(tmp_path), format="safetensors").warnings

    (tmp_path / "model.safetensors").unlink()
    assert estimate(str(tmp_path)).format == "gguf"
    with pytest.raises(RuntimeError, match="No Safetensors weights"):
        run(str(tmp_path), format="safetensors")
    with pytest.raises(
        RuntimeError, match="No ONNX weights found .* which has GGUF weights, use `--format gguf`"
    ):
        run(str(tmp_path), format="onnx")
    with pytest.raises(RuntimeError, match="not valid"):
        run(str(tmp_path), format="pt")

    (tmp_path / "pytorch_model.bin").write_bytes(b"\0" * 10)
    (tmp_path / "model-Q4_K_M.gguf").unlink()
    with pytest.raises(RuntimeError, match="only has PyTorch pickle weights"):
        run(str(tmp_path), format="safetensors")
    with pytest.raises(RuntimeError, match="only has PyTorch pickle weights"):
        run(str(tmp_path), format="gguf")


def test_quantization_tags_and_mmproj(tmp_path: Path) -> None:
    for name in ("m-Q4_K_M", "m-UD-Q4_K_M", "m-BF16", "mmproj-m-BF16", "mmproj-m-Q8_0"):
        write_gguf(str(tmp_path / f"{name}.gguf"), {**ARCH, **TOKENIZER}, TENSORS)

    write_gguf(
        str(tmp_path / "mmproj-m-Q8_0.gguf"),
        {
            "general.architecture": "clip",
            "clip.vision.embedding_length": 64,
            "clip.vision.image_size": 224,
            "clip.vision.patch_size": 14,
        },
        TENSORS,
    )
    q4 = estimate(str(tmp_path), file="q4_k_m")
    assert q4.file == "m-Q4_K_M.gguf" and list(q4.components) == ["Model", "mmproj"]
    assert some(q4.overhead).encoder == int(7.2 * 16**2 * 4 * 64)
    assert estimate(str(tmp_path), file="UD-Q4_K_M").file == "m-UD-Q4_K_M.gguf"
    assert (
        _gguf.mmproj(
            dict.fromkeys(["m-Q4_K_M.gguf", "mmproj-m-BF16.gguf", "mmproj-m-Q8_0.gguf"], 1), "m-Q4_K_M.gguf"
        )
        == "mmproj-m-Q8_0.gguf"
    )
    assert (
        _gguf.mmproj(
            dict.fromkeys(["m-BF16.gguf", "mmproj-m-BF16.gguf", "mmproj-m-Q8_0.gguf"], 1), "m-BF16.gguf"
        )
        == "mmproj-m-BF16.gguf"
    )
    assert _gguf.mmproj({"m-BF16.gguf": 1}, "m-BF16.gguf") is None
    files = {some(f.file): f for f in listing(str(tmp_path)).files}
    assert (
        files["m-Q4_K_M.gguf"].weights.bytes
        > estimate(str(tmp_path), file="m-Q4_K_M.gguf").components["Model"].bytes
    )
    with pytest.raises(RuntimeError, match="No GGUF file matching"):
        estimate(str(tmp_path), file="Q3_K_S")
