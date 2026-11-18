from pathlib import Path

import pytest
from helpers import estimate, llama_config, write_json, write_safetensors

from hf_mem import _safetensors


@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("prefix", ["", "language_model."])
def test_tied_dense_tables_are_shared_across_checkpoint_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pooled: bool, tied: bool, prefix: str
) -> None:
    first = write_safetensors(
        str(tmp_path / "model-00001-of-00002.safetensors"),
        {prefix + "model.embed_tokens.weight": ("BF16", [1000, 256])},
    )
    second = write_safetensors(
        str(tmp_path / "model-00002-of-00002.safetensors"),
        {prefix + "lm_head.weight": ("BF16", [1000, 256]), prefix + "model.norm.weight": ("F32", [256])},
    )
    config = llama_config(tie_word_embeddings=tied)
    if prefix:
        config = {"architectures": ["LlavaForConditionalGeneration"], "text_config": config}
    write_json(str(tmp_path / "config.json"), config)
    if pooled:
        monkeypatch.setattr(_safetensors, "POOL_MIN_FILES", 1)
    result = estimate(str(tmp_path))
    assert result.weights.bytes == first + second - (512000 if tied else 0)
    assert result.weights.params == (256000 if tied else 512000) + 256
    assert result.weights.dtypes["F32"].bytes == 1024
    assert any("counted once" in warning for warning in result.warnings) == tied


@pytest.mark.parametrize(
    "tensors",
    [
        {"lm_head.weight": ("BF16", [1000, 256])},
        {"model.embed_tokens.weight": ("BF16", [1000, 256]), "lm_head.weight": ("BF16", [1000, 128])},
        {"model.embed_tokens.weight": ("F32", [1000, 256]), "lm_head.weight": ("BF16", [1000, 256])},
        {"model.embed_tokens.weight": ("BF16", [1000, 256]), "lm_head.weight": ("I8", [1000, 256])},
        {"model.embed_tokens.weight": ("BF16", [1000, 256]), "other.lm_head.weight": ("BF16", [1000, 256])},
    ],
)
def test_tying_does_not_guess_missing_incompatible_or_packed_tables(
    tmp_path: Path, tensors: dict[str, tuple[str, list[int]]]
) -> None:
    raw = write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config(tie_word_embeddings=True))
    result = estimate(str(tmp_path))
    assert result.weights.bytes == raw
    assert not any("counted once" in warning for warning in result.warnings)


def test_main_model_tying_does_not_apply_to_separate_auxiliary_components(tmp_path: Path) -> None:
    tables = {"model.embed_tokens.weight": ("BF16", [1000, 256]), "lm_head.weight": ("BF16", [1000, 256])}
    main = write_safetensors(str(tmp_path / "model.safetensors"), tables)
    auxiliary = write_safetensors(str(tmp_path / "codec/model.safetensors"), tables)
    write_json(str(tmp_path / "config.json"), llama_config(tie_word_embeddings=True))
    result = estimate(str(tmp_path))
    assert result.components["Transformer"].bytes == main // 2
    assert result.components["codec"].bytes == auxiliary
    assert result.weights.bytes == main // 2 + auxiliary


@pytest.mark.parametrize("family", ["qwen3_5_text", "qwen3_5_moe_text", "llama"])
@pytest.mark.parametrize("pooled", [False, True])
def test_only_verified_family_root_mtp_weights_are_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, pooled: bool
) -> None:
    main = write_safetensors(
        str(tmp_path / "model.safetensors"),
        {"model.norm.weight": ("F32", [256]), "mtp.layers.0.weight": ("BF16", [16, 256])},
    )
    auxiliary = write_safetensors(
        str(tmp_path / "codec/model.safetensors"), {"mtp.layers.0.weight": ("BF16", [16, 256])}
    )
    write_json(str(tmp_path / "config.json"), llama_config(model_type=family))
    if pooled:
        monkeypatch.setattr(_safetensors, "POOL_MIN_FILES", 1)
    result = estimate(str(tmp_path))
    assert result.weights.bytes == main + auxiliary - (8192 if family != "llama" else 0)
    assert any("prediction head" in warning for warning in result.warnings) == (family != "llama")
    assert result.components["codec"].bytes == auxiliary
