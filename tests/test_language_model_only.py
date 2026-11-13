import asyncio
from pathlib import Path
from typing import Any

import pytest
from helpers import llama_config, llama_tensors, some, write_gguf, write_json, write_safetensors
from test_gguf import ARCH, TENSORS

from hf_mem import _safetensors
from hf_mem._render import table
from hf_mem.estimate import Estimate, arun, run


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_language_only_skips_identified_weights_and_activations_but_text_requests_keep_weights(
    tmp_path: Path, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            **llama_tensors(),
            "model.visual.patch_embed.weight": ("F32", [256]),
            "model.multi_modal_projector.weight": ("F16", [32, 32]),
            # A decoder tensor containing the word visual is not an encoder namespace.
            "model.layers.0.visual_projection.weight": ("F16", [8, 8]),
        },
    )
    write_json(
        str(tmp_path / "config.json"),
        {
            **llama_config(),
            "architectures": ["MuseGlimmerForConditionalGeneration"]
            if engine == "sglang"
            else ["LlamaForCausalLM"],
            "vision_config": {"hidden_size": 256, "patch_size": 16, "image_size": 224},
        },
    )
    monkeypatch.setattr(_safetensors, "POOL_MIN_FILES", 1)
    options: dict[str, Any] = dict(engine=engine, gpu_type="nvidia-a10g", gpu_memory="2x24", max_num_seqs=8)
    full = run(str(tmp_path), **options)
    text = run(str(tmp_path), text_only=True, **options)
    lm = run(str(tmp_path), language_model_only=True, **options)
    assert isinstance(full, Estimate) and isinstance(text, Estimate) and isinstance(lm, Estimate)
    assert lm.weights.bytes == full.weights.bytes - (256 * 4 + 32 * 32 * 2)
    assert lm.weights.params == full.weights.params - (256 + 32 * 32)
    assert text.weights == full.weights
    assert lm.kv_cache == full.kv_cache == text.kv_cache
    assert some(lm.overhead).encoder == some(text.overhead).encoder == 0 < some(full.overhead).encoder
    assert some(lm.fit).required < some(text.fit).required
    assert lm.to_dict()["workload"] == "language-model-only"
    omitted = 256 * 4 + 32 * 32 * 2
    assert lm.startup_reserve_per_gpu == (omitted if engine == "vllm" else 0)
    if engine == "vllm":
        assert some(lm.fit).startup_required == some(lm.fit).required + 2 * omitted
        assert "STARTUP RESERVE" in table(lm)
    asynchronous = asyncio.run(arun(str(tmp_path), language_model_only=True, **options))
    assert asynchronous.to_dict() == lm.to_dict()


def test_unknown_multimodal_weight_names_are_retained(tmp_path: Path) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"), {**llama_tensors(), "unknown_encoder.weight": ("F32", [256])}
    )
    write_json(str(tmp_path / "config.json"), {**llama_config(), "vision_config": {"hidden_size": 256}})
    full = run(str(tmp_path))
    lm = run(str(tmp_path), language_model_only=True)
    assert isinstance(full, Estimate) and isinstance(lm, Estimate)
    assert full.weights == lm.weights
    assert any("could not identify" in warning for warning in lm.warnings)


def test_language_only_gguf_omits_projector_weights_and_encoder(tmp_path: Path) -> None:
    write_gguf(str(tmp_path / "model-F16.gguf"), ARCH, TENSORS)
    write_gguf(str(tmp_path / "mmproj-model-F16.gguf"), {"clip.vision.embedding_length": 256}, TENSORS)
    full = run(str(tmp_path), file="model-F16.gguf")
    lm = run(str(tmp_path), file="model-F16.gguf", language_model_only=True)
    assert isinstance(full, Estimate) and isinstance(lm, Estimate)
    assert full.weights.bytes - lm.weights.bytes == full.components["mmproj"].bytes
    assert full.kv_cache == lm.kv_cache
    assert some(lm.overhead).encoder == 0 < some(full.overhead).encoder
    assert lm.to_dict()["workload"] == "language-model-only"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_sglang_rejects_an_unsupported_language_only_architecture(tmp_path: Path, asynchronous: bool) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), {"architectures": ["Qwen3_5ForConditionalGeneration"]})
    with pytest.raises(RuntimeError, match="engine rejects this launch"):
        if asynchronous:
            asyncio.run(arun(str(tmp_path), engine="sglang", language_model_only=True))
        else:
            run(str(tmp_path), engine="sglang", language_model_only=True)
