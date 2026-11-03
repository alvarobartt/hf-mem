import os

import pytest
from helpers import llama_config, llama_tensors, write_json, write_safetensors

from hf_mem import Estimate, _safetensors, run


def test_dense_model_with_kv_cache(tmp_path):
    nbytes = write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    result = run(str(tmp_path))

    assert isinstance(result, Estimate)
    assert result.weights.bytes == nbytes
    assert set(result.weights.dtypes) == {"BF16", "F32"}
    assert result.kv_cache is not None
    assert result.kv_cache.bytes == 2 * 4 * 2 * 32 * 1024 * 2
    assert result.total == nbytes + result.kv_cache.bytes + result.overhead.bytes


def test_kv_cache_options(tmp_path):
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    result = run(str(tmp_path), max_model_len=100, batch_size=3, kv_cache_dtype="fp8")

    assert result.kv_cache.bytes == 2 * 4 * 2 * 32 * 112 * 3
    assert result.kv_cache.dtype == "F8_E4M3"
    with pytest.raises(RuntimeError, match="not valid for Safetensors"):
        run(str(tmp_path), kv_cache_dtype="Q4_K")


def test_sharded_model_uses_the_index(tmp_path):
    tensors = list(llama_tensors().items())
    first = write_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"), dict(tensors[:5]))
    second = write_safetensors(str(tmp_path / "model-00002-of-00002.safetensors"), dict(tensors[5:]))
    write_safetensors(str(tmp_path / "consolidated.safetensors"), dict(tensors))
    write_json(
        str(tmp_path / "model.safetensors.index.json"),
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
    )
    write_json(str(tmp_path / "config.json"), llama_config())

    result = run(str(tmp_path))

    assert result.weights.bytes == first + second
    assert list(result.components) == ["Transformer"]


def test_complete_shard_listing_skips_the_index(tmp_path):
    tensors = list(llama_tensors().items())
    first = write_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"), dict(tensors[:5]))
    second = write_safetensors(str(tmp_path / "model-00002-of-00002.safetensors"), dict(tensors[5:]))
    (tmp_path / "model.safetensors.index.json").write_text("not json")
    write_json(str(tmp_path / "config.json"), llama_config())

    assert run(str(tmp_path)).weights.bytes == first + second

    write_safetensors(str(tmp_path / "model-00001-of-00003.safetensors"), dict(tensors))
    write_json(
        str(tmp_path / "model.safetensors.index.json"),
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
    )
    assert run(str(tmp_path)).weights.bytes == first + second


def test_many_shards_are_parsed_in_a_process_pool(tmp_path, monkeypatch):
    for i in range(4):
        write_safetensors(
            str(tmp_path / f"model-0000{i + 1}-of-00004.safetensors"),
            {f"model.layers.{i}.mlp.experts.{e}.down_proj.weight": ("BF16", [64, 32]) for e in range(4)}
            | {f"model.layers.{i}.self_attn.q_proj.weight": ("BF16", [64, 64])},
        )
    write_json(str(tmp_path / "config.json"), llama_config(num_local_experts=4, num_experts_per_tok=2))

    inline = run(str(tmp_path))
    monkeypatch.setattr(_safetensors, "POOL_MIN_FILES", 1)
    pooled = run(str(tmp_path))

    assert pooled.to_dict() == inline.to_dict()
    assert pooled.moe is not None and pooled.moe.experts == 4
    assert pooled.moe.expert.bytes == 4 * 64 * 32 * 2


def test_diffusers_pipeline_prefers_the_default_variant(tmp_path):
    transformer = write_safetensors(
        str(tmp_path / "transformer/diffusion_pytorch_model.safetensors"), {"w": ("BF16", [128, 128])}
    )
    write_safetensors(
        str(tmp_path / "transformer/diffusion_pytorch_model.fp16.safetensors"), {"w": ("F16", [128, 128])}
    )
    vae = write_safetensors(
        str(tmp_path / "vae/diffusion_pytorch_model.fp16.safetensors"), {"w": ("F32", [64])}
    )
    write_json(
        str(tmp_path / "model_index.json"),
        {
            "_class_name": "Pipeline",
            "transformer": ["diffusers", "T"],
            "vae": ["diffusers", "V"],
            "scheduler": ["d", "S"],
        },
    )

    result = run(str(tmp_path))

    assert {n: c.bytes for n, c in result.components.items()} == {"transformer": transformer, "vae": vae}
    assert result.kv_cache is None and result.overhead is None


def test_sentence_transformers_adds_dense_modules(tmp_path):
    main = write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [100, 64])})
    dense = write_safetensors(str(tmp_path / "2_Dense/model.safetensors"), {"linear.weight": ("F32", [64, 32])})
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    write_json(str(tmp_path / "modules.json"), [{"path": ""}, {"path": "1_Pooling"}, {"path": "2_Dense"}])
    write_json(str(tmp_path / "config.json"), llama_config(architectures=["Gemma3TextModel"]))

    result = run(str(tmp_path))

    assert {n: c.bytes for n, c in result.components.items()} == {"0_Transformer": main, "2_Dense": dense}
    assert result.kv_cache is None


def test_unindexed_files_are_separate_components(tmp_path):
    write_safetensors(str(tmp_path / "flux-dev.safetensors"), {"w": ("BF16", [64, 64])})
    write_safetensors(str(tmp_path / "flux-schnell-00001-of-00002.safetensors"), {"a": ("BF16", [32])})
    write_safetensors(str(tmp_path / "flux-schnell-00002-of-00002.safetensors"), {"b": ("BF16", [32])})

    result = run(str(tmp_path))
    assert sorted(result.components) == ["flux-dev", "flux-schnell"]
    assert any("separate component" in w for w in result.warnings)

    single = run(str(tmp_path), file="flux-schnell-00001-of-00002.safetensors")
    assert single.weights.bytes == 128
    with pytest.raises(RuntimeError, match="No Safetensors file matching"):
        run(str(tmp_path), file="missing.safetensors")


def test_moe_breakdown(tmp_path):
    tensors = {"model.embed_tokens.weight": ("BF16", [100, 64])}
    for layer in range(2):
        for expert in range(4):
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = ("BF16", [128, 64])
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config(num_local_experts=4, num_experts_per_tok=2))

    moe = run(str(tmp_path)).moe

    assert moe is not None
    assert (moe.experts, moe.active) == (4, 2)
    assert moe.expert.bytes == 2 * 128 * 64 * 2
    assert moe.experts_total.bytes == 4 * moe.expert.bytes
    assert moe.base.bytes == 100 * 64 * 2


def test_moe_breakdown_skipped_for_non_uniform_experts(tmp_path):
    tensors = {
        "model.layers.0.mlp.experts.0.w.weight": ("BF16", [128, 64]),
        "model.layers.0.mlp.experts.1.w.weight": ("BF16", [64, 64]),
    }
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config())

    result = run(str(tmp_path))

    assert result.moe is None
    assert any("not uniform" in w for w in result.warnings)


def test_weights_from_file_sizes(tmp_path):
    for name, size in {"pytorch_model.bin": 1000, "optimizer.pt": 99, "README.md": 5}.items():
        with open(tmp_path / name, "wb") as f:
            f.truncate(size)

    result = run(str(tmp_path))

    assert result.format == "file-sizes"
    assert result.weights.bytes == 1000


def test_no_weights(tmp_path):
    write_json(str(tmp_path / "config.json"), llama_config())
    with pytest.raises(RuntimeError, match="No Safetensors or GGUF weights"):
        run(str(tmp_path))


def test_local_file_reads_the_sibling_config(tmp_path):
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    result = run(os.path.join(str(tmp_path), "model.safetensors"))

    assert result.file == "model.safetensors"
    assert result.kv_cache is not None


def test_mistral_params_json_warns(tmp_path):
    write_safetensors(str(tmp_path / "consolidated.safetensors"), llama_tensors())
    write_json(str(tmp_path / "params.json"), {"dim": 64, "n_layers": 4})

    result = run(str(tmp_path))

    assert result.kv_cache is None
    assert any("params.json" in w for w in result.warnings)
