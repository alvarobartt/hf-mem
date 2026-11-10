import os
from pathlib import Path

import pytest
from helpers import estimate, kimi_config, llama_config, llama_tensors, some, write_json, write_safetensors

from hf_mem import Estimate, _config, _safetensors
from hf_mem._render import markdown, table
from hf_mem._source import LocalSource
from hf_mem._weights import Weights


def test_dense_model_with_kv_cache(tmp_path: Path) -> None:
    nbytes = write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path))

    assert isinstance(result, Estimate)
    assert result.weights.bytes == nbytes
    assert set(result.weights.dtypes) == {"BF16", "F32"}
    assert result.kv_cache is not None
    assert some(result.kv_cache).bytes == 2 * 4 * 2 * 32 * 1024 * 2
    assert result.total == nbytes + some(result.kv_cache).bytes + some(result.overhead).bytes


def test_kv_cache_options(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path), max_model_len=100, batch_size=3, kv_cache_dtype="fp8")

    assert some(result.kv_cache).bytes == 2 * 4 * 2 * 32 * 112 * 3
    assert some(result.kv_cache).dtype == "F8_E4M3"
    with pytest.raises(RuntimeError, match="not a vLLM KV cache dtype"):
        estimate(str(tmp_path), kv_cache_dtype="Q4_K")


def test_sharded_model_uses_the_index(tmp_path: Path) -> None:
    tensors = list(llama_tensors().items())
    first = write_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"), dict(tensors[:5]))
    second = write_safetensors(str(tmp_path / "model-00002-of-00002.safetensors"), dict(tensors[5:]))
    write_safetensors(str(tmp_path / "consolidated.safetensors"), dict(tensors))
    write_json(
        str(tmp_path / "model.safetensors.index.json"),
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
    )
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path))

    assert result.weights.bytes == first + second
    assert list(result.components) == ["Transformer"]


def test_complete_shard_listing_skips_the_index(tmp_path: Path) -> None:
    tensors = list(llama_tensors().items())
    first = write_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"), dict(tensors[:5]))
    second = write_safetensors(str(tmp_path / "model-00002-of-00002.safetensors"), dict(tensors[5:]))
    (tmp_path / "model.safetensors.index.json").write_text("not json")
    write_json(str(tmp_path / "config.json"), llama_config())

    assert estimate(str(tmp_path)).weights.bytes == first + second

    write_safetensors(str(tmp_path / "model-00001-of-00003.safetensors"), dict(tensors))
    write_json(
        str(tmp_path / "model.safetensors.index.json"),
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
    )
    assert estimate(str(tmp_path)).weights.bytes == first + second


def test_many_shards_are_parsed_in_a_process_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for i in range(4):
        write_safetensors(
            str(tmp_path / f"model-0000{i + 1}-of-00004.safetensors"),
            {f"model.layers.{i}.mlp.experts.{e}.down_proj.weight": ("BF16", [64, 32]) for e in range(4)}
            | {f"model.layers.{i}.self_attn.q_proj.weight": ("BF16", [64, 64])},
        )
    write_json(str(tmp_path / "config.json"), llama_config(num_local_experts=4, num_experts_per_tok=2))

    inline = estimate(str(tmp_path))
    monkeypatch.setattr(_safetensors, "POOL_MIN_FILES", 1)
    pooled = estimate(str(tmp_path))

    assert pooled.to_dict() == inline.to_dict()
    assert pooled.moe is not None and some(pooled.moe).experts == 4
    assert some(pooled.moe).expert.bytes == 4 * 64 * 32 * 2


def test_diffusers_pipeline_prefers_the_default_variant(tmp_path: Path) -> None:
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

    result = estimate(str(tmp_path))

    assert {n: c.bytes for n, c in result.components.items()} == {"transformer": transformer, "vae": vae}
    assert result.kv_cache is None and result.overhead is None


def test_sentence_transformers_adds_dense_modules(tmp_path: Path) -> None:
    main = write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [100, 64])})
    dense = write_safetensors(str(tmp_path / "2_Dense/model.safetensors"), {"linear.weight": ("F32", [64, 32])})
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    write_json(str(tmp_path / "modules.json"), [{"path": ""}, {"path": "1_Pooling"}, {"path": "2_Dense"}])
    write_json(str(tmp_path / "config.json"), llama_config(architectures=["Gemma3TextModel"]))

    result = estimate(str(tmp_path))

    assert {n: c.bytes for n, c in result.components.items()} == {"0_Transformer": main, "2_Dense": dense}
    assert result.kv_cache is None


def test_unindexed_files_are_separate_components(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "flux-dev.safetensors"), {"w": ("BF16", [64, 64])})
    write_safetensors(str(tmp_path / "flux-schnell-00001-of-00002.safetensors"), {"a": ("BF16", [32])})
    write_safetensors(str(tmp_path / "flux-schnell-00002-of-00002.safetensors"), {"b": ("BF16", [32])})

    result = estimate(str(tmp_path))
    assert sorted(result.components) == ["flux-dev", "flux-schnell"]
    assert any("separate component" in w for w in result.warnings)

    single = estimate(str(tmp_path), file="flux-schnell-00001-of-00002.safetensors")
    assert single.weights.bytes == 128
    with pytest.raises(RuntimeError, match="No Safetensors file matching"):
        estimate(str(tmp_path), file="missing.safetensors")


def test_moe_breakdown(tmp_path: Path) -> None:
    tensors = {"model.embed_tokens.weight": ("BF16", [100, 64])}
    for layer in range(2):
        for expert in range(4):
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = ("BF16", [128, 64])
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config(num_local_experts=4, num_experts_per_tok=2))

    result = estimate(str(tmp_path))
    moe = result.moe

    assert moe is not None
    assert "EXPERTS (4 x 16.38K PARAMS, 0.00 GiB)" in table(result)
    assert (moe.experts, moe.active) == (4, 2)
    assert moe.expert.bytes == 2 * 128 * 64 * 2
    assert moe.experts_total.bytes == 4 * moe.expert.bytes
    assert moe.base.bytes == 100 * 64 * 2


def test_moe_breakdown_skipped_for_non_uniform_experts(tmp_path: Path) -> None:
    tensors = {
        "model.layers.0.mlp.experts.0.w.weight": ("BF16", [128, 64]),
        "model.layers.0.mlp.experts.1.w.weight": ("BF16", [64, 64]),
    }
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path))

    assert result.moe is None
    assert any("not uniform" in w for w in result.warnings)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_mxfp4_parameters_include_both_packed_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, pooled: bool
) -> None:
    quantization = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized"}
    config = llama_config(num_local_experts=2, num_experts_per_tok=1, quantization_config=quantization)
    write_json(
        str(tmp_path / "config.json"),
        {"architectures": ["KimiK31ForConditionalGeneration"], "text_config": config} if nested else config,
    )
    first = write_safetensors(
        str(tmp_path / "model-00001-of-00002.safetensors"),
        {"model.embed_tokens.weight": ("BF16", [100, 64]), "model.buffer": ("U8", [8])}
        | {f"model.layers.0.mlp.experts.{e}.w1.weight_packed": ("U8", [32, 16]) for e in range(2)},
    )
    second = write_safetensors(
        str(tmp_path / "model-00002-of-00002.safetensors"),
        {f"model.layers.0.mlp.experts.{e}.w1.weight_scale": ("U8", [32, 1]) for e in range(2)},
    )
    if pooled:
        monkeypatch.setattr(_safetensors, "POOL_MIN_FILES", 1)

    result = estimate(str(tmp_path))

    assert result.weights.bytes == first + second == 13896
    assert result.weights.params == 8456
    assert result.weights.dtypes["MXFP4"].params == 2048
    assert result.weights.dtypes["MXFP4"].bytes == 1088
    assert result.weights.dtypes["U8"].params == result.weights.dtypes["U8"].bytes == 8
    assert some(result.moe).expert.params == 1024
    assert some(result.moe).expert.bytes == 544
    assert some(result.moe).base.params == 6408
    assert "MXFP4" in table(result)


@pytest.mark.parametrize("quantization", [{}, {"format": "pack-quantized"}])
def test_u8_tensors_are_not_assumed_to_be_mxfp4(tmp_path: Path, quantization: dict[str, str]) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {"layer.weight_packed": ("U8", [32, 16]), "layer.weight_scale": ("U8", [32, 1])},
    )
    write_json(str(tmp_path / "config.json"), llama_config(quantization_config=quantization))

    result = estimate(str(tmp_path))

    assert result.weights.params == result.weights.bytes == 544
    assert set(result.weights.dtypes) == {"U8"}


def test_gpt_oss_mxfp4_blocks_count_both_values(tmp_path: Path) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "model.embed_tokens.weight": ("BF16", [16, 32]),
            "model.layers.0.mlp.experts.gate_up_proj_blocks": ("U8", [2, 32, 1, 16]),
            "model.layers.0.mlp.experts.gate_up_proj_scales": ("U8", [2, 32, 1]),
            "model.layers.0.mlp.experts.gate_up_proj_bias": ("BF16", [2, 32]),
            "model.buffer": ("U8", [5]),
        },
    )
    write_json(str(tmp_path / "config.json"), llama_config(quantization_config={"quant_method": "mxfp4"}))

    result = estimate(str(tmp_path))

    assert result.weights.params == 2629
    assert result.weights.bytes == 2245
    assert result.weights.dtypes["MXFP4"].params == 2048
    assert result.weights.dtypes["MXFP4"].bytes == 1088
    assert result.weights.dtypes["BF16"].params == 576
    assert result.weights.dtypes["U8"].params == 5


@pytest.mark.parametrize("expert_dtype,params", [("fp4", 1024), ("fp8", 512)])
def test_deepseek_fp4_experts_keep_the_fp8_weights(tmp_path: Path, expert_dtype: str, params: int) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "layers.0.ffn.experts.0.w1.weight": ("I8", [32, 16]),
            "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [32, 1]),
            "layers.0.attn.wq.weight": ("I8", [16, 32]),
            "layers.0.attn.wq.scale": ("F8_E8M0", [1, 1]),
        },
    )
    quantization = {"quant_method": "fp8", "expert_dtype": expert_dtype}
    write_json(str(tmp_path / "config.json"), llama_config(quantization_config=quantization))

    result = estimate(str(tmp_path))

    assert result.weights.bytes == 1057
    assert result.weights.params == params + 512
    assert some(result.moe).expert.params == params
    assert some(result.moe).expert.bytes == 544
    assert some(result.moe).base.params == 512
    assert some(result.moe).base.bytes == 513


def test_fp8_scales_are_storage_not_parameters(tmp_path: Path) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "model.layers.0.attn.weight": ("F8_E4M3", [32, 64]),
            "model.layers.0.attn.weight_scale_inv": ("F32", [1, 2]),
            "model.layers.0.norm.scale": ("F32", [32]),
            "model.layers.1.engram.embed.weight": ("F8_E4M3", [8, 8]),
            "model.layers.1.engram.embed.scale": ("F8_E8M0", [1, 1]),
        },
    )
    write_json(str(tmp_path / "config.json"), llama_config(quantization_config={"quant_method": "fp8"}))

    result = estimate(str(tmp_path))

    assert result.weights.params == 2080
    assert result.weights.bytes == 2184
    assert result.weights.dtypes["F32"].params == 32
    assert result.weights.dtypes["F32"].bytes == 136
    assert some(result.offloaded).params == 64
    assert some(result.offloaded).bytes == 65


def test_pickle_weights_are_not_estimated(tmp_path: Path) -> None:
    for name, size in {"pytorch_model.bin": 1000, "optimizer.pt": 99, "README.md": 5}.items():
        with open(tmp_path / name, "wb") as f:
            f.truncate(size)

    with pytest.raises(RuntimeError, match="only has PyTorch pickle weights \\(e.g. `pytorch_model.bin`\\)"):
        estimate(str(tmp_path))


def test_no_weights(tmp_path: Path) -> None:
    write_json(str(tmp_path / "config.json"), llama_config())
    with pytest.raises(RuntimeError, match="No Safetensors, GGUF or ONNX weights"):
        estimate(str(tmp_path))


def test_local_file_reads_the_sibling_config(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(os.path.join(str(tmp_path), "model.safetensors"))

    assert result.file == "model.safetensors"
    assert result.kv_cache is not None


def test_mistral_params_json_warns(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "consolidated.safetensors"), llama_tensors())
    write_json(str(tmp_path / "params.json"), {"dim": 64, "n_layers": 4})

    result = estimate(str(tmp_path))

    assert result.kv_cache is None
    assert any("params.json" in w for w in result.warnings)


def test_max_num_batched_tokens_selects_the_vllm_profile(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config(intermediate_size=128))

    default = some(estimate(str(tmp_path)).overhead)
    small = some(estimate(str(tmp_path), max_num_batched_tokens=2048).overhead)

    assert default.activations > small.activations
    assert default.runtime > small.runtime


def test_vision_encoder_overhead(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    vision = {"hidden_size": 64, "intermediate_size": 128, "patch_size": 14, "image_size": 896}
    config = {
        "architectures": ["LlavaForConditionalGeneration"],
        "text_config": llama_config(),
        "vision_config": vision,
    }
    write_json(str(tmp_path / "config.json"), config)
    write_json(str(tmp_path / "preprocessor_config.json"), {"max_pixels": 14 * 14 * 1024})

    overhead = some(estimate(str(tmp_path)).overhead)

    assert overhead.encoder == 1024 * 2 * (4 * 64 + 2 * 128)


def test_video_models_warn_about_the_vllm_video_profile(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    vision = {"hidden_size": 64, "intermediate_size": 128, "patch_size": 14, "temporal_patch_size": 2}
    config = {
        "architectures": ["Qwen2VLForConditionalGeneration"],
        "text_config": llama_config(),
        "vision_config": vision,
    }
    write_json(str(tmp_path / "config.json"), config)

    assert any("largest video" in w for w in estimate(str(tmp_path)).warnings)


def test_index_with_missing_files_uses_the_uploaded_ones(tmp_path: Path) -> None:
    tensors = list(llama_tensors().items())
    first = write_safetensors(str(tmp_path / "model-00001-of-00003.safetensors"), dict(tensors[:5]))
    weight_map = {f"t{i}": f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)}
    write_json(str(tmp_path / "model.safetensors.index.json"), {"weight_map": weight_map})
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path))

    assert result.weights.bytes == first
    assert any("2 files that are not in the repository" in w for w in result.warnings)


def test_invalid_files_are_skipped_in_multi_file_repositories(tmp_path: Path) -> None:
    good = write_safetensors(str(tmp_path / "lora-a.safetensors"), {"w": ("BF16", [64, 64])})
    (tmp_path / "lora-b.safetensors").write_bytes(b"\x10\0\0\0\0\0\0\0not json")

    result = estimate(str(tmp_path))

    assert result.weights.bytes == good
    assert any("Skipping 1 invalid" in w for w in result.warnings)
    with pytest.raises(RuntimeError, match="not a valid Safetensors"):
        estimate(str(tmp_path), file="lora-b.safetensors")


def test_80_gib_gpus_warn_about_a100(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    assert any("A100" in w for w in estimate(str(tmp_path), gpu_memory="80").warnings)
    assert not any("A100" in w for w in estimate(str(tmp_path), gpu_memory="141").warnings)


def test_malformed_and_oversized_headers_are_skipped(tmp_path: Path) -> None:
    good = write_safetensors(str(tmp_path / "lora-a.safetensors"), {"w": ("BF16", [64, 64])})
    (tmp_path / "lora-b.safetensors").write_bytes(b'\x0b\0\0\0\0\0\0\0{"w": {}}')
    (tmp_path / "lora-c.safetensors").write_bytes(b"\xff" * 8 + b"{}")

    result = estimate(str(tmp_path))

    assert result.weights.bytes == good
    assert any("Skipping 2 invalid" in w for w in result.warnings)


def test_file_prefers_the_exact_path_over_the_basename(tmp_path: Path) -> None:
    root = write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_safetensors(str(tmp_path / "other" / "model.safetensors"), {"w": ("BF16", [64, 64])})
    write_json(str(tmp_path / "config.json"), llama_config())

    assert estimate(str(tmp_path), file="model.safetensors").weights.bytes == root
    assert estimate(str(tmp_path / "model.safetensors")).weights.bytes == root


def test_local_file_lists_only_its_folder(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_safetensors(str(tmp_path / "nested" / "deep" / "model.safetensors"), llama_tensors())
    source = LocalSource.open(str(tmp_path), recursive=False)
    assert list(source.files) == ["model.safetensors"]


def test_fit_is_not_capped_below_the_requested_length(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config(max_position_embeddings=1024))

    result = estimate(str(tmp_path), max_model_len=4096, gpu_memory="80")

    assert some(result.fit).fits and some(result.fit).max_model_len >= 4096


def test_audio_tokenizers_are_added_as_components(tmp_path: Path) -> None:
    model = write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    tokenizer = write_safetensors(
        str(tmp_path / "audio_tokenizer" / "model.safetensors"), {"w": ("BF16", [64, 64])}
    )
    write_safetensors(str(tmp_path / "variant" / "model.safetensors"), {"w": ("BF16", [64, 64])})
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path))

    assert list(result.components) == ["Transformer", "audio_tokenizer"]
    assert result.weights.bytes == model + tokenizer and result.kv_cache is not None


def test_engram_and_ngram_tables_are_offloaded_to_cpu(tmp_path: Path) -> None:
    tensors = {
        **llama_tensors(),
        "layers.1.engram.embed.weight": ("F8_E4M3", [1000, 64]),
        "layers.1.engram.embed.scale": ("F8_E4M3", [1000, 2]),
        "layers.1.engram.wkv.weight": ("BF16", [64, 64]),
        "model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight": ("BF16", [500, 64]),
    }
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path))

    offloaded = some(result.offloaded)
    assert offloaded.bytes == 1000 * 64 + 1000 * 2 + 500 * 64 * 2
    assert "BF16" in result.weights.dtypes and result.weights.bytes == write_safetensors(
        str(tmp_path / "gpu" / "model.safetensors"),
        {**llama_tensors(), "layers.1.engram.wkv.weight": ("BF16", [64, 64])},
    )
    assert any("pinned host memory" in w for w in result.warnings)
    assert result.to_dict()["offloaded"]["bytes"] == offloaded.bytes


def test_suggested_context_fits_the_actual_cache_pages(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config(max_position_embeddings=1000000))
    result = estimate(str(tmp_path), gpu_memory="2")
    length = some(result.fit).max_model_len
    assert 0 < length < 1000000
    suggested = estimate(str(tmp_path), gpu_memory="2", max_model_len=length)
    assert some(suggested.fit).fits
    assert some(suggested.fit).required <= some(suggested.fit).usable
    longer = estimate(str(tmp_path), gpu_memory="2", max_model_len=length + 1)
    assert some(longer.fit).required > some(longer.fit).usable


def test_suggested_context_for_saturated_sliding_cache(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(
        str(tmp_path / "config.json"),
        llama_config(
            max_position_embeddings=100000,
            layer_types=["sliding_attention"] * 4,
            sliding_window=128,
        ),
    )
    result = estimate(str(tmp_path), gpu_memory="2")
    assert some(result.fit).fits
    assert some(result.fit).max_model_len == 100000


@pytest.mark.parametrize("gpu_memory", ["4", "2x4", "8x4"])
def test_kimi_fit_uses_the_cache_rows_for_each_tensor_parallel_rank(tmp_path: Path, gpu_memory: str) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    config = kimi_config()
    write_json(str(tmp_path / "config.json"), config)
    result = estimate(str(tmp_path), gpu_memory=gpu_memory)
    cache = some(result.kv_cache)
    fit = some(result.fit)
    ranks = fit.gpus or 1
    expected = _config.kv_cache(config, cache.dtype, cache.max_model_len, 1, tensor_parallel_size=ranks)[0]
    assert fit.required == result.weights.bytes + expected + ranks * some(result.overhead).bytes
    assert cache.dtype == "FP8_DS_MLA"
    assert 0 < fit.max_model_len < cache.max_model_len
    suggested = estimate(str(tmp_path), gpu_memory=gpu_memory, max_model_len=fit.max_model_len)
    assert some(suggested.fit).fits
    longer = estimate(str(tmp_path), gpu_memory=gpu_memory, max_model_len=fit.max_model_len + 1)
    assert not some(longer.fit).fits
    if ranks == 8:
        assert some(longer.fit).needed == 4
        assert "DOES NOT FIT" in table(longer)
        assert "fits on 4 GPUs" in markdown(longer)


@pytest.mark.parametrize("dtype,size", [("F32", 4), ("BF16", 2), ("F16", 2)])
def test_encoder_warmup_keeps_weights_and_scales_with_tokens(tmp_path: Path, dtype: str, size: int) -> None:
    stored = write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "w": (dtype, [1000, 64]),
            "norm.weight": ("F32", [64]),
            "position_ids": ("I64", [512]),
            "norm.weight": ("F32", [64]),
        },
    )
    write_json(
        str(tmp_path / "config.json"),
        llama_config(architectures=["BertModel"], model_type="bert", torch_dtype=None),
    )
    small = estimate(str(tmp_path), max_num_batched_tokens=512)
    large = estimate(str(tmp_path), max_num_batched_tokens=2048)
    assert small.weights.bytes == large.weights.bytes == stored
    assert set(small.weights.dtypes) == {dtype, "F32", "I64"}
    assert small.kv_cache is large.kv_cache is None
    per_token = 7 * 256 + 2 * 1024
    if size == 2:
        assert some(large.overhead).activations == 4 * some(small.overhead).activations
        assert some(small.overhead).activations // size == 512 * per_token
    else:
        # F32 runs eager: a single sequence keeps the scores and the probabilities, a batch also a padding mask
        assert some(small.overhead).activations // size == 512 * per_token + 2 * 8 * 512**2
        assert some(large.overhead).activations // size == 2048 * per_token + 3 * 8 * 2 * 1024**2
    assert small.total == small.weights.bytes + some(small.overhead).bytes
    assert any("Weights are estimated as stored" in w for w in small.warnings)


def test_encoder_f32_activations_add_the_attention_tensors(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [1000, 64])})
    config = llama_config(architectures=["BertModel"], model_type="bert", torch_dtype="float32")
    write_json(str(tmp_path / "config.json"), config)
    single = estimate(str(tmp_path), max_num_batched_tokens=512)
    write_json(str(tmp_path / "config.json"), {**config, "torch_dtype": "float16"})
    half = estimate(str(tmp_path), max_num_batched_tokens=512)
    assert half.weights.to_dict() == single.weights.to_dict()
    per_token = 7 * 256 + 2 * 1024
    assert some(half.overhead).activations == 512 * per_token * 2
    assert some(single.overhead).activations == (512 * per_token + 2 * 8 * 512**2) * 4
    assert any("eager attention" in w and "half precision" in w for w in single.warnings)
    assert any("memory-efficient attention" in w for w in half.warnings)


def test_f32_modernbert_keeps_both_attention_masks_and_does_not_fit(tmp_path: Path) -> None:
    # nomic-ai/modernbert-embed-base with `--dtype float32` peaks at 4.70 GiB with a 4096 token input on an L4
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [149_014_272])})
    config = {
        "architectures": ["ModernBertModel"],
        "model_type": "modernbert",
        "hidden_size": 768,
        "intermediate_size": 1152,
        "num_attention_heads": 12,
        "num_hidden_layers": 22,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "max_position_embeddings": 8192,
        "torch_dtype": "float32",
    }
    write_json(str(tmp_path / "config.json"), config)
    options = {"max_num_batched_tokens": 4096, "max_model_len": 4096, "gpu_memory": "4"}
    full = estimate(str(tmp_path), dtype="float32", **options)
    half = estimate(str(tmp_path), dtype="float16", **options)
    assert some(full.overhead).activations == (4096 * (7 * 768 + 5 * 1152) + 4 * 12 * 4096**2) * 4
    assert not some(full.fit).fits and some(half.fit).fits
    assert any("eager attention" in w for w in full.warnings)


def test_f16_override_counts_the_loading_peak_of_the_largest_tensor(tmp_path: Path) -> None:
    # intfloat/multilingual-e5-base with `--dtype float16` peaks at 1326 MiB while loading and 820 MiB once ready
    embeddings = 250002 * 768
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {"embeddings.word_embeddings.weight": ("F32", [250002, 768]), "encoder.w": ("F32", [86_000_000])},
    )
    write_json(
        str(tmp_path / "config.json"),
        llama_config(
            architectures=["XLMRobertaModel"],
            model_type="xlm-roberta",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            max_position_embeddings=514,
            pad_token_id=1,
            torch_dtype="float32",
        ),
    )
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    cast = estimate(str(tmp_path), dtype="float16", max_num_batched_tokens=128)
    stored = estimate(str(tmp_path), max_num_batched_tokens=128)
    overhead = some(cast.overhead)
    # The embeddings load first, uploaded in F32 and cast to F16 on the device
    assert overhead.loading > 0
    assert cast.total == overhead.runtime + embeddings * (4 + 2)
    assert some(stored.overhead).loading == 0
    assert "LOADING" in table(cast)
    assert any("while loading" in w for w in cast.warnings)


def test_f32_alibi_keeps_the_bias_table(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [1000, 64])})
    write_json(
        str(tmp_path / "config.json"),
        llama_config(
            architectures=["JinaBertForMaskedLM"],
            model_type="bert",
            position_embedding_type="alibi",
            max_position_embeddings=8192,
            torch_dtype="float32",
        ),
    )
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    full = estimate(str(tmp_path), max_num_batched_tokens=128)
    half = estimate(str(tmp_path), max_num_batched_tokens=128, dtype="float16")
    assert some(full.overhead).buffers == 8 * 8192**2 * 4
    assert some(half.overhead).buffers == 0
    assert full.total == full.weights.bytes + some(full.overhead).bytes
    assert "BUFFERS" in table(full)
    assert any("ALiBi" in w for w in full.warnings)


def test_encoder_eager_warmup_uses_all_tokens_and_sequence_override(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [1000, 64])})
    write_json(str(tmp_path / "config.json"), llama_config(architectures=["MPNetModel"], model_type="mpnet"))
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    write_json(str(tmp_path / "sentence_bert_config.json"), {"max_seq_length": 384})
    original = estimate(str(tmp_path), max_num_batched_tokens=768)
    shorter = estimate(str(tmp_path), max_num_batched_tokens=768, max_model_len=128)
    bigger = estimate(str(tmp_path), max_num_batched_tokens=1536)
    assert some(shorter.overhead).activations < some(original.overhead).activations
    assert some(bigger.overhead).activations == 2 * some(original.overhead).activations
    assert any("up to 128 tokens" in w for w in shorter.warnings)
    assert original.weights.bytes == shorter.weights.bytes == bigger.weights.bytes


def test_encoder_runtime_and_replica_fit(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("BF16", [1024, 1024, 2048])})
    write_json(str(tmp_path / "config.json"), llama_config(architectures=["BertModel"], model_type="bert"))
    result = estimate(str(tmp_path), gpu_memory="2x3")
    assert (some(result.fit).gpus, some(result.fit).needed, some(result.fit).fits) == (1, None, False)
    assert any("one replica per GPU" in w for w in result.warnings)
    assert some(result.overhead).runtime > 0
    assert some(result.fit).required == result.total


def test_sentence_transformers_decoder_uses_forward_overhead(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config(hidden_act="silu", rms_norm_eps=1e-6))
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    write_json(str(tmp_path / "generation_config.json"), {})
    result = estimate(str(tmp_path), max_num_batched_tokens=512)
    assert result.kv_cache is None and some(result.overhead).activations > 0


def test_masked_language_model_counts_logits_unless_embedding_pooling(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [1000, 64])})
    write_json(
        str(tmp_path / "config.json"),
        llama_config(architectures=["BertForMaskedLM"], model_type="bert", vocab_size=30522),
    )
    masked = estimate(str(tmp_path), max_num_batched_tokens=512)
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    pooled = estimate(str(tmp_path), max_num_batched_tokens=512)
    write_json(str(tmp_path / "document_1_SpladePooling/config.json"), {})
    sparse = estimate(str(tmp_path), max_num_batched_tokens=512)
    assert (
        some(sparse.overhead).activations
        > some(masked.overhead).activations
        > some(pooled.overhead).activations
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"is_encoder_decoder": True},
        {"vision_config": {"hidden_size": 64}},
        {"audio_config": {"hidden_size": 64}},
    ],
)
def test_other_models_do_not_take_the_encoder_path(tmp_path: Path, extra: dict[str, object]) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config(architectures=["T5Model"], **extra))
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    result = estimate(str(tmp_path))
    assert result.overhead is None
    assert not any("Encoder warmup" in w for w in result.warnings)


def test_unknown_encoder_length_skips_warmup_until_overridden(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("BF16", [1000, 64])})
    write_json(
        str(tmp_path / "config.json"), llama_config(architectures=["BertModel"], max_position_embeddings=None)
    )
    result = estimate(str(tmp_path))
    assert result.overhead is None and any("sequence length is unknown" in w for w in result.warnings)
    assert estimate(str(tmp_path), max_model_len=512).overhead is not None


@pytest.mark.parametrize("extra", [{"num_experts": 8}, {"layer_types": ["mamba"] * 4}, {"kv_lora_rank": 64}])
def test_encoder_warmup_skips_unmodeled_architectures(tmp_path: Path, extra: dict[str, object]) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(
        str(tmp_path / "config.json"), llama_config(architectures=["LlamaForSequenceClassification"], **extra)
    )
    result = estimate(str(tmp_path))
    assert result.overhead is None and any("Encoder warmup estimation skipped" in w for w in result.warnings)


@pytest.mark.parametrize(
    "dtype,target,size", [("float16", "F16", 2), ("float32", "F32", 4), ("bfloat16", "BF16", 2)]
)
def test_encoder_precision_override_keeps_components_and_buffer_storage(
    tmp_path: Path, dtype: str, target: str, size: int
) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {"w": ("F32", [1000, 64]), "norm.weight": ("F32", [64]), "position_ids": ("I64", [512])},
    )
    write_safetensors(str(tmp_path / "2_Dense/model.safetensors"), {"linear.weight": ("BF16", [64, 32])})
    write_json(
        str(tmp_path / "config.json"),
        llama_config(architectures=["BertModel"], model_type="bert", torch_dtype="float32"),
    )
    write_json(str(tmp_path / "config_sentence_transformers.json"), {})
    write_json(str(tmp_path / "modules.json"), [{"path": ""}, {"path": "1_Pooling"}, {"path": "2_Dense"}])
    before = estimate(str(tmp_path), max_num_batched_tokens=768, max_model_len=384)
    result = estimate(str(tmp_path), max_num_batched_tokens=768, max_model_len=384, dtype=dtype)
    assert result.weights.params == before.weights.params
    assert result.weights.bytes == (64000 + 64 + 2048) * size + 512 * 8
    assert set(result.weights.dtypes) == {target, "I64"}
    assert result.weights.dtypes["I64"].to_dict() == before.weights.dtypes["I64"].to_dict()
    assert {name: w.bytes for name, w in result.components.items()} == {
        "0_Transformer": (64000 + 64) * size + 512 * 8,
        "2_Dense": 2048 * size,
    }
    # Two sequences of 384 tokens, which only keep the attention tensors in F32
    per_token, attention = 7 * 256 + 2 * 1024, 3 * 8 * 2 * 384**2
    assert some(before.overhead).activations == (768 * per_token + attention) * 4
    assert some(result.overhead).activations == (768 * per_token + (attention if size == 4 else 0)) * size
    assert result.total == result.weights.bytes + some(result.overhead).bytes
    assert (
        estimate(str(tmp_path), max_num_batched_tokens=768, max_model_len=384).weights.to_dict()
        == before.weights.to_dict()
    )


@pytest.mark.parametrize("model_type", ["bert", "mpnet"])
def test_encoder_dtype_override_keeps_attention_and_warmup_shape(tmp_path: Path, model_type: str) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), {"w": ("F32", [1000, 64])})
    write_json(
        str(tmp_path / "config.json"),
        llama_config(architectures=["BertModel"], model_type=model_type, torch_dtype="float32"),
    )
    half = estimate(str(tmp_path), max_num_batched_tokens=768, max_model_len=384, dtype="float16")
    full = estimate(str(tmp_path), max_num_batched_tokens=768, max_model_len=384, dtype="float32")
    # MPNet always runs eager, while BERT only materializes the attention scores in F32
    if model_type == "mpnet":
        assert some(full.overhead).activations == 2 * some(half.overhead).activations
    else:
        assert some(full.overhead).activations > 2 * some(half.overhead).activations
    assert some(full.overhead).runtime == some(half.overhead).runtime
    assert ("eager attention" in " ".join(half.warnings)) == (model_type == "mpnet")
    assert "eager attention" in " ".join(full.warnings)


@pytest.mark.parametrize(
    "config",
    [
        llama_config(),
        llama_config(architectures=["BertModel"], quantization_config={"quant_method": "fp8"}),
        {
            "architectures": ["LlavaForConditionalGeneration"],
            "text_config": llama_config(),
            "vision_config": {"hidden_size": 64},
        },
    ],
)
def test_dtype_override_rejects_other_workloads_without_changing_defaults(
    tmp_path: Path, config: dict[str, object]
) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), config)
    before = estimate(str(tmp_path)).to_dict()
    with pytest.raises(RuntimeError, match="only supports unquantized Safetensors text encoders"):
        estimate(str(tmp_path), dtype="float16")
    assert estimate(str(tmp_path)).to_dict() == before


def test_precision_cast_preserves_non_parameter_bytes_without_mutating_weights() -> None:
    weights = Weights()
    weights.add("F32", 0, 4096)
    weights.add("BF16", 1024, 2048)
    weights.add("U8", 512, 512)
    before = weights.to_dict()
    result = weights.cast("float16")
    assert result.bytes == weights.bytes and result.params == weights.params
    assert result.dtypes["F32"].bytes == 4096 and result.dtypes["F32"].params == 0
    assert result.dtypes["F16"].bytes == 2048 and result.dtypes["U8"].bytes == 512
    assert weights.to_dict() == before
