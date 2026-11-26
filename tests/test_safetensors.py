import json
import os
import struct
from pathlib import Path
from typing import Any

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


def test_repeated_layers_expand_cache_without_duplicating_stored_weights(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    config = llama_config(layer_types=["full_attention"] * 4)
    write_json(str(tmp_path / "config.json"), config)
    single = estimate(str(tmp_path))
    config["loop_config"] = {"core_start": 1, "core_end": 3, "num_passes": 3}
    write_json(str(tmp_path / "config.json"), config)

    repeated = estimate(str(tmp_path))

    assert repeated.weights == single.weights
    assert some(repeated.kv_cache).bytes == 2 * some(single.kv_cache).bytes
    assert some(repeated.kv_cache).layers == {"full": 8, "sliding": 0, "state": 0, "none": 0}
    assert some(repeated.recommended_gpu_memory) > some(single.recommended_gpu_memory)


def test_inkling2_config_estimates_the_decoder_and_local_layers(tmp_path: Path) -> None:
    # https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/main/config.json
    config = json.loads((Path(__file__).parent / "fixtures" / "inkling2.json").read_text())
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), config)

    result = estimate(str(tmp_path), max_num_batched_tokens=8192, engine="vllm")

    cache = some(result.kv_cache)
    assert cache.max_model_len == 1048576 and cache.dtype == "BF16"
    assert cache.layers == {"full": 8, "sliding": 40, "state": 0, "none": 0}
    window_tokens = (-(-(1024 - 1 + 2 * 8192) // 16) + 1) * 16
    attention = (8 * 1048576 + 40 * window_tokens) * 2 * 8 * 128 * 2
    assert cache.bytes == attention + 48 * 3 * (4 * 8 * 128 + 2 * 4608) * 2
    assert not any("KV cache estimation skipped" in warning for warning in result.warnings)

    override = some(estimate(str(tmp_path), max_model_len=128, batch_size=2, engine="vllm").kv_cache)
    assert override.bytes == 48 * 128 * 2 * 8 * 128 * 2 * 2 + 48 * 3 * (4 * 8 * 128 + 2 * 4608) * 2 * 2


def test_kv_cache_options(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    result = estimate(str(tmp_path), max_model_len=100, batch_size=3, kv_cache_dtype="fp8")

    assert some(result.kv_cache).bytes == 2 * 4 * 2 * 32 * 112 * 3
    assert some(result.kv_cache).dtype == "F8_E4M3"
    with pytest.raises(RuntimeError, match="not a vLLM KV cache dtype"):
        estimate(str(tmp_path), kv_cache_dtype="Q4_K")


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("compressed", [False, True])
def test_compressed_cache_fit_warns_about_runtime_reservations(
    tmp_path: Path, engine: str, compressed: bool
) -> None:
    config = llama_config()
    if compressed:
        config.update(compress_ratios=[4, 128, 4, 128], index_head_dim=32)
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), config)

    result = estimate(str(tmp_path), engine=engine, gpu_memory="80", gpu_type="nvidia-h100")

    assert some(result.fit).fits
    assert any("page pools and workspace reservations" in warning for warning in result.warnings) == compressed


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_compressed_attention_projections_are_replicated_across_ranks(tmp_path: Path, engine: str) -> None:
    tensors = {
        "model.layers.0.self_attn.q_a_proj.weight": ("BF16", [32, 64]),
        "model.layers.0.self_attn.compressor.wkv.weight": ("BF16", [64, 64]),
        "model.layers.0.self_attn.indexer.weights_proj.weight": ("BF16", [8, 64]),
        "model.layers.0.mlp.gate.weight": ("BF16", [4, 64]),
        "model.layers.0.input_layernorm.weight": ("BF16", [64]),
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("BF16", [128, 64]),
    }
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(
        str(tmp_path / "config.json"),
        llama_config(compress_ratios=[4] * 4, head_dim=32, hidden_size=64),
    )
    single = estimate(str(tmp_path), engine=engine, gpu_memory="1x80")
    parallel = estimate(str(tmp_path), engine=engine, gpu_memory="4x80")

    assert single.weights == parallel.weights
    assert some(single.overhead).weight_adjustment == 0
    assert some(parallel.overhead).weight_adjustment == 3 * 2 * (32 * 64 + 64 * 64 + 8 * 64 + 4 * 64 + 64)


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


@pytest.mark.parametrize("active_key", ["num_experts_per_tok", "num_experts_per_token", "top_k"])
@pytest.mark.parametrize("active, expected", [(0, None), (True, None), (1, 1), (2, 2)])
def test_moe_breakdown(tmp_path: Path, active_key: str, active: int, expected: int | None) -> None:
    tensors = {"model.embed_tokens.weight": ("BF16", [100, 64])}
    for layer in range(2):
        for expert in range(4):
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = ("BF16", [128, 64])
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    config = llama_config(num_local_experts=4)
    config[active_key] = active
    write_json(str(tmp_path / "config.json"), config)

    result = estimate(str(tmp_path))
    moe = result.moe

    assert moe is not None
    assert "EXPERTS (4 x 16.38K PARAMS, 0.00 GiB)" in table(result)
    assert (moe.experts, moe.active) == (4, expected)
    assert result.to_dict()["moe"]["active"] == expected
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
@pytest.mark.parametrize("prefix", ["experts", "local_experts", "routed_experts"])
def test_fused_experts_match_individually_stored_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, pooled: bool, prefix: str
) -> None:
    config = llama_config(n_routed_experts=4, num_experts_per_tok=2)
    if nested:
        config = {"architectures": ["LlamaForConditionalGeneration"], "text_config": config}
    tensors: dict[str, tuple[str, list[int]]] = {"model.embed_tokens.weight": ("BF16", [100, 64])}
    separate = dict(tensors)
    for layer in range(2):
        tensors[f"model.layers.{layer}.mlp.{prefix}.gate_up_proj"] = ("BF16", [4, 128, 64])
        tensors[f"model.layers.{layer}.mlp.{prefix}.down_proj"] = ("BF16", [4, 64, 64])
        tensors[f"model.layers.{layer}.mlp.{prefix}.bias"] = ("BF16", [4, 64])
        for expert in range(4):
            separate[f"model.layers.{layer}.mlp.experts.{expert}.gate_up_proj.weight"] = ("BF16", [128, 64])
            separate[f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"] = ("BF16", [64, 64])
            separate[f"model.layers.{layer}.mlp.experts.{expert}.bias"] = ("BF16", [64])
    write_json(str(tmp_path / "fused/config.json"), config)
    write_json(str(tmp_path / "separate/config.json"), config)
    write_safetensors(str(tmp_path / "fused/model.safetensors"), tensors)
    write_safetensors(str(tmp_path / "separate/model.safetensors"), separate)
    if pooled:
        monkeypatch.setattr(_safetensors, "POOL_MIN_FILES", 1)

    fused = estimate(str(tmp_path / "fused"))
    indexed = estimate(str(tmp_path / "separate"))

    assert fused.weights == indexed.weights
    assert fused.moe == indexed.moe
    moe = some(fused.moe)
    assert (moe.experts, moe.active) == (4, 2)
    assert moe.base.bytes == 100 * 64 * 2
    assert moe.expert.params == 2 * (128 * 64 + 64 * 64 + 64)
    assert "EXPERTS (4 x" in table(fused)


@pytest.mark.parametrize("count", [None, 3])
def test_fused_experts_require_a_matching_configured_count(tmp_path: Path, count: int | None) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"), {"model.layers.0.mlp.experts.gate_up_proj": ("BF16", [4, 32, 16])}
    )
    write_json(str(tmp_path / "config.json"), llama_config(n_routed_experts=count))

    result = estimate(str(tmp_path))

    assert result.moe is None
    assert result.weights.params == 4 * 32 * 16
    assert result.weights.bytes == 4 * 32 * 16 * 2


def test_fused_u8_experts_without_quantization_metadata_are_not_assumed_to_be_fp4(tmp_path: Path) -> None:
    nbytes = write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "model.layers.0.mlp.experts.w13_weight": ("U8", [4, 64, 16]),
            "model.layers.0.mlp.experts.w13_weight_scale": ("F8_E4M3", [4, 64, 2]),
        },
    )
    write_json(str(tmp_path / "config.json"), llama_config(n_routed_experts=4))

    result = estimate(str(tmp_path))

    assert result.weights.bytes == nbytes
    assert result.weights.params == 4 * 64 * 16 + 4 * 64 * 2
    assert set(result.weights.dtypes) == {"U8", "F8_E4M3"}
    assert some(result.moe).experts == 4


@pytest.mark.parametrize(
    "declared", [None, "invalid", "[]", '{"quant_method":"modelopt","quant_algo":"NVFP4"}']
)
def test_quantization_declared_in_the_safetensors_header(tmp_path: Path, declared: str | None) -> None:
    path = tmp_path / "model.safetensors"
    nbytes = write_safetensors(
        str(path),
        {
            "model.layers.0.mlp.experts.w13_weight": ("U8", [4, 64, 16]),
            "model.layers.0.mlp.experts.w13_weight_scale": ("F8_E4M3", [4, 64, 2]),
        },
    )
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : size + 8])
    if declared is not None:
        header["__metadata__"]["quantization_config"] = declared
    metadata = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(metadata)) + metadata + raw[8 + size :])
    write_json(str(tmp_path / "config.json"), llama_config(n_routed_experts=4))

    result = estimate(str(tmp_path))

    assert result.weights.bytes == nbytes
    assert some(result.moe).experts == 4
    if declared and "NVFP4" in declared:
        assert result.weights.params == 2 * 4 * 64 * 16
        assert result.weights.dtypes["NVFP4"].params == 2 * 4 * 64 * 16
    else:
        assert result.weights.params == 4 * 64 * 16 + 4 * 64 * 2
        assert set(result.weights.dtypes) == {"U8", "F8_E4M3"}


@pytest.mark.parametrize("standalone", [False, True])
def test_mixed_precision_parameter_groups_preserve_bytes_and_count_packed_values(
    tmp_path: Path, standalone: bool
) -> None:
    packed, floating = "model.layers.0.mlp.experts.w13_weight", "model.layers.1.mlp.experts.w13_weight"
    nbytes = write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            packed: ("U8", [4, 64, 16]),
            packed + "_scale": ("F8_E4M3", [4, 64, 2]),
            packed + "_amax": ("F32", [4]),
            floating: ("F8_E4M3", [4, 64, 32]),
            floating + "_scale": ("F8_E4M3", [4, 2, 1]),
            floating + "_outer_scale_denominator": ("F32", [1]),
            "model.norm.scale": ("BF16", [64]),
            "model.buffer": ("U8", [3]),
        },
    )
    quantization = {
        "model": {
            "parameter_groups": [
                {"compute_dtype": "nvfp4", "output_dtype": "uint8", "tensors": [packed]},
                {"compute_dtype": "mxfp8", "output_dtype": "float8_e4m3fn", "tensors": [floating]},
                {"compute_dtype": "bfloat16", "tensors": ["model.norm.scale"]},
            ]
        }
    }
    config = llama_config(n_routed_experts=4)
    if standalone:
        write_json(str(tmp_path / "hf_quant_config.json"), quantization)
    else:
        config["quantization_config"] = quantization
    write_json(str(tmp_path / "config.json"), config)

    result = estimate(str(tmp_path))

    assert result.weights.bytes == nbytes
    assert result.weights.params == 2 * 4 * 64 * 16 + 4 * 64 * 32 + 64 + 3
    assert result.weights.dtypes["F8_E4M3"].params == 4 * 64 * 32
    assert result.weights.dtypes["F32"].params == 0
    assert some(result.moe).experts == 4
    assert some(result.moe).expert.params == 2 * 64 * 16 + 64 * 32


@pytest.mark.parametrize("selectors", ["exact", "regex"])
@pytest.mark.parametrize("nested", [False, True])
def test_compressed_tensor_groups_count_mixed_packed_weights(
    tmp_path: Path, selectors: str, nested: bool
) -> None:
    expert = "model.layers.0.mlp.experts.0.up_proj"
    dense = "model.layers.0.self_attn.q_proj"
    tensors = {
        expert + ".weight_packed": ("U8", [32, 16]),
        expert + ".weight_scale": ("F8_E4M3", [32, 2]),
        expert + ".weight_global_scale": ("F32", [1]),
        expert + ".input_global_scale": ("F32", [1]),
        dense + ".weight": ("F8_E4M3", [32, 32]),
        dense + ".weight_scale": ("U8", [32, 1]),
        "model.buffer": ("U8", [7]),
    }
    nbytes = write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    quantization = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "packed": {
                "format": "nvfp4-pack-quantized",
                "targets": [expert] if selectors == "exact" else [r"re:.*mlp\.experts\.\d+\."],
            },
            "floating": {
                "format": "mxfp8-quantized",
                "targets": [dense] if selectors == "exact" else [r"re:.*self_attn\."],
            },
        },
    }
    config = llama_config(n_routed_experts=1, quantization_config=quantization)
    if nested:
        config = {"architectures": ["LlamaForConditionalGeneration"], "text_config": config}
    write_json(str(tmp_path / "config.json"), config)

    result = estimate(str(tmp_path))

    assert result.weights.bytes == nbytes
    assert result.weights.params == 2 * 32 * 16 + 32 * 32 + 7
    assert result.weights.dtypes["NVFP4"].params == 1024
    assert result.weights.dtypes["F8_E4M3"].params == 1024
    assert result.weights.dtypes["F32"].params == 0
    assert result.weights.dtypes["U8"].params == 7
    assert some(result.moe).expert.params == 1024
    assert some(result.moe).expert.bytes == 32 * 16 + 32 * 2 + 8


@pytest.mark.parametrize("encoding", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("target", ["Linear", "model.proj", "re:model\\.proj$"])
@pytest.mark.parametrize("ignore", ["model.skip", "re:model\\.skip$"])
def test_compressed_tensor_group_targets_and_ignores(
    tmp_path: Path, encoding: str, target: str, ignore: str
) -> None:
    tensors = {}
    for module in ["model.proj", "model.skip", "model.norm"]:
        tensors[module + ".weight_packed"] = ("U8", [32, 16])
        tensors[module + ".weight_scale"] = ("U8", [32, 1])
    nbytes = write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    quantization = {
        "quant_method": "compressed-tensors",
        "format": encoding + "-pack-quantized",
        "ignore": [ignore],
        "config_groups": {"group": {"targets": [target]}},
    }
    write_json(str(tmp_path / "config.json"), llama_config(quantization_config=quantization))

    result = estimate(str(tmp_path))

    assert result.weights.bytes == nbytes == 1632
    assert result.weights.params == 1024 + 2 * 544
    assert result.weights.dtypes[encoding.upper()].params == 1024
    assert result.weights.dtypes["U8"].params == 2 * 544


@pytest.mark.parametrize("contents", ["invalid", "[]"])
def test_invalid_standalone_quantization_keeps_the_stored_weights(tmp_path: Path, contents: str) -> None:
    nbytes = write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    (tmp_path / "hf_quant_config.json").write_text(contents)

    result = estimate(str(tmp_path))

    assert result.weights.bytes == nbytes
    assert any("hf_quant_config.json" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "quantization", [{"quant_method": "modelopt", "quant_algo": "NVFP4"}, {"quant_method": "nvfp4"}]
)
@pytest.mark.parametrize("nested", [False, True])
def test_nvfp4_fused_experts_count_weights_and_preserve_scale_storage(
    tmp_path: Path, quantization: dict[str, str], nested: bool
) -> None:
    nbytes = write_safetensors(
        str(tmp_path / "model-00001-of-00002.safetensors"),
        {
            "model.embed_tokens.weight": ("BF16", [100, 64]),
            "model.layers.0.mlp.gate.weight": ("BF16", [4, 64]),
            "model.layers.0.mlp.experts.w13_weight": ("U8", [4, 64, 16]),
            "model.layers.0.mlp.experts.w2_weight": ("U8", [4, 32, 16]),
            "model.layers.0.mlp.shared_experts.w13_weight": ("U8", [64, 16]),
            "model.layers.0.attn.weight": ("F8_E4M3", [32, 32]),
            "model.buffer": ("U8", [7]),
        },
    )
    nbytes += write_safetensors(
        str(tmp_path / "model-00002-of-00002.safetensors"),
        {
            "model.layers.0.mlp.experts.w13_weight_scale": ("F8_E4M3", [4, 64, 2]),
            "model.layers.0.mlp.experts.w2_weight_scale": ("F8_E4M3", [4, 32, 2]),
            "model.layers.0.mlp.experts.w13_weight_amax": ("F32", [4]),
            "model.layers.0.mlp.experts.w13_weight_outer_scale_denominator": ("F32", [1]),
            "model.layers.0.attn.weight_scale_inv": ("F32", [1, 1]),
            "model.layers.0.attn.q_l2_clip.scale": ("BF16", [32]),
        },
    )
    config = llama_config(n_routed_experts=4, num_experts_per_tok=1, quantization_config=quantization)
    if nested:
        config = {"architectures": ["LlamaForConditionalGeneration"], "text_config": config}
    write_json(str(tmp_path / "config.json"), config)

    result = estimate(str(tmp_path))
    moe = some(result.moe)

    assert result.weights.bytes == nbytes
    assert (
        result.weights.params
        == 100 * 64 + 4 * 64 + 2 * (4 * 64 * 16 + 4 * 32 * 16 + 64 * 16) + 32 * 32 + 7 + 32
    )
    assert result.weights.dtypes["NVFP4"].params == 2 * (4 * 64 * 16 + 4 * 32 * 16 + 64 * 16)
    assert result.weights.dtypes["F8_E4M3"].params == 32 * 32
    assert result.weights.dtypes["F32"].params == 0
    assert result.weights.dtypes["U8"].params == 7
    assert (moe.experts, moe.active) == (4, 1)
    assert moe.expert.params == 2 * (64 * 16 + 32 * 16)
    assert moe.expert.bytes == 64 * 16 + 32 * 16 + 64 * 2 + 32 * 2 + 4
    assert moe.experts_total.bytes + moe.base.bytes == result.weights.bytes
    assert moe.experts_total.params + moe.base.params == result.weights.params


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


@pytest.mark.parametrize(
    "quantization",
    [{}, {"format": "pack-quantized"}, {"config_groups": {"group": {"targets": ["Linear"]}}}],
)
def test_u8_tensors_are_not_assumed_to_be_mxfp4(tmp_path: Path, quantization: dict[str, Any]) -> None:
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
@pytest.mark.parametrize("field", ["config", "quantization_config"])
@pytest.mark.parametrize("nested", [False, True])
def test_fp4_expert_precision_keeps_the_fp8_weights(
    tmp_path: Path, expert_dtype: str, params: int, field: str, nested: bool
) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "layers.0.ffn.experts.0.w1.weight": ("I8", [32, 16]),
            "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [32, 1]),
            "layers.0.attn.wq.weight": ("I8", [16, 32]),
            "layers.0.attn.wq.scale": ("F8_E8M0", [1, 1]),
        },
    )
    config = llama_config(quantization_config={"quant_method": "fp8"})
    (config if field == "config" else config["quantization_config"])["expert_dtype"] = expert_dtype
    if nested:
        config = {"architectures": ["LlamaForConditionalGeneration"], "text_config": config}
    write_json(str(tmp_path / "config.json"), config)

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


def test_block_fp8_fused_experts_keep_unsigned_scales_out_of_parameter_counts(tmp_path: Path) -> None:
    nbytes = write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "model.layers.0.mlp.experts.w13_weight": ("F8_E4M3", [4, 64, 32]),
            "model.layers.0.mlp.experts.w13_weight_scale": ("U8", [4, 64, 1]),
            "model.layers.0.attn.q_l2_clip.scale": ("BF16", [32]),
            "model.buffer": ("U8", [3]),
        },
    )
    write_json(
        str(tmp_path / "config.json"),
        llama_config(n_routed_experts=4, quantization_config={"quant_method": "fp8", "scale_fmt": "ue8m0"}),
    )

    result = estimate(str(tmp_path))

    assert result.weights.bytes == nbytes
    assert result.weights.params == 4 * 64 * 32 + 32 + 3
    assert result.weights.dtypes["U8"].params == 3
    assert result.weights.dtypes["U8"].bytes == 4 * 64 + 3
    assert some(result.moe).expert.params == 64 * 32
    assert some(result.moe).expert.bytes == 64 * 32 + 64


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

    large = some(estimate(str(tmp_path), max_num_batched_tokens=8192).overhead)
    assert default == small
    assert large.activations > small.activations
    assert large.bytes > small.bytes
    assert small.runtime > small.context


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

    assert overhead.encoder == 1024 * 3 * (4 * 64 + 2 * 128)


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
    result = estimate(str(tmp_path), gpu_memory="1.5")
    length = some(result.fit).max_model_len
    assert 0 < length < 1000000
    suggested = estimate(str(tmp_path), gpu_memory="1.5", max_model_len=length)
    assert some(suggested.fit).fits
    assert some(suggested.fit).required <= some(suggested.fit).usable
    longer = estimate(str(tmp_path), gpu_memory="1.5", max_model_len=length + 1)
    longer_fit = some(longer.fit)
    assert longer_fit.startup_required is not None
    assert longer_fit.startup_required > longer_fit.usable


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


@pytest.mark.parametrize("gpu_memory", ["4", "2x3", "8x3"])
def test_kimi_fit_uses_the_cache_rows_for_each_tensor_parallel_rank(tmp_path: Path, gpu_memory: str) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    config = kimi_config()
    write_json(str(tmp_path / "config.json"), config)
    result = estimate(str(tmp_path), gpu_memory=gpu_memory)
    cache = some(result.kv_cache)
    fit = some(result.fit)
    ranks = fit.gpus or 1
    expected = _config.kv_cache(config, cache.dtype, cache.max_model_len, 1, tensor_parallel_size=ranks)[0]
    assert fit.required == result.weights.bytes + expected + ranks * some(fit.overhead_per_gpu)
    assert cache.dtype == "FP8_DS_MLA"
    assert 0 < fit.max_model_len < cache.max_model_len
    suggested = estimate(str(tmp_path), gpu_memory=gpu_memory, max_model_len=fit.max_model_len)
    assert some(suggested.fit).fits
    longer = estimate(str(tmp_path), gpu_memory=gpu_memory, max_model_len=fit.max_model_len + 1)
    assert not some(longer.fit).fits
    if ranks == 8:
        needed = some(longer.fit).needed
        assert needed is not None and needed < ranks
        assert "DOES NOT FIT" in table(longer)
        assert f"fits on {needed} GPU" in markdown(longer)


def test_sm90_packing_preserves_nvfp4_scales_and_expands_mxfp8_scales() -> None:
    from hf_mem._safetensors import inkling_weight_adjustment

    def tensor(dtype: str, shape: list[int], size: int) -> dict[str, Any]:
        return {"dtype": dtype, "shape": shape, "data_offsets": [0, size]}

    tensors = {
        "model.llm.layers.0.mlp.experts.w13_weight": tensor("U8", [4, 32, 16], 2048),
        "model.llm.layers.0.mlp.experts.w13_weight_scale": tensor("F8_E4M3", [4, 32, 2], 256),
        "model.llm.layers.1.mlp.experts.w2_weight": tensor("F8_E4M3", [4, 32, 64], 8192),
        "model.llm.layers.1.mlp.experts.w2_weight_scale": tensor("U8", [4, 32, 2], 256),
        "model.visual.weight": tensor("BF16", [8, 8], 128),
        "model.llm.layers.0.mlp.gate.weight": tensor("BF16", [4, 8], 64),
    }
    assert inkling_weight_adjustment(tensors, 4, ("model.visual.",)) == 256 + 3 * (128 + 64)
    assert inkling_weight_adjustment(tensors, 4, ("model.visual.",), marlin=False) == 3 * (128 + 64)
    del tensors["model.llm.layers.1.mlp.experts.w2_weight"]
    assert inkling_weight_adjustment(tensors, 4, ("model.visual.",)) == 3 * (128 + 64)


@pytest.mark.parametrize("dtype,size", [("F32", 4), ("BF16", 2), ("F16", 2)])
def test_encoder_warmup_keeps_weights_and_scales_with_tokens(tmp_path: Path, dtype: str, size: int) -> None:
    stored = write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "w": (dtype, [1000, 64]),
            "norm.weight": ("F32", [64]),
            "position_ids": ("I64", [512]),
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
        assert some(small.overhead).activations // size == 512 * per_token + 3 * 8 * 512**2
        assert some(large.overhead).activations // size == 2048 * per_token + 4 * 8 * 2 * 1024**2
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
    assert some(single.overhead).activations == (512 * per_token + 3 * 8 * 512**2) * 4
    assert any("eager attention" in w and "half precision" in w for w in single.warnings)
    assert any("memory-efficient attention" in w for w in half.warnings)


def test_f32_modernbert_keeps_both_attention_masks_and_does_not_fit(tmp_path: Path) -> None:
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
    options = {"max_num_batched_tokens": 4096, "max_model_len": 4096, "gpu_memory": "4.6"}
    full = estimate(str(tmp_path), dtype="float32", **options)
    half = estimate(str(tmp_path), dtype="float16", **options)
    assert some(full.overhead).activations == (4096 * (7 * 768 + 5 * 1152) + 5 * 12 * 4096**2) * 4
    assert not some(full.fit).fits and some(half.fit).fits
    assert any("eager attention" in w for w in full.warnings)


def test_f16_override_counts_the_loading_peak_of_the_largest_tensor(tmp_path: Path) -> None:
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
    assert overhead.loading > 0
    assert "loading" in markdown(cast)
    assert cast.total == overhead.runtime + cast.weights.bytes + embeddings * 4
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
    assert "buffers" in markdown(full)
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
    per_token, attention = 7 * 256 + 2 * 1024, 4 * 8 * 2 * 384**2
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


def test_fp32_decoder_reference_casts_weights_and_preserves_integer_buffers(tmp_path: Path) -> None:
    stored = write_safetensors(
        str(tmp_path / "model.safetensors"),
        {"model.embed_tokens.weight": ("F32", [100, 64]), "position_ids": ("I64", [3])},
    )
    write_json(str(tmp_path / "config.json"), llama_config(torch_dtype="float32"))
    result = estimate(str(tmp_path))
    assert result.weights.bytes == (stored - 24) // 2 + 24
    assert set(result.weights.dtypes) == {"F16", "I64"}
    assert some(result.kv_cache).dtype == "F16"
    assert any("16-bit CUDA serving reference" in warning for warning in result.warnings)


def test_fp32_expert_storage_stays_consistent_with_the_decoder_reference(tmp_path: Path) -> None:
    tensors = {"model.embed_tokens.weight": ("F32", [100, 64])}
    for index in range(4):
        tensors[f"model.layers.0.mlp.experts.{index}.up_proj.weight"] = ("F32", [128, 64])
    stored = write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(
        str(tmp_path / "config.json"),
        llama_config(torch_dtype="float32", num_local_experts=4, num_experts_per_tok=1),
    )
    result = estimate(str(tmp_path))
    moe = some(result.moe)
    assert result.weights.bytes == stored // 2
    assert moe.base.bytes + moe.experts_total.bytes == result.weights.bytes
    assert moe.active == 1
