from pathlib import Path
from typing import Any

import pytest
from helpers import estimate, llama_config, llama_tensors, write_json, write_safetensors

from hf_mem import _overhead
from hf_mem._render import markdown, table

GIB = 1 << 30


@pytest.mark.parametrize("engine,static", [("vllm", None), ("sglang", None), ("sglang", 0.858)])
@pytest.mark.parametrize("fused", [False, True])
def test_fused_packed_weight_allocations_follow_the_allocator_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str, static: float | None, fused: bool
) -> None:
    tensors = llama_tensors()
    for layer in range(2):
        names = (
            {"gate_up_proj": 5760, "down_proj": 2880}
            if fused
            else {"gate_proj": 2880, "up_proj": 2880, "down_proj": 2880}
        )
        for name, width in names.items():
            prefix = f"model.layers.{layer}.mlp.experts.{name}"
            tensors[prefix + "_blocks"] = ("U8", [2, width, 90, 16])
            tensors[prefix + "_scales"] = ("U8", [2, width, 90])
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(
        str(tmp_path / "config.json"),
        llama_config(
            num_hidden_layers=2,
            hidden_size=2880,
            intermediate_size=2880,
            num_local_experts=2,
            quantization_config={"quant_method": "mxfp4"},
        ),
    )
    arguments = dict(
        engine=engine,
        gpu_memory="95",
        gpu_type="nvidia-rtx-pro-6000",
        mem_fraction_static=static,
    )
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    native = estimate(str(tmp_path), **arguments)
    monkeypatch.setenv(
        "PYTORCH_ALLOC_CONF" if static is None else "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    expandable = estimate(str(tmp_path), **arguments)

    assert native.weights == expandable.weights
    assert native.kv_cache == expandable.kv_cache
    assert native.recommended_gpu_memory is not None and expandable.recommended_gpu_memory is not None
    if engine == "sglang" and fused:
        assert native.total > expandable.total
        assert native.recommended_gpu_memory > expandable.recommended_gpu_memory
    else:
        assert native.total == expandable.total
        assert native.recommended_gpu_memory == expandable.recommended_gpu_memory


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "quantization,ranks",
    [({"quant_method": "fp8", "weight_block_size": [128, 128]}, 8), ({"quant_method": "mxfp4"}, 4)],
)
def test_quantized_partition_warning_preserves_byte_fit(
    tmp_path: Path, engine: str, quantization: dict[str, Any], ranks: int
) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    config = llama_config(intermediate_size=1280, quantization_config=quantization)
    write_json(str(tmp_path / "config.json"), config)

    result = estimate(str(tmp_path), engine=engine, gpu_memory=f"{ranks}x24", gpu_type="nvidia-l4")

    assert result.fit is not None and result.fit.fits
    assert any(f"TP={ranks}" in warning and "128-value blocks" in warning for warning in result.warnings)
    config["intermediate_size"] = 2048
    write_json(str(tmp_path / "config.json"), config)
    aligned = estimate(str(tmp_path), engine=engine, gpu_memory=f"{ranks}x24", gpu_type="nvidia-l4")
    assert not any("supports this partition" in warning for warning in aligned.warnings)


def test_capacity_only_profiles_are_optional_and_identified(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())

    generic = estimate(str(tmp_path))
    capacity = estimate(str(tmp_path), gpu_memory="95")
    hardware = estimate(str(tmp_path), gpu_memory="95", gpu_type="nvidia-rtx-pro-6000")

    assert not any("inferred from memory capacity" in warning for warning in generic.warnings)
    assert any("inferred from memory capacity" in warning for warning in capacity.warnings)
    assert not any("inferred from memory capacity" in warning for warning in hardware.warnings)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("gpu_memory", [None, "2x16", "8x16"])
def test_hybrid_gpu_fit_with_explicit_layout_and_nested_state_dimensions(
    tmp_path: Path, engine: str, gpu_memory: str | None
) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    config = llama_config(
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        layer_types=["linear_attention", "deepseek_sparse_attention"] * 2,
        linear_attn_config={
            "full_attn_layers": [1, 3],
            "kda_layers": [0, 2],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        },
    )
    canonical = {
        **config,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
    }
    arguments = dict(
        engine=engine,
        gpu_memory=gpu_memory,
        max_model_len=32768,
        max_num_batched_tokens=128,
        max_num_seqs=4,
    )
    write_json(
        str(tmp_path / "config.json"),
        {"architectures": ["LlamaForConditionalGeneration"], "text_config": canonical},
    )
    expected = estimate(str(tmp_path), **arguments)
    write_json(
        str(tmp_path / "config.json"),
        {"architectures": ["LlamaForConditionalGeneration"], "text_config": config},
    )

    actual = estimate(str(tmp_path), **arguments)

    assert actual.weights == expected.weights
    assert actual.kv_cache == expected.kv_cache
    assert actual.fit == expected.fit
    assert actual.recommended_gpu_memory == expected.recommended_gpu_memory
    assert actual.kv_cache is not None and actual.kv_cache.layers["state"] == 2


@pytest.mark.parametrize("ranks", [1, 2, 8])
@pytest.mark.parametrize("utilization", [0.5, 0.9, 1.0])
@pytest.mark.parametrize("context", [0, GIB // 2, 4 * GIB])
def test_recommendation_is_the_capacity_threshold(ranks: int, utilization: float, context: int) -> None:
    weights = 12 * GIB
    allocation = weights + ranks * context
    required = _overhead.recommended_memory(allocation, utilization)
    args: dict[str, Any] = dict(
        weights=weights,
        kv_fixed=0,
        kv_per_token=0,
        max_model_len=0,
        context_length=0,
        overhead=context,
        utilization=utilization,
    )
    above = _overhead.fit(gpu_memory=f"{ranks}x{(required / ranks + 4) / GIB}", **args)
    below = _overhead.fit(gpu_memory=f"{ranks}x{(required / ranks - 4) / GIB}", **args)
    assert above.fits
    assert not below.fits


def test_model_only_recommendation_includes_requested_headroom(tmp_path: Path) -> None:
    tensors = llama_tensors()
    tensors["model.embed_tokens.weight"] = ("BF16", [262144, 32768])
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config())
    regular = estimate(str(tmp_path))
    reserved = estimate(str(tmp_path), gpu_memory_utilization=0.5)
    assert regular.total <= reserved.total
    assert regular.recommended_gpu_memory is not None
    assert reserved.recommended_gpu_memory is not None
    assert regular.total < regular.recommended_gpu_memory < reserved.recommended_gpu_memory
    assert reserved.to_dict()["gpu_memory_utilization"] == 0.5
    assert "GPU CAPACITY" in table(reserved)
    assert "TOTAL MEMORY" not in table(reserved) and "ALLOCATIONS" not in table(reserved)
    assert "Recommended GPU capacity" in markdown(reserved)
    assert "| **Total** |" not in markdown(reserved)


def test_topology_recommendation_uses_rank_allocations(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    result = estimate(str(tmp_path), gpu_memory="4x80")
    assert result.fit is not None and result.overhead is not None
    assert result.recommended_gpu_memory == _overhead.recommended_memory(
        result.fit.required + 4 * result.startup_reserve_per_gpu, 0.9
    )
    assert "ACROSS 4 GPUS" in table(result)


def test_model_only_selects_small_gpu_defaults_without_a_hardware_flag(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    inferred = estimate(str(tmp_path))
    small_gpu = estimate(str(tmp_path), gpu_memory="16")
    large_gpu = estimate(str(tmp_path), gpu_memory="141")
    assert inferred.overhead == small_gpu.overhead
    assert inferred.kv_cache == small_gpu.kv_cache
    assert inferred.total < large_gpu.total


def test_explicit_cache_pool_ratio_changes_fit_without_changing_weights(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(
        str(tmp_path / "config.json"),
        llama_config(
            max_position_embeddings=131072,
            sliding_window=128,
            layer_types=["full_attention"] + ["sliding_attention"] * 3,
        ),
    )
    logical = estimate(str(tmp_path), engine="sglang", gpu_memory="24", max_num_batched_tokens=512)
    partitioned = estimate(
        str(tmp_path), gpu_memory="24", max_num_batched_tokens=512, swa_full_tokens_ratio=0.8
    )
    assert partitioned.weights == logical.weights
    assert partitioned.overhead == logical.overhead
    assert partitioned.total > logical.total
    assert partitioned.to_dict()["kv_cache"]["swa_full_tokens_ratio"] == 0.8
    assert partitioned.fit is not None and partitioned.fit.required == partitioned.total
    assert any("swa-full-tokens-ratio" in warning for warning in logical.warnings)
    assert any("mem-fraction-static" in warning for warning in partitioned.warnings)


def test_sglang_reserves_shared_layer_slots_without_a_ratio_override(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(
        str(tmp_path / "config.json"),
        llama_config(
            num_kv_shared_layers=2,
            sliding_window=128,
            layer_types=["full_attention", "sliding_attention"] * 2,
        ),
    )
    common = estimate(str(tmp_path), engine="vllm")
    pooled = estimate(str(tmp_path), engine="sglang")
    assert common.kv_cache is not None and pooled.kv_cache is not None
    assert common.kv_cache.layers == {"full": 1, "sliding": 1, "state": 0, "none": 2}
    assert pooled.kv_cache.layers == {"full": 2, "sliding": 2, "state": 0, "none": 0}
    assert pooled.kv_cache.bytes == 2 * common.kv_cache.bytes


@pytest.mark.parametrize("ratio", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_cache_pool_ratios_are_rejected(ratio: float) -> None:
    with pytest.raises(RuntimeError, match="swa-full-tokens-ratio"):
        estimate("unused/model", swa_full_tokens_ratio=ratio)


def test_static_budget_changes_capacity_without_changing_allocations(tmp_path: Path) -> None:
    tensors = llama_tensors()
    tensors["model.embed_tokens.weight"] = ("BF16", [262144, 32768])
    write_safetensors(str(tmp_path / "model.safetensors"), tensors)
    write_json(str(tmp_path / "config.json"), llama_config())
    regular = estimate(str(tmp_path), engine="sglang", gpu_memory="2x16")
    limited = estimate(str(tmp_path), gpu_memory="2x16", mem_fraction_static=0.5)
    assert regular.total == limited.total
    assert regular.overhead == limited.overhead
    assert regular.fit is not None and regular.fit.fits
    assert limited.fit is not None and not limited.fit.fits
    assert limited.fit.required == limited.total
    assert limited.fit.capacity_required == limited.recommended_gpu_memory
    assert limited.to_dict()["mem_fraction_static"] == 0.5


def test_static_budget_is_checked_during_context_search() -> None:
    result = _overhead.fit(
        weights=GIB,
        kv_fixed=0,
        kv_per_token=0,
        max_model_len=4096,
        context_length=4096,
        overhead=0,
        gpu_memory="4",
        utilization=0.9,
        capacity_at_context=lambda n, length: 2 * GIB + length * (1 << 20),
    )
    assert not result.fits and result.needed == 2
    assert result.required == GIB
    assert result.max_model_len == 2048


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_static_budgets_are_rejected(fraction: float) -> None:
    with pytest.raises(RuntimeError, match="mem-fraction-static"):
        estimate("unused/model", mem_fraction_static=fraction)


def test_fused_experts_still_require_moe_workspaces(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    dense = estimate(str(tmp_path), engine="sglang", gpu_memory="2x80")
    write_json(str(tmp_path / "config.json"), llama_config(n_routed_experts=4, num_experts_per_tok=2))
    fused = estimate(str(tmp_path), engine="sglang", gpu_memory="2x80")
    assert fused.moe is None
    assert dense.overhead is not None and fused.overhead is not None
    assert fused.overhead.graphs > dense.overhead.graphs
    assert fused.fit is not None and dense.fit is not None
    assert fused.fit.required > dense.fit.required


def test_language_only_startup_can_prevent_a_steady_state_fit() -> None:
    result = _overhead.fit(
        weights=10 * GIB,
        kv_fixed=0,
        kv_per_token=1024,
        max_model_len=1000,
        context_length=1000,
        overhead=GIB,
        gpu_memory="16",
        utilization=0.9,
        startup_reserve=4 * GIB,
    )
    assert result.required < result.usable
    assert result.startup_required is not None and result.startup_required > result.usable
    assert not result.fits
    assert result.needed == 2
    assert result.max_model_len == 0


def test_inkling_defaults_to_its_calibrated_engine(tmp_path: Path) -> None:
    import json

    config = json.loads((Path(__file__).parent / "fixtures/inkling2.json").read_text())
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), config)
    default = estimate(str(tmp_path), max_model_len=128)
    selected = estimate(str(tmp_path), max_model_len=128, engine="sglang")
    assert default.overhead is not None and default.overhead.reference_engine == "sglang"
    assert default.total == selected.total
    assert default.recommended_gpu_memory == selected.recommended_gpu_memory
    assert "SGLANG REFERENCE" not in table(default)
    other = estimate(str(tmp_path), max_model_len=128, engine="vllm")
    assert any("vLLM layout is not validated" in w for w in other.warnings)


def test_small_decoder_capacity_includes_shared_initialization_workspace(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    result = estimate(str(tmp_path), gpu_memory="4x24")
    assert result.overhead is not None and result.fit is not None
    assert result.overhead.activations // 4 + result.startup_reserve_per_gpu >= GIB
    assert result.fit.startup_required == result.fit.required + 4 * result.startup_reserve_per_gpu


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("ranks", [1, 2, 4, 8])
@pytest.mark.parametrize("heads", [1, 4])
def test_selected_topology_breakdown_matches_the_fit(
    tmp_path: Path, engine: str, ranks: int, heads: int
) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config(num_key_value_heads=heads))
    result = estimate(str(tmp_path), engine=engine, gpu_memory=f"{ranks}x24")
    assert result.fit is not None and result.kv_cache is not None and result.overhead is not None
    assert result.total == result.fit.required
    assert result.total == result.weights.bytes + result.kv_cache.bytes + result.overhead.bytes
    assert result.kv_cache.bytes == result.fit.cache_bytes
    assert result.fit.overhead_per_gpu is not None
    assert result.overhead.bytes == ranks * result.fit.overhead_per_gpu
    assert "Engine reference" not in markdown(result)
    assert f"{engine.upper()} REFERENCE" not in table(result)
    assert "ALLOCATION BUDGET" not in table(result)
    assert result.to_dict()["total"] == result.fit.required
