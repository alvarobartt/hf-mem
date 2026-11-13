"""Capacity recommendations must agree with deployment fit decisions."""

from pathlib import Path
from typing import Any

import pytest
from helpers import estimate, llama_config, llama_tensors, write_json, write_safetensors

from hf_mem import _overhead
from hf_mem._render import markdown, table

GIB = 1 << 30


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
    # Use a few bytes either side to avoid decimal CLI round-trip rounding.
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
    assert "ALLOCATIONS" in table(reserved)
    assert "Recommended GPU capacity" in markdown(reserved)


def test_topology_recommendation_uses_rank_allocations(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    result = estimate(str(tmp_path), gpu_memory="4x80")
    assert result.fit is not None and result.overhead is not None
    assert result.recommended_gpu_memory == _overhead.recommended_memory(result.fit.required, 0.9)
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


def test_native_engine_budget_can_prevent_an_allocation_only_fit() -> None:
    result = _overhead.fit(
        weights=10 * GIB,
        kv_fixed=0,
        kv_per_token=1024,
        max_model_len=1000,
        context_length=1000,
        overhead=GIB,
        gpu_memory="20",
        utilization=0.9,
        minimum_capacity=lambda ranks, length: 30 * GIB + length * 1024,
    )
    assert result.required < result.usable
    assert not result.fits
    assert result.needed == 2
    assert result.minimum_capacity is not None and result.minimum_capacity > result.gpu_memory
    assert result.max_model_len == 0


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


def test_native_conv_pool_matches_independent_state_geometry() -> None:
    import json

    config = json.loads((Path(__file__).parent / "fixtures/inkling2.json").read_text())["text_config"]
    pool = _overhead.sglang_conv_pool(
        config,
        weights=396545310944,
        attention=11496587264,
        tokens=8192,
        ranks=4,
        gpu_memory="4x139.80072021484375",
        gpu_type="h200",
        max_num_seqs=64,
        batch_size=1,
        multimodal=True,
    )
    # Native config: 48 layers; six BF16 convolution streams per layer, with
    # TP-local K/V widths and replicated 4608-wide hidden streams. The native
    # automatic pool has 845 usable slots plus one guard on each rank.
    native_per_slot = 48 * 2 * 3 * (4 * 256 + 2 * 4608)
    assert pool.bytes_per_gpu == (pool.slots + 1) * native_per_slot
    assert abs(pool.slots - 845) <= 20
    assert pool.requests == 64
    assert pool.minimum_capacity_per_gpu > 110 * GIB


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
    assert "SGLANG REFERENCE" in table(default)
    other = estimate(str(tmp_path), max_model_len=128, engine="vllm")
    assert any("vLLM layout is not validated" in w for w in other.warnings)
