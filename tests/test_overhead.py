from typing import Any

import pytest

from hf_mem import _overhead

GIB = _overhead.GIB


def test_vllm_overhead_follows_the_gpu_class() -> None:
    config = {"vocab_size": 128256, "intermediate_size": 14336, "hidden_size": 4096, "num_hidden_layers": 32}
    small = _overhead.vllm(config, moe=False, hybrid=False, tokens=2048)
    large = _overhead.vllm(config, moe=False, hybrid=False, tokens=8192)
    assert 1.2 * GIB < small.activations < 1.6 * GIB
    assert 2.6 * GIB < large.activations < 3.0 * GIB
    assert 0.35 * GIB < small.cuda_graphs < 0.5 * GIB
    assert (small.non_torch, large.non_torch) == (int(0.27 * GIB), int(0.87 * GIB))
    moe = _overhead.vllm(
        {**config, "moe_intermediate_size": 768, "num_experts_per_tok": 8}, moe=True, hybrid=False, tokens=2048
    )
    assert moe.cuda_graphs > small.cuda_graphs
    assert (
        _overhead.vllm(config, moe=False, hybrid=False, tokens=2048, encoder_bytes=GIB).bytes
        == small.bytes + GIB
    )


def test_encoder_patches() -> None:
    qwen = {
        "vision_config": {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "patch_size": 16,
            "hidden_act": "gelu",
        }
    }
    assert _overhead.encoder(qwen, {"size": {"longest_edge": 16777216}}) == 65536 * 2 * (4 * 1152 + 2 * 4304)
    gated = {"vision_config": {**qwen["vision_config"], "patch_size": 14, "hidden_act": "silu"}}
    assert _overhead.encoder(gated, {"max_pixels": 12845056}) == 65536 * 2 * (4 * 1152 + 3 * 4304)
    siglip = {"vision_config": {**qwen["vision_config"], "patch_size": 14, "image_size": 896}}
    assert _overhead.encoder(siglip, {}) == 4096 * 2 * (4 * 1152 + 2 * 4304)
    tiles = {
        **siglip,
        "max_dynamic_patch": 12,
        "use_thumbnail": True,
        "vision_config": {**siglip["vision_config"], "image_size": 448},
    }
    assert _overhead.encoder(tiles, {}) == 13 * 1024 * 2 * (4 * 1152 + 2 * 4304)
    pooled = {
        "vision_config": {**qwen["vision_config"], "pooling_kernel_size": 3},
        "vision_soft_tokens_per_image": 280,
    }
    assert _overhead.encoder(pooled, {}) == 2520 * 2 * (4 * 1152 + 2 * 4304)
    audio = {"audio_config": {"d_model": 1280, "encoder_ffn_dim": 5120, "max_source_positions": 1500}}
    assert _overhead.encoder(audio, {}) == 1500 * 2 * (4 * 1280 + 2 * 5120)
    assert _overhead.encoder({}, {}) == 0


def test_batched_tokens_follow_the_gpu_size() -> None:
    assert _overhead.batched_tokens(None) == _overhead.batched_tokens("8x80") == 8192
    assert _overhead.batched_tokens("24") == 2048
    assert _overhead.batched_tokens("180") == 16384


@pytest.mark.parametrize(
    "value, expected",
    [("80", (None, 80 * GIB)), ("8x80", (8, 80 * GIB)), ("2x24GiB", (2, 24 * GIB)), ("0.5", (None, GIB // 2))],
)
def test_parse_gpu_memory(value: str, expected: tuple[int | None, int]) -> None:
    assert _overhead.parse_gpu_memory(value) == expected


@pytest.mark.parametrize("value", ["", "x80", "0x80", "8x", "eighty", "-1"])
def test_parse_gpu_memory_errors(value: str) -> None:
    with pytest.raises(RuntimeError):
        _overhead.parse_gpu_memory(value)


def test_fit() -> None:
    args: dict[str, Any] = dict(
        kv_fixed=0, kv_per_token=1024, max_model_len=1000, context_length=10**9, overhead=GIB
    )
    fit = _overhead.fit(weights=10 * GIB, gpu_memory="20", utilization=0.9, **args)
    assert (fit.needed, fit.fits) == (1, True)
    assert fit.max_model_len == int((18 * GIB - GIB - 10 * GIB) / 1024)

    fit = _overhead.fit(weights=30 * GIB, gpu_memory="20", utilization=0.9, **args)
    assert (fit.needed, fit.fits, fit.max_model_len) == (2, False, 0)
    assert _overhead.fit(weights=30 * GIB, gpu_memory="2x20", utilization=0.9, **args).fits
    assert (
        _overhead.fit(
            weights=30 * GIB, gpu_memory="2x20", utilization=0.9, **{**args, "context_length": 5}
        ).max_model_len
        == 5
    )
    assert _overhead.fit(weights=10**15, gpu_memory="20", utilization=0.9, **args).needed is None


def test_fit_keeps_the_cuda_context_outside_the_utilization() -> None:
    args: dict[str, Any] = dict(
        weights=0, kv_fixed=0, kv_per_token=1024, max_model_len=1000, context_length=10**9, gpu_memory="20"
    )
    with_context = _overhead.fit(overhead=2 * GIB, utilization=0.9, context=GIB, **args)
    without = _overhead.fit(overhead=GIB, utilization=0.9, **args)
    assert with_context.max_model_len == without.max_model_len
    assert _overhead.fit(overhead=2 * GIB, utilization=1.0, context=GIB, **args).max_model_len == int(
        18 * GIB / 1024
    )


def test_fit_replicates_kv_heads_across_gpus() -> None:
    args: dict[str, Any] = dict(
        kv_fixed=0, kv_per_token=GIB / 1000, max_model_len=1000, context_length=10**9, overhead=0
    )
    split = _overhead.fit(weights=0, gpu_memory="8x10", utilization=1.0, kv_heads=8, **args)
    replicated = _overhead.fit(weights=0, gpu_memory="8x10", utilization=1.0, kv_heads=2, **args)
    assert split.max_model_len == 4 * replicated.max_model_len
    assert _overhead.fit(
        weights=0, gpu_memory="8x10", utilization=1.0, kv_heads=None, **args
    ).max_model_len == (split.max_model_len)


def test_batched_tokens_override() -> None:
    assert _overhead.batched_tokens("8x80", 2048) == 2048
