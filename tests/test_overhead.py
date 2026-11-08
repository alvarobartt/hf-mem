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
    assert 0.35 * GIB < small.graphs < 0.5 * GIB
    assert (small.runtime, large.runtime) == (int(0.27 * GIB) + small.context, int(0.87 * GIB) + large.context)
    moe = _overhead.vllm(
        {**config, "moe_intermediate_size": 768, "num_experts_per_tok": 8}, moe=True, hybrid=False, tokens=2048
    )
    assert moe.graphs > small.graphs
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


@pytest.mark.parametrize("value", ["", "x80", "0x80", "-2x80", "8x", "eighty", "-1", "inf", "1e400"])
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
    assert (fit.usable, fit.required) == (int(18 * GIB), 10 * GIB + 1024 * 1000 + GIB)
    replicated = _overhead.fit(weights=10 * GIB, gpu_memory="4x20", utilization=0.9, kv_heads=1, **args)
    assert replicated.required == 10 * GIB + 4 * 1024 * 1000 + 4 * GIB

    fit = _overhead.fit(weights=30 * GIB, gpu_memory="20", utilization=0.9, **args)
    assert (fit.needed, fit.fits, fit.max_model_len) == (2, False, 0)
    assert _overhead.fit(weights=30 * GIB, gpu_memory="2x20", utilization=0.9, **args).fits
    limited: dict[str, Any] = {**args, "context_length": 5}
    assert _overhead.fit(weights=30 * GIB, gpu_memory="2x20", utilization=0.9, **limited).max_model_len == 5
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


V = "clip.vision."


@pytest.mark.parametrize(
    "metadata, mib",
    [
        # llama.cpp `server-cuda` on an A100, with the largest image: gemma-3-4b-it, Qwen2.5-VL-7B, LFM3-VL-2B,
        # LFM2.5-VL-3B and the ultravox-v0_5-llama-3_2-1b audio encoder
        (
            {
                V + "embedding_length": 1152,
                V + "feed_forward_length": 4304,
                V + "image_size": 896,
                V + "patch_size": 14,
            },
            121.25,
        ),
        (
            {
                V + "embedding_length": 1280,
                V + "feed_forward_length": 3420,
                V + "patch_size": 14,
                V + "n_wa_pattern": 8,
                "clip.use_silu": True,
            },
            1459.7,
        ),
        (
            {
                V + "embedding_length": 1024,
                V + "feed_forward_length": 2560,
                V + "patch_size": 16,
                V + "spatial_merge_size": 2,
                V + "image_max_pixels": 16777216,
                "clip.use_silu": True,
            },
            718.1,
        ),
        (
            {
                V + "embedding_length": 1152,
                V + "feed_forward_length": 4304,
                V + "image_size": 256,
                V + "patch_size": 16,
                V + "projector.scale_factor": 2,
                "clip.projector_type": "lfm2",
            },
            30.31,
        ),
        ({"clip.audio.embedding_length": 1280, "clip.audio.feed_forward_length": 5120}, 51.27),
    ],
)
def test_mmproj_encoder_matches_llama_cpp(metadata: dict[str, Any], mib: float) -> None:
    assert abs(_overhead.mmproj_encoder(metadata) / (1 << 20) / mib - 1) < 0.05
    assert _overhead.llama_cpp(metadata).encoder == _overhead.mmproj_encoder(metadata)
    assert _overhead.llama_cpp().encoder == 0


def test_mxfp4_experts_are_padded_to_the_vllm_tiles() -> None:
    # gpt-oss-120b on 1 GPU (2880 -> 2944 intermediate, 2880 -> 3072 hidden) and DeepSeek-V4.1 on 8 GPUs (288 -> 384)
    gpt_oss = _overhead.mxfp4_padding(100, 2880, 2880)
    assert gpt_oss(1) == int(100 * (2944 / 2880 * 3072 / 2880 - 1))
    deepseek = _overhead.mxfp4_padding(300, 2304, 5120)
    assert 99 <= deepseek(8) <= 100 and deepseek(1) == 0
    args: dict[str, Any] = dict(kv_fixed=0, kv_per_token=0, max_model_len=0, context_length=0, overhead=0)
    plain = _overhead.fit(weights=60 * GIB, gpu_memory="64", utilization=1.0, **args)
    padded = _overhead.fit(
        weights=60 * GIB, gpu_memory="64", utilization=1.0, padding=lambda n: 8 * GIB, **args
    )
    assert plain.fits and not padded.fits and padded.required == 68 * GIB


def test_fit_searches_non_linear_cache_and_replicates_fixed_windows() -> None:
    # A sliding cache saturates at 100 pages. Both its fixed and growing parts replicate when KV heads < TP.
    def cache(length: int, n: int) -> int:
        return min(-(-length // 16), 100) * 1024 * n

    args: dict[str, Any] = dict(
        weights=0,
        kv_fixed=100 * 1024,
        kv_per_token=0,
        max_model_len=10000,
        context_length=10000,
        overhead=0,
        kv_heads=1,
        kv_bytes=cache,
        utilization=1.0,
    )
    fit = _overhead.fit(gpu_memory=str(80 * 1024 / GIB), **args)
    assert not fit.fits and fit.max_model_len == 80 * 16
    replicated = _overhead.fit(gpu_memory=f"4x{80 * 1024 / GIB}", **args)
    assert replicated.required == 4 * 100 * 1024
    assert replicated.max_model_len == fit.max_model_len
    roomy = _overhead.fit(gpu_memory=str(100 * 1024 / GIB), **args)
    assert roomy.fits and roomy.max_model_len == 10000


def test_fit_checks_the_requested_gpu_count_between_power_of_two_sizes() -> None:
    result = _overhead.fit(
        weights=25 * GIB,
        kv_fixed=0,
        kv_per_token=0,
        max_model_len=0,
        context_length=0,
        overhead=0,
        gpu_memory="3x10",
        utilization=1.0,
    )
    assert result.fits and result.needed == 3


def test_encoder_attention_workspace_tracks_sequence_boundaries() -> None:
    config = {"model_type": "mpnet", "hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12}
    single = _overhead.encoder_warmup(config, {}, 384, 384, 4)
    double = _overhead.encoder_warmup(config, {}, 768, 384, 4)
    partial = _overhead.encoder_warmup(config, {}, 512, 384, 4)
    short = _overhead.encoder_warmup(config, {}, 768, 128, 4)
    assert double.activations == 2 * single.activations
    assert single.activations < partial.activations < double.activations
    assert short.activations < double.activations
    assert double.runtime == single.runtime == partial.runtime
    fused = _overhead.encoder_warmup({**config, "_attn_implementation": "sdpa"}, {}, 768, 384, 4)
    assert fused.activations < short.activations


def test_encoder_precision_and_gemma_have_no_server_specific_rules() -> None:
    config = {
        "model_type": "gemma3_text",
        "hidden_size": 768,
        "intermediate_size": 1152,
        "num_attention_heads": 3,
        "head_dim": 256,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
    }
    half = _overhead.encoder_warmup(config, {}, 2048, 2048, 2)
    full = _overhead.encoder_warmup(config, {}, 2048, 2048, 4)
    assert full.activations == 2 * half.activations
    assert not _overhead.encoder_eager(config)
