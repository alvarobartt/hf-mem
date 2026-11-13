from typing import Any

import pytest

from hf_mem import _overhead

GIB = _overhead.GIB


def test_vllm_overhead_follows_the_gpu_class() -> None:
    config = {"vocab_size": 128256, "intermediate_size": 14336, "hidden_size": 4096, "num_hidden_layers": 32}
    small = _overhead.vllm(config, moe=False, hybrid=False, tokens=2048)
    large = _overhead.vllm(config, moe=False, hybrid=False, tokens=8192)
    assert 0.4 * GIB < small.activations < 0.6 * GIB
    assert 1.1 * GIB < large.activations < 1.4 * GIB
    assert 0.2 * GIB < small.graphs < 0.4 * GIB
    assert small.context == large.context
    assert small.runtime > small.context and large.runtime > large.context
    assert small.max_num_seqs == large.max_num_seqs == 256
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
    assert _overhead.encoder(qwen, {"size": {"longest_edge": 16777216}}) == 65536 * 3 * (4 * 1152 + 2 * 4304)
    gated = {"vision_config": {**qwen["vision_config"], "patch_size": 14, "hidden_act": "silu"}}
    assert _overhead.encoder(gated, {"max_pixels": 12845056}) == 65536 * 3 * (4 * 1152 + 3 * 4304)
    siglip = {"vision_config": {**qwen["vision_config"], "patch_size": 14, "image_size": 896}}
    assert _overhead.encoder(siglip, {}) == 4096 * 3 * (4 * 1152 + 2 * 4304)
    tiles = {
        **siglip,
        "max_dynamic_patch": 12,
        "use_thumbnail": True,
        "vision_config": {**siglip["vision_config"], "image_size": 448},
    }
    assert _overhead.encoder(tiles, {}) == 13 * 1024 * 3 * (4 * 1152 + 2 * 4304)
    pooled = {
        "vision_config": {**qwen["vision_config"], "pooling_kernel_size": 3},
        "vision_soft_tokens_per_image": 280,
    }
    assert _overhead.encoder(pooled, {}) == 2520 * 3 * (4 * 1152 + 2 * 4304)
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


def test_fit_budgets_the_entire_runtime_including_cuda_context() -> None:
    args: dict[str, Any] = dict(
        weights=0, kv_fixed=0, kv_per_token=1024, max_model_len=1000, context_length=10**9, gpu_memory="20"
    )
    with_context = _overhead.fit(overhead=2 * GIB, utilization=0.9, **args)
    without = _overhead.fit(overhead=GIB, utilization=0.9, **args)
    assert with_context.max_model_len == without.max_model_len - GIB // 1024
    assert _overhead.fit(overhead=2 * GIB, utilization=1.0, **args).max_model_len == int(18 * GIB / 1024)


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


def test_scheduler_capacity_scales_overhead_without_removing_fixed_memory() -> None:
    config = {"vocab_size": 128256, "intermediate_size": 14336, "hidden_size": 4096, "num_hidden_layers": 32}
    default = _overhead.vllm(config, False, False, 8192, encoder_bytes=GIB)
    reference = _overhead.vllm(config, False, False, 8192, encoder_bytes=GIB, max_num_seqs=1024)
    one = _overhead.vllm(config, False, False, 8192, encoder_bytes=GIB, max_num_seqs=1)
    many = _overhead.vllm(config, False, False, 8192, encoder_bytes=GIB, max_num_seqs=256)
    assert reference.bytes == default.bytes
    assert one.activations < many.activations == default.activations
    assert 0 < one.graphs < many.graphs == default.graphs
    assert one.runtime >= one.context == many.context == default.context
    assert one.encoder == many.encoder == default.encoder == GIB
    assert default.to_dict()["max_num_seqs"] == 256
    assert one.to_dict()["max_num_seqs"] == 1


def test_scheduler_calibration_reduces_error_against_independent_eight_model_reference() -> None:
    import json
    from pathlib import Path
    from statistics import median

    fixture = json.loads((Path(__file__).parent / "fixtures/overhead_scheduler.json").read_text())
    total_errors = []
    overhead_errors = []
    for row in fixture["rows"]:
        overhead = _overhead.vllm(
            row["config"],
            row["moe"],
            row["hybrid"],
            row["tokens"],
            encoder_bytes=row["encoder_bytes"],
            max_num_seqs=1,
        )
        measured = row["native_runtime_footprint_bytes"]
        base = row["estimated_weights_bytes"] + row["estimated_kv_bytes"]
        before = abs((base + row["original_overhead_bytes"]) / measured - 1)
        after = abs((base + overhead.bytes) / measured - 1)
        assert after < before, row["case"]["name"]
        total_errors.append(after)
        measured_overhead = measured - row["native_parameter_bytes"] - row["native_cache_bytes"]
        overhead_errors.append(abs(overhead.bytes / measured_overhead - 1))
    assert len(total_errors) == 8
    assert median(total_errors) < 0.05
    assert median(overhead_errors) < 0.5


def test_fit_recomputes_overhead_for_gpu_search_and_context_fit() -> None:
    result = _overhead.fit(
        weights=10 * GIB,
        kv_fixed=0,
        kv_per_token=1024,
        max_model_len=1000,
        context_length=10**9,
        overhead=lambda n: (9 if n == 2 else 1) * GIB,
        gpu_memory="2x10",
        utilization=1.0,
    )
    assert result.needed == 4 and not result.fits
    assert result.max_model_len == 0
    assert result.required == 28 * GIB + 1024 * 1000
    assert result.to_dict()["overhead_per_gpu"] == 9 * GIB


def test_tp_calibration_reduces_error_on_all_measured_multi_rank_topologies() -> None:
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/overhead_tensor_parallel.json").read_text())
    base = fixture["estimated_weights_bytes"] + fixture["estimated_cache_bytes"]
    single = _overhead.vllm(
        fixture["config"], False, False, 8192, max_num_seqs=1, gpu_memory="140", gpu_type="h200"
    )
    for observation in fixture["observations"]:
        n = observation["tensor_parallel_size"]
        overhead = _overhead.vllm(
            fixture["config"],
            False,
            False,
            8192,
            max_num_seqs=1,
            tensor_parallel_size=n,
            gpu_memory="140",
            gpu_type="h200",
        )
        assert overhead.context == single.context and overhead.encoder == single.encoder
        actual = observation["native_runtime_footprint_bytes"]
        error = abs((base + n * overhead.bytes) / actual - 1)
        if n > 1:
            assert error < abs((base + n * single.bytes) / actual - 1)
            assert GIB <= overhead.runtime - single.runtime < 1.01 * GIB
            assert overhead.activations < single.activations
            assert error < 0.04
        else:
            assert overhead.bytes == single.bytes


def test_llamacpp_hardware_runtime_profiles_match_pinned_t4_h200_observations() -> None:
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/overhead_llamacpp_hardware.json").read_text())
    for row in fixture["observations"]:
        estimate = _overhead.llama_cpp(gpu_memory=row["gpu_memory"])
        actual = row["native_runtime_footprint_bytes"]
        base = row["estimated_weights_bytes"] + row["estimated_cache_bytes"]
        assert abs((base + estimate.bytes) / actual - 1) < abs(
            (base + row["original_overhead_bytes"]) / actual - 1
        )
        assert abs((base + estimate.bytes) / actual - 1) < 0.04
    assert _overhead.llama_cpp().bytes == _overhead.llama_cpp(gpu_memory="80").bytes
    assert _overhead.llama_cpp(gpu_memory="2x15").context == int(0.13 * GIB)


def test_context_dependent_scheduler_metadata_is_recomputed_during_fit_search() -> None:
    # The cache and request-index map both grow with context. A frozen overhead
    # would return a context that exceeds this independently computed budget.
    result = _overhead.fit(
        weights=0,
        kv_fixed=0,
        kv_per_token=0,
        max_model_len=1000,
        context_length=100000,
        overhead=0,
        gpu_memory="0.001",
        utilization=1,
        kv_bytes=lambda length, ranks: length * 8,
        overhead_at_context=lambda ranks, length: length * 4,
    )
    budget = int(0.001 * _overhead.GIB)
    assert result.max_model_len == budget // 12
    assert result.required == 12000


def test_spatially_merged_image_reserve_cannot_exceed_entire_context() -> None:
    config = {
        "vision_config": {
            "hidden_size": 896,
            "intermediate_size": 3584,
            "hidden_act": "silu",
            "patch_size": 16,
            "spatial_merge_size": 2,
        }
    }
    processor = {"max_pixels": 16777216}
    original = _overhead.encoder(config, processor)
    assert _overhead.encoder(config, processor, 8192) == original // 2
    assert _overhead.encoder(config, processor, 32768) == original


@pytest.mark.parametrize(
    "gpu_type,memory",
    [
        ("nvidia-t4", "14.74127"),
        ("nvidia-a10g", "2x22.30127"),
        ("nvidia-l4", "22.0343"),
        ("nvidia-l40s", "44.39215"),
        ("nvidia-a100", "79.24975"),
        ("nvidia-a100", "40"),
        ("nvidia-h100", "94"),
        ("nvidia-h100", "96"),
        ("nvidia-h200", "8x141"),
        ("nvidia-h200", "8x139.80072GiB"),
        ("nvidia-h200", "8x150.10988032GB"),
        ("nvidia-rtx-pro-6000", "94.97052"),
    ],
)
def test_named_gpu_accepts_board_variants_and_measured_driver_totals(gpu_type: str, memory: str) -> None:
    _overhead.validate_gpu_memory(memory, gpu_type)


@pytest.mark.parametrize(
    "gpu_type,memory",
    [
        ("nvidia-h200", "8x80"),
        ("nvidia-a100", "48"),
        ("nvidia-t4", "24"),
        ("nvidia-h100", "141"),
        ("nvidia-h200", "141.01"),
    ],
)
def test_named_gpu_rejects_incompatible_capacity(gpu_type: str, memory: str) -> None:
    with pytest.raises(RuntimeError, match="does not match"):
        _overhead.validate_gpu_memory(memory, gpu_type)


def test_gpu_capacity_units_are_explicit() -> None:
    assert _overhead.parse_gpu_memory("8x141GB") == (8, 141_000_000_000)
    assert _overhead.parse_gpu_memory("8x141GiB") == (8, 141 * GIB)
    assert _overhead.GPU_TYPES["h200"][0] == 141
    assert _overhead.hardware("8x141", "NVIDIA-H200") == _overhead.hardware("8x141", "h200")
