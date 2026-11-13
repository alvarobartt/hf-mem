import json
from pathlib import Path
from statistics import median

import pytest
from helpers import estimate, llama_config, llama_tensors, some, write_json, write_safetensors

from hf_mem import _overhead


def test_engine_profiles_against_independent_native_reference_bytes() -> None:
    reference = json.loads((Path(__file__).parent / "fixtures/overhead_crossengine.json").read_text())
    errors, totals = [], []
    for row in reference["rows"]:
        ranks = row["tensor_parallel_size"]
        overhead = (
            _overhead.vllm(
                row["config"],
                False,
                False,
                row["tokens"],
                max_num_seqs=row["max_num_seqs"],
                tensor_parallel_size=ranks,
                gpu_memory=row["gpu_memory"],
                gpu_type=row["gpu_type"],
                engine=row["engine"],
            ).bytes
            * ranks
        )
        total = (
            row["checkpoint_weights_bytes"]
            - row["tied_duplicate_bytes"]
            + row["logical_cache_bytes"]
            + overhead
        )
        errors.append(abs(overhead / row["native_residual_bytes"] - 1))
        totals.append(abs(total / row["native_runtime_bytes"] - 1))
    # Measured denominators come from registered storage, physical pools and NVML,
    # independently of hf-mem. These are retrospective regressions, not holdouts.
    assert median(errors) < 0.05 and max(errors) < 0.15
    assert median(totals) < 0.01 and max(totals) < 0.03


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_calibrated_token_budget_has_no_profile_boundary_jump(engine: str) -> None:
    config = {"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36, "vocab_size": 151936}
    for capacity in [1, 16, 256]:
        for boundary in [2048, 8192]:
            values = [
                _overhead.vllm(
                    config,
                    False,
                    False,
                    tokens,
                    max_num_seqs=capacity,
                    gpu_memory="48",
                    engine=engine,
                ).bytes
                for tokens in [boundary - 1, boundary, boundary + 1]
            ]
            assert all(abs(left - right) < _overhead.MIB for left, right in zip(values, values[1:]))


def test_public_engine_and_hardware_controls_reach_topology_fitting(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    options = dict(gpu_memory="2x24", gpu_type="A10G", max_num_seqs=16, max_num_batched_tokens=4096)
    vllm = estimate(str(tmp_path), engine="vllm", **options)
    sglang = estimate(str(tmp_path), engine="SGLang", **options)
    assert vllm.weights == sglang.weights and vllm.kv_cache == sglang.kv_cache
    assert some(sglang.overhead).bytes > some(vllm.overhead).bytes
    for result, engine in [(vllm, "vllm"), (sglang, "sglang")]:
        overhead = some(result.overhead)
        assert overhead.reference_engine == engine and overhead.gpu_type == "nvidia-a10g"
        expected = _overhead.vllm(
            llama_config(),
            False,
            False,
            4096,
            max_num_seqs=16,
            tensor_parallel_size=2,
            gpu_memory="2x24",
            gpu_type="a10g",
            engine=engine,
            context_length=some(result.kv_cache).max_model_len,
        )
        assert some(result.fit).overhead_per_gpu == expected.bytes


def test_named_hardware_overrides_ambiguous_capacity_and_a100_prefill_default() -> None:
    assert _overhead.hardware("80", "h100")[2]
    assert not _overhead.hardware("80", "a100")[2]
    assert not _overhead.hardware("96", "rtx-pro-6000")[1]
    assert _overhead.batched_tokens("80", gpu_type="a100") == 2048
    assert _overhead.batched_tokens("80", gpu_type="h100") == 8192


def test_sglang_gemma_default_selects_observed_full_pool_without_changing_attention(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(
        str(tmp_path / "config.json"),
        llama_config(model_type="gemma3_text", max_position_embeddings=10000, sliding_window_pattern=2),
    )
    options = dict(max_model_len=10000, max_num_batched_tokens=2048, max_num_seqs=1)
    vllm = estimate(str(tmp_path), **options)
    sglang = estimate(str(tmp_path), engine="sglang", **options)
    full = estimate(str(tmp_path), full_sliding_window_cache=True, **options)
    assert sglang.kv_cache == full.kv_cache
    assert some(sglang.kv_cache).bytes > some(vllm.kv_cache).bytes
    assert some(sglang.kv_cache).layers == some(vllm.kv_cache).layers


@pytest.mark.parametrize("options", [{"engine": "invalid"}, {"gpu_type": "invalid"}])
def test_invalid_serving_profiles_are_rejected_before_model_access(options: dict[str, str]) -> None:
    with pytest.raises(RuntimeError, match="--engine|--gpu-type"):
        estimate("not/a/real/model", **options)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_expanded_query_projection_is_local_to_tensor_parallel_rank(engine: str) -> None:
    common = {"hidden_size": 1024, "intermediate_size": 3072, "vocab_size": 151936}
    expanded = {**common, "num_attention_heads": 16, "head_dim": 128}

    def memory(config: dict[str, int], ranks: int) -> int:
        return _overhead.vllm(
            config,
            False,
            False,
            8192,
            max_num_seqs=1,
            tensor_parallel_size=ranks,
            gpu_memory="96",
            gpu_type="rtx-pro-6000",
            engine=engine,
        ).bytes

    assert memory(expanded, 1) > memory(common, 1)
    assert memory(expanded, 2) == memory(common, 2)


def test_native_engine_defaults_keep_request_slots_distinct_from_graph_limits() -> None:
    assert _overhead.default_sequences("140", "h200", "vllm") == 1024
    assert _overhead.default_sequences("96", "rtx-pro-6000", "vllm") == 1024
    assert _overhead.default_sequences("80", "a100", "sglang") == 2048
    for size, prefill, graph in [(16, 2048, 8), (24, 4096, 48), (48, 4096, 32), (80, 8192, 256)]:
        assert _overhead.batched_tokens(str(size), engine="sglang") == prefill
        assert _overhead.graph_capacity(str(size), None, 1, "sglang") == graph
    assert _overhead.graph_capacity("80", "a100", 4, "sglang") == 512
    config = {"hidden_size": 1024, "intermediate_size": 3072, "vocab_size": 151936}
    small = _overhead.vllm(config, False, False, 8192, max_num_seqs=1024, gpu_memory="140")
    concurrent = _overhead.vllm(config, False, False, 8192, max_num_seqs=1024, gpu_memory="140", batch_size=512)
    assert concurrent.bytes > small.bytes


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_text_workload_keeps_loaded_multimodal_weights_and_cache(tmp_path: Path, engine: str) -> None:
    write_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            **llama_tensors(),
            "visual.norm.weight": ("F32", [256]),
        },
    )
    config = llama_config()
    config["vision_config"] = {
        "hidden_size": 256,
        "intermediate_size": 1024,
        "patch_size": 16,
        "image_size": 224,
        "spatial_merge_size": 2,
    }
    write_json(str(tmp_path / "config.json"), config)
    options = dict(engine=engine, gpu_type="a100", gpu_memory="80", max_num_seqs=8)
    full = estimate(str(tmp_path), **options)
    text = estimate(str(tmp_path), text_only=True, **options)
    assert text.weights == full.weights and text.kv_cache == full.kv_cache
    assert some(text.overhead).encoder == 0 < some(full.overhead).encoder
    assert some(text.fit).required == some(full.fit).required - some(full.overhead).encoder
    assert text.text_only


def test_unhinted_runtime_uses_the_same_calibration_as_generic_ampere() -> None:
    config = {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "head_dim": 64,
    }
    unknown = _overhead.vllm(config, False, False, 8192)
    named = _overhead.vllm(config, False, False, 8192, gpu_memory="80", gpu_type="a100")
    assert unknown.bytes == named.bytes
    assert unknown.max_num_seqs == 256


def test_v3_profiles_against_native_default_moe_and_workload_references() -> None:
    from hf_mem import _config

    fixture = json.loads((Path(__file__).parent / "fixtures/overhead_crossengine_v3.json").read_text())
    errors = []
    for row in fixture["rows"]:
        config = row["config"]
        encoder = (
            0
            if row["text_only"]
            else _overhead.encoder(row["root_config"], row["processor"], row["context_length"], row["engine"])
        )
        predicted = (
            _overhead.vllm(
                config,
                bool(config.get("num_experts_per_tok")),
                "state" in _config.layer_kinds(config),
                row["tokens"],
                encoder,
                row["max_num_seqs"],
                row["tensor_parallel_size"],
                row["gpu_memory"],
                row["gpu_type"],
                row["engine"],
                context_length=row["context_length"],
            ).bytes
            * row["tensor_parallel_size"]
        )
        errors.append(abs(predicted / row["native_residual_bytes"] - 1))
    # Independent native denominators; these are training/regression references.
    # Paired text/image phases share a deployment and are not holdout replicates.
    assert len(errors) == 46
    assert median(errors) < 0.05 and max(errors) < 0.15


def test_estimate_preserves_original_positional_warnings_argument() -> None:
    from hf_mem._weights import Weights
    from hf_mem.estimate import Estimate

    result = Estimate(
        "model", None, "url", "safetensors", Weights(), {}, None, None, None, None, None, None, ["warning"]
    )
    assert result.warnings == ["warning"] and not result.text_only


def test_fused_hopper_gdn_scratch_has_a_separate_bound() -> None:
    config = {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "head_dim": 256,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
    }
    small = _overhead.vllm(config, False, True, 4096, max_num_seqs=8, gpu_type="h200", engine="sglang")
    large = _overhead.vllm(config, False, True, 8192, max_num_seqs=8, gpu_type="h200", engine="sglang")
    # Hidden/output buffers still grow. The residual fused-kernel reserve does
    # not also grow as if it retained the unfused Triton intermediate tensors.
    assert 0 < large.activations - small.activations < 0.5 * _overhead.GIB
    fallback = _overhead.vllm(
        {**config, "mamba_ssm_dtype": "bfloat16"},
        False,
        True,
        8192,
        max_num_seqs=8,
        gpu_type="h200",
        engine="sglang",
    )
    assert fallback.activations > large.activations
