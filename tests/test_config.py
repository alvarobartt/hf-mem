from typing import Any

import pytest
from helpers import kimi_config, llama_config, some

from hf_mem import _config
from hf_mem._weights import Weights


def kv(
    config: dict[str, Any], dtype: str = "BF16", length: int = 1024, batch: int = 1
) -> tuple[int, dict[str, int]]:
    return _config.kv_cache(config, dtype, length, batch)


def test_gqa() -> None:
    size, layers = kv(llama_config(head_dim=64))
    assert size == 4 * 1024 * 2 * 2 * 64 * 2
    assert layers == {"full": 4, "sliding": 0, "state": 0, "none": 0}


@pytest.mark.parametrize("passes", [1, 2, 4])
@pytest.mark.parametrize("ranks", [1, 4])
def test_repeated_layers_match_explicit_cache_slots(passes: int, ranks: int) -> None:
    config = llama_config(
        layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
        sliding_window=128,
        loop_config={"core_start": 1, "core_end": 3, "num_passes": passes},
    )
    kinds = config["layer_types"] + config["layer_types"][1:3] * (passes - 1)
    explicit = {**config, "num_hidden_layers": len(kinds), "layer_types": kinds}
    explicit.pop("loop_config")

    result = _config.kv_cache(config, "BF16", 10000, 2, 64, ranks)

    assert result == _config.kv_cache(explicit, "BF16", 10000, 2, 64, ranks)
    assert result[1] == {"full": passes + 1, "sliding": passes + 1, "state": 0, "none": 0}
    assert _config.partitioned_kv_cache(
        config, "BF16", 10000, 2, 64, ranks, 0.5
    ) == _config.partitioned_kv_cache(explicit, "BF16", 10000, 2, 64, ranks, 0.5)
    assert _config.sglang_kv_cache(config, "BF16", 10000, 2, 64, ranks) == result


def test_repeated_layer_cache_normalization_preserves_overrides_and_sharing() -> None:
    config = llama_config(
        layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
        loop_config={"core_start": 1, "core_end": 3, "num_passes": 2},
        per_layer_config={"1": {"head_dim": 64}},
        indexer_types=["none", "none", "full", "none"],
        kv_source_layers=[0, 0, 2, 3],
        num_kv_shared_layers=1,
    )

    expanded = _config.cache_config(config)

    assert expanded["num_hidden_layers"] == 6
    assert expanded["per_layer_config"] == {1: {"head_dim": 64}, 4: {"head_dim": 64}}
    assert expanded["indexer_types"] == ["none", "none", "full", "none", "none", "full"]
    assert expanded["kv_sharing_modes"] == [None, "reuse", None, "reuse", "reuse", None]
    assert expanded == _config.cache_config(expanded)
    assert config["num_hidden_layers"] == 4 and config["loop_config"]["num_passes"] == 2
    assert config["per_layer_config"] == {"1": {"head_dim": 64}}


@pytest.mark.parametrize(
    "loop",
    [
        {"core_start": -1, "core_end": 3, "num_passes": 2},
        {"core_start": 1, "core_end": 5, "num_passes": 2},
        {"core_start": 1, "core_end": 1, "num_passes": 2},
        {"core_start": 1, "core_end": 3, "num_passes": 0},
        {"core_start": True, "core_end": 3, "num_passes": 2},
    ],
)
def test_invalid_repeated_layer_geometry_is_rejected(loop: dict[str, int]) -> None:
    with pytest.raises(RuntimeError, match="Repeated layers require"):
        kv(llama_config(loop_config=loop))


def test_mla_caches_the_latent() -> None:
    config = llama_config(kv_lora_rank=512, qk_rope_head_dim=64)
    assert kv(config)[0] == 4 * 1024 * (512 + 64) * 2


def sliding_tokens(window: int, length: int = 100_000) -> int:
    tokens = min(window - 1 + 2 * _config.MAX_NUM_BATCHED_TOKENS, length)
    return (-(-tokens // 16) + (tokens < length)) * 16


def test_sliding_window_layers_reserve_the_window_plus_batched_tokens() -> None:
    pure = llama_config(sliding_window=128, layer_types=["sliding_attention"] * 4)
    assert kv(pure, length=100_000)[0] == 4 * sliding_tokens(128) * 2 * 2 * 32 * 2

    hybrid = llama_config(sliding_window=128, layer_types=["sliding_attention"] * 3 + ["full_attention"])
    assert kv(hybrid, length=100_000)[0] == (100_000 + 3 * sliding_tokens(128)) * 2 * 2 * 32 * 2
    assert kv(hybrid, length=1024)[0] == 4 * 1024 * 2 * 2 * 32 * 2


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("dtype,width", [("BF16", 2), ("F8_E4M3", 1)])
def test_partitioned_cache_reserves_both_pools(ranks: int, batch: int, dtype: str, width: int) -> None:
    config = llama_config(
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_global_key_value_heads=1,
        head_dim=32,
        global_head_dim=64,
        sliding_window=128,
        layer_types=["full_attention", "sliding_attention", "sliding_attention"],
    )
    size, layers = _config.partitioned_kv_cache(config, dtype, 8192, batch, 512, ranks, 0.5)
    full = 8192 * batch * 2 * 64 * width * ranks
    sliding = 4096 * batch * 2 * 2 * (8 // ranks) * 32 * width * ranks
    assert size == full + sliding
    assert layers == {"full": 1, "sliding": 2, "state": 0, "none": 0}


def test_partitioned_cache_protects_the_sliding_prefill_capacity() -> None:
    config = llama_config(sliding_window=128, layer_types=["full_attention"] + ["sliding_attention"] * 3)
    logical = _config.kv_cache(config, "BF16", 8192, 1, 512)[0]
    balanced = _config.partitioned_kv_cache(config, "BF16", 8192, 1, 512, 1, 0.5)[0]
    small_window_pool = _config.partitioned_kv_cache(config, "BF16", 8192, 1, 512, 1, 0.01)[0]
    assert small_window_pool > balanced > logical
    assert _config.partitioned_kv_cache(config, "BF16", 0, 1, 512, 1, 0.5)[0] == 0


def test_partitioned_cache_reserves_slots_for_each_attention_layer() -> None:
    config = llama_config(
        num_hidden_layers=4,
        sliding_window=128,
        layer_types=["full_attention", "sliding_attention"] * 2,
        num_kv_shared_layers=2,
    )
    shared, layers = _config.partitioned_kv_cache(config, "BF16", 8192, 1, 512, 2, 0.5)
    separate = _config.partitioned_kv_cache(
        {**config, "num_kv_shared_layers": 0},
        "BF16",
        8192,
        1,
        512,
        2,
        0.5,
    )[0]
    assert shared == separate
    assert shared > _config.kv_cache(config, "BF16", 8192, 1, 512, 2)[0]
    assert layers == {"full": 2, "sliding": 2, "state": 0, "none": 0}


def test_pool_ratio_does_not_change_single_attention_cache() -> None:
    config = llama_config(num_kv_shared_layers=2)
    assert _config.partitioned_kv_cache(config, "BF16", 8192, 1, 512, 2, 0.5) == _config.kv_cache(
        config, "BF16", 8192, 1, 512, 2
    )


def test_gemma3_defaults() -> None:
    config = {
        "model_type": "gemma3_text",
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "hidden_size": 3840,
        "sliding_window": 1024,
    }
    size, layers = kv(config, length=32768)
    assert layers == {"full": 2, "sliding": 10, "state": 0, "none": 0}
    assert size == (2 * 32768 + 10 * sliding_tokens(1024, 32768)) * 2 * 4 * 256 * 2


def test_sliding_window_pattern() -> None:
    config = llama_config(num_hidden_layers=6, sliding_window=64, sliding_window_pattern=3)
    assert _config.layer_kinds(_config.normalize(config)) == ["sliding", "sliding", "full"] * 2


def test_linear_attention_hybrid_pads_states_to_attention_pages() -> None:
    config = llama_config(
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        mamba_ssm_dtype="float32",
    )
    state = (2 * 2 * 16 + 4 * 16) * 3 * 2 + 4 * 16 * 16 * 4
    token = 2 * 2 * 32 * 2
    block = -(-(-(-state // token)) // 16) * 16
    size, layers = kv(config)
    assert layers == {"full": 1, "sliding": 0, "state": 3, "none": 0}
    assert size == (-(-1024 // block) + 3 * 2) * block * token


def test_mamba_hybrid_pattern() -> None:
    config = llama_config(
        num_hidden_layers=None,
        hybrid_override_pattern="M-M*",
        mamba_num_heads=8,
        mamba_head_dim=16,
        ssm_state_size=32,
        n_groups=1,
        conv_kernel=4,
    )
    size, layers = kv(config)
    assert layers == {"full": 1, "sliding": 0, "state": 2, "none": 1}
    state = (8 * 16 + 2 * 32) * 3 * 2 + 8 * 16 * 32 * 2
    token = 2 * 2 * 32 * 2
    block = -(-(-(-state // token)) // 16) * 16
    assert size == (-(-1024 // block) + 2 * 3) * block * token


def test_pure_state_model() -> None:
    config = llama_config(layer_types=["mamba"] * 4, ssm_state_size=16, mamba_expand=2, conv_kernel=4)
    inner = 2 * 256
    assert kv(config)[0] == 4 * 3 * ((inner + 2 * 16) * 3 * 2 + inner * 16 * 2)


def test_text_config_resolution() -> None:
    nested: dict[str, Any] = {
        "architectures": ["LlavaForConditionalGeneration"],
        "torch_dtype": "float16",
        "text_config": llama_config(),
    }
    assert some(_config.text_config(nested))["torch_dtype"] == "bfloat16"
    del nested["text_config"]["torch_dtype"]
    assert some(_config.text_config(nested))["torch_dtype"] == "float16"

    aliases = {
        "architectures": ["GPT2LMHeadModel"],
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 64,
        "n_positions": 128,
    }
    resolved = some(_config.text_config(aliases))
    assert (resolved["num_hidden_layers"], resolved["max_position_embeddings"]) == (2, 128)

    vision = {
        "architectures": ["X"],
        "vision_config": llama_config(hidden_size=4096),
        "decoder": llama_config(),
    }
    assert some(_config.text_config(vision))["hidden_size"] == 256

    assert _config.text_config({**llama_config(), "architectures": ["BertForMaskedLM"]}) is None


@pytest.mark.parametrize(
    "key", ["max_position_embeddings", "model_max_length", "n_positions", "max_seq_len", "seq_length"]
)
@pytest.mark.parametrize("model_type", ["llama", "qwen2", "gemma3_text"])
def test_context_aliases_override_architecture_defaults(key: str, model_type: str) -> None:
    config = llama_config(model_type=model_type)
    del config["max_position_embeddings"]
    config[key] = 65536
    assert _config.normalize(config)["max_position_embeddings"] == 65536
    assert "max_position_embeddings" not in config or key == "max_position_embeddings"


def test_explicit_legacy_context_takes_precedence_over_aliases() -> None:
    assert _config.normalize(llama_config(model_max_length=65536))["max_position_embeddings"] == 1024
    assert (
        _config.normalize(llama_config(max_position_embeddings=None, model_max_length=65536))[
            "max_position_embeddings"
        ]
        == 65536
    )


@pytest.mark.parametrize("child_key", ["dtype", "torch_dtype"])
@pytest.mark.parametrize("parent_key", ["dtype", "torch_dtype"])
def test_nested_dtype_precedes_the_parent_alias(child_key: str, parent_key: str) -> None:
    child = llama_config()
    del child["torch_dtype"]
    child[child_key] = "bfloat16"
    config = some(_config.text_config({parent_key: "float16", "text_config": child}))
    assert _config.kv_dtype(config, "auto", Weights(), []) == "BF16"


def test_local_layers_use_sliding_attention_dimensions() -> None:
    config = llama_config(
        head_dim=64,
        swa_head_dim=32,
        swa_num_key_value_heads=4,
        local_layer_ids=[0, 2],
        sliding_window_size=128,
    )
    size, layers = kv(config, length=100000)
    assert layers == {"full": 2, "sliding": 2, "state": 0, "none": 0}
    assert size == (2 * 100000 * 2 * 2 * 64 + 2 * sliding_tokens(128) * 2 * 4 * 32) * 2
    explicit = {**config, "layer_types": ["full_attention"] * 4}
    assert kv(explicit, length=100000)[1]["full"] == 4


def test_per_layer_attention_overrides_and_zero_padded_indices() -> None:
    config = llama_config(
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=128,
        head_dim=32,
        per_layer_config={
            "01": {"head_dim": 64, "num_key_value_heads": 1},
            "02": {"sliding_window": 256},
            "03": {"head_dim": None, "num_attention_heads": 4},
        },
    )
    size, layers = kv(config, length=100000, batch=2)
    expected = (sliding_tokens(128) + sliding_tokens(256)) * 2 * 2 * 32
    expected += 100000 * (2 * 1 * 64 + 2 * 2 * 64)
    assert size == expected * 2 * 2
    assert layers == {"full": 2, "sliding": 2, "state": 0, "none": 0}
    assert _config.kv_heads(config) == 2
    # TP=4 replicates the one-head layer four times and the two-head layers twice.
    replicated, _ = _config.kv_cache(config, "BF16", 100000, 2, tensor_parallel_size=4)
    assert replicated == (expected * 2 + 100000 * 2 * 1 * 64 * 2) * 2 * 2


def test_per_layer_windows_can_define_the_layer_kinds() -> None:
    config = llama_config(per_layer_config={"1": {"sliding_window": 128}, "3": {"attention_chunk_size": 256}})
    assert _config.layer_kinds(_config.normalize(config)) == ["full", "sliding", "full", "sliding"]


@pytest.mark.parametrize(
    "sharing", [{"num_kv_shared_layers": 2}, {"kv_sharing_modes": ["store", None, "reuse", "reuse"]}]
)
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_shared_layers_do_not_allocate_another_cache(sharing: dict[str, Any], heterogeneous: bool) -> None:
    config = llama_config(**sharing)
    if heterogeneous:
        config["per_layer_config"] = {"1": {"head_dim": 64}, "2": {"head_dim": 128}}
    size, layers = kv(config)
    assert layers == {"full": 2, "sliding": 0, "state": 0, "none": 2}
    assert size == 1024 * 2 * 2 * (32 + (64 if heterogeneous else 32)) * 2


def test_dummy_layers_do_not_allocate_cache() -> None:
    assert kv(llama_config(layer_types=["dummy"] * 4))[0] == 0


@pytest.mark.parametrize("explicit_modes", [False, True])
def test_shared_mla_retains_independent_indexer_caches(explicit_modes: bool) -> None:
    # Transformers #50305: a KV consumer may still own an indexer cache.
    config = llama_config(
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_source_layers=[0, 1, 0, 1],
        no_rope_layers=[1, 0, 1, 0],
        index_head_dim=128,
        indexer_types=["full", "full", "full", "shared"],
    )
    if explicit_modes:
        config["kv_sharing_modes"] = ["store", "store", "reuse_but_indexer", "reuse"]
    size, layers = kv(config)
    assert layers == {"full": 2, "sliding": 0, "state": 0, "none": 2}
    assert size == 1024 * ((512 + 64 + 512) * 2 + 3 * (128 + 4))
    assert _config.kv_cache(config, "BF16", 1024, 1, tensor_parallel_size=4)[0] == size * 4


def test_is_generative() -> None:
    assert _config.is_generative({"architectures": ["Qwen3ForCausalLM"]}, {})
    assert _config.is_generative({"architectures": ["Custom"]}, {"generation_config.json": 1})
    assert not _config.is_generative({"architectures": ["BertModel"]}, {})


@pytest.mark.parametrize(
    "architectures, files, expected",
    [
        (["BertForMaskedLM"], {}, True),
        (["XLMRobertaForSequenceClassification"], {}, True),
        (["BertForTokenClassification"], {}, True),
        (["BertModel"], {}, True),
        (["Qwen3ForCausalLM"], {"config_sentence_transformers.json": 1, "generation_config.json": 1}, True),
        (["LlamaForSequenceClassification"], {"generation_config.json": 1}, True),
        (["LlamaModel"], {"generation_config.json": 1}, False),
        (["LlamaForCausalLM"], {}, False),
        (["Dinov2Model"], {"preprocessor_config.json": 1}, False),
        (["Wav2Vec2ForSequenceClassification"], {"preprocessor_config.json": 1}, False),
    ],
)
def test_is_text_encoder(architectures: list[str], files: dict[str, int], expected: bool) -> None:
    assert _config.is_text_encoder(llama_config(architectures=architectures), files) is expected


def test_is_text_encoder_needs_a_top_level_text_config() -> None:
    clip = {"architectures": ["CLIPModel"], "text_config": llama_config()}
    assert not _config.is_text_encoder(clip, {"config_sentence_transformers.json": 1})


@pytest.mark.parametrize(
    "config, requested, expected",
    [
        ({"torch_dtype": "bfloat16"}, "auto", "BF16"),
        ({"dtype": "float16"}, "auto", "F16"),
        ({"dtype": "float16", "torch_dtype": "bfloat16"}, "auto", "F16"),
        ({"torch_dtype": "bfloat16", "quantization_config": {"quant_method": "fp8"}}, "auto", "BF16"),
        ({"quantization_config": {"kv_cache_scheme": {"num_bits": 8}}}, "auto", "F8_E4M3"),
        ({"quantization_config": {"kv_cache_quant_algo": "FP8"}}, "auto", "F8_E4M3"),
        ({"torch_dtype": "bfloat16"}, "fp8_e5m2", "F8_E5M2"),
        ({"torch_dtype": "bfloat16"}, "fp8", "F8_E4M3"),
        ({"torch_dtype": "bfloat16"}, "float16", "F16"),
    ],
)
def test_kv_dtype(config: dict[str, Any], requested: str, expected: str) -> None:
    assert _config.kv_dtype(config, requested, Weights(), []) == expected


def test_kv_dtype_falls_back_to_the_dominant_weight_dtype() -> None:
    weights, warnings = Weights(), list[str]()
    weights.add("F16", 10, 20)
    weights.add("F8_E4M3", 100, 100)
    assert _config.kv_dtype({}, "auto", weights, warnings) == "F16"
    assert warnings


def test_global_attention_layers_use_their_own_kv_heads_and_head_dim() -> None:
    config = llama_config(
        num_key_value_heads=16,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
        sliding_window=1024,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
    )
    size, _ = kv(config, length=131072)
    assert size == (131072 * 2 * 4 * 512 + 3 * sliding_tokens(1024, 131072) * 2 * 16 * 256) * 2


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize(
    "head,rope,index,dtype,width,index_width",
    [(512, 64, 128, "F8_E4M3", 584, 132), (256, 32, 64, "F8_E4M3", 296, 68), (256, 32, 64, "BF16", 512, 68)],
)
def test_shared_compressed_kv_cache(
    auxiliary: bool, head: int, rope: int, index: int, dtype: str, width: int, index_width: int
) -> None:
    config = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "hidden_size": 64,
        "head_dim": head,
        "qk_rope_head_dim": rope,
        "sliding_window": 128,
        "compress_ratios": [0, 2, 1, 0],
        "kv_source_layer_ids": [1, 2],
        "index_source_layer_ids": [1, 2],
        "index_head_dim": index,
    }
    if auxiliary:
        config["compress_ratios"].append(2)
        config["kv_source_layer_ids"].append(4)
    size, layers = _config.kv_cache(config, dtype, 4096, 1, batched_tokens=1024)
    sliding = 4 * 2176 * width
    per_token = (width + index_width) * 1.5
    assert size == int(sliding + per_token * 4096)
    assert layers["compressed"] == 2 and layers["sliding"] == 4
    assert _config.kv_dtype(config, "auto", Weights(), []) == "F8_E4M3"
    without_index, _ = _config.kv_cache(
        {**config, "index_source_layer_ids": []}, dtype, 4096, 1, batched_tokens=1024
    )
    assert size - without_index == index_width * 6144


@pytest.mark.parametrize("ranks", [1, 2, 8])
@pytest.mark.parametrize("batch", [1, 3])
def test_unshared_compressed_cache_includes_compression_states(ranks: int, batch: int) -> None:
    config = {
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "hidden_size": 64,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [4, 128, 0, 128],
    }
    size, layers = _config.kv_cache(config, "F8_E4M3", 32768, batch, 32, ranks)
    longer, _ = _config.kv_cache(config, "F8_E4M3", 65536, batch, 32, ranks)
    wider, _ = _config.kv_cache(config, "F8_E4M3", 32768, batch, 64, ranks)
    half, _ = _config.kv_cache(config, "BF16", 32768, batch, 32, ranks)

    assert size == 7956992 * ranks * batch
    assert longer - size == 6014976 * ranks * batch
    assert wider - size == 1029632 * ranks * batch
    assert half == 11927552 * ranks * batch
    assert layers == {"compressed": 2, "sliding": 3, "full": 0, "state": 0, "none": 0}
    assert config["compress_ratios"] == [4, 128, 0, 128]


def test_compressed_cache_honors_index_sources_and_full_retention() -> None:
    config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "hidden_size": 64,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [4, 128],
    }
    indexed, _ = _config.kv_cache(config, "F8_E4M3", 32768, 1, 32)
    plain, _ = _config.kv_cache({**config, "index_source_layer_ids": []}, "F8_E4M3", 32768, 1, 32)
    full, _ = _config.kv_cache(config, "F8_E4M3", 32768, 1, 32, full_sliding_window_cache=True)

    assert indexed - plain == 1245184
    assert full > indexed


def test_kimi_linear_attention_config() -> None:
    config = {
        "model_type": "kimi_linear",
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_size": 64,
        "kv_lora_rank": 32,
        "qk_rope_head_dim": 8,
        "linear_attn_config": {
            "full_attn_layers": [4],
            "kda_layers": [1, 2, 3],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        },
    }
    assert _config.layer_kinds(_config.normalize(config)) == ["state", "state", "state", "full"]
    size, layers = _config.kv_cache(config, "BF16", 1024, 1)
    assert (layers["full"], layers["state"]) == (1, 3)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_explicit_hybrid_layout_preserves_nested_recurrent_geometry(tp: int) -> None:
    layout = ["linear_attention", "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"]
    config = llama_config(
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        layer_types=layout,
        linear_attn_config={
            "full_attn_layers": [1, 3],
            "kda_layers": [0, 2],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        },
    )
    geometry = {
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
    }

    normalized = _config.normalize(config)

    assert normalized["layer_types"] == layout
    assert _config.layer_kinds(normalized) == ["state", "full", "state", "full"]
    assert {key: normalized.get(key) for key in geometry} == geometry
    assert _config._state(normalized, 2) == (3 * 4 * 16 * 3 * 2 + 4 * 16 * 16 * 4, 3)
    assert _config.normalize(normalized) == normalized
    assert "linear_num_value_heads" not in config
    expected = {**config, **geometry}
    assert _config.kv_cache(config, "BF16", 32768, 2, tensor_parallel_size=tp) == _config.kv_cache(
        expected, "BF16", 32768, 2, tensor_parallel_size=tp
    )
    assert _config.sglang_kv_cache(
        config, "BF16", 32768, 2, 128, tp, state_capacity=4
    ) == _config.sglang_kv_cache(expected, "BF16", 32768, 2, 128, tp, state_capacity=4)


def test_nested_recurrent_defaults_preserve_explicit_geometry() -> None:
    config = llama_config(
        layer_types=["linear_attention"] * 4,
        linear_attn_config={"num_heads": 8, "head_dim": 32, "short_conv_kernel_size": 4},
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=3,
        mamba_ssm_dtype="bfloat16",
    )

    normalized = _config.normalize(config)

    for key in (
        "linear_num_value_heads",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "mamba_ssm_dtype",
    ):
        assert normalized[key] == config[key]


@pytest.mark.parametrize("index,width", [(64, 68), (128, 132), (256, 264)])
def test_indexer_cache_only_on_indexer_layers(index: int, width: int) -> None:
    base = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "hidden_size": 64,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    plain = _config.kv_cache(base, "BF16", 1024, 1)[0]
    every = _config.kv_cache({**base, "index_head_dim": index}, "BF16", 1024, 1)[0]
    some_layers = _config.kv_cache(
        {**base, "index_head_dim": index, "indexer_types": ["full", "shared", "shared", "full"]},
        "BF16",
        1024,
        1,
    )[0]
    assert every - plain == 4 * 1024 * width and some_layers - plain == 2 * 1024 * width
    dsa = _config.kv_cache(
        {**base, "dsa_config": {"dsa_layers": [2, 4], "index_head_dim": index}}, "BF16", 1024, 1
    )[0]
    assert dsa - plain == 2 * 1024 * width
    assert _config.kv_dtype({**base, "dsa_config": {}}, "auto", Weights(), []) == "FP8_DS_MLA"
    assert _config.kv_cache(base, "FP8_DS_MLA", 1024, 1)[0] == 4 * 1024 * 656


# vLLM kimi-k31 (266e88c05): packed cache planner, with one KDA prefill checkpoint.
@pytest.mark.parametrize(
    "full, state, tp, dtype, length, expected",
    [
        (3, 3, 1, "FP8_DS_MLA", 131072, 426725376),
        (3, 3, 1, "FP8_DS_MLA", 1048576, 2599145472),
        (3, 3, 8, "FP8_DS_MLA", 131072, 2970451968),
        (3, 3, 8, "FP8_DS_MLA", 1048576, 22832578560),
        (3, 9, 1, "FP8_DS_MLA", 16383, 387932160),
        (3, 9, 1, "FP8_DS_MLA", 16384, 387932160),
        (3, 9, 1, "FP8_DS_MLA", 16385, 426725376),
        (3, 9, 1, "FP8_DS_MLA", 131072, 659484672),
        (3, 9, 1, "FP8_DS_MLA", 1048576, 2831904768),
        (3, 9, 1, "BF16", 131072, 790732800),
        (3, 9, 1, "F8_E4M3", 131072, 592846848),
        (3, 9, 8, "FP8_DS_MLA", 131072, 3236462592),
        (3, 9, 8, "FP8_DS_MLA", 1048576, 23098589184),
        (3, 9, 8, "BF16", 131072, 4444323840),
        (3, 9, 8, "F8_E4M3", 131072, 2604957696),
        (3, 15, 1, "FP8_DS_MLA", 131072, 892243968),
        (3, 15, 1, "FP8_DS_MLA", 1048576, 3064664064),
        (3, 15, 8, "FP8_DS_MLA", 131072, 3502473216),
        (3, 15, 8, "FP8_DS_MLA", 1048576, 23364599808),
        (24, 69, 1, "FP8_DS_MLA", 131072, 3879321600),
        (24, 69, 1, "FP8_DS_MLA", 1048576, 21258682368),
        (24, 69, 8, "FP8_DS_MLA", 131072, 21258682368),
        (24, 69, 8, "FP8_DS_MLA", 1048576, 160293568512),
    ],
)
def test_kimi_sparse_cache_matches_vllm_rows(
    full: int, state: int, tp: int, dtype: str, length: int, expected: int
) -> None:
    config = kimi_config(full, state)
    size, layers = _config.kv_cache(config, dtype, length, 1, tensor_parallel_size=tp)
    assert size == expected
    assert (layers["full"], layers["state"]) == (full, state)
    assert _config.kv_cache(config, dtype, length, 2, tensor_parallel_size=tp)[0] == 2 * expected


def test_kimi_dense_cache_keeps_the_prefill_checkpoint() -> None:
    config = kimi_config(sparse=False)
    assert _config.kv_cache(config, "BF16", 131072, 1)[0] == 637009920
    assert _config.kv_cache(config, "BF16", 1048576, 1)[0] == 3822059520


def test_full_sliding_window_cache_matches_observed_sglang_gemma_pool_geometry() -> None:
    config = {
        "model_type": "gemma3_text",
        "hidden_size": 2560,
        "num_hidden_layers": 34,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "sliding_window": 1024,
        "sliding_window_pattern": 6,
    }
    bounded, kinds = _config.kv_cache(config, "BF16", 32768, 1, batched_tokens=8192)
    full, same_kinds = _config.kv_cache(
        config, "BF16", 32768, 1, batched_tokens=8192, full_sliding_window_cache=True
    )
    assert full == 34 * 32768 * 2 * 4 * 256 * 2
    assert full > bounded and same_kinds == kinds and kinds["sliding"] == 29
    assert 4599193600 == 34 * 33025 * 2 * 4 * 256 * 2
    assert abs(full / 4599193600 - 1) < 0.01
    assert (
        _config.kv_cache(config, "BF16", 32768, 2, tensor_parallel_size=8, full_sliding_window_cache=True)[0]
        == full * 4
    )


def test_full_window_extent_does_not_change_pure_full_attention_or_state_caches() -> None:
    for config in [llama_config(), llama_config(layer_types=["mamba"] * 4, ssm_state_size=16)]:
        assert _config.kv_cache(config, "BF16", 10000, 1) == _config.kv_cache(
            config, "BF16", 10000, 1, full_sliding_window_cache=True
        )


def test_sglang_gdn_state_slots_shard_independently_from_attention_heads() -> None:
    config = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "layer_types": ["linear_attention"] * 18 + ["full_attention"] * 6,
    }
    for ranks, native_pool in [(1, 435965952), (2, 435965952), (4, 539787264)]:
        total, kinds = _config.sglang_kv_cache(config, "BF16", 8192, 1, 4096, ranks, state_capacity=16)
        extra_attention = 257 * 6 * 2 * 2 * 256 * 2 * max(1, ranks // 2)
        assert total == native_pool - extra_attention
        assert kinds == {"full": 6, "sliding": 0, "state": 18, "none": 0}


def test_explicit_local_layer_ids_and_model_context_aliases() -> None:
    config = llama_config()
    del config["max_position_embeddings"]
    config.update(model_max_length=1048576, local_layer_ids=[0, 1, 3], sliding_window_size=1024)
    normalized = _config.normalize(config)
    assert normalized["max_position_embeddings"] == 1048576 and normalized["sliding_window"] == 1024
    assert _config.layer_kinds(normalized) == ["sliding", "sliding", "full", "sliding"]
    size, layers = _config.kv_cache(config, "BF16", 262144, 1)
    assert layers == {"full": 1, "sliding": 3, "state": 0, "none": 0}
    assert size < _config.kv_cache(config, "BF16", 262144, 1, full_sliding_window_cache=True)[0]
    assert (
        _config.layer_kinds(_config.normalize({**config, "layer_types": ["full_attention"] * 4}))
        == ["full"] * 4
    )
    for invalid in [[-1], [4], [True], ["0"]]:
        assert _config.layer_kinds(_config.normalize({**config, "local_layer_ids": invalid})) == ["full"] * 4


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("dtype", ["BF16", "F8_E4M3"])
def test_short_conv_state_retains_local_kv_and_replicated_hidden_streams(
    ranks: int, batch: int, dtype: str
) -> None:
    config = llama_config(sconv_kernel_size=4, head_dim=32)
    without = {**config, "sconv_kernel_size": 0}
    actual = _config.kv_cache(config, dtype, 1024, batch, tensor_parallel_size=ranks)[0]
    attention = _config.kv_cache(without, dtype, 1024, batch, tensor_parallel_size=ranks)[0]
    local_heads = -(-config["num_key_value_heads"] // ranks)
    per_layer = (4 * local_heads * 32 + 2 * config["hidden_size"]) * 3 * 2
    assert actual - attention == config["num_hidden_layers"] * per_layer * ranks * batch


@pytest.mark.parametrize("config", [{"dtype": "float32"}, {"torch_dtype": "float32"}, {}])
def test_fp32_auto_cache_uses_a_shared_16_bit_reference(config: dict[str, Any]) -> None:
    weights = Weights()
    weights.add("F32", 10, 40)
    assert _config.kv_dtype(config, "auto", weights, []) == "F16"
    assert _config.kv_dtype(config, "bfloat16", weights, []) == "BF16"


def test_tensor_parallel_replication_uses_each_attention_types_head_count() -> None:
    config = llama_config(
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        head_dim=128,
        global_head_dim=256,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=128,
    )
    single = _config.kv_cache(config, "BF16", 4096, 1, 256)[0]
    parallel = _config.kv_cache(config, "BF16", 4096, 1, 256, tensor_parallel_size=8)[0]
    full_cache = 4096 * 4 * 256 * 2 * 2
    assert parallel == single + full_cache
    assert parallel < 2 * single
