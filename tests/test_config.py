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


def test_deepseek_v41_compressed_kv_cache() -> None:
    config = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "hidden_size": 64,
        "kv_lora_rank": 512,
        "sliding_window": 128,
        "compress_ratios": [0, 2, 1, 0],
        "kv_source_layer_ids": [1, 2],
        "index_source_layer_ids": [1, 2],
        "index_head_dim": 128,
    }
    size, layers = _config.kv_cache(config, "F8_E4M3", 4096, 1, batched_tokens=1024)
    sliding = 4 * _config._cdiv(127 + 2048, 32) * 32 * _config.DSV4_RECORD
    per_token = (_config.DSV4_RECORD + _config.DSV4_INDEX_RECORD) * 1.5
    assert size == int(sliding + per_token * 4096)
    assert layers["compressed"] == 2 and layers["sliding"] == 4
    assert _config.kv_dtype(config, "auto", Weights(), []) == "F8_E4M3"


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


def test_indexer_cache_only_on_indexer_layers() -> None:
    base = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "hidden_size": 64,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    plain = _config.kv_cache(base, "BF16", 1024, 1)[0]
    every = _config.kv_cache({**base, "index_head_dim": 128}, "BF16", 1024, 1)[0]
    some_layers = _config.kv_cache(
        {**base, "index_head_dim": 128, "indexer_types": ["full", "shared", "shared", "full"]}, "BF16", 1024, 1
    )[0]
    assert every - plain == 4 * 1024 * 132 and some_layers - plain == 2 * 1024 * 132
    dsa = _config.kv_cache(
        {**base, "dsa_config": {"dsa_layers": [2, 4], "index_head_dim": 128}}, "BF16", 1024, 1
    )[0]
    assert dsa - plain == 2 * 1024 * 132
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
