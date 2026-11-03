import pytest

from hf_mem import _overhead

GIB = _overhead.GIB


def test_vllm_overhead():
    config = {"vocab_size": 128256, "intermediate_size": 14336, "hidden_size": 4096}
    dense = _overhead.vllm(config, moe=False, tokens=8192)
    assert dense.activations == 1024 * 128256 * 8 + 8192 * 14336 * 16
    assert (dense.cuda_graphs, dense.runtime) == (int(0.25 * GIB), int(0.75 * GIB))
    moe = _overhead.vllm(
        {"vocab_size": 1000, "moe_intermediate_size": 768, "num_experts_per_tok": 8}, moe=True, tokens=4096
    )
    assert moe.activations == 1024 * 1000 * 8 + 4096 * 768 * 8 * 16
    assert moe.cuda_graphs == int(0.36 * GIB)


def test_batched_tokens_follow_the_gpu_size():
    assert _overhead.batched_tokens(None) == _overhead.batched_tokens("8x80") == 8192
    assert _overhead.batched_tokens("24") == 2048


@pytest.mark.parametrize(
    "value, expected",
    [("80", (None, 80 * GIB)), ("8x80", (8, 80 * GIB)), ("2x24GiB", (2, 24 * GIB)), ("0.5", (None, GIB // 2))],
)
def test_parse_gpu_memory(value, expected):
    assert _overhead.parse_gpu_memory(value) == expected


@pytest.mark.parametrize("value", ["", "x80", "0x80", "8x", "eighty", "-1"])
def test_parse_gpu_memory_errors(value):
    with pytest.raises(RuntimeError):
        _overhead.parse_gpu_memory(value)


def test_fit():
    args = dict(kv_fixed=0, kv_per_token=1024, max_model_len=1000, context_length=10**9, overhead=GIB)
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


def test_fit_replicates_kv_heads_across_gpus():
    args = dict(kv_fixed=0, kv_per_token=GIB / 1000, max_model_len=1000, context_length=10**9, overhead=0)
    split = _overhead.fit(weights=0, gpu_memory="8x10", utilization=1.0, kv_heads=8, **args)
    replicated = _overhead.fit(weights=0, gpu_memory="8x10", utilization=1.0, kv_heads=2, **args)
    assert split.max_model_len == 4 * replicated.max_model_len
    assert _overhead.fit(
        weights=0, gpu_memory="8x10", utilization=1.0, kv_heads=None, **args
    ).max_model_len == (split.max_model_len)


def test_batched_tokens_override():
    assert _overhead.batched_tokens("8x80", 2048) == 2048
