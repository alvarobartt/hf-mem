import pytest

from hf_mem import _overhead

GIB = _overhead.GIB


def test_vllm_overhead():
    dense = _overhead.vllm({"vocab_size": 128256, "intermediate_size": 14336, "hidden_size": 4096}, moe=False)
    assert dense.activations == 1024 * 128256 * 8 + 4096 * 14336 * 8
    assert (dense.cuda_graphs, dense.runtime) == (int(0.25 * GIB), int(0.75 * GIB))
    moe = _overhead.vllm({"vocab_size": 1000, "moe_intermediate_size": 768, "num_experts_per_tok": 8}, moe=True)
    assert moe.activations == 1024 * 1000 * 8 + 4096 * 768 * 8 * 8
    assert moe.cuda_graphs == int(0.36 * GIB)


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
    assert (fit.needed, fit.fits) == (2, False)
    assert _overhead.fit(weights=30 * GIB, gpu_memory="2x20", utilization=0.9, **args).fits
    assert (
        _overhead.fit(
            weights=30 * GIB, gpu_memory="20", utilization=0.9, **{**args, "context_length": 5}
        ).max_model_len
        == 5
    )
    assert _overhead.fit(weights=10**15, gpu_memory="20", utilization=0.9, **args).needed is None
