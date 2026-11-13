import asyncio
from pathlib import Path
from typing import Any

import pytest

from hf_mem import arun, run


@pytest.mark.parametrize(
    "options, message",
    [
        ({"batch_size": 0}, "must be positive"),
        ({"batch_size": -1}, "must be positive"),
        ({"max_num_seqs": 0}, "must be positive"),
        ({"max_num_seqs": -1}, "must be positive"),
        ({"batch_size": 2, "max_num_seqs": 1}, "at least"),
        ({"max_model_len": 0}, "must be positive"),
        ({"max_model_len": -1}, "must be positive"),
        ({"max_num_batched_tokens": 0}, "must be positive"),
        ({"max_num_batched_tokens": -1}, "must be positive"),
        ({"gpu_memory_utilization": 0}, "must be in"),
        ({"gpu_memory_utilization": -0.1}, "must be in"),
        ({"gpu_memory_utilization": 1.1}, "must be in"),
        ({"gpu_memory_utilization": float("nan")}, "must be in"),
        ({"gpu_memory_utilization": float("inf")}, "must be in"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_api_rejects_invalid_options_before_opening_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], message: str, asynchronous: bool
) -> None:
    async def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Invalid options must fail before opening the model")

    monkeypatch.setattr("hf_mem.estimate.open_source", unexpected)
    with pytest.raises(RuntimeError, match=message):
        if asynchronous:
            asyncio.run(arun(str(tmp_path), **options))
        else:
            run(str(tmp_path), **options)
