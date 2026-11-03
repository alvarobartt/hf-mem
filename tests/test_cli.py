import json
import sys
from pathlib import Path

import pytest
from helpers import llama_config, llama_tensors, write_gguf, write_json, write_safetensors
from test_gguf import ARCH, TENSORS

from hf_mem import __version__
from hf_mem.cli import main


@pytest.fixture
def model(tmp_path: Path) -> str:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    return str(tmp_path)


def cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int | str | None, str, str]:
    monkeypatch.setattr(sys, "argv", ["hf-mem", *args])
    try:
        main()
        code: int | str | None = 0
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out, out.err


def test_table(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, err = cli(monkeypatch, capsys, model)
    assert code == 0
    assert "TOTAL MEMORY" in out and "KV CACHE (1024 TOKENS" in out and "MODEL (" in out
    assert f"hf-mem v{__version__}" in out
    assert "\x1b[" not in out and err == ""
    assert len({len(line) for line in out.splitlines()}) == 1


def test_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "--json", "--max-model-len", "10", "--batch-size", "2")
    data = json.loads(out)
    assert code == 0
    keys = ["hf_mem", "model", "revision", "format", "total", "weights", "kv_cache", "overhead", "warnings"]
    assert list(data) == keys
    assert data["kv_cache"]["max_model_len"] == 10 and data["kv_cache"]["batch_size"] == 2
    assert data["total"] == data["weights"]["bytes"] + data["kv_cache"]["bytes"] + data["overhead"]["bytes"]


def test_gpu_fit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "2x4", "--json")
    fit = json.loads(out)["fit"]
    assert code == 0 and fit["gpus"] == 2 and fit["needed"] == 1 and fit["fits"]
    assert fit["max_model_len"] == 1024
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "4")
    assert code == 0 and "FITS ON 1 x 4.00 GiB" in out
    code, _, err = cli(monkeypatch, capsys, model, "--gpu-memory", "lots")
    assert code == 1 and "--gpu-memory" in err


def test_markdown(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "--markdown")
    assert code == 0
    assert "| **Total** |" in out and "| KV cache (BF16, 1024 tokens x 1) |" in out


def test_legacy_flags(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, err = cli(
        monkeypatch, capsys, "--model-id", model, "--experimental", "--json-output", "--details"
    )
    assert code == 0
    assert json.loads(out)["kv_cache"] is not None
    assert "--experimental" in err


def test_gguf_listing_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for quant in ("Q4_K_M", "Q8_0"):
        write_gguf(str(tmp_path / f"m-{quant}.gguf"), ARCH, TENSORS)
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path), "--gguf-file", "m-Q8_0.gguf", "--json")
    assert code == 0 and json.loads(out)["file"] == "m-Q8_0.gguf"
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path))
    assert code == 0 and "FILES" in out and "m-Q4_K_M.gguf" in out


def test_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, _, err = cli(monkeypatch, capsys, str(tmp_path / "missing"))
    assert code == 1 and err.startswith("hf-mem: error:")
    code, _, err = cli(monkeypatch, capsys)
    assert code == 2 and "required" in err
    code, _, _ = cli(monkeypatch, capsys, str(tmp_path), "--batch-size", "0")
    assert code == 2


def test_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = cli(monkeypatch, capsys, "--version")
    assert code == 0 and out.strip() == f"hf-mem {__version__}"
