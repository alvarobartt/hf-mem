import json
import re
import sys
from pathlib import Path

import pytest
from helpers import llama_config, llama_tensors, write_gguf, write_json, write_safetensors
from test_gguf import ARCH, TENSORS

from hf_mem import __version__
from hf_mem._render import _bar, _where
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
    code, out, _ = cli(monkeypatch, capsys, model, "-o", "json", "--max-model-len", "10", "--batch-size", "2")
    data = json.loads(out)
    assert code == 0
    keys = [
        "version",
        "model",
        "revision",
        "file",
        "format",
        "total",
        "weights",
        "components",
        "kv_cache",
        "overhead",
        "moe",
        "fit",
        "warnings",
    ]
    assert list(data) == keys
    assert data["version"] == f"hf-mem@{__version__}"
    assert data["kv_cache"]["max_model_len"] == 10 and data["kv_cache"]["batch_size"] == 2
    assert data["total"] == data["weights"]["bytes"] + data["kv_cache"]["bytes"] + data["overhead"]["bytes"]


def test_gpu_fit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "2x4", "-o", "json")
    fit = json.loads(out)["fit"]
    assert code == 0 and fit["gpus"] == 2 and fit["needed"] == 1 and fit["fits"]
    assert fit["max_model_len"] == 1024
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "4")
    assert code == 0 and "GPU MEMORY" in out and "/ 4.00 GiB [" in out and "│ FITS " in out
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "2x4")
    assert "FITS ON 1 GPU" in out and "/ 8.00 GiB [" in out
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "2x4", "--output", "markdown")
    assert "| Overhead (activations, graphs, runtime) |" in out
    assert "Fits on 2 x 4.00 GiB, needs 1 GPU, up to 1024 tokens." in out
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "0.5")
    assert "⚠ NEEDS 2 GPUS" in out or "⚠ DOES NOT FIT" in out
    code, _, err = cli(monkeypatch, capsys, model, "--gpu-memory", "lots")
    assert code == 1 and "--gpu-memory" in err


def test_markdown(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "--output", "markdown")
    assert code == 0
    assert "| **Total** |" in out and "| KV cache (BF16, 1024 tokens x 1) |" in out


def test_removed_flags(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    for flag in (
        "--model-id",
        "--experimental",
        "--json",
        "--markdown",
        "--json-output",
        "--details",
        "--gguf-file",
    ):
        code, _, err = cli(monkeypatch, capsys, model, flag, "x")
        assert code == 2 and "unrecognized arguments" in err


def test_gguf_listing_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for quant in ("Q4_K_M", "Q8_0"):
        write_gguf(str(tmp_path / f"m-{quant}.gguf"), ARCH, TENSORS)
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path), "--file", "m-Q8_0.gguf", "-o", "json")
    assert code == 0 and json.loads(out)["file"] == "m-Q8_0.gguf"
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path))
    assert code == 0 and "FILES" in out and "m-Q4_K_M.gguf" in out
    assert "GiB WEIGHTS +" in out and "GiB OVERHEAD)" in out


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


def test_titles_shorten_paths_and_commits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _where(str(tmp_path / "cwd" / "m"), "a" * 40) == "m @ aaaaaaa"
    assert _where(str(tmp_path / "home" / "m"), None) == "~/m"
    assert _where("https://hf.co/org/model", "main") == "https://hf.co/org/model @ main"
    long = _where("https://hf.co/org/" + "x" * 200, None)
    assert len(long) == 72 and long.startswith("...") and long.endswith("x")


def test_version_matches_pyproject() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    match = re.search(r'^version = "(.+)"$', pyproject, re.M)
    assert match is not None and match[1] == __version__


def test_file_size_weights_are_shown_with_the_kv_cache(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"\0" * 1000)
    write_json(str(tmp_path / "config.json"), llama_config())
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path))
    assert code == 0 and "MODEL (0.00 GiB)" in out and "FILE SIZE" in out and "KV CACHE" in out


def test_bars_show_small_fractions() -> None:
    assert _bar(0.0, 4) == "░░░░"
    assert _bar(0.001, 4) == "▏░░░"
    assert _bar(0.5, 4) == "██░░"
    assert _bar(0.3, 4) == "█▎░░"
    assert _bar(1.0, 4) == "████"
    assert all(len(_bar(f / 100, 7)) == 7 for f in range(101))
