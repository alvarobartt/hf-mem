import json
import re
import sys
from pathlib import Path

import pytest
from helpers import llama_config, llama_tensors, write_gguf, write_json, write_safetensors
from test_gguf import ARCH, TENSORS

from hf_mem import __version__
from hf_mem._render import _where
from hf_mem.cli import main


@pytest.fixture
def model(tmp_path):
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    return str(tmp_path)


def cli(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["hf-mem", *args])
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out, out.err


def test_table(monkeypatch, capsys, model):
    code, out, err = cli(monkeypatch, capsys, model)
    assert code == 0
    assert "TOTAL MEMORY" in out and "KV CACHE (1024 TOKENS" in out and "MODEL (" in out
    assert f"hf-mem v{__version__}" in out
    assert "\x1b[" not in out and err == ""
    assert len({len(line) for line in out.splitlines()}) == 1


def test_json(monkeypatch, capsys, model):
    code, out, _ = cli(monkeypatch, capsys, model, "--json", "--max-model-len", "10", "--batch-size", "2")
    data = json.loads(out)
    assert code == 0
    assert list(data) == ["hf_mem", "model", "revision", "format", "total", "weights", "kv_cache", "warnings"]
    assert data["kv_cache"]["max_model_len"] == 10 and data["kv_cache"]["batch_size"] == 2
    assert data["total"] == data["weights"]["bytes"] + data["kv_cache"]["bytes"]


def test_markdown(monkeypatch, capsys, model):
    code, out, _ = cli(monkeypatch, capsys, model, "--markdown")
    assert code == 0
    assert "| **Total** |" in out and "| KV cache (BF16, 1024 tokens x 1) |" in out


def test_legacy_flags(monkeypatch, capsys, model):
    code, out, err = cli(
        monkeypatch, capsys, "--model-id", model, "--experimental", "--json-output", "--details"
    )
    assert code == 0
    assert json.loads(out)["kv_cache"] is not None
    assert "--experimental" in err


def test_gguf_listing_table(monkeypatch, capsys, tmp_path):
    for quant in ("Q4_K_M", "Q8_0"):
        write_gguf(str(tmp_path / f"m-{quant}.gguf"), ARCH, TENSORS)
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path), "--gguf-file", "m-Q8_0.gguf", "--json")
    assert code == 0 and json.loads(out)["file"] == "m-Q8_0.gguf"
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path))
    assert code == 0 and "FILES" in out and "m-Q4_K_M.gguf" in out


def test_errors(monkeypatch, capsys, tmp_path):
    code, _, err = cli(monkeypatch, capsys, str(tmp_path / "missing"))
    assert code == 1 and err.startswith("hf-mem: error:")
    code, _, err = cli(monkeypatch, capsys)
    assert code == 2 and "required" in err
    code, _, _ = cli(monkeypatch, capsys, str(tmp_path), "--batch-size", "0")
    assert code == 2


def test_version(monkeypatch, capsys):
    code, out, _ = cli(monkeypatch, capsys, "--version")
    assert code == 0 and out.strip() == f"hf-mem {__version__}"


def test_titles_shorten_paths_and_commits(monkeypatch, tmp_path):
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
