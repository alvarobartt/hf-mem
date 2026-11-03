import os

import pytest
from helpers import llama_config, llama_tensors, write_json, write_safetensors

from hf_mem import _source, run


@pytest.mark.parametrize(
    "model, expected",
    [
        ("Qwen/Qwen3-8B", ("Qwen/Qwen3-8B", None, None)),
        ("Qwen/Qwen3-8B@refs/pr/1", ("Qwen/Qwen3-8B", "refs/pr/1", None)),
        ("https://huggingface.co/Qwen/Qwen3-8B", ("Qwen/Qwen3-8B", None, None)),
        ("https://hf.co/Qwen/Qwen3-8B/tree/dev", ("Qwen/Qwen3-8B", "dev", None)),
        (
            "https://huggingface.co/org/m-GGUF/blob/main/q/m-Q4_K_M.gguf",
            ("org/m-GGUF", "main", "q/m-Q4_K_M.gguf"),
        ),
    ],
)
def test_parse_model(model, expected):
    assert _source.parse_model(model) == expected


def test_parse_model_rejects_other_urls():
    with pytest.raises(RuntimeError, match="not a valid"):
        _source.parse_model("https://huggingface.co/datasets")


def test_token_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert _source._token(None) is None
    (tmp_path / "token").write_text("hf_file\n")
    assert _source._token(None) == "hf_file"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert _source._token(None) == "hf_env"
    assert _source._token("hf_arg") == "hf_arg"


def _cache(tmp_path, model_id="org/model", sha="a" * 40):
    repo = tmp_path / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / sha
    write_safetensors(str(snapshot / "model.safetensors"), llama_tensors())
    write_json(str(snapshot / "config.json"), llama_config())
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(sha)
    return snapshot


def test_offline_mode_reads_the_cache(tmp_path, monkeypatch):
    _cache(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = run("org/model")

    assert (result.model, result.revision, result.url) == ("org/model", "main", "https://hf.co/org/model")
    assert result.kv_cache is not None
    assert run("org/model", revision="a" * 40).total == result.total
    with pytest.raises(RuntimeError, match="not in the local cache"):
        run("org/other")


def test_cache_snapshot_resolution(tmp_path, monkeypatch):
    snapshot = _cache(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert _source._cache_snapshot("org/model", "main") == str(snapshot)
    assert _source._cache_snapshot("org/model", "dev") is None
    assert _source._cache_snapshot("org/missing", "main") is None


def test_local_listing_skips_hidden_files(tmp_path):
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_safetensors(str(tmp_path / ".cache" / "model.safetensors"), llama_tensors())
    (tmp_path / ".hidden").write_text("x")
    source = _source.LocalSource.open(str(tmp_path))
    assert list(source.files) == ["model.safetensors"]
    assert os.path.isabs(source.name)


def test_invalid_model(tmp_path):
    with pytest.raises(RuntimeError, match="neither a local path"):
        run(str(tmp_path / "missing"))
