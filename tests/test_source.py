import hashlib
import json
import os
from pathlib import Path
from typing import Any

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
    assert source.revision is None


def test_local_dir_revision_from_download_metadata(tmp_path: Path) -> None:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "model.safetensors.metadata").write_text("c" * 40 + "\netag\n1.0\n")
    (metadata / "config.json.metadata").write_text("d" * 40 + "\netag\n1.0\n")
    assert _source.LocalSource.open(str(tmp_path)).revision == "d" * 40


def test_invalid_model(tmp_path):
    with pytest.raises(RuntimeError, match="neither a local path"):
        run(str(tmp_path / "missing"))


class _Response:
    def __init__(self, content: bytes, headers: dict[str, str]) -> None:
        self.content, self.headers, self.links = content, headers, dict[str, Any]()


def _hub_cache(tmp_path: Path, sha: str) -> dict[str, str]:
    staging, repo = tmp_path / "staging", tmp_path / "models--org--model"
    write_safetensors(str(staging / "model.safetensors"), llama_tensors())
    write_json(str(staging / "config.json"), llama_config())
    oids = {}
    for path in ("model.safetensors", "config.json"):
        oid = hashlib.sha256((staging / path).read_bytes()).hexdigest()
        (repo / "blobs").mkdir(parents=True, exist_ok=True)
        (staging / path).rename(repo / "blobs" / oid)
        (repo / "snapshots" / sha).mkdir(parents=True, exist_ok=True)
        (repo / "snapshots" / sha / path).symlink_to(Path("..", "..", "blobs", oid))
        oids[path] = oid
    return oids


def test_online_mode_reads_validated_files_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "b" * 40
    oids = _hub_cache(tmp_path, sha)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    tree = [
        {
            "type": "file",
            "path": "model.safetensors",
            "size": 1,
            "lfs": {"oid": oids["model.safetensors"], "size": 1 << 30},
        },
        {"type": "file", "path": "config.json", "size": 1, "oid": oids["config.json"]},
    ]
    requested: list[str] = []

    async def get(self: _source.HubSource, url: str, headers: dict[str, str] | None = None) -> _Response:
        requested.append(url)
        if "/api/models/" not in url:
            raise _source.HubError(f"unexpected request to {url}")
        return _Response(json.dumps(tree).encode(), {"x-repo-commit": sha})

    monkeypatch.setattr(_source.HubSource, "_get", get)
    result = run("org/model")
    assert result.kv_cache is not None and len(requested) == 1

    tree[1]["oid"] = "stale"
    requested.clear()
    assert run("org/model").kv_cache is None
    assert [url.rsplit("/", 1)[1] for url in requested[1:]] == ["config.json"]
