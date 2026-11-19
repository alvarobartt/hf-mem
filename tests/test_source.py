import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from helpers import estimate, llama_config, llama_tensors, write_json, write_safetensors

from hf_mem import Estimate, _source, arun


def _transport(monkeypatch: pytest.MonkeyPatch, handler: Callable[..., Any]) -> list[httpx2.Request]:
    requests: list[httpx2.Request] = []
    client = httpx2.AsyncClient

    async def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        response = handler(request)
        return await response if hasattr(response, "__await__") else response

    def create(**kwargs: Any) -> httpx2.AsyncClient:
        return client(**kwargs, transport=httpx2.MockTransport(handle))

    monkeypatch.setattr(httpx2, "AsyncClient", create)
    return requests


def _tree(paths: list[str], lfs: bool = True, size: int = 1024) -> httpx2.Response:
    return httpx2.Response(
        200,
        json=[
            {"type": "file", "path": path, "size": size, **({"lfs": {"size": size}} if lfs else {})}
            for path in paths
        ],
    )


@pytest.mark.parametrize(
    "model, expected",
    [
        ("Qwen/Qwen3-8B", ("Qwen/Qwen3-8B", None, None)),
        ("Qwen/Qwen3-8B@refs/pr/1", ("Qwen/Qwen3-8B", "refs/pr/1", None)),
        ("https://huggingface.co/Qwen/Qwen3-8B", ("Qwen/Qwen3-8B", None, None)),
        ("https://hf.co/Qwen/Qwen3-8B/tree/dev", ("Qwen/Qwen3-8B", "dev", None)),
        ("https://hf.co/org/m/tree/refs%2Fpr%2F1", ("org/m", "refs/pr/1", None)),
        ("org/m-GGUF:Q4_K_M", ("org/m-GGUF", None, "Q4_K_M")),
        ("org/m-GGUF@dev:UD-Q4_K_XL", ("org/m-GGUF", "dev", "UD-Q4_K_XL")),
        ("https://hf.co/org/m/blob/main/a%20b.gguf", ("org/m", "main", "a b.gguf")),
        (
            "https://huggingface.co/org/m-GGUF/blob/main/q/m-Q4_K_M.gguf",
            ("org/m-GGUF", "main", "q/m-Q4_K_M.gguf"),
        ),
    ],
)
def test_parse_model(model: str, expected: tuple[str, str | None, str | None]) -> None:
    assert _source.parse_model(model) == expected


def test_parse_model_rejects_other_urls() -> None:
    with pytest.raises(RuntimeError, match="not a valid"):
        _source.parse_model("https://huggingface.co/datasets")


def test_token_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert _source._token(None) is None
    (tmp_path / "token").write_text("hf_file\n")
    assert _source._token(None) == "hf_file"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert _source._token(None) == "hf_env"
    assert _source._token("hf_arg") == "hf_arg"


def _cache(tmp_path: Path, model_id: str = "org/model", sha: str = "a" * 40) -> Path:
    repo = tmp_path / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / sha
    write_safetensors(str(snapshot / "model.safetensors"), llama_tensors())
    write_json(str(snapshot / "config.json"), llama_config())
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(sha)
    return snapshot


def test_offline_mode_reads_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cache(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    result = estimate("org/model")

    assert (result.model, result.revision, result.url) == ("org/model", "main", "https://hf.co/org/model")
    assert result.kv_cache is not None
    assert estimate("org/model", revision="a" * 40).total == result.total
    with pytest.raises(RuntimeError, match="not in the local cache"):
        estimate("org/other")


def test_cache_snapshot_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = _cache(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert _source._cache_snapshot("org/model", "main") == str(snapshot)
    assert _source._cache_snapshot("org/model", "dev") is None
    assert _source._cache_snapshot("org/missing", "main") is None


def test_local_listing_skips_hidden_files(tmp_path: Path) -> None:
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


def test_invalid_model(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        estimate(str(tmp_path / "missing"))


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
    result = estimate("org/model")
    assert result.kv_cache is not None and len(requested) == 1

    tree[1]["oid"] = "stale"
    requested.clear()
    assert estimate("org/model").kv_cache is None
    assert [url.rsplit("/", 1)[1] for url in requested[1:]] == ["config.json"]


def test_range_reads_reuse_cdn_urls_without_forwarding_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(["model.safetensors"])
        if request.url.host == "cdn.example":
            assert "authorization" not in request.headers
            assert request.headers["user-agent"].startswith("hf-mem/")
            start, end = map(int, request.headers["range"][6:].split("-"))
            return httpx2.Response(206, content=bytes(range(start, end + 1)))
        assert request.method == "HEAD" and "range" not in request.headers
        assert request.headers["authorization"] == "Bearer hf_test"
        return httpx2.Response(302, headers={"location": "https://cdn.example/model?signature=test"})

    requests = _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", "hf_test", _source.Progress())
        try:
            await source.list()
            assert await source.read("model.safetensors", 0, 8) == bytes(range(8))
            chunks = await asyncio.gather(*(source.read("model.safetensors", n, n + 8) for n in (8, 16, 24)))
            assert chunks == [bytes(range(n, n + 8)) for n in (8, 16, 24)]
            assert [r.method for r in requests] == ["GET", "HEAD", "GET", "GET", "GET", "GET"]
            assert source.progress.done == source.progress.total == 5
        finally:
            await source.close()

    asyncio.run(check())


def test_range_reads_follow_relative_hub_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(["model.gguf"])
        if request.url.path == "/intermediate":
            assert request.method == "HEAD"
            assert request.headers["authorization"] == "Bearer hf_test"
            return httpx2.Response(302, headers={"location": "https://cdn.example/model"})
        if request.url.host == "cdn.example":
            assert request.method == "GET" and "authorization" not in request.headers
            return httpx2.Response(206, content=b"abcdefgh")
        return httpx2.Response(307, headers={"location": "/intermediate"})

    requests = _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", "hf_test", _source.Progress())
        try:
            await source.list()
            await source.read("model.gguf", 0, 8)
            await source.read("model.gguf", 8, 16)
            assert [r.method for r in requests] == ["GET", "HEAD", "HEAD", "GET", "GET"]
        finally:
            await source.close()

    asyncio.run(check())


def test_non_lfs_range_reads_need_no_head_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(["small.onnx"], lfs=False)
        assert request.method == "GET" and request.headers["range"] == "bytes=0-7"
        return httpx2.Response(206, content=b"abcdefgh")

    requests = _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            await source.list()
            assert await source.read("small.onnx", 0, 8) == b"abcdefgh"
            assert len(requests) == 2
        finally:
            await source.close()

    asyncio.run(check())


def test_expired_cdn_urls_are_refreshed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    resolutions, downloads = 0, 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal resolutions, downloads
        if "/api/models/" in request.url.path:
            return _tree(["model.gguf"])
        if request.method == "HEAD":
            resolutions += 1
            return httpx2.Response(302, headers={"location": f"https://cdn.example/model?v={resolutions}"})
        if request.url.host == "cdn.example":
            downloads += 1
            assert "authorization" not in request.headers
            return httpx2.Response(
                403 if downloads > 1 and request.url.params["v"] == "1" else 206, content=b"abcdefgh"
            )
        return httpx2.Response(200, json={"ok": True})

    _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", "hf_test", _source.Progress())
        try:
            await source.list()
            assert await source.read("model.gguf", 0, 8) == b"abcdefgh"
            assert await source.read("model.gguf", 8, 16) == b"abcdefgh"
            assert await source.json("config.json") == {"ok": True}
            assert resolutions == 2 and downloads == 3
        finally:
            await source.close()

    asyncio.run(check())


def test_cdn_denial_does_not_retry_indefinitely_or_expose_signed_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(["model.gguf"])
        if request.method == "HEAD":
            return httpx2.Response(302, headers={"location": "https://cdn.example/model?secret=signature"})
        return httpx2.Response(403)

    requests = _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            await source.list()
            with pytest.raises(_source.HubError, match="download server") as error:
                await source.read("model.gguf", 0, 8)
            assert "signature" not in str(error.value)
            assert len(requests) == 5
        finally:
            await source.close()

    asyncio.run(check())


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("lfs", [False, True])
def test_gated_shards_stop_after_the_first_denial(
    monkeypatch: pytest.MonkeyPatch, status: int, lfs: bool
) -> None:
    paths = [f"model-{n:05}-of-00064.safetensors" for n in range(1, 65)]

    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(paths, lfs=lfs)
        if request.url.path.endswith("config.json"):
            return httpx2.Response(200, json={})
        return httpx2.Response(status, headers={"x-error-message": "Access restricted"})

    requests = _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            await source.list()
            await source.json("config.json")
            with pytest.raises(_source.HubError, match="gated") as error:
                await _source.gather(*(source.read(path, 0, 8) for path in paths), return_exceptions=True)
            assert error.value.status == status
            assert len(requests) == 3
            with pytest.raises(_source.HubError):
                await source.json("another.json")
            assert len(requests) == 3
            assert source.progress.done == source.progress.total
        finally:
            await source.close()

    asyncio.run(check())


@pytest.mark.parametrize("format", ["safetensors", "gguf", "onnx"])
@pytest.mark.parametrize("denied_config", [False, True])
def test_estimation_propagates_access_denials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, format: str, denied_config: bool
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    paths = (
        [f"model-{n:05}-of-00064.safetensors" for n in range(1, 65)]
        if format == "safetensors"
        else [f"model-{n}.{format}" for n in range(64)]
    )

    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(["config.json", *paths], lfs=False)
        if request.url.path.endswith("config.json") and not denied_config:
            return httpx2.Response(200, json={})
        return httpx2.Response(403, headers={"x-error-message": "Access restricted"})

    requests = _transport(monkeypatch, handle)
    progress = _source.Progress()
    with pytest.raises(_source.HubError, match="gated"):
        asyncio.run(arun("org/model", format=format, progress=progress))
    denied = [
        r for r in requests if not r.url.path.startswith("/api/") and (denied_config or "range" in r.headers)
    ]
    assert len(denied) == 1
    assert progress.done == progress.total


def test_gather_cancels_pending_reads_and_preserves_optional_errors() -> None:
    async def check() -> None:
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def slow() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def denied() -> None:
            await started.wait()
            raise _source.HubError("gated", 403)

        async def optional() -> None:
            raise ValueError("invalid optional metadata")

        with pytest.raises(_source.HubError, match="gated"):
            await _source.gather(slow(), denied(), return_exceptions=True)
        assert cancelled.is_set()
        assert isinstance((await _source.gather(optional(), return_exceptions=True))[0], ValueError)

    asyncio.run(check())


def test_denial_stops_queued_requests_after_a_successful_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        return (
            httpx2.Response(206, content=b"abcdefgh")
            if request.url.path.endswith("first")
            else httpx2.Response(403)
        )

    requests = _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        source._semaphore = asyncio.Semaphore(1)
        try:
            await source.read("first", 0, 8)
            with pytest.raises(_source.HubError):
                await _source.gather(*(source.read(f"shard-{n}", 0, 8) for n in range(64)))
            assert len(requests) == 2
            assert source.progress.done == source.progress.total
        finally:
            await source.close()

    asyncio.run(check())


def test_parallel_reads_resume_after_the_first_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def check() -> None:
        started = 0
        parallel = asyncio.Event()

        async def handle(request: httpx2.Request) -> httpx2.Response:
            nonlocal started
            started += 1
            if started > 1:
                if started == 5:
                    parallel.set()
                await parallel.wait()
            return httpx2.Response(206, content=b"abcdefgh")

        _transport(monkeypatch, handle)
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            results = await asyncio.wait_for(
                _source.gather(*(source.read(f"shard-{n}", 0, 8) for n in range(5))), timeout=1
            )
            assert results == [b"abcdefgh"] * 5
        finally:
            await source.close()

    asyncio.run(check())


def test_cancellation_cleans_up_pending_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    async def check() -> None:
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def handle(request: httpx2.Request) -> httpx2.Response:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return httpx2.Response(206)

        requests = _transport(monkeypatch, handle)
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            task = asyncio.create_task(_source.gather(*(source.read(f"shard-{n}", 0, 8) for n in range(64))))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set() and len(requests) == 1
            assert source.progress.done == source.progress.total
        finally:
            await source.close()

    asyncio.run(check())


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, None])
def test_transient_download_errors_still_retry(monkeypatch: pytest.MonkeyPatch, status: int | None) -> None:
    attempts = 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            if status is None:
                raise httpx2.ConnectError("connection failed")
            return httpx2.Response(status)
        return httpx2.Response(206, content=b"abcdefgh")

    async def sleep(delay: float) -> None:
        pass

    _transport(monkeypatch, handle)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            assert await source.read("model.gguf", 0, 8) == b"abcdefgh"
            assert attempts == 3
            assert source.progress.done == source.progress.total == 1
        finally:
            await source.close()

    asyncio.run(check())


@pytest.mark.parametrize("status", [200, 404])
def test_optional_config_errors_do_not_prevent_weight_estimation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(str(path), llama_tensors())
    data = path.read_bytes()

    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(["config.json", "model.safetensors"], lfs=False, size=len(data))
        if request.url.path.endswith("config.json"):
            return httpx2.Response(status, content=b"invalid json")
        return httpx2.Response(206, content=data)

    _transport(monkeypatch, handle)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    result = asyncio.run(arun("org/model"))
    assert isinstance(result, Estimate)
    assert result.weights.params > 0
    assert any("config.json" in warning for warning in result.warnings)


def test_hub_endpoints_without_head_support_fall_back_to_get(monkeypatch: pytest.MonkeyPatch) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if "/api/models/" in request.url.path:
            return _tree(["model.gguf"])
        if request.method == "HEAD":
            return httpx2.Response(405)
        return httpx2.Response(206, content=b"abcdefgh")

    requests = _transport(monkeypatch, handle)

    async def check() -> None:
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            await source.list()
            assert await source.read("model.gguf", 0, 8) == b"abcdefgh"
            assert await source.read("model.gguf", 8, 16) == b"abcdefgh"
            assert [r.method for r in requests] == ["GET", "HEAD", "GET", "GET"]
        finally:
            await source.close()

    asyncio.run(check())


def test_parallel_reads_resolve_a_new_file_once(monkeypatch: pytest.MonkeyPatch) -> None:
    async def check() -> None:
        head_started, release_head = asyncio.Event(), asyncio.Event()

        async def handle(request: httpx2.Request) -> httpx2.Response:
            if "/api/models/" in request.url.path:
                return _tree(["model.gguf"])
            if request.method == "HEAD":
                head_started.set()
                await release_head.wait()
                return httpx2.Response(302, headers={"location": "https://cdn.example/model"})
            return httpx2.Response(206, content=b"abcdefgh")

        requests = _transport(monkeypatch, handle)
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            await source.list()
            await source.read("warmup", 0, 8)
            task = asyncio.create_task(_source.gather(*(source.read("model.gguf", n, n + 8) for n in range(8))))
            await head_started.wait()
            release_head.set()
            assert await task == [b"abcdefgh"] * 8
            assert sum(r.method == "HEAD" for r in requests) == 1
        finally:
            await source.close()

    asyncio.run(check())


def test_denial_stops_a_pending_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async def check() -> None:
        retrying, release_retry = asyncio.Event(), asyncio.Event()

        def handle(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(503 if request.url.path.endswith("retry.json") else 403)

        async def sleep(delay: float) -> None:
            retrying.set()
            await release_retry.wait()

        requests = _transport(monkeypatch, handle)
        monkeypatch.setattr(asyncio, "sleep", sleep)
        source = _source.HubSource("org/model", "main", None, _source.Progress())
        try:
            task = asyncio.create_task(source.json("retry.json"))
            await retrying.wait()
            with pytest.raises(_source.HubError) as denied:
                await source.json("denied.json")
            release_retry.set()
            with pytest.raises(_source.HubError) as retried:
                await task
            assert retried.value is denied.value
            assert len(requests) == 2
            assert source.progress.done == source.progress.total
        finally:
            await source.close()

    asyncio.run(check())
