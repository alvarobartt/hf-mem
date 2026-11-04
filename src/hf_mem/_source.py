import asyncio
import json
import os
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

from hf_mem._version import __version__

ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CONCURRENCY = int(os.getenv("MAX_WORKERS", 32))
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))
RETRIES = 3

_URL = re.compile(r"^/(?P<id>[^/]+/[^/]+)(?:/(?:tree|blob|resolve)/(?P<rev>[^/]+)(?:/(?P<file>.+))?)?/?$")


class HubError(RuntimeError):
    pass


def _truthy(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _hf_home() -> str:
    return os.path.expanduser(os.getenv("HF_HOME") or os.path.join("~", ".cache", "huggingface"))


def _token(token: str | None) -> str | None:
    if token or (token := os.getenv("HF_TOKEN")):
        return token
    path = os.getenv("HF_TOKEN_PATH") or os.path.join(_hf_home(), "token")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    return None


def _cache_repo(model_id: str) -> str:
    root = os.getenv("HF_HUB_CACHE") or os.path.join(_hf_home(), "hub")
    return os.path.join(os.path.expanduser(root), "models--" + model_id.replace("/", "--"))


def _cache_snapshot(model_id: str, revision: str) -> str | None:
    repo = _cache_repo(model_id)
    sha = revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        ref = os.path.join(repo, "refs", revision)
        if not os.path.isfile(ref):
            return None
        with open(ref, encoding="utf-8") as f:
            sha = f.read().strip()
    path = os.path.join(repo, "snapshots", sha)
    return path if os.path.isdir(path) else None


def parse_model(model: str, revision: str | None = None) -> tuple[str, str | None, str | None]:
    if model.startswith(("http://", "https://")):
        url = urlparse(model)
        if match := _URL.match(url.path):
            rev, file = (unquote(match[k]) if match[k] else None for k in ("rev", "file"))
            return match["id"], revision or rev, file
        raise RuntimeError(f"`{model}` is not a valid Hugging Face Hub model URL.")
    if "@" in model and not os.path.exists(model):
        model, _, rev = model.partition("@")
        revision = revision or rev
    return model, revision, None


class Progress:
    def __init__(self, callback: Callable[[int, int], None] | None = None) -> None:
        self.done, self.total, self.callback = 0, 0, callback

    def start(self) -> None:
        self.total += 1
        if self.callback:
            self.callback(self.done, self.total)

    def finish(self) -> None:
        self.done += 1
        if self.callback:
            self.callback(self.done, self.total)


def _local_revision(root: str) -> str | None:
    folder = os.path.join(root, ".cache", "huggingface", "download")
    if not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder), key=lambda n: n != "config.json.metadata"):
        if name.endswith(".metadata"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                return f.readline().strip() or None
    return None


class LocalSource:
    def __init__(self, root: str, files: dict[str, int], name: str, progress: Progress) -> None:
        self.root, self.files, self.name, self.progress = root, files, name, progress
        self.revision: str | None = None
        self.url = name

    @classmethod
    def open(
        cls, path: str, name: str | None = None, progress: Progress | None = None, recursive: bool = True
    ) -> "LocalSource":
        root, files = os.path.abspath(os.path.expanduser(path)), {}
        for directory, dirs, names in os.walk(root, followlinks=True):
            if not recursive:
                dirs.clear()
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in names:
                full = os.path.join(directory, filename)
                if not filename.startswith(".") and os.path.isfile(full):
                    files[os.path.relpath(full, root).replace(os.sep, "/")] = os.path.getsize(full)
        source = cls(root, files, name or root, progress or Progress())
        source.revision = _local_revision(root)
        return source

    async def json(self, path: str) -> Any:
        self.progress.start()
        try:
            with open(os.path.join(self.root, path), "rb") as f:
                return json.loads(f.read())
        finally:
            self.progress.finish()

    async def read(self, path: str, start: int, end: int) -> bytes:
        self.progress.start()
        try:
            with open(os.path.join(self.root, path), "rb") as f:
                f.seek(start)
                return f.read(end - start)
        finally:
            self.progress.finish()

    async def close(self) -> None:
        pass


class HubSource:
    def __init__(self, model_id: str, revision: str, token: str | None, progress: Progress) -> None:
        import httpx2

        self._httpx2 = httpx2
        self.name, self.revision, self.progress = model_id, revision, progress
        self.url = f"https://hf.co/{model_id}"
        self.files: dict[str, int] = {}
        self._oids: dict[str, str] = {}
        self._snapshot: str | None = None
        self._base = f"{ENDPOINT}/{model_id}/resolve/{quote(revision, safe='')}"
        self._headers = {
            "User-Agent": f"hf-mem/{__version__}; id={uuid4()}; model_id={model_id}; revision={revision}"
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._semaphore = asyncio.Semaphore(CONCURRENCY)
        self._client = httpx2.AsyncClient(
            limits=httpx2.Limits(max_keepalive_connections=CONCURRENCY, max_connections=CONCURRENCY),
            timeout=httpx2.Timeout(TIMEOUT),
            http2=True,
            follow_redirects=True,
        )

    async def _get(self, url: str, headers: dict[str, str] | None = None) -> Any:
        httpx2 = self._httpx2
        self.progress.start()
        async with self._semaphore:
            try:
                for attempt in range(RETRIES):
                    try:
                        response = await self._client.get(url, headers={**self._headers, **(headers or {})})
                    except httpx2.TransportError as e:
                        if attempt == RETRIES - 1:
                            raise HubError(f"Request to {url} failed: {e!r}") from e
                    else:
                        if response.status_code not in {429, 500, 502, 503, 504} or attempt == RETRIES - 1:
                            if response.is_error:
                                raise HubError(
                                    response.headers.get("x-error-message")
                                    or f"HTTP {response.status_code} for {url}"
                                )
                            return response
                    await asyncio.sleep(0.5 * 2**attempt)
            finally:
                self.progress.finish()

    async def list(self) -> None:
        url: str | None = (
            f"{ENDPOINT}/api/models/{self.name}/tree/{quote(self.revision, safe='')}?recursive=true"
        )
        commit = None
        while url:
            response = await self._get(url)
            commit = commit or response.headers.get("x-repo-commit")
            for entry in json.loads(response.content):
                if entry.get("type") == "file":
                    lfs = entry.get("lfs") or {}
                    self.files[entry["path"]] = lfs.get("size", entry.get("size", 0))
                    self._oids[entry["path"]] = lfs.get("oid") or entry.get("oid", "")
            url = response.links.get("next", {}).get("url")
        self._snapshot = _cache_snapshot(self.name, commit or self.revision)

    def _cached(self, path: str) -> str | None:
        if self._snapshot is None:
            return None
        full = os.path.join(self._snapshot, path)
        if os.path.islink(full) and os.path.basename(os.readlink(full)) == self._oids.get(path):
            return full
        return None

    async def json(self, path: str) -> Any:
        if cached := self._cached(path):
            with open(cached, "rb") as f:
                return json.loads(f.read())
        return json.loads((await self._get(f"{self._base}/{path}")).content)

    async def read(self, path: str, start: int, end: int) -> bytes:
        end = min(end, self.files.get(path, end))
        if start >= end:
            return b""
        if cached := self._cached(path):
            with open(cached, "rb") as f:
                f.seek(start)
                return f.read(end - start)
        response = await self._get(f"{self._base}/{path}", {"Range": f"bytes={start}-{end - 1}"})
        return response.content[: end - start]

    async def close(self) -> None:
        await self._client.aclose()


async def open_source(
    model: str,
    revision: str | None = None,
    token: str | None = None,
    progress: Progress | None = None,
) -> tuple[LocalSource | HubSource, str | None]:
    progress = progress or Progress()
    path = os.path.expanduser(model)
    if os.path.isfile(path):
        root = os.path.dirname(os.path.abspath(path))
        return LocalSource.open(root, progress=progress, recursive=False), os.path.basename(path)
    if os.path.isdir(path):
        return LocalSource.open(path, progress=progress), None

    model_id, revision, file = parse_model(model, revision)
    revision = revision or "main"
    if model_id.count("/") != 1:
        raise RuntimeError(f"`{model}` is neither a local path nor a Hugging Face Hub model ID (`org/name`).")

    if _truthy("HF_HUB_OFFLINE"):
        if snapshot := _cache_snapshot(model_id, revision):
            source = LocalSource.open(snapshot, name=model_id, progress=progress)
            source.revision, source.url = revision, f"https://hf.co/{model_id}"
            return source, file
        raise RuntimeError(f"`HF_HUB_OFFLINE` is set but {model_id} @ {revision} is not in the local cache.")

    hub = HubSource(model_id, revision, _token(token), progress)
    try:
        await hub.list()
    except BaseException:
        await hub.close()
        raise
    return hub, file
