"""Sample native llama.cpp deployments without inferring GPU memory from GGUF disk size."""

import hashlib
import ctypes
import ctypes.util
import importlib.metadata
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pynvml

LOAD_ONLY = os.environ.get("HFMEM_LOAD_ONLY") == "1"
VERBOSITY = os.environ.get("HFMEM_LOG_VERBOSITY", "4")


def emit(kind: str, **data: Any) -> None:
    print("HFMEM_RESULT " + json.dumps({"kind": kind, "time": time.time(), **data}), flush=True)


def memory() -> dict[str, Any]:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "device_used_bytes": info.used,
        "device_total_bytes": info.total,
        "processes": [
            {"pid": p.pid, "used_bytes": p.usedGpuMemory}
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ],
    }


class MemoryBudget:
    """Hold a declared CUDA allocation so a native server sees bounded free VRAM."""

    def __init__(self, target: int) -> None:
        self.pointer = ctypes.c_void_p()
        candidates = [ctypes.util.find_library("cudart"), "libcudart.so.13", "libcudart.so.12", "libcudart.so"]
        for name in candidates:
            if not name:
                continue
            try:
                self.runtime = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("CUDA runtime library is unavailable for the bounded-memory probe")
        self.check(self.runtime.cudaSetDevice(0))
        self.check(self.runtime.cudaFree(ctypes.c_void_p()))
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.check(self.runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)))
        before = free.value
        reserve = before - target
        if reserve <= 0:
            raise RuntimeError("Requested free-memory budget exceeds available CUDA memory")
        self.check(self.runtime.cudaMalloc(ctypes.byref(self.pointer), ctypes.c_size_t(reserve)))
        self.check(self.runtime.cudaMemset(self.pointer, 0, ctypes.c_size_t(reserve)))
        self.check(self.runtime.cudaDeviceSynchronize())
        self.check(self.runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)))
        self.audit = {
            "target_free_bytes": target,
            "cuda_total_bytes": total.value,
            "cuda_free_before_bytes": before,
            "held_allocation_bytes": reserve,
            "cuda_free_after_bytes": free.value,
            "reservation_pid": os.getpid(),
        }
        if not 0 < free.value <= target + (2 << 20):
            self.close()
            raise RuntimeError("Bounded-memory reservation did not establish the declared limit")

    @staticmethod
    def check(code: int) -> None:
        if code:
            raise RuntimeError(f"CUDA reservation API failed with code {code}")

    def close(self) -> None:
        self.check(self.runtime.cudaFree(self.pointer))
        self.check(self.runtime.cudaDeviceReset())


def execute(case: dict[str, Any]) -> None:
    context = int(case.get("context_length", 32768))
    parallel = int(case.get("parallel", 1))
    prompt_tokens = int(case.get("prompt_tokens", context // parallel - 8))
    target = Path("/tmp") / case["file"]
    source = f"https://huggingface.co/{case['model']}/resolve/{case['revision']}/{case['file']}"
    digest = hashlib.sha256()
    with urllib.request.urlopen(source, timeout=120) as response, target.open("wb") as file:
        while chunk := response.read(8 << 20):
            digest.update(chunk)
            file.write(chunk)
    assert digest.hexdigest() == case["lfs_sha256"], "Downloaded file does not match pinned LFS SHA"
    emit("download", case=case, file_sha256=digest.hexdigest(), file_size=target.stat().st_size)
    command = [
        "/app/llama-server",
        "-lv",
        VERBOSITY,
        "-m",
        str(target),
        "-ngl",
        "999",
        "-c",
        str(context),
        "-np",
        str(parallel),
        "-b",
        "2048",
        "-ub",
        "512",
        "-fa",
        "auto",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        *case.get("extra", []),
    ]
    deadline = time.monotonic() + 30
    while memory()["processes"] and time.monotonic() < deadline:
        time.sleep(1)
    baseline = memory()
    assert not baseline["processes"], "Nonempty GPU baseline"
    budget = MemoryBudget(case["memory_budget_bytes"]) if case.get("memory_budget_bytes") else None
    if budget:
        baseline = memory()
        assert {p["pid"] for p in baseline["processes"]} == {os.getpid()}, (
            "Unexpected process in bounded baseline"
        )
    emit("start", case=case, arguments=command, baseline=baseline)
    samples: list[dict[str, Any]] = []
    phase = "startup"
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            samples.append({"time": time.time(), "phase": phase, **memory()})
            stop.wait(0.01)

    log = Path("/tmp") / (case["name"] + ".log")
    with log.open("w") as file:
        process = subprocess.Popen(command, stdout=file, stderr=subprocess.STDOUT, start_new_session=True)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    result: dict[str, Any] = {
        "case": case,
        "arguments": command,
        "baseline": baseline,
        "file_sha256": digest.hexdigest(),
        "file_size": target.stat().st_size,
        "healthy": False,
        "load_only": LOAD_ONLY,
        "requested_prompt_tokens": prompt_tokens,
        "requested_completion_tokens": 8,
        "memory_budget": budget.audit if budget else None,
    }
    try:
        deadline = time.monotonic() + 300
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
                    health = json.load(response)
                if health.get("status") == "ok":
                    result["healthy"] = True
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        if result["healthy"] and LOAD_ONLY:
            phase = "ready"
            time.sleep(3)
            result["ready"] = memory()
        if result["healthy"] and not LOAD_ONLY:
            phase = "ready"
            time.sleep(3)
            result["ready"] = memory()
            phase = "full_context_request"
            payload = {
                "prompt": [100] * prompt_tokens,
                "n_predict": 8,
                "ignore_eos": True,
                "temperature": 0,
                "cache_prompt": False,
                "stream": False,
            }
            request = urllib.request.Request(
                "http://127.0.0.1:8080/completion",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            try:
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=300) as response:
                    answer = json.load(response)
                keys = [
                    "tokens_evaluated",
                    "tokens_predicted",
                    "stop_type",
                    "stopped_eos",
                    "stopped_limit",
                    "truncated",
                    "timings",
                    "tokens_cached",
                    "n_predict",
                ]
                result["request"] = {
                    "success": True,
                    "elapsed_seconds": time.monotonic() - started,
                    **{key: answer.get(key) for key in keys},
                }
            except Exception as error:
                result["request"] = {
                    "success": False,
                    "error": f"{type(error).__name__}: {error}",
                    "body": getattr(error, "read", lambda: b"")()[:1500].decode(errors="replace"),
                }
            phase = "after_request"
            time.sleep(2)
            result["after_request"] = memory()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stop.set()
        sampler.join(timeout=3)
        result["exit_code"] = process.returncode
        result["samples"] = samples
        result["phase_summary"] = {
            name: {
                "peak_device_used_bytes": max(s["device_used_bytes"] for s in samples if s["phase"] == name),
                "sample_count": sum(s["phase"] == name for s in samples),
            }
            for name in sorted({s["phase"] for s in samples})
        }
        if budget:
            budget.close()
        emit("case", **result)
        print("HFMEM_LOG_START " + case["name"], flush=True)
        print(log.read_text(errors="replace"), flush=True)
        print("HFMEM_LOG_END " + case["name"], flush=True)
        time.sleep(3)


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
emit(
    "environment",
    gpu=pynvml.nvmlDeviceGetName(handle),
    driver=pynvml.nvmlSystemGetDriverVersion(),
    nvml_python_version=importlib.metadata.version("nvidia-ml-py"),
    engine_version=subprocess.check_output(
        ["/app/llama-server", "--version"], stderr=subprocess.STDOUT, text=True
    ),
    sampling_interval_seconds=0.01,
)
for case in json.loads(Path("/tmp/cases.json").read_text()):
    execute(case)
emit("done")
