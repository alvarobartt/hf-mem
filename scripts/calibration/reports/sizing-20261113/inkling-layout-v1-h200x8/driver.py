"""Run an exact native server argument vector with per-device NVML sampling."""

import gzip
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

lock = threading.Lock()


def emit(kind: str, **data: Any) -> None:
    with lock:
        print(
            "HFMEM_RESULT " + json.dumps({"kind": kind, "time": time.time(), **data}, default=str), flush=True
        )


def get(path: str, timeout: int = 5) -> Any:
    with urllib.request.urlopen("http://127.0.0.1:30000" + path, timeout=timeout) as response:
        return json.load(response)


spec = json.loads(Path("/tmp/specification.json").read_text())
pynvml.nvmlInit()
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(spec["gpus"])]


def memory() -> dict[str, Any]:
    devices = []
    for i, handle in enumerate(handles):
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        devices.append(
            {
                "index": i,
                "used": info.used,
                "total": info.total,
                "processes": [{"pid": p.pid, "used": p.usedGpuMemory} for p in procs],
            }
        )
    return {
        "used": sum(d["used"] for d in devices),
        "total": sum(d["total"] for d in devices),
        "devices": devices,
    }


versions = {}
for name in ["sglang", "torch", "transformers", "flash-attn", "flashinfer-python", "sgl-kernel"]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
baseline = memory()
assert all(not d["processes"] for d in baseline["devices"]), "Nonempty GPU baseline"
shm = os.statvfs("/dev/shm")
emit(
    "environment",
    versions=versions,
    baseline=baseline,
    gpu_names=[pynvml.nvmlDeviceGetName(h) for h in handles],
    driver=pynvml.nvmlSystemGetDriverVersion(),
    shm_total_bytes=shm.f_blocks * shm.f_frsize,
    interconnect=subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout,
    specification=spec,
)
# Preserve exact source files from the actual image for native geometry diagnosis.
try:
    import importlib.util

    package = Path(importlib.util.find_spec("sglang").origin).parent
    for relative in [
        "srt/models/inkling2.py",
        "srt/models/inkling.py",
        "srt/layers/attention/inkling_attention.py",
    ]:
        path = package / relative
        if path.is_file():
            emit("native_source", path=relative, content=path.read_text())
except Exception as exc:
    emit("source_observation_error", error=str(exc))
args = ["python3", "-m", "sglang.launch_server", *spec["server_arguments"]]
path = Path("/tmp/native-server.log")
phase = "startup"
samples = []
stopped = threading.Event()


def sample() -> None:
    while not stopped.is_set():
        try:
            samples.append({"time": time.time(), "phase": phase, **memory()})
        except pynvml.NVMLError:
            pass
        stopped.wait(0.01)


emit("start", arguments=args, baseline=baseline)
proc = subprocess.Popen(
    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, bufsize=1
)


def read_output() -> None:
    assert proc.stdout is not None
    with path.open("w") as file:
        for line in proc.stdout:
            file.write(line)
            file.flush()
            with lock:
                print("HFMEM_LIVE " + line.rstrip(), flush=True)


reader = threading.Thread(target=read_output, daemon=True)
sampler = threading.Thread(target=sample, daemon=True)
reader.start()
sampler.start()
requests = []
healthy = False
try:
    deadline = time.monotonic() + spec.get("startup_timeout_seconds", 1800)
    while proc.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2):
                pass
            healthy = True
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2)
    emit("health", healthy=healthy, process_exit_code=proc.poll())
    if healthy:
        phase = "ready"
        time.sleep(3)
        try:
            info = get("/server_info")
            # Never export API keys, credentials or unrelated configuration.
            fields = [
                "context_length",
                "max_running_requests",
                "max_total_tokens",
                "chunked_prefill_size",
                "max_prefill_tokens",
                "page_size",
                "dtype",
                "kv_cache_dtype",
                "mem_fraction_static",
                "tp_size",
                "ep_size",
                "attention_backend",
                "enable_multimodal",
                "disable_hybrid_swa_memory",
                "swa_full_tokens_ratio",
                "cuda_graph_max_bs",
                "cuda_graph_max_bs_decode",
            ]
            settings = info.get("server_args") or info
            emit(
                "resolved_controls",
                controls={k: settings.get(k) for k in fields},
                internal_states=[
                    {
                        k: state.get(k)
                        for k in fields
                        + [
                            "rank",
                            "tp_rank",
                            "max_req_len",
                            "max_req_input_len",
                            "full_max_total_num_tokens",
                            "swa_max_total_num_tokens",
                        ]
                        if k in state
                    }
                    for state in (info.get("internal_states") or [])
                    if isinstance(state, dict)
                ],
                ready_memory=memory(),
            )
        except Exception as exc:
            emit("server_info_error", error=str(exc))
        for i, count in enumerate(spec["prompt_lengths"]):
            phase = f"request_{i}_{count}_tokens"
            payload = {
                "input_ids": [100 + i] * count,
                "sampling_params": {"max_new_tokens": 8, "ignore_eos": True, "temperature": 0},
            }
            request = urllib.request.Request(
                "http://127.0.0.1:30000/generate",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json"},
            )
            started = time.monotonic()
            observation = {
                "requested_prompt_tokens": count,
                "requested_completion_tokens": 8,
                "prompt_token_id": 100 + i,
            }
            try:
                with urllib.request.urlopen(
                    request, timeout=spec.get("request_timeout_seconds", 900)
                ) as response:
                    data = json.load(response)
                observation.update(
                    success=True, meta_info=data.get("meta_info", {}), output_text=data.get("text", "")[:200]
                )
            except Exception as exc:
                body = getattr(exc, "read", lambda: b"")()[:2000].decode(errors="replace")
                observation.update(success=False, error=f"{type(exc).__name__}: {exc}", body=body)
            observation["elapsed_seconds"] = time.monotonic() - started
            requests.append(observation)
            emit("request", index=i, **observation)
            phase = "after_request_" + str(i)
            time.sleep(3)
            if not observation["success"]:
                break
finally:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    stopped.set()
    sampler.join(timeout=5)
    reader.join(timeout=5)
    summaries = {}
    for item in samples:
        key = item["phase"]
        previous = summaries.get(key)
        if previous is None or item["used"] > previous["peak_used"]:
            summaries[key] = {
                "peak_used": item["used"],
                "devices_at_peak": item["devices"],
                "sample_time": item["time"],
            }
    emit(
        "result",
        healthy=healthy,
        process_exit_code=proc.returncode,
        requests=requests,
        phase_summary=summaries,
        sample_count=len(samples),
        baseline=baseline,
    )
    # Smaller independent chunks retain exact samples without giant wrapped records.
    for offset in range(0, len(samples), 128):
        emit("samples", offset=offset, samples=samples[offset : offset + 128])
    print("HFMEM_LOG_START native", flush=True)
    print(path.read_text(errors="replace"), flush=True)
    print("HFMEM_LOG_END native", flush=True)
    emit("done")
