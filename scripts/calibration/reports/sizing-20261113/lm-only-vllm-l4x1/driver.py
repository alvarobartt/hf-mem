import base64
import hashlib
import json
import os
import re
import signal
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

import pynvml

output_lock = threading.Lock()
port = int(os.environ.get("HFMEM_PORT", "8000"))
max_num_seqs = int(os.environ.get("HFMEM_MAX_NUM_SEQS", "1"))
tp_size = int(os.environ.get("HFMEM_TP_SIZE", "1"))


def emit(kind: str, **data: Any) -> None:
    with output_lock:
        print(
            "HFMEM_RESULT "
            + json.dumps(
                {
                    "kind": kind,
                    "time": time.time(),
                    "max_num_seqs": max_num_seqs,
                    "tensor_parallel_size": tp_size,
                    **data,
                },
                default=str,
            ),
            flush=True,
        )


def get(path: str) -> Any:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}" + path, timeout=3) as response:
        return json.load(response)


def memory() -> dict[str, Any]:
    devices = []
    for index, handle in zip(gpu_indices, handles):
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        devices.append(
            {
                "device_index": index,
                "device_used_bytes": info.used,
                "device_total_bytes": info.total,
                "processes": [{"pid": p.pid, "used_bytes": p.usedGpuMemory} for p in processes],
            }
        )
    return {
        "device_used_bytes": sum(d["device_used_bytes"] for d in devices),
        "device_total_bytes": sum(d["device_total_bytes"] for d in devices),
        "processes": [p for d in devices for p in d["processes"]],
        "devices": devices,
    }


def synthetic_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, value: bytes) -> bytes:
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
        )

    pixels = (b"\0" + b"\x80\x80\x80" * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels, 9))
        + chunk(b"IEND", b"")
    )


def execute(case: dict[str, Any], mode: str, blocks: int | None) -> dict[str, Any]:
    context_length = int(case.get("max_model_len", 32768))
    prompt_tokens = int(case.get("prompt_tokens", context_length - 8))
    batch_size = int(case.get("batch_size", 1))
    prefill_tokens = int(case.get("max_num_batched_tokens", 8192))
    args = [
        "vllm",
        "serve",
        case["model"],
        "--revision",
        case["revision"],
        "--tokenizer-revision",
        case["revision"],
        "--max-num-seqs",
        str(max_num_seqs),
        "--tensor-parallel-size",
        str(tp_size),
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        str(context_length),
        "--max-num-batched-tokens",
        str(prefill_tokens),
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.92",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        *case.get("extra", []),
    ]
    if blocks is not None:
        args += ["--num-gpu-blocks-override", str(blocks)]
    # Zero in the trial specification declares an omitted native control. Keep
    # the requested value in the records and audit the resolved engine value.
    omitted = []
    if case.get("native_scheduler_defaults"):
        omitted += ["--max-num-seqs"]
    if case.get("native_prefill_defaults"):
        omitted += ["--max-num-batched-tokens"]
    if case.get("native_context_defaults"):
        omitted += ["--max-model-len"]
    if case.get("native_cache_defaults"):
        omitted += ["--gpu-memory-utilization", "--block-size"]
    for flag in omitted:
        index = args.index(flag)
        del args[index : index + 2]
    env = os.environ.copy()
    env["HFMEM_PLAN_ONLY"] = "1" if mode == "plan" else "0"
    path = Path("/tmp") / f"{case['name']}-{mode}-{port}.log"
    samples = []
    phase = "startup"
    stopped = threading.Event()

    def sample() -> None:
        while not stopped.is_set():
            try:
                samples.append({"time": time.time(), "phase": phase, **memory()})
            except pynvml.NVMLError:
                pass
            stopped.wait(0.01)

    deadline = time.monotonic() + 30
    while memory()["processes"] and time.monotonic() < deadline:
        time.sleep(1)
    baseline = memory()
    assert not baseline["processes"], "Nonempty GPU baseline"
    emit("start", case=case, mode=mode, arguments=args, baseline=baseline)
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def consume() -> None:
        assert proc.stdout is not None
        with path.open("w") as logfile:
            for line in proc.stdout:
                logfile.write(line)
                logfile.flush()
                with output_lock:
                    print("HFMEM_LIVE " + case["name"] + " " + mode + " " + line.rstrip(), flush=True)

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    healthy = False
    result = {
        "case": case,
        "mode": mode,
        "arguments": args,
        "baseline": baseline,
        "requested_prompt_tokens": prompt_tokens * batch_size,
        "requested_completion_tokens": 8 * batch_size,
    }
    if case.get("image_probe"):
        result["requested_prompt_tokens"] = None
        result["minimum_prompt_tokens"] = case["image_probe"]["minimum_prompt_tokens"]
    try:
        deadline = time.monotonic() + int(case.get("startup_timeout", 1100))
        while proc.poll() is None and time.monotonic() < deadline:
            if mode != "plan":
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        healthy = True
                    break
                except (urllib.error.URLError, TimeoutError):
                    pass
            time.sleep(1)
        result["healthy"] = healthy
        if healthy:
            phase = "ready"
            time.sleep(3)
            result["ready"] = memory()
            result["models"] = [
                {"id": m["id"], "max_model_len": m.get("max_model_len")} for m in get("/v1/models")["data"]
            ]
            results = []
            repeats = int(case.get("repeats", 1))
            for index in range(repeats):
                suffix = f"_{index}" if repeats > 1 else ""
                phase = "full_context_request" + suffix
                payload = {
                    "model": case["model"],
                    "prompt": [100 + index] * prompt_tokens,
                    "max_tokens": 8,
                    "temperature": 0,
                    "ignore_eos": True,
                }
                if batch_size > 1:
                    payload["prompt"] = [
                        [100 + index * batch_size + b] * prompt_tokens for b in range(batch_size)
                    ]
                image_data = None
                endpoint = "/v1/completions"
                probe = case.get("image_probe")
                if probe and (not probe.get("after_text") or index > 0):
                    png = synthetic_png(probe["width"], probe["height"])
                    image_data = {
                        "width": probe["width"],
                        "height": probe["height"],
                        "sha256": hashlib.sha256(png).hexdigest(),
                        "png_base64": base64.b64encode(png).decode(),
                        "zlib_version": zlib.ZLIB_VERSION,
                    }
                    payload.pop("prompt")
                    payload["messages"] = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64," + image_data["png_base64"]},
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        }
                    ]
                    endpoint = "/v1/chat/completions"
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}" + endpoint,
                    json.dumps(payload).encode(),
                    {"Content-Type": "application/json"},
                )
                started = time.monotonic()
                try:
                    with urllib.request.urlopen(
                        request, timeout=int(case.get("request_timeout", 120))
                    ) as response:
                        response_data = json.load(response)
                    observation = {
                        "success": True,
                        "usage": response_data.get("usage"),
                        "sequence_count": len(response_data.get("choices", [])),
                        "finish_reasons": [
                            item.get("finish_reason") for item in response_data.get("choices", [])
                        ],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                except Exception as exc:
                    body = getattr(exc, "read", lambda: b"")()[:1000].decode(errors="replace")
                    observation = {"success": False, "error": f"{type(exc).__name__}: {exc}", "body": body}
                observation["workload"] = "image" if image_data is not None else "text"
                if image_data is not None:
                    observation["synthetic_image"] = image_data
                observation["repeat_index"] = index
                observation["prompt_token_id"] = 100 + index
                results.append(observation)
                result["request"] = observation
                phase = "after_request" + suffix
                time.sleep(2)
                result["after_request"] = memory()
            result["requests"] = results
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        stopped.set()
        sampler.join(timeout=3)
        result["exit_code"] = proc.returncode
        reader.join(timeout=5)
        native = path.read_text(errors="replace")
        events = []
        for match in re.finditer(r"HFMEM_NATIVE (?=\{)", native):
            try:
                events.append(json.JSONDecoder().raw_decode(native, match.end())[0])
            except json.JSONDecodeError:
                pass
        result["native"] = events
        result["samples"] = samples
        result["phase_summary"] = {
            name: {
                "peak_device_used_bytes": max(s["device_used_bytes"] for s in samples if s["phase"] == name),
                "sample_count": sum(s["phase"] == name for s in samples),
            }
            for name in sorted({s["phase"] for s in samples})
        }
        result["native_log_tail"] = native[-10000:] if not healthy and mode != "plan" else None
        emit("case", **result)
        print("HFMEM_LOG_START " + case["name"] + " " + mode, flush=True)
        print(native, flush=True)
        print("HFMEM_LOG_END " + case["name"] + " " + mode, flush=True)
        time.sleep(3)
    return result


pynvml.nvmlInit()
gpu_indices = [
    int(i) for i in os.environ.get("HFMEM_GPU_INDICES", os.environ.get("HFMEM_GPU_INDEX", "0")).split(",")
]
assert len(gpu_indices) == tp_size
handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]
emit(
    "environment",
    gpu=pynvml.nvmlDeviceGetName(handles[0]),
    gpu_indices=gpu_indices,
    device_count=len(handles),
    driver=pynvml.nvmlSystemGetDriverVersion(),
    interconnect=subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout,
    compute_capabilities=[pynvml.nvmlDeviceGetCudaComputeCapability(handle) for handle in handles],
    sampling_interval_seconds=0.01,
    versions=subprocess.check_output(
        [
            "python3",
            "-c",
            "import importlib.metadata,json;print(json.dumps({x:importlib.metadata.version(x) for x in ['vllm','torch','transformers']}))",
        ],
        text=True,
    ).strip(),
)
for case in json.loads(Path(os.environ.get("HFMEM_CASES_PATH", "/tmp/cases.json")).read_text()):
    blocks = None if case.get("native_cache_defaults") else case.get("num_gpu_blocks_override", 2049)
    if case.get("native_plan"):
        planned = execute(case, "plan", 1)
        plans = [event for event in planned["native"] if event.get("kind") == "cache_plan"]
        if not plans:
            emit("plan_failed", case=case)
            continue
        blocks = plans[-1]["required_blocks_including_null"]
    execute(case, "single_sequence", blocks)
emit("done")
