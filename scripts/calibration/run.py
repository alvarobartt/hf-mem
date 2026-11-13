"""Preview or launch pinned, bounded inference-memory calibration Jobs.

Default is a preview. Credentials are passed only through Hub Jobs secrets.
Four-GPU vLLM runs contain independent one-GPU servers, not tensor parallelism.
"""

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

IMAGES = {
    "vllm": "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3",
    "sglang": "lmsysorg/sglang@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5",
}
CASES = [
    {
        "name": "llama31-8b",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "extra": [],
    },
    {
        "name": "qwen3-8b",
        "model": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "extra": [],
    },
]
MULTI = """import json
import os
import subprocess
import threading
import pynvml
pynvml.nvmlInit()
lock = threading.Lock()
trials = json.load(open("/tmp/trials.json"))
def run(index, trial):
    uuid = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index))
    if isinstance(uuid, bytes): uuid = uuid.decode()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "HFMEM_GPU_INDEX": str(index),
           "HFMEM_MAX_NUM_SEQS": str(trial["max_num_seqs"]),
           "HFMEM_PORT": str(8000 + index * 100), "HFMEM_CASES_PATH": f"/tmp/cases-{index}.json"}
    process = subprocess.Popen(["python3", "-u", "/tmp/driver.py"], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        with lock: print(line.rstrip(), flush=True)
    process.wait()
    with lock:
        print("HFMEM_RESULT " + json.dumps({"kind": "suite_exit", "trial": index,
              "case": trial["case"]["name"], "max_num_seqs": trial["max_num_seqs"],
              "exit_code": process.returncode}), flush=True)
threads = [threading.Thread(target=run, args=(i, trial)) for i, trial in enumerate(trials)]
for t in threads: t.start()
for t in threads: t.join()
"""


def prepare(engine: str, output: Path) -> dict[str, Any]:
    sources = Path(__file__).parent
    output.mkdir(parents=True, exist_ok=True)
    if (output / "job.json").exists():
        raise RuntimeError("Existing Job handle: inspect/collect it before creating a new experiment directory")
    for source, target in [(f"{engine}_driver.py", "driver.py"), (f"{engine}_hooks.py", "sitecustomize.py")]:
        (output / target).write_bytes((sources / source).read_bytes())
    trials = [{"case": case, "max_num_seqs": n} for case in CASES for n in [1, 256]] if engine == "vllm" else []
    if engine == "vllm":
        (output / "multi.py").write_text(MULTI)
        (output / "trials.json").write_text(json.dumps(trials, indent=2))
        for index, trial in enumerate(trials):
            (output / f"cases-{index}.json").write_text(json.dumps([trial["case"]], indent=2))
    else:
        (output / "cases.json").write_text(json.dumps(CASES, indent=2))
    names = [
        "driver.py",
        "sitecustomize.py",
        *(["multi.py", "trials.json", *[f"cases-{i}.json" for i in range(4)]] if trials else ["cases.json"]),
    ]
    spec = {
        "engine": engine,
        "image": IMAGES[engine],
        "flavor": "a100x4" if engine == "vllm" else "a100-large",
        "timeout": "20m",
        "cases": CASES,
        "trials": trials,
        "tensor_parallel_size": 1,
        "concurrent_requests_per_server": 1,
        "controls": {
            "context_length": 32768,
            "prompt_tokens": 32760 if engine == "vllm" else 32736,
            "requested_completion_tokens": 8,
            "prefill_tokens": 8192,
            "gpu_memory_utilization": 0.92,
        },
        "cache_pool": {"blocks": 2049, "block_size": 16}
        if engine == "vllm"
        else {"max_total_tokens": 33024, "extra_admission_slots": 256},
        "source_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names},
    }
    (output / "specification.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=IMAGES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    spec = prepare(args.engine, args.output)
    if not args.launch:
        print(json.dumps(spec, indent=2))
        return
    from huggingface_hub import HfApi, get_token

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in spec["source_sha256"]:
            archive.add(
                args.output / name,
                arcname="hfmem-hooks/sitecustomize.py" if name == "sitecustomize.py" else name,
            )
    payload = base64.b64encode(buffer.getvalue()).decode()
    entrypoint = "multi.py" if args.engine == "vllm" else "driver.py"
    token = get_token()
    if not token:
        raise RuntimeError("Authenticate to Hugging Face before launching gated model Jobs")
    job = HfApi().run_job(
        image=spec["image"],
        command=[
            "bash",
            "-c",
            f"set -eu; echo {payload} | base64 -d | tar -xz -C /tmp; exec python3 -u /tmp/{entrypoint}",
        ],
        flavor=spec["flavor"],
        timeout=spec["timeout"],
        name=f"hf-mem-{args.engine}-scheduler-calibration",
        env={
            "PYTHONPATH": "/tmp/hfmem-hooks",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        secrets={"HF_TOKEN": token},
    )
    (args.output / "job.json").write_text(
        json.dumps({"job_id": job.id, "flavor": spec["flavor"]}, indent=2) + "\n"
    )
    print(json.dumps({"job_id": job.id, "stage": job.status.stage}))


if __name__ == "__main__":
    main()
