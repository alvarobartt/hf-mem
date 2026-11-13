"""Preview or explicitly replay an exported native benchmark from pinned source hashes."""

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


def prepare(directory: Path) -> tuple[dict[str, Any], str, dict[str, str]]:
    spec = json.loads((directory / "specification.json").read_text())
    if "@sha256:" not in spec["image"]:
        raise ValueError("Replay requires an immutable container digest")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, expected in spec["source_sha256"].items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Source names must be relative and contained in the bundle")
            file = directory / "executed" / name
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
            archive.add(file, arcname="hfmem-hooks/sitecustomize.py" if name == "sitecustomize.py" else name)
    payload = base64.b64encode(buffer.getvalue()).decode()
    env = {"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
    command = f"set -eu; echo {payload} | base64 -d | tar -xz -C /tmp; "
    if spec["engine"] == "vllm":
        sources = spec["source_sha256"]
        if "topology.py" in sources:
            command += "exec python3 -u /tmp/topology.py"
        elif all(name in sources for name in ["driver.py", "cases.json", "sitecustomize.py"]):
            count = spec.get("tensor_parallel_size")
            capacity = spec.get("max_num_seqs")
            if not isinstance(count, int) or count < 1 or not isinstance(capacity, int) or capacity < 1:
                raise ValueError("Explicit positive topology and scheduler capacity are required")
            command += "exec python3 -u /tmp/driver.py"
            env.update(
                {
                    "HFMEM_TP_SIZE": str(count),
                    "HFMEM_GPU_INDICES": ",".join(map(str, range(count))),
                    "HFMEM_MAX_NUM_SEQS": str(capacity),
                }
            )
        else:
            raise ValueError("Replay requires a topology wrapper or explicit single-server settings")
        env.update(
            {
                "PYTHONPATH": "/tmp/hfmem-hooks",
                "TOKENIZERS_PARALLELISM": "false",
                "HFMEM_AUDIT_WEIGHTS": "1",
            }
        )
    elif spec["engine"] == "llama.cpp":
        version = spec["nvml_python_version"]
        if not all(character.isdigit() or character == "." for character in version):
            raise ValueError("Unexpected NVML package version")
        command += (
            "apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv >/tmp/python-install.log 2>&1; "
            "python3 -m venv /tmp/hfmem-python; "
            f"/tmp/hfmem-python/bin/pip install --quiet nvidia-ml-py=={version}; "
            "exec /tmp/hfmem-python/bin/python -u /tmp/driver.py"
        )
        env["HFMEM_LOG_VERBOSITY"] = str(spec.get("log_verbosity", 3))
        if spec.get("mode") == "load_only_component_audit":
            env["HFMEM_LOAD_ONLY"] = "1"
    else:
        raise ValueError("Unsupported exported engine")
    return spec, command, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if (args.output / "job.json").exists():
        raise RuntimeError("Existing Job handle; inspect it instead of relaunching")
    spec, command, env = prepare(args.directory)
    if not args.launch:
        print(json.dumps({key: spec[key] for key in ["engine", "image", "flavor", "timeout"]}, indent=2))
        return
    from huggingface_hub import HfApi, get_token

    args.output.mkdir(parents=True, exist_ok=True)
    secrets = None
    if spec["engine"] == "vllm":
        token = get_token()
        if not token:
            raise RuntimeError("Authenticate before replaying the gated model")
        secrets = {"HF_TOKEN": token}
    job = HfApi().run_job(
        image=spec["image"],
        command=["bash", "-c", command],
        flavor=spec["flavor"],
        timeout=spec["timeout"],
        env=env,
        secrets=secrets,
    )
    result = {"job_id": job.id, "stage": job.status.stage, "flavor": spec["flavor"]}
    (args.output / "job.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
