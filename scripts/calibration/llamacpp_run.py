"""Preview or launch a source-verified native llama.cpp experiment."""

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any


def prepare(directory: Path) -> tuple[dict[str, Any], list[str]]:
    spec = json.loads((directory / "specification.json").read_text())
    if spec["engine"] != "llama.cpp" or not re.fullmatch(r".+@sha256:[a-f0-9]{64}", spec["image"]):
        raise ValueError("An immutable llama.cpp image is required")
    for case in spec["cases"]:
        if not re.fullmatch(r"[a-f0-9]{40}", case["revision"]) or not re.fullmatch(
            r"[a-f0-9]{64}", case["lfs_sha256"]
        ):
            raise ValueError("Immutable model revision and GGUF SHA-256 are required")
    if set(spec["source_sha256"]) != {"driver.py", "cases.json"}:
        raise ValueError("Unexpected executed source set")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, digest in spec["source_sha256"].items():
            source = directory / name
            if not source.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Executed source must stay in the experiment directory")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {name}")
            if name == "cases.json" and json.loads(data) != spec["cases"]:
                raise ValueError("Executed cases differ from the specification")
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        requirements = b"nvidia-ml-py==13.615.71 --hash=sha256:959bf4adf6fe1308e4bd739e722236b0d1ec8392e2cefad33ff70c311380b9b6\n"
        item = tarfile.TarInfo("requirements.txt")
        item.size = len(requirements)
        archive.addfile(item, io.BytesIO(requirements))
    payload = base64.b64encode(buffer.getvalue()).decode()
    command = [
        "bash",
        "-c",
        f"set -eu; echo {payload} | base64 -d | tar -xz -C /tmp; "
        "apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv >/tmp/python-install.log 2>&1; "
        "python3 -m venv /tmp/hfmem-python; /tmp/hfmem-python/bin/pip install --quiet --require-hashes -r /tmp/requirements.txt; "
        "exec /tmp/hfmem-python/bin/python -u /tmp/driver.py",
    ]
    return spec, command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch and (args.directory / "job.json").exists():
        raise RuntimeError("Existing Job handle; inspect or collect it instead of relaunching")
    spec, command = prepare(args.directory)
    if not args.launch:
        print(json.dumps({k: spec[k] for k in ["engine", "image", "flavor", "timeout"]}))
        return
    from huggingface_hub import HfApi, get_token

    if not get_token():
        raise RuntimeError("Authenticate to Hugging Face before launching Jobs")
    job = HfApi().run_job(
        image=spec["image"],
        command=command,
        flavor=spec["flavor"],
        timeout=spec["timeout"],
        attempts=1,
        name="hf-mem-" + args.directory.name,
    )
    result = {"job_id": job.id, "stage": job.status.stage, "flavor": spec["flavor"]}
    (args.directory / "job.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
