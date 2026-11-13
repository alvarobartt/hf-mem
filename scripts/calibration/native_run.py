"""Source-verified exact-argument native Job launcher. No public uploads."""

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
    spec_bytes = (directory / "specification.json").read_bytes()
    spec = json.loads(spec_bytes)
    if not re.fullmatch(r".+@sha256:[a-f0-9]{64}", spec["image"]):
        raise ValueError("Immutable image digest required")
    if not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
        raise ValueError("Immutable model revision required")
    args = spec["server_arguments"]
    for flag, expected in [
        ("--model-path", spec["model"]),
        ("--revision", spec["revision"]),
        ("--tp", str(spec["gpus"])),
        ("--ep", str(spec["gpus"])),
        ("--context-length", str(spec["context_length"])),
    ]:
        if args.count(flag) != 1 or args[args.index(flag) + 1] != expected:
            raise ValueError(f"Mismatched native control: {flag}")
    if spec["flavor"] != f"h200x{spec['gpus']}":
        raise ValueError("Topology/flavor mismatch")
    buffer = io.BytesIO()
    sources = {"specification.json": spec_bytes}
    if set(spec["source_sha256"]) != {"driver.py", "hooks/sitecustomize.py"}:
        raise ValueError("Required executed source set differs")
    for name, digest in spec["source_sha256"].items():
        source = directory / name
        if not source.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Source must stay inside the experiment directory")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Source hash mismatch: {name}")
        sources[name] = data
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in sources.items():
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o644
            archive.addfile(entry, io.BytesIO(data))
    payload = base64.b64encode(buffer.getvalue()).decode()
    return spec, [
        "bash",
        "-c",
        f"set -eu; echo {payload} | base64 -d | tar -xz -C /tmp; exec python3 -u /tmp/driver.py",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch and (args.directory / "job.json").exists():
        raise RuntimeError("Existing Job handle; inspect it instead of relaunching")
    spec, command = prepare(args.directory)
    if not args.launch:
        print(json.dumps({k: spec[k] for k in ["image", "flavor", "timeout", "server_arguments"]}, indent=2))
        return
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise RuntimeError("Authenticate before launching Jobs")
    job = HfApi().run_job(
        image=spec["image"],
        command=command,
        flavor=spec["flavor"],
        timeout=spec["timeout"],
        attempts=1,
        name="hf-mem-inkling-native-" + spec["flavor"],
        env={
            "PYTHONPATH": "/tmp/hooks",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        secrets={"HF_TOKEN": token},
    )
    handle = {"job_id": job.id, "stage_at_submission": job.status.stage, "flavor": spec["flavor"]}
    (args.directory / "job.json").write_text(json.dumps(handle, indent=2) + "\n")
    print(json.dumps(handle))


if __name__ == "__main__":
    main()
