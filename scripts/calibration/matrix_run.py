"""Preview or launch a source-verified matrix Job; never publishes bucket artifacts."""

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


def prepare(directory: Path) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    specification = json.loads((directory / "specification.json").read_text())
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", specification["image"]) or specification["engine"] not in {
        "vllm",
        "sglang",
    }:
        raise ValueError("Matrix replay requires an immutable supported engine image")
    for trial in specification["trials"]:
        revision = trial["case"]["revision"]
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("Every model revision must be a full commit SHA")
    sources = directory / "executed" if (directory / "executed").is_dir() else directory
    required = {"matrix.py", "driver.py", "sitecustomize.py", "trials.json"}
    required.update(f"cases-{index}.json" for index in range(len(specification["trials"])))
    if not required <= specification["source_sha256"].keys():
        raise ValueError("Missing required executed sources")
    if not sources.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Executed directory must stay inside replay bundle")
    buffer = io.BytesIO()
    verified: dict[str, bytes] = {}
    with tarfile.open(fileobj=buffer, mode="w:gz", dereference=True) as archive:
        for name, digest in specification["source_sha256"].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source paths must stay inside the replay bundle")
            source = sources / name
            if not source.resolve().is_relative_to(sources.resolve()):
                raise ValueError("Source symlinks must stay inside the replay bundle")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Source hash mismatch: {name}")
            verified[name] = data
            member = tarfile.TarInfo("hfmem-hooks/sitecustomize.py" if name == "sitecustomize.py" else name)
            member.size, member.mode = len(data), 0o644
            archive.addfile(member, io.BytesIO(data))
    if json.loads(verified["trials.json"]) != specification["trials"]:
        raise ValueError("Executed trials differ from the specification")
    for index, trial in enumerate(specification["trials"]):
        if json.loads(verified[f"cases-{index}.json"]) != [trial["case"]]:
            raise ValueError("Executed case differs from its declared trial")
    payload = base64.b64encode(buffer.getvalue()).decode()
    command = [
        "bash",
        "-c",
        f"set -eu; echo {payload} | base64 -d | tar -xz -C /tmp; exec python3 -u /tmp/matrix.py",
    ]
    environment = {
        "PYTHONPATH": "/tmp/hfmem-hooks",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return specification, command, environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch and (args.directory / "job.json").exists():
        raise RuntimeError("Existing Job handle; inspect or collect it instead of relaunching")
    specification, command, environment = prepare(args.directory)
    if not args.launch:
        print(json.dumps({key: specification[key] for key in ["engine", "image", "flavor", "timeout"]}))
        return
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        raise RuntimeError("Authenticate to Hugging Face before creating paid Jobs")
    job = HfApi().run_job(
        image=specification["image"],
        command=command,
        flavor=specification["flavor"],
        timeout=specification["timeout"],
        name="hf-mem-" + args.directory.name,
        env=environment,
        secrets={"HF_TOKEN": token},
    )
    result = {"job_id": job.id, "stage": job.status.stage, "flavor": specification["flavor"]}
    (args.directory / "job.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
