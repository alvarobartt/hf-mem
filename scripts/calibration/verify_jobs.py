"""Verify native Job payloads and pre-submission registrations without exporting secrets."""

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def registration(directory: Path, spec: dict[str, Any]) -> tuple[str | None, str | None, bool | None]:
    prediction = directory / "candidate-predictions.json"
    saved = json.loads(prediction.read_text()) if prediction.exists() else None
    for path in sorted(directory.parent.glob("*registration.json")):
        data = json.loads(path.read_text())
        entries = data.get("experiments", [])
        if "specification" in data:
            entries = [data]
        for entry in entries:
            match = (
                entry == directory.name
                if isinstance(entry, str)
                else (
                    Path(entry.get("directory", "")).name == directory.name
                    or entry.get("specification") == spec
                )
            )
            if not match:
                continue
            frozen = data.get("frozen_at_utc") or data.get("prepared_at_utc")
            forecast = None if isinstance(entry, str) else entry.get("forecasts", entry.get("forecast"))
            if forecast is not None and saved is not None:
                matches = saved == forecast or (isinstance(saved, list) and saved == [forecast])
            else:
                matches = None
            return frozen, path.name, matches
    return spec.get("frozen_at_utc"), "specification.json", None


def verify(directory: Path, api: Any) -> dict[str, Any]:
    spec = json.loads((directory / "specification.json").read_text())
    handle = json.loads((directory / "job.json").read_text())
    job = api.inspect_job(job_id=handle["job_id"])
    command = job.command[-1]
    match = re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", command)
    if not match:
        raise ValueError("Cannot identify the executed source payload")
    members = {}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(match[1])), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError("Unexpected non-file in source payload")
            source = archive.extractfile(member)
            assert source is not None
            members[member.name] = source.read()
    source_checks = {}
    for name, expected in spec["source_sha256"].items():
        local = directory / name
        if not local.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Local executed source escapes the experiment directory")
        packed = "hfmem-hooks/sitecustomize.py" if name == "sitecustomize.py" else name
        source_checks[name] = digest(local.read_bytes()) == digest(members[packed]) == expected
    embedded_spec = (
        json.loads(members["specification.json"]) == spec if "specification.json" in members else None
    )
    frozen, protocol, prediction_matches = registration(directory, spec)
    before_submission = (
        datetime.fromisoformat(frozen.replace("Z", "+00:00")) < job.created_at if frozen else False
    )
    result = {
        "directory": directory.name,
        "job_id": job.id,
        "url": job.url,
        "stage": job.status.stage,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "frozen_at": frozen,
        "registration": protocol,
        "registration_forecast_matches": prediction_matches,
        "registered_before_submission": before_submission,
        "image_matches": job.docker_image == spec["image"],
        "flavor_matches": job.flavor == spec["flavor"],
        "owner": job.owner.name,
        "source_checks": source_checks,
        "embedded_spec_matches": embedded_spec,
        "payload_sha256": digest(base64.b64decode(match[1])),
    }
    primary = directory.name.startswith(
        (
            "capacity-v2-",
            "batch-v3-",
            "gguf-v2-",
            "gguf-v3-",
            "inkling-layout-",
            "lm-only-",
            "bounded-",
            "startup-",
        )
    )
    result["verified"] = bool(
        all(source_checks.values())
        and result["image_matches"]
        and result["flavor_matches"]
        and embedded_spec is not False
        and before_submission
        and (not primary or prediction_matches is True)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    rows = [
        verify(path.parent, api)
        for path in sorted(args.directory.glob("*/job.json"))
        if (path.parent / "specification.json").exists()
    ]
    result = {
        "scope": "Read-only Job API verification. No environment or secrets are exported. Exact executed source bytes, immutable image, flavor and pre-submission forecast registration are audited separately from measurement completeness.",
        "verified": bool(rows) and all(row["verified"] for row in rows),
        "jobs": rows,
    }
    (args.directory / "job-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verified": result["verified"],
                "jobs": len(rows),
                "failures": [r["directory"] for r in rows if not r["verified"]],
            }
        )
    )


if __name__ == "__main__":
    main()
