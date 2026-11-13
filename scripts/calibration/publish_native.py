"""Publish only sanitized native measurements; exclude all source and estimator forecasts."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

from collect import SECRET, checksum

NATIVE_FILES = {
    "aggregate.csv",
    "ranks.csv",
    "results.jsonl",
    "environment.jsonl",
    "samples.jsonl.gz",
    "manifest.json",
    "specification.json",
    "native.log",
    "planning.jsonl",
}


def prepare(directory: Path, destination: Path, token: str | None) -> dict:
    source = directory / "public"
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Only complete native matrices can enter this gold bundle")
    destination.mkdir(parents=True, exist_ok=False)
    for name in NATIVE_FILES:
        path = source / name
        if not path.exists():
            continue
        data = path.read_bytes()
        plain = gzip.decompress(data) if name.endswith(".gz") else data
        text = plain.decode()
        if SECRET.search(text) or (token and token in text):
            raise ValueError(f"Credential scan failed: {name}")
        # Full logs are retained, compressed losslessly. No executed .py files,
        # baseline/candidate forecasts, fitted comparisons, or source snapshots.
        if name in {"native.log", "results.jsonl", "planning.jsonl"}:
            data = gzip.compress(data, mtime=0)
            name += ".gz"
        (destination / name).write_bytes(data)
    job = json.loads((directory / "job.json").read_text())
    job["stage_at_submission"] = job.pop("stage", None)
    (destination / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    note = (
        f"# Native engine measurements: {directory.name}\n\n"
        "Complete rank/token/sample gates passed. This bundle contains only native engine "
        "measurements and their specifications, not estimator source, predictions or accuracy claims. "
        "Raw logs, requests and native parameter/cache/graph audits are retained losslessly. "
        "The specification records exact immutable model/container revisions and executed-harness hashes. "
        "Executed source and estimator forecasts remain local pending separate publication approval; "
        "this native-only bundle is not a complete standalone replay package.\n\n"
        "Two sequential prompts share a warmed server. Aggregate NVML readings query devices sequentially "
        "within an iteration; sampled peaks are lower bounds. Stage peaks must not be added. "
        "Residual memory subtracts audited runtime parameter storage and physical pools, not broad loading deltas.\n"
    )
    (destination / "README.md").write_text(note)
    checksum(destination)
    return {"dataset": directory.name, "files": len(list(destination.iterdir())), "job_id": job["job_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, get_token

    token = get_token()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = [prepare(directory, args.output / directory.name, token) for directory in args.directories]
    (args.output / "index.json").write_text(json.dumps(entries, indent=2) + "\n")
    checksum(args.output)
    # Include per-dataset manifests too; the generic collector helper excludes
    # every SHA256SUMS basename when it constructs an aggregate manifest.
    root_manifest = args.output / "SHA256SUMS"
    with root_manifest.open("a") as file:
        for nested in sorted(args.output.rglob("SHA256SUMS")):
            if nested != root_manifest:
                file.write(
                    f"{hashlib.sha256(nested.read_bytes()).hexdigest()}  {nested.relative_to(args.output)}\n"
                )
    print(json.dumps({"datasets": len(entries), "publication": bool(args.publish)}), flush=True)
    if not args.publish:
        return
    api = HfApi()
    api.sync_bucket(
        str(args.output), "hf://buckets/alvarobartt/hf-mem-validation/" + args.prefix, quiet=True, delete=False
    )
    anonymous = args.output.with_name(args.output.name + "-anonymous")
    pairs = []
    for line in (args.output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe remote manifest path")
        pairs.append((args.prefix + "/" + name, anonymous / name))
    pairs.append((args.prefix + "/SHA256SUMS", anonymous / "SHA256SUMS"))
    api.download_bucket_files(
        "alvarobartt/hf-mem-validation", files=pairs, token=False, raise_on_missing_files=True
    )
    for path in args.output.rglob("*"):
        if path.is_file():
            assert (
                hashlib.sha256(path.read_bytes()).digest()
                == hashlib.sha256((anonymous / path.relative_to(args.output)).read_bytes()).digest()
            )
    print(json.dumps({"prefix": args.prefix, "all_files_anonymously_verified": True}), flush=True)


if __name__ == "__main__":
    main()
