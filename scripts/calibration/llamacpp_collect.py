"""Preserve llama.cpp runtime samples and separately identified native buffer audits."""

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any

from collect import SECRET, TERMINAL, checksum, parse_records, sanitize, unique_records, write_csv


def buffers(text: str) -> dict[str, Any]:
    pattern = r"(CUDA\d+|CUDA_Host|CPU_Mapped|CPU)\s+(model|KV|RS|compute) buffer size\s*=\s*([\d.]+) MiB"
    matches = list(re.finditer(pattern, text))
    model = {}
    start = None
    for match in matches:
        device, kind, amount = match.groups()
        if kind == "model" and float(amount) > 0:
            model[device] = int(float(amount) * 2**20)
            if re.fullmatch(r"CUDA\d+", device):
                start = match.start()
    result: dict[str, Any] = {
        "gpu_model_bytes": sum(value for device, value in model.items() if re.fullmatch(r"CUDA\d+", device))
        or None,
        "host_mapped_model_bytes": model.get("CPU_Mapped"),
        "native_rounding": "Native MiB values are rounded to 0.01 MiB; conversion does not increase precision.",
    }
    for kind, name in [("KV", "gpu_kv_bytes"), ("RS", "gpu_recurrent_bytes"), ("compute", "gpu_compute_bytes")]:
        items = [
            match
            for match in matches
            if start is not None
            and match.start() > start
            and match[2] == kind
            and re.fullmatch(r"CUDA\d+", match[1])
        ]
        if kind == "compute":
            per_device: dict[str, int] = {}
            for match in items:
                per_device[match[1]] = max(per_device.get(match[1], 0), int(float(match[3]) * 2**20))
            result[name] = sum(per_device.values()) if per_device else None
        else:
            result[name] = sum(int(float(match[3]) * 2**20) for match in items) if items else None
    return result


def completed_request(record: dict[str, Any]) -> bool:
    request = record.get("request", {})
    return bool(
        record.get("healthy")
        and request.get("success")
        and request.get("tokens_evaluated") == record.get("requested_prompt_tokens", 32760)
        and request.get("tokens_predicted") == record.get("requested_completion_tokens", 8)
        and request.get("stop_type") == "limit"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--publish")
    args = parser.parse_args()
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    job = api.inspect_job(job_id=args.job_id)
    if job.status.stage not in TERMINAL:
        print(json.dumps({"stage": job.status.stage, "action": "Keep the exact existing handle"}))
        return
    raw = (
        args.log_file.read_text()
        if args.log_file
        else "\n".join(api.fetch_job_logs(job_id=args.job_id, tail=1000000))
    )
    (args.directory / "private.log").write_text(raw)
    text = sanitize(raw, get_token())
    records, invalid = parse_records(text)
    records, duplicate_records = unique_records(records)
    cases = [r for r in records if r.get("kind") == "case"]
    blocks = list(dict.fromkeys(re.findall(r"HFMEM_LOG_START [^\n]+\n(.*?)\nHFMEM_LOG_END [^\n]+", text, re.S)))
    assert len(blocks) == len(cases), "Incomplete native log blocks"
    output = args.directory / "public"
    output.mkdir(exist_ok=True)
    (output / "native.log").write_text(text)
    spec = json.loads((args.directory / "specification.json").read_text())
    load_only = spec.get("mode") == "load_only_component_audit"
    rows = []
    for case, native in zip(cases, blocks):
        baseline = case["baseline"]
        budget = case.get("memory_budget")
        if budget:
            assert {p["pid"] for p in baseline["processes"]} == {budget["reservation_pid"]}
            assert 0 < budget["cuda_free_after_bytes"] <= budget["target_free_bytes"] + (2 << 20)
        else:
            assert not baseline["processes"]
        assert case["file_sha256"] == case["case"]["lfs_sha256"]
        assert case["file_size"] == case["case"]["file_size"]
        samples = case.get("samples", [])
        assert len(samples) == sum(p["sample_count"] for p in case["phase_summary"].values())
        runtime = [s["device_used_bytes"] for s in samples if s["phase"] != "startup"]
        case["native_buffers"] = buffers(native)
        rows.append(
            {
                "case": case["case"]["name"],
                "model": case["case"]["model"],
                "revision": case["case"]["revision"],
                "file": case["case"]["file"],
                "load_only": load_only,
                "healthy": case.get("healthy"),
                "completed_full_token_counts": completed_request(case),
                "native_context_boundary_flag": case.get("request", {}).get("truncated"),
                "sampled_runtime_or_ready_footprint_bytes": max(runtime) - baseline["device_used_bytes"]
                if runtime
                else None,
                "sampled_startup_footprint_bytes": max(
                    s["device_used_bytes"] for s in samples if s["phase"] == "startup"
                )
                - baseline["device_used_bytes"],
                "baseline_bytes": baseline["device_used_bytes"],
                "bounded_free_memory_bytes": budget["cuda_free_after_bytes"] if budget else None,
                "file_sha256_verified": True,
                **case["native_buffers"],
            }
        )
    with gzip.open(output / "samples.jsonl.gz", "wt") as file:
        for case in cases:
            for sample in case.pop("samples", []):
                file.write(json.dumps({"case": case["case"]["name"], **sample}) + "\n")
    write_csv(output / "measurements.csv", rows)
    (output / "results.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    (output / "environment.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records if r.get("kind") == "environment")
    )
    expected = {case["name"] for case in spec["cases"]}
    seen = {case["case"]["name"] for case in cases}
    complete = (
        seen == expected
        and not invalid
        and all(
            row["healthy"]
            and (
                bool(row["gpu_model_bytes"]) and bool(row["gpu_kv_bytes"]) and bool(row["gpu_compute_bytes"])
                if load_only
                else row["completed_full_token_counts"]
            )
            for row in rows
        )
    )
    manifest = {
        "job_stage": job.status.stage,
        "mode": "load_only_component_audit" if load_only else "full_context_inference",
        "expected_cases": sorted(expected),
        "observed_cases": sorted(seen),
        "complete": complete,
        "invalid_records": invalid,
        "duplicate_transport_records": duplicate_records,
        "scope": "One T4. Runtime and load-only component audits are separate processes/jobs. All models use immutable revisions and full GGUF SHA-256 checks. Sampled peaks are lower bounds; CPU-mapped model bytes are not GPU memory. The native context-boundary flag is retained; completion validity requires every requested prompt and decode token.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "specification.json").write_text(json.dumps(spec, indent=2))
    (output / "executed").mkdir(exist_ok=True)
    for name in spec["source_sha256"]:
        (output / "executed" / name).write_text(sanitize((args.directory / name).read_text(), get_token()))
    for name in ["collect.py", "llamacpp_collect.py"]:
        (output / name).write_text(Path(__file__).with_name(name).read_text())
    for name in [
        "estimates.json",
        "image.json",
        "image-config.json",
        "image-manifest.json",
        "nvml-package.json",
    ]:
        if (args.directory / name).exists():
            (output / name).write_text(sanitize((args.directory / name).read_text(), get_token()))
    checksum(output)
    if args.publish:
        if not complete:
            raise RuntimeError("Incomplete runtime or component evidence")
        for path in output.rglob("*"):
            if path.is_file():
                content = (
                    gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
                )
                if SECRET.search(content) or (get_token() and get_token() in content):
                    raise RuntimeError("Credential detected")
        api.sync_bucket(str(output), args.publish, delete=False, quiet=True)
    print(json.dumps({"manifest": manifest, "measurements": rows}))


if __name__ == "__main__":
    main()
