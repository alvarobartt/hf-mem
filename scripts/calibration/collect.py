"""Export controlled Hub Job observations without treating failed requests as calibration data.

Requires huggingface_hub only when fetching/publishing, not for offline processing.
The output includes complete sanitized native logs and separate NVML time series.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL = {"COMPLETED", "ERROR", "FAILED", "CANCELED", "DELETED"}
SECRET = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}|sk-[A-Za-z0-9_-]{20,})\b"
)


def sanitize(text: str, token: str | None = None) -> str:
    if token:
        text = text.replace(token, "[REDACTED]")
    text = SECRET.sub("[REDACTED]", text)
    text = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = re.sub(r"\bGPU-[a-fA-F0-9-]{20,}\b", "[GPU_ID]", text)
    return text


def parse_records(text: str) -> tuple[list[dict[str, Any]], int]:
    """Job log transport may wrap a long JSON record across multiple log entries."""
    records = []
    invalid = 0
    pending: list[str] = []
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        if line.startswith("HFMEM_RESULT "):
            if pending:
                invalid += 1
            pending = [line.removeprefix("HFMEM_RESULT ")]
        elif pending:
            if line.startswith("HFMEM_LOG"):
                invalid += 1
                pending = []
                continue
            pending.append(line)
        else:
            continue
        # Every emitted record is a JSON object. Transport fragments ending in
        # the middle of an array/string cannot complete it; avoid reparsing a
        # multi-megabyte sample record at every 16 KiB boundary.
        if not line.rstrip().endswith("}"):
            continue
        try:
            record, end = decoder.raw_decode("".join(pending))
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or "".join(pending)[end:].strip():
            invalid += 1
        else:
            records.append(record)
        pending = []
    return records, invalid + bool(pending)


def parse_native(text: str) -> tuple[list[dict[str, Any]], int]:
    # Different workers can emit adjacent JSON objects on the same physical line.
    # Decode at each marker rather than rejecting the entire joined line.
    records = []
    invalid = 0
    decoder = json.JSONDecoder()
    for match in re.finditer(r"HFMEM_NATIVE (?=\{)", text):
        try:
            record, _ = decoder.raw_decode(text, match.end())
            records.append(record)
        except json.JSONDecodeError:
            invalid += 1
    return records, invalid


def unique_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse byte-equivalent JSON events replayed by the Job log transport.

    Distinct timestamps, samples or outcomes remain distinct, so a conflicting
    second execution cannot silently replace the first one.
    """
    seen = set()
    unique = []
    for record in records:
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).digest()
        if digest not in seen:
            seen.add(digest)
            unique.append(record)
    return unique, len(records) - len(unique)


def request_succeeded(case: dict[str, Any]) -> bool:
    requests = case.get("requests")
    if requests is None:
        requests = [case.get("request") or {}]
    expected_repeats = case.get("case", {}).get("repeats", len(requests))
    if not case.get("healthy") or len(requests) != expected_repeats or not requests:
        return False
    for index, request in enumerate(requests):
        usage = request.get("usage") or request
        reason = request.get("finish_reason") or {}
        if isinstance(reason, dict) and reason.get("type") in {"abort", "error"}:
            return False
        reasons = request.get("finish_reasons", [])
        expected_prompt = case.get("requested_prompt_tokens", 32760)
        batch_size = case.get("case", {}).get("batch_size", 1)
        if batch_size > 1:
            if request.get("sequence_count") != batch_size:
                return False
            if any(
                r.get("prompt_tokens") != expected_prompt // batch_size or r.get("completion_tokens") != 8
                for r in request.get("sequence_usage", [])
            ):
                return False
        if index == 0 and case.get("case", {}).get("image_probe", {}).get("after_text"):
            expected_prompt = case["case"]["prompt_tokens"]
        prompt = usage.get("prompt_tokens")
        prompt_matches = prompt == expected_prompt
        if expected_prompt is None:
            minimum = case.get("minimum_prompt_tokens")
            maximum = case.get("case", {}).get("max_model_len", 32768) - 8
            prompt_matches = (
                isinstance(prompt, int) and isinstance(minimum, int) and minimum <= prompt <= maximum
            )
        if not (
            request.get("success")
            and prompt_matches
            and usage.get("completion_tokens", 0) > 0
            and (
                case.get("requested_completion_tokens") is None
                or usage.get("completion_tokens") == case["requested_completion_tokens"]
            )
            and all(reason not in {"abort", "error"} for reason in reasons)
        ):
            return False
    return True


def summarize(case: dict[str, Any], engine: str) -> dict[str, Any]:
    if case.get("tensor_parallel_size", 1) > 1:
        raise ValueError("Tensor-parallel data requires tp_collect.py; a final rank is not an aggregate")
    phases = case.get("phase_summary", {})
    baseline = case.get("baseline", {})
    empty = not baseline.get("processes", ["unknown"])
    peak = max((v["peak_device_used_bytes"] for k, v in phases.items() if k != "startup"), default=None)
    native = case.get("native", [])
    row = {
        "case": case["case"]["name"],
        "model": case["case"]["model"],
        "revision": case["case"]["revision"],
        "mode": case["mode"],
        "engine": engine,
        "max_num_seqs": case.get("max_num_seqs", 1),
        "request_completed": request_succeeded(case),
        "request_count": len(case.get("requests", [case.get("request")])),
        "empty_device_baseline": empty,
        "sampled_runtime_footprint_bytes": peak - baseline["device_used_bytes"]
        if peak is not None and empty
        else None,
        "sampled_startup_footprint_bytes": phases["startup"]["peak_device_used_bytes"]
        - baseline["device_used_bytes"]
        if "startup" in phases and empty
        else None,
        "allocated_cache_bytes": None,
        "native_weight_load_delta_bytes": None,
        "native_activation_headroom_bytes": None,
        "native_graph_estimate_bytes": None,
        "native_consumed_bytes": None,
        "registered_parameter_bytes": None,
        "registered_buffer_bytes": None,
    }
    for event in native:
        if event.get("kind") == "cache_allocation":
            row["allocated_cache_bytes"] = event["planned_device_pool_bytes"]
        if event.get("kind") == "weight_storage_audit":
            row["registered_parameter_bytes"] = event["parameter_storage_bytes"]
            row["registered_buffer_bytes"] = event["buffer_storage_bytes"]
        cache = event.get("cache") or {}
        if cache.get("allocated_cache_bytes"):
            row["allocated_cache_bytes"] = cache["allocated_cache_bytes"]
        storage = event.get("model_storage") or {}
        if storage:
            row["registered_parameter_bytes"] = storage["parameter_storage_bytes"]
            row["registered_buffer_bytes"] = storage["buffer_storage_bytes"]
        for source, dest in [
            ("model_memory_usage_bytes", "native_weight_load_delta_bytes"),
            ("native_weight_load_delta_bytes", "native_weight_load_delta_bytes"),
            ("peak_activation_memory_bytes", "native_activation_headroom_bytes"),
            ("cudagraph_memory_estimate_bytes", "native_graph_estimate_bytes"),
            ("total_consumed_bytes", "native_consumed_bytes"),
        ]:
            if source in event:
                row[dest] = event[source]
    plans = [event for event in native if event.get("kind") == "cache_plan"]
    row["native_provenance_matches"] = engine != "vllm" or bool(
        plans and all(event.get("max_num_seqs") == row["max_num_seqs"] for event in plans)
    )
    row["usable_for_runtime_comparison"] = bool(
        row["native_provenance_matches"]
        and row["request_completed"]
        and empty
        and peak is not None
        and row["allocated_cache_bytes"]
    )
    # A native loading delta is broader than parameters; keep its residual named accordingly.
    row["residual_after_native_loading_and_cache_bytes"] = (
        row["sampled_runtime_footprint_bytes"]
        - row["native_weight_load_delta_bytes"]
        - row["allocated_cache_bytes"]
        if row["usable_for_runtime_comparison"] and row["native_weight_load_delta_bytes"] is not None
        else None
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(source: Path, output: Path, engine: str, token: str | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    raw = source.read_text(errors="replace")
    text = sanitize(raw, token)
    records, invalid = parse_records(text)
    planning_cases = [
        record for record in records if record.get("kind") == "case" and record.get("mode") == "plan"
    ]
    cases = [record for record in records if record.get("kind") == "case" and record.get("mode") != "plan"]
    rows = [summarize(case, engine) for case in cases]
    sample_counts: dict[tuple[str, str, int], int] = {}
    for record in records:
        if record.get("kind") not in {"case", "samples"}:
            continue
        name = record["case"]["name"] if isinstance(record["case"], dict) else record["case"]
        key = (name, record["mode"], record.get("max_num_seqs", 1))
        sample_counts[key] = sample_counts.get(key, 0) + len(record.get("samples", record.get("rows", [])))
    incomplete_series = []
    for case in cases:
        key = (case["case"]["name"], case["mode"], case.get("max_num_seqs", 1))
        phases = list(case.get("phase_summary", {}).values())
        expected = (
            sum(phase["sample_count"] for phase in phases)
            if phases and all("sample_count" in phase for phase in phases)
            else None
        )
        if sample_counts.get(key, 0) != expected:
            incomplete_series.append(
                {
                    "case": key[0],
                    "mode": key[1],
                    "max_num_seqs": key[2],
                    "expected_samples": expected,
                    "retained_samples": sample_counts.get(key, 0),
                }
            )
    (output / "native.log").write_text(text)
    # Driver samples are removed only after being retained as individual time-series rows.
    with gzip.open(output / "timeseries.csv.gz", "wt") as file:
        fields = [
            "case",
            "mode",
            "max_num_seqs",
            "time",
            "phase",
            "device_used_bytes",
            "device_total_bytes",
            "process_used_bytes",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record.get("kind") == "samples":
                samples = record.get("rows", [])
                name = record["case"]
            elif record.get("kind") == "case":
                samples = record.pop("samples", [])
                name = record["case"]["name"]
            else:
                continue
            for sample in samples:
                writer.writerow(
                    {
                        "case": name,
                        "mode": record["mode"],
                        "max_num_seqs": record.get("max_num_seqs", 1),
                        **{
                            key: sample.get(key)
                            for key in ["time", "phase", "device_used_bytes", "device_total_bytes"]
                        },
                        "process_used_bytes": sum(p["used_bytes"] for p in sample.get("processes", [])),
                    }
                )
    for filename, data in [
        ("results.jsonl", cases),
        ("planning.jsonl", planning_cases),
        ("environment.jsonl", [record for record in records if record.get("kind") == "environment"]),
        (
            "failures.jsonl",
            [case for case, row in zip(cases, rows) if not row["usable_for_runtime_comparison"]],
        ),
    ]:
        (output / filename).write_text("".join(json.dumps(row) + "\n" for row in data))
    write_csv(output / "measurements.csv", rows)
    manifest = {
        "schema_version": 1,
        "engine": engine,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "case_records": len(cases),
        "planning_records": len(planning_cases),
        "usable_runtime_comparisons": sum(row["usable_for_runtime_comparison"] for row in rows),
        "invalid_records": invalid,
        "environment_records": sum(record.get("kind") == "environment" for record in records),
        "start_records": sum(record.get("kind") == "start" for record in records),
        "incomplete_timeseries_cases": incomplete_series,
        "raw_source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "scope": "Single physical GPU per server. Runtime sampled peaks are lower bounds. Failed requests are excluded from runtime accuracy. Loading delta is not parameter storage. Stage peaks must not be added together.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def checksum(output: Path) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}\n" for path in files
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specification", type=Path)
    parser.add_argument("--publish", help="Destination hf://buckets/owner/bucket/experiment")
    args = parser.parse_args()
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    token = get_token()
    status = None
    if args.job_id:
        job = api.inspect_job(job_id=args.job_id)
        status = job.status.stage
        if status not in TERMINAL:
            print(json.dumps({"stage": status, "action": "Wait for this exact handle; do not relaunch"}))
            return
        args.output.mkdir(parents=True, exist_ok=True)
        args.source = args.output.parent / (args.output.name + "-private.log")
        args.source.write_text("\n".join(api.fetch_job_logs(job_id=args.job_id, tail=1000000)))
    if args.source is None:
        parser.error("Provide --source or --job-id")
    manifest = export(args.source, args.output, args.engine, token)
    manifest["job_stage"] = status
    if args.specification:
        specification = json.loads(args.specification.read_text())
        trials = specification.get("trials") or [
            {"case": case, "max_num_seqs": 1} for case in specification.get("cases", [])
        ]
        observed, _ = parse_records(args.source.read_text())
        identities = {
            (r["case"]["name"], r.get("max_num_seqs", 1))
            for r in observed
            if r.get("kind") == "case" and r.get("mode") != "plan"
        }
        manifest["missing_cases"] = [
            {"case": trial["case"]["name"], "max_num_seqs": trial["max_num_seqs"]}
            for trial in trials
            if (trial["case"]["name"], trial["max_num_seqs"]) not in identities
        ]
        (args.output / "specification.json").write_text(sanitize(args.specification.read_text(), token))
    manifest["complete_log_coverage"] = bool(
        manifest["environment_records"]
        and manifest["start_records"] >= manifest["case_records"]
        and not manifest["incomplete_timeseries_cases"]
        and not manifest.get("missing_cases")
        and not manifest["invalid_records"]
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ["collect.py", f"{args.engine}_driver.py", f"{args.engine}_hooks.py"]:
        (args.output / name).write_text(sanitize(Path(__file__).with_name(name).read_text(), token))
    checksum(args.output)
    if args.publish:
        if not manifest["complete_log_coverage"]:
            raise RuntimeError(
                "Incomplete log coverage: preserve as explicitly labeled partial evidence, not a gold release"
            )
        for path in args.output.rglob("*"):
            if path.is_file():
                content = (
                    gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
                )
                if SECRET.search(content) or (token and token in content):
                    raise RuntimeError(f"Credential pattern detected in {path.name}")
        api.sync_bucket(str(args.output), args.publish, delete=False, quiet=True)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
