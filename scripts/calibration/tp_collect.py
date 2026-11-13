"""Collect true tensor-parallel observations without treating one rank as the whole server."""

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any

from collect import (
    SECRET,
    TERMINAL,
    checksum,
    parse_native,
    parse_records,
    request_succeeded,
    sanitize,
    write_csv,
)


def summarize(case: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = case["tensor_parallel_size"]
    devices = case["baseline"]["devices"]
    assert len(devices) == count
    assert not any(device["processes"] for device in devices), "Device baseline is not empty"
    native = case.get("native", [])
    phases = [
        "init_device",
        "load_model",
        "determine_available_memory",
        "initialize_from_config",
        "compile_or_warm_up_model",
    ]
    missing = []
    rank_set = {event.get("rank") for event in native if event.get("kind") == "worker_memory"}
    allocation_set = {event.get("worker_index") for event in native if event.get("kind") == "cache_allocation"}
    if rank_set != set(range(count)) or allocation_set != set(range(count)):
        missing.append("unexpected_or_missing_rank_set")
    if any(
        event.get("tensor_parallel_size") != count for event in native if event.get("kind") == "worker_memory"
    ):
        missing.append("native_topology_mismatch")
    rank_rows = []
    all_samples = case.get("samples", [])
    runtime = [sample for sample in all_samples if sample["phase"] != "startup"]
    peak = max((sample["device_used_bytes"] for sample in runtime), default=None)
    for rank, baseline in enumerate(devices):
        stages = {
            phase: [
                event
                for event in native
                if event.get("kind") == "worker_memory"
                and event.get("rank") == rank
                and event.get("phase") == phase
            ]
            for phase in phases
        }
        audits = [
            event
            for event in native
            if event.get("kind") == "weight_storage_audit" and event.get("rank") == rank
        ]
        allocations = [
            event
            for event in native
            if event.get("kind") == "cache_allocation" and event.get("worker_index") == rank
        ]
        graphs = [
            event for event in native if event.get("kind") == "graph_allocation" and event.get("rank") == rank
        ]
        for phase, events in stages.items():
            if len(events) != 1:
                missing.append(f"rank{rank}:{phase}:{len(events)}")
        if len(audits) != 1 or len(allocations) != 1 or not graphs:
            missing.append(f"rank{rank}:audit/cache/graphs")
        if not audits or not allocations:
            continue
        observed = [
            next(d for d in sample["devices"] if d["device_index"] == baseline["device_index"])[
                "device_used_bytes"
            ]
            for sample in runtime
        ]
        used = max(observed) - baseline["device_used_bytes"] if observed else None
        audit = audits[-1]
        cache = allocations[-1]["planned_device_pool_bytes"]
        params = audit["parameter_storage_bytes"]
        row = {
            "case": case["case"]["name"],
            "tensor_parallel_size": count,
            "rank": rank,
            "device_index": baseline["device_index"],
            "sampled_runtime_footprint_bytes": used,
            "parameter_storage_bytes": params,
            "buffer_storage_bytes": audit["buffer_storage_bytes"],
            "physical_cache_bytes": cache,
            "residual_after_parameters_and_cache_bytes": used - params - cache if used is not None else None,
            "actual_graph_bytes": graphs[-1]["actual_graph_bytes"] if graphs else None,
        }
        for phase, events in stages.items():
            if events:
                for field in [
                    "torch_allocated_bytes",
                    "torch_reserved_bytes",
                    "torch_peak_allocated_bytes",
                    "cuda_free_bytes",
                    "cuda_total_bytes",
                    "model_memory_usage_bytes",
                ]:
                    row[phase + "_" + field] = events[-1].get(field)
        rank_rows.append(row)
    plans = [event for event in native if event.get("kind") == "cache_plan"]
    if len(plans) != 1 or plans[0].get("worker_spec_count") != count:
        missing.append("global_native_plan_worker_count")
    if plans and plans[0].get("max_num_seqs") != case.get("max_num_seqs"):
        missing.append("scheduler_capacity_mismatch")
    expected_samples = sum(phase["sample_count"] for phase in case.get("phase_summary", {}).values())
    if expected_samples != len(all_samples):
        missing.append("sample_count_mismatch")
    if any(len(sample["devices"]) != count for sample in all_samples):
        missing.append("sample_device_count_mismatch")
    weights = sum(row["parameter_storage_bytes"] for row in rank_rows)
    cache = sum(row["physical_cache_bytes"] for row in rank_rows)
    footprint = peak - case["baseline"]["device_used_bytes"] if peak is not None else None
    summary = {
        "case": case["case"]["name"],
        "model": case["case"]["model"],
        "revision": case["case"]["revision"],
        "tensor_parallel_size": count,
        "max_num_seqs": case.get("max_num_seqs"),
        "request_completed": request_succeeded(case),
        "request_count": len(case.get("requests", [case.get("request")])),
        "sample_count": len(all_samples),
        "complete_rank_evidence": not missing and len(rank_rows) == count,
        "missing_rank_evidence": missing,
        "sampled_simultaneous_runtime_footprint_bytes": footprint,
        "audited_parameter_storage_sum_bytes": weights,
        "registered_buffer_storage_sum_bytes": sum(row["buffer_storage_bytes"] for row in rank_rows),
        "native_physical_cache_sum_bytes": cache,
        "simultaneous_residual_after_parameters_and_cache_bytes": footprint - weights - cache
        if footprint is not None
        else None,
        "sum_individual_rank_peak_footprints_bytes": sum(
            row["sampled_runtime_footprint_bytes"] or 0 for row in rank_rows
        ),
    }
    summary["usable_for_accuracy"] = bool(summary["request_completed"] and summary["complete_rank_evidence"])
    return summary, rank_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--publish")
    args = parser.parse_args()
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    job = api.inspect_job(job_id=args.job_id)
    if job.status.stage not in TERMINAL:
        print(json.dumps({"stage": job.status.stage, "action": "Keep the exact existing handle"}))
        return
    text = "\n".join(api.fetch_job_logs(job_id=args.job_id, tail=1000000))
    (args.directory / "private.log").write_text(text)
    text = sanitize(text, get_token())
    records, invalid = parse_records(text)
    output = args.directory / "public"
    output.mkdir(exist_ok=True)
    (output / "native.log").write_text(text)
    cases = [r for r in records if r.get("kind") == "case"]
    blocks = re.findall(r"HFMEM_LOG_START [^\n]+\n(.*?)\nHFMEM_LOG_END [^\n]+", text, re.S)
    if len(blocks) != len(cases):
        raise RuntimeError("Native log block count does not match trial count")
    for case, block in zip(cases, blocks):
        recovered, native_invalid = parse_native(block)
        if native_invalid:
            raise RuntimeError("Malformed native event; do not silently lose rank evidence")
        case["native_event_recovery"] = {
            "driver_parsed_count": len(case["native"]),
            "raw_log_parsed_count": len(recovered),
            "source": "matching complete native log block",
        }
        case["native"] = recovered
    summaries, ranks = [], []
    for case in cases:
        summary, rows = summarize(case)
        summaries.append(summary)
        ranks.extend(rows)
    with gzip.open(output / "samples.jsonl.gz", "wt") as file:
        for case in cases:
            for sample in case.pop("samples", []):
                file.write(
                    json.dumps(
                        {
                            "case": case["case"]["name"],
                            "tensor_parallel_size": case["tensor_parallel_size"],
                            **sample,
                        }
                    )
                    + "\n"
                )
    (output / "results.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    (output / "environment.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records if r.get("kind") == "environment")
    )
    write_csv(output / "aggregate.csv", summaries)
    write_csv(output / "ranks.csv", ranks)
    spec = json.loads((args.directory / "specification.json").read_text())
    expected = set(spec["topologies"])
    seen = {row["tensor_parallel_size"] for row in summaries}
    manifest = {
        "job_stage": job.status.stage,
        "expected_topologies": sorted(expected),
        "observed_topologies": sorted(seen),
        "invalid_records": invalid,
        "complete": expected == seen and not invalid and all(row["usable_for_accuracy"] for row in summaries),
        "scope": "True tensor parallelism, one concurrent request; sequential repetitions may be specified. Aggregate peak is measured by summing same-sample device readings, not individual rank maxima. NVML samples are sequential within each sampling iteration, not hardware-synchronized instantaneous readings. Residual includes CUDA context, buffers, allocator and communication memory.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "specification.json").write_text(sanitize(json.dumps(spec, indent=2), get_token()))
    (output / "executed").mkdir(exist_ok=True)
    for name in spec["source_sha256"]:
        (output / "executed" / name).write_text(sanitize((args.directory / name).read_text(), get_token()))
    (output / "tp_collect.py").write_text(Path(__file__).read_text())
    (output / "collect.py").write_text(Path(__file__).with_name("collect.py").read_text())
    checksum(output)
    if args.publish:
        if not manifest["complete"]:
            raise RuntimeError("Missing rank/topology/request evidence; cannot publish a complete gold release")
        for path in output.rglob("*"):
            if path.is_file():
                content = (
                    gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
                )
                if SECRET.search(content) or (get_token() and get_token() in content):
                    raise RuntimeError("Credential detected")
        api.sync_bucket(str(output), args.publish, quiet=True, delete=False)
    print(json.dumps({"manifest": manifest, "summaries": summaries}))


if __name__ == "__main__":
    main()
