"""Collect exact-native Job evidence locally; distinguish auto-reserved pools from workload storage."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from collect import parse_native, parse_records, sanitize


def collect(directory: Path, log_file: Path, token: str | None = None) -> dict[str, Any]:
    spec = json.loads((directory / "specification.json").read_text())
    for name, digest in spec.get("source_sha256", {}).items():
        source = directory / name
        if (
            not source.resolve().is_relative_to(directory.resolve())
            or hashlib.sha256(source.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Executed source audit failed")
    text = sanitize(log_file.read_text(errors="replace"), token)
    records, invalid = parse_records(text)
    output = directory / "evidence"
    output.mkdir(exist_ok=True)
    (output / "native.log.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
    result = next((x for x in reversed(records) if x["kind"] == "result"), None)
    done = any(x["kind"] == "done" for x in records)
    samples = [x for r in records if r["kind"] == "samples" for x in r["samples"]]
    offsets = [r["offset"] for r in records if r["kind"] == "samples"]
    samples_complete = (
        offsets == list(range(0, len(samples), 128))
        and result is not None
        and len(samples) == result["sample_count"]
    )
    raw = (
        text.split("HFMEM_LOG_START native\n", 1)[1].split("HFMEM_LOG_END native", 1)[0]
        if "HFMEM_LOG_START native\n" in text and "HFMEM_LOG_END native" in text
        else text
    )
    native, native_invalid = parse_native(raw)
    runners = [r for r in native if r.get("kind") == "runner_memory"]
    by_rank = {rank: [x for x in runners if x.get("rank") == rank] for rank in range(spec["gpus"])}
    audits = []
    for rank, events in by_rank.items():
        models = [x for x in events if "model_storage" in x]
        pools = [x for x in events if "allocated_cache_bytes" in x.get("cache", {})]
        request_pools = [x["request_pool_storage"] for x in events if x.get("request_pool_storage")]
        audits.append(
            {
                "rank": rank,
                "phases": sorted(set(x["phase"] for x in events)),
                "cuda_device_indices": sorted(set(x["cuda_device_index"] for x in events)),
                "model_storage": models[-1]["model_storage"] if models else None,
                "cache": pools[-1]["cache"] if pools else None,
                "request_pool_storage": request_pools[-1] if request_pools else None,
                "last_stage": events[-1] if events else None,
            }
        )
    for audit in audits:
        events = by_rank[audit["rank"]]
        load = next(
            (x for x in reversed(events) if x["phase"] == "load_model" and x.get("model_storage")), None
        )
        initial = next((x for x in events if x["phase"] == "init_torch_distributed"), None)
        audit["registered_weight_storage_complete"] = False
        if load and initial:
            model = load["model_storage"]
            registered = model.get(
                "model_storage_bytes", model["parameter_storage_bytes"] + model["buffer_storage_bytes"]
            )
            allocated = load["torch_allocated_bytes"] - initial["torch_allocated_bytes"]
            gap = allocated - registered
            audit["load_torch_allocated_delta_bytes"] = allocated
            audit["unregistered_or_nonparameter_load_bytes"] = gap
            audit["registered_weight_storage_complete"] = abs(gap) <= max(0.5 * 2**30, 0.05 * registered)
    trace_valid = all(
        len(x["devices"]) == spec["gpus"]
        and {d["index"] for d in x["devices"]} == set(range(spec["gpus"]))
        and x["used"] == sum(d["used"] for d in x["devices"])
        for x in samples
    )
    baseline = result.get("baseline", {}) if result else {}
    baseline_devices = baseline.get("devices", [])
    baseline_valid = (
        len(baseline_devices) == spec["gpus"]
        and not any(d.get("processes") for d in baseline_devices)
        and baseline.get("used") == sum(d["used"] for d in baseline_devices)
    )
    rank_startup_complete = all(
        any(
            x["phase"] == "init_torch_distributed" and x.get("tensor_parallel_size") == spec["gpus"]
            for x in events
        )
        for events in by_rank.values()
    )
    requests_ok = (
        result is not None
        and len(result["requests"]) == len(spec["prompt_lengths"])
        and all(
            x.get("success")
            and x.get("meta_info", {}).get("prompt_tokens") == x["requested_prompt_tokens"]
            and x.get("meta_info", {}).get("completion_tokens") == 8
            for x in result["requests"]
        )
    )
    weight_attention_complete = all(
        x["model_storage"] and x["cache"] and x["registered_weight_storage_complete"] for x in audits
    )
    auxiliary_cache_expected = any("ShortConv" in x.get("backend_type", "") for x in runners)
    auxiliary_complete = not auxiliary_cache_expected or all(x["request_pool_storage"] for x in audits)
    component_audits_complete = weight_attention_complete and auxiliary_complete
    summary: dict[str, Any] = {
        "done": done,
        "device_trace_valid": trace_valid,
        "empty_baseline_valid": baseline_valid,
        "all_rank_startup_audited": rank_startup_complete,
        "runtime_measurement_complete": bool(
            done
            and invalid == 0
            and samples_complete
            and trace_valid
            and baseline_valid
            and rank_startup_complete
            and requests_ok
            and result
            and result["healthy"]
        ),
        "invalid_transport_records": invalid,
        "invalid_native_records": native_invalid,
        "sample_count": len(samples),
        "samples_complete": samples_complete,
        "all_declared_requests_passed": requests_ok,
        "component_audits_complete": component_audits_complete,
        "weight_attention_audits_complete": weight_attention_complete,
        "auxiliary_pool_audit_required": auxiliary_cache_expected,
        "auxiliary_pool_audit_complete": auxiliary_complete,
        "healthy": result["healthy"] if result else None,
        "native_source": str(log_file),
        "scope": "Exact native user configuration with automatic cache reservation retained. Native pooled storage is not the estimator minimum workload KV. Runtime peaks use aggregate devices in each sampling iteration after an empty baseline. Device reads are sequential and 10 ms sampling is a lower bound. Text-only probes on a multimodal-enabled server do not establish the maximum image workload footprint.",
        "rank_audits": audits,
    }
    if result:
        baseline = result["baseline"]["used"]
        runtime = [v["peak_used"] for key, v in result["phase_summary"].items() if key != "startup"]
        summary.update(
            result=result,
            baseline_bytes=baseline,
            startup_peak_bytes=result["phase_summary"].get("startup", {}).get("peak_used", baseline) - baseline,
            runtime_peak_bytes=max(runtime) - baseline if runtime else None,
        )
    if all(x["cache"] for x in audits):
        summary["native_reported_cache_bytes"] = sum(x["cache"]["allocated_cache_bytes"] for x in audits)
    if all(x["request_pool_storage"] for x in audits):
        summary["native_request_state_pool_bytes"] = sum(
            x["request_pool_storage"]["model_storage_bytes"] for x in audits
        )
    if all(x["model_storage"] for x in audits):
        summary["registered_parameter_bytes"] = sum(
            x["model_storage"]["parameter_storage_bytes"] for x in audits
        )
        summary["registered_buffer_bytes"] = sum(x["model_storage"]["buffer_storage_bytes"] for x in audits)
    if not component_audits_complete:
        summary["residual_overhead_score"] = (
            "Omitted: incomplete module-storage/cache audit. Custom quantized weights can live outside registered parameters/buffers; load deltas are not exact weight storage."
        )
    if component_audits_complete:
        summary["native_parameter_bytes"] = sum(x["model_storage"]["parameter_storage_bytes"] for x in audits)
        summary["native_model_storage_bytes"] = sum(
            x["model_storage"].get(
                "model_storage_bytes",
                x["model_storage"]["parameter_storage_bytes"] + x["model_storage"]["buffer_storage_bytes"],
            )
            for x in audits
        )
        summary["native_buffer_bytes"] = sum(x["model_storage"]["buffer_storage_bytes"] for x in audits)
        summary["native_reported_cache_bytes"] = sum(x["cache"]["allocated_cache_bytes"] for x in audits)
        if summary.get("runtime_peak_bytes") is not None:
            summary["unattributed_runtime_residual_bytes"] = (
                summary["runtime_peak_bytes"]
                - summary["native_model_storage_bytes"]
                - summary["native_reported_cache_bytes"]
                - summary.get("native_request_state_pool_bytes", 0)
            )
            if summary["unattributed_runtime_residual_bytes"] < 0:
                summary["component_audits_complete"] = False
                summary["residual_overhead_score"] = (
                    "Omitted: measured footprint is below the audited weight/cache sum."
                )
                del summary["unattributed_runtime_residual_bytes"]

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "records.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in records if x["kind"] != "samples")
    )
    (output / "native-events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in native))
    (output / "samples.jsonl.gz").write_bytes(
        gzip.compress("".join(json.dumps(x) + "\n" for x in samples).encode(), mtime=0)
    )
    for record in records:
        if record["kind"] == "native_source":
            path = output / "native-source" / record["path"]
            if not path.resolve().is_relative_to((output / "native-source").resolve()):
                raise ValueError("Source path must stay inside evidence directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(record["content"])
    manifest = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (output / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {k: v for k, v in summary.items() if k not in {"rank_audits", "result"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import get_token

        token = get_token()
    except ImportError:
        token = None
    print(json.dumps(collect(args.directory, args.log_file, token), indent=2))


if __name__ == "__main__":
    main()
