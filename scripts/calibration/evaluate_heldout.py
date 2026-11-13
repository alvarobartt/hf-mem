"""Compare preregistered predictions with complete native held-out deployments offline."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from collect import request_succeeded, write_csv


def evaluate(bundle: Path, forecasts: dict[str, Any], aliases: dict[str, str]) -> list[dict[str, Any]]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest.get("complete_log_coverage", manifest.get("complete")), "Incomplete native evidence"
    specification = json.loads((bundle / "specification.json").read_text())
    for name, digest in specification["source_sha256"].items():
        assert hashlib.sha256((bundle / "executed" / name).read_bytes()).hexdigest() == digest, name
    rows = []
    for line in (bundle / "results.jsonl").read_text().splitlines():
        case = json.loads(line)
        assert case.get("mode") != "plan" and request_succeeded(case), "Incomplete workload"
        settings = case["case"]
        name = settings["name"]
        registered = forecasts[aliases.get(name, name)]
        prediction = registered["estimate"]
        args = registered["arguments"]
        assert prediction["revision"] == settings["revision"]
        assert prediction["model"] == settings["model"]
        for setting in ["max_model_len", "max_num_batched_tokens"]:
            assert args[setting] == settings[setting], setting
        assert args["max_num_seqs"] == case["max_num_seqs"]
        count = case["tensor_parallel_size"]
        baseline = case["baseline"]
        assert not baseline["processes"] and len(baseline["devices"]) == count
        native = case["native"]
        audits = [event for event in native if event["kind"] == "weight_storage_audit"]
        allocations = [event for event in native if event["kind"] == "cache_allocation"]
        graphs = [event for event in native if event["kind"] == "graph_allocation"]
        plans = [event for event in native if event["kind"] == "cache_plan"]
        assert len(plans) == 1 and plans[0]["worker_spec_count"] == count
        for setting in ["max_model_len", "max_num_batched_tokens", "max_num_seqs"]:
            assert plans[0][setting] == args[setting], setting
        for events, rank_field in [(audits, "rank"), (allocations, "worker_index"), (graphs, "rank")]:
            assert len(events) == count and {event[rank_field] for event in events} == set(range(count))
        for rank in range(count):
            for phase in [
                "init_device",
                "load_model",
                "determine_available_memory",
                "initialize_from_config",
                "compile_or_warm_up_model",
            ]:
                stages = [
                    event
                    for event in native
                    if event["kind"] == "worker_memory" and event["rank"] == rank and event["phase"] == phase
                ]
                assert len(stages) == 1 and stages[0]["tensor_parallel_size"] == count
        parameters = sum(event["parameter_storage_bytes"] for event in audits)
        cache = sum(event["planned_device_pool_bytes"] for event in allocations)
        phases = case["phase_summary"]
        runtime = max(v["peak_device_used_bytes"] for k, v in phases.items() if k != "startup")
        runtime -= baseline["device_used_bytes"]
        requests = case.get("requests", [case["request"]])
        repeated = len(requests) > 1
        peaks = []
        for index in range(len(requests)):
            suffix = f"_{index}" if repeated else ""
            names = ["full_context_request" + suffix, "after_request" + suffix]
            assert all(key in phases and phases[key]["sample_count"] > 0 for key in names)
            peaks.append(max(phases[key]["peak_device_used_bytes"] for key in names))
        peaks = [value - baseline["device_used_bytes"] for value in peaks]
        if repeated:
            token_ids = [request["prompt_token_id"] for request in requests]
            assert len(set(token_ids)) == len(requests), "Repeated requests share their entire prefix"
        predicted = prediction["fit"]["required"]
        assert (prediction["fit"].get("gpus") or 1) == count
        overhead = (prediction["fit"].get("overhead_per_gpu") or prediction["overhead"]["bytes"]) * count
        residual = runtime - parameters - cache
        row = {
            "case": name,
            "forecast_case": aliases.get(name, name),
            "model": settings["model"],
            "revision": settings["revision"],
            "tensor_parallel_size": count,
            "prefill_tokens": settings["max_num_batched_tokens"],
            "max_model_len": settings["max_model_len"],
            "request_count": len(requests),
            "prompt_tokens": [request["usage"]["prompt_tokens"] for request in requests],
            "completion_tokens": [request["usage"]["completion_tokens"] for request in requests],
            "sampled_runtime_bytes": runtime,
            "sampled_startup_bytes": phases["startup"]["peak_device_used_bytes"]
            - baseline["device_used_bytes"],
            "per_request_sampled_footprints_bytes": peaks,
            "mean_request_footprint_bytes": mean(peaks),
            "request_footprint_range_bytes": max(peaks) - min(peaks),
            "request_footprint_stddev_bytes": pstdev(peaks),
            "native_parameter_bytes": parameters,
            "native_buffer_bytes": sum(event["buffer_storage_bytes"] for event in audits),
            "native_physical_cache_bytes": cache,
            "actual_graph_bytes": sum(event["actual_graph_bytes"] for event in graphs),
            "native_residual_after_parameters_and_cache_bytes": residual,
            "frozen_predicted_total_bytes": predicted,
            "frozen_predicted_overhead_bytes": overhead,
            "total_error_pct": 100 * (predicted / runtime - 1),
            "overhead_error_pct": 100 * (overhead / residual - 1),
            "weight_error_bytes": prediction["weights"]["bytes"] - parameters,
            "logical_vs_physical_cache_error_pct": 100 * (prediction["kv_cache"]["bytes"] / cache - 1),
            "workload": "image" if settings.get("image_probe") else "text",
            "forecast_includes_encoder": bool(prediction["overhead"]["encoder"]),
        }
        if row["forecast_includes_encoder"] and row["workload"] == "text":
            encoder = count * prediction["overhead"]["encoder"]
            row["counterfactual_excluding_encoder_total_error_pct"] = 100 * (
                (predicted - encoder) / runtime - 1
            )
            row["counterfactual_excluding_encoder_overhead_error_pct"] = 100 * (
                (overhead - encoder) / residual - 1
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--extra-forecast", type=Path, action="append", default=[])
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    forecasts = {row["case"]: row for row in json.loads(args.forecasts.read_text())}
    for path in args.extra_forecast:
        record = json.loads(path.read_text())
        forecasts[record["specification"]["cases"][0]["name"]] = record
    aliases = dict(alias.split("=", 1) for alias in args.alias)
    rows = [row for bundle in args.bundles for row in evaluate(bundle, forecasts, aliases)]
    report = {
        "scope": "Predictions were frozen before submission. Samples are NVML lower bounds after an empty-device baseline; stage peaks are not additive. Repetitions share a warmed server and are not independent deployment replicates. Text-only use of a vision model is not a matched maximum-image workload; excluding its encoder reserve is an explicitly labeled counterfactual, not the actual hf-mem forecast. Logical cache excludes engine guard blocks and may differ by backend alignment or state checkpoints.",
        "observations": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(
        json.dumps(
            [{key: row[key] for key in ["case", "total_error_pct", "overhead_error_pct"]} for row in rows]
        )
    )


if __name__ == "__main__":
    main()
