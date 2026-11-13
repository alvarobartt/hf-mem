"""Offline comparison of frozen forecasts with audited cross-engine matrix bytes."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from collect import write_csv


def error(predicted: int, observed: int) -> float:
    return 100 * (predicted / observed - 1)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for engine in ["all", "vllm", "sglang"]:
        selected = rows if engine == "all" else [row for row in rows if row["engine"] == engine]
        if not selected:
            continue
        values = {}
        for key in [
            "total_error_pct",
            "overhead_error_pct",
            "capacity_only_total_error_pct",
            "capacity_only_overhead_error_pct",
            "baseline_total_error_pct",
            "baseline_overhead_error_pct",
        ]:
            errors = sorted(abs(row[key]) for row in selected if row.get(key) is not None)
            if errors:
                values[key] = {
                    "median_absolute": median(errors),
                    "p90_absolute": errors[math.ceil(0.9 * len(errors)) - 1],
                    "max_absolute": max(errors),
                }
        result[engine] = {"configurations": len(selected), **values}
    return result


def evaluate(directory: Path, forecasts_name: str = "candidate-predictions.json") -> list[dict[str, Any]]:
    manifest = json.loads((directory / "public/manifest.json").read_text())
    forecasts = {row["case"]: row for row in json.loads((directory / forecasts_name).read_text())}
    baseline = {}
    if (directory / "baseline-predictions.json").exists():
        baseline = {
            row["case"]: row for row in json.loads((directory / "baseline-predictions.json").read_text())
        }
    spec = json.loads((directory / "specification.json").read_text())
    result = []
    for row in csv.DictReader((directory / "public/aggregate.csv").open()):
        if row["usable_for_accuracy"] != "True":
            continue
        prediction = forecasts[row["case"]]
        args = prediction["arguments"]
        estimate = prediction["estimate"]
        native = int(row["sampled_runtime_bytes"])
        residual = int(row["native_residual_bytes"])
        count = int(row["tensor_parallel_size"])
        assert args["model_id"] == row["model"] and args["revision"] == row["revision"]
        assert args["engine"] == row["engine"] == spec["engine"]
        for field, measured in [
            ("max_model_len", "max_model_len"),
            ("max_num_batched_tokens", "prefill_tokens"),
            ("max_num_seqs", "max_num_seqs"),
        ]:
            if args.get(field) is None and field != "max_model_len":
                flag = "native_scheduler_defaults" if field == "max_num_seqs" else "native_prefill_defaults"
                assert row.get(flag) == "True" and int(row[measured]) == 0, field
            else:
                assert args[field] == int(row[measured]), field
        if args.get("text_only"):
            assert row.get("workload") == "text", "Text-only forecast must match a text-only native workload"
        assert estimate["fit"]["gpus"] == count
        required = estimate["fit"]["required"]
        overhead = estimate["fit"]["overhead_per_gpu"] * count
        observed = {
            **row,
            "dataset": directory.name,
            "complete_dataset": manifest["complete"],
            "gpu_type": args["gpu_type"],
            "frozen_total_bytes": required,
            "frozen_overhead_bytes": overhead,
            "total_error_pct": error(required, native),
            "overhead_error_pct": error(overhead, residual),
            "overhead_error_MiB": (overhead - residual) / (1 << 20),
            "weight_error_bytes": estimate["weights"]["bytes"] - int(row["native_parameter_bytes"]),
            "reference_cache_error_bytes": estimate["kv_cache"]["bytes"]
            - int(row["native_physical_cache_bytes"]),
            # The top-level KV breakdown is a TP1 reference. Selected-topology
            # cache includes replication when TP exceeds the KV-head count.
            # Fitting can also add packed-weight tile padding; keep that caveat
            # explicit instead of silently calling every remainder cache.
            "topology_cache_and_weight_padding_error_bytes": required
            - estimate["weights"]["bytes"]
            - overhead
            - int(row["native_physical_cache_bytes"]),
        }
        capacity = prediction.get("capacity_only_estimate")
        if capacity:
            observed["capacity_only_total_error_pct"] = error(capacity["fit"]["required"], native)
            observed["capacity_only_overhead_error_pct"] = error(
                capacity["fit"]["overhead_per_gpu"] * count, residual
            )
        if previous := baseline.get(row["case"]):
            old = previous["estimate"]
            observed["baseline_total_error_pct"] = error(old["fit"]["required"], native)
            observed["baseline_overhead_error_pct"] = error(old["fit"]["overhead_per_gpu"] * count, residual)
        result.append(observed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--forecasts-name", default="candidate-predictions.json")
    args = parser.parse_args()
    if args.registration:
        registration = json.loads(args.registration.read_text())
        experiments = {row["directory"]: row for row in registration["experiments"]}
        for directory in args.directories:
            assert (
                json.loads((directory / "candidate-predictions.json").read_text())
                == experiments[directory.name]["candidate_predictions"]
            ), "Frozen forecast changed"
    rows = [row for directory in args.directories for row in evaluate(directory, args.forecasts_name)]
    report = {
        "scope": (
            "Candidate predictions were frozen locally before Job submission; publication was held pending explicit approval. "
            if args.registration
            else "Offline comparison of stored forecasts; prospective versus retrospective status follows each experiment's registration. "
        )
        + "Native complete-rank/token gates are independent of the forecasts. This evaluator uses stored forecasts, not current coefficients. Unsuccessful or partial datasets cannot establish complete coverage. Stage peaks are not additive; NVML peaks are sampled lower bounds; repeated requests share a warmed server. Logical versus physical cache and checkpoint versus runtime weight storage are reported separately.",
        "complete": all(
            json.loads((directory / "public/manifest.json").read_text())["complete"]
            for directory in args.directories
        ),
        "metrics": metrics(rows),
        "observations": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps({"complete": report["complete"], "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
