"""Evaluate frozen native sizing experiments without equating spare cache with requirements."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from hf_mem import Estimate, run


def metrics(values: list[float]) -> dict[str, Any]:
    absolute = sorted(abs(x) for x in values)
    return {
        "count": len(values),
        "median_absolute_error_pct": statistics.median(absolute) if absolute else None,
        "maximum_absolute_error_pct": max(absolute) if absolute else None,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def current_estimate(forecast: dict[str, Any], hardware: bool = True) -> Estimate:
    args = dict(forecast["arguments"])
    stored = forecast.get("hardware") or forecast.get("estimate")
    if hardware and stored.get("fit") and "gpu_memory" not in args:
        fit = stored["fit"]
        args["gpu_memory"] = f"{fit['gpus'] or 1}x{fit['gpu_memory'] / 2**30}"
        if gpu := (stored.get("overhead") or {}).get("gpu_type"):
            args["gpu_type"] = gpu
    result = run(**args)
    if not isinstance(result, Estimate):
        raise ValueError("Sizing replay requires a selected model file")
    return result


def block_text(directory: Path, case: str) -> str:
    text = (directory / "public/native.log").read_text(errors="replace")
    match = re.search(r"HFMEM_LOG_START " + re.escape(case) + r"[^\n]*\n(.*?)\nHFMEM_LOG_END", text, re.S)
    return match[1] if match else ""


def failure_reason(text: str) -> tuple[str, str | None]:
    if "assert override == blocks" in text:
        return "instrumentation_failure", "The old hook rejected a valid multi-request cache override."
    if "--language-model-only does not support" in text:
        return "unsupported_option", next(
            (x.strip() for x in text.splitlines() if "ValueError: --language-model-only" in x), None
        )
    patterns = [
        r"CUDA out of memory",
        r"out of memory",
        r"failed to allocate",
        r"not enough memory",
        r"available.*KV cache.*memory",
        r"KV cache.*(?:larger|needed|insufficient)",
        r"exceeds the maximum allowed length",
        r"No available memory for the cache blocks",
    ]
    for line in text.splitlines():
        if len(line) < 2000 and any(re.search(p, line, re.I) for p in patterns):
            return "memory_or_context_limit", line.strip()[:1000]
    return "inconclusive_failure", None


def unstarted_cancellation(directory: Path, verification: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Keep hardware-queue cancellations without turning them into engine outcomes."""
    job = next((j for j in verification["jobs"] if j["directory"] == directory.name), {})
    if (
        not job.get("verified")
        or job.get("stage") != "CANCELED"
        or "started_at" not in job
        or job["started_at"] is not None
    ):
        return None
    frozen = read_json(directory / "candidate-predictions.json")
    if isinstance(frozen, dict):
        frozen = [{"case": "native", **frozen}]
    spec = read_json(directory / "specification.json")
    return [
        {
            "directory": directory.name,
            "case": f["case"],
            "engine": spec.get("engine", "sglang"),
            "arguments": f["arguments"],
            "frozen_estimate": f.get("hardware") or f.get("estimate"),
            "cohort": "diagnostic",
            "outcome": "canceled_before_start",
            "request_completed": False,
            "measurement_complete": False,
            "failure_reason": "Job canceled while waiting for hardware; no engine execution or memory outcome.",
        }
        for f in frozen
    ]


def evaluate(directory: Path) -> list[dict[str, Any]]:
    spec = read_json(directory / "specification.json")
    predictions = directory / "candidate-predictions.json"
    if not predictions.exists():
        return []
    frozen = read_json(predictions)
    if isinstance(frozen, dict):
        frozen = [{"case": "native", **frozen}]
    forecasts = {f["case"]: f for f in frozen}
    engine = spec.get("engine", "sglang")
    native = (directory / "evidence/summary.json").exists()
    gguf = engine == "llama.cpp"
    if native:
        summary = read_json(directory / "evidence/summary.json")
        measurements = [
            {
                "case": "native",
                "request_completed": summary["runtime_measurement_complete"],
                "runtime_bytes": summary.get("runtime_peak_bytes"),
                "startup_bytes": summary.get("startup_peak_bytes"),
                "measurement_complete": summary["runtime_measurement_complete"],
                "cache_defaults": False,
            }
        ]
    elif gguf:
        if (directory / "collection-result.json").exists():
            gguf_rows = read_json(directory / "collection-result.json")["measurements"]
        else:
            gguf_rows = list(csv.DictReader((directory / "public/measurements.csv").open()))
            for item in gguf_rows:
                item["completed_full_token_counts"] = item["completed_full_token_counts"] == "True"
                for key in (
                    "sampled_runtime_or_ready_footprint_bytes",
                    "sampled_startup_footprint_bytes",
                    "bounded_free_memory_bytes",
                ):
                    item[key] = int(item[key]) if item.get(key) else None
        measurements = [
            {
                "case": row["case"],
                "request_completed": row["completed_full_token_counts"],
                "runtime_bytes": row["sampled_runtime_or_ready_footprint_bytes"],
                "startup_bytes": row["sampled_startup_footprint_bytes"],
                "measurement_complete": row["completed_full_token_counts"],
                "cache_defaults": False,
                "bounded_free_bytes": row.get("bounded_free_memory_bytes"),
            }
            for row in gguf_rows
        ]
    else:
        manifest = read_json(directory / "public/manifest.json")
        if manifest["invalid_records"] or manifest["missing_cases"] or manifest["unexpected_cases"]:
            raise ValueError("Incomplete transport or case set: " + directory.name)
        results = {
            x["case"]["name"]: x
            for x in map(json.loads, (directory / "public/results.jsonl").read_text().splitlines())
        }
        measurements = []
        for row in csv.DictReader((directory / "public/aggregate.csv").open()):
            case = results[row["case"]]
            measurements.append(
                {
                    "case": row["case"],
                    "request_completed": row["request_completed"] == "True",
                    "runtime_bytes": int(row["sampled_runtime_bytes"])
                    if row["sampled_runtime_bytes"]
                    else None,
                    "startup_bytes": case["phase_summary"]["startup"]["peak_device_used_bytes"]
                    - case["baseline"]["device_used_bytes"],
                    "native_cache_bytes": int(row["native_physical_cache_bytes"]),
                    "measurement_complete": row["usable_for_accuracy"] == "True",
                    "cache_defaults": bool(case["case"].get("native_cache_defaults")),
                    "request_error": (case.get("request") or {}).get("error"),
                    "request_body": (case.get("request") or {}).get("body"),
                }
            )
    output = []
    for observed in measurements:
        forecast = forecasts[observed["case"]]
        stored = forecast.get("hardware") or forecast.get("estimate")
        row = {
            "directory": directory.name,
            "case": observed["case"],
            "engine": engine,
            "arguments": forecast["arguments"],
            "frozen_estimate": stored,
            **observed,
        }
        row["cohort"] = "development" if directory.name.startswith("capacity-v1-") else "prospective"
        if "long-timeout" in row["case"] or directory.name.startswith("bounded-"):
            row["cohort"] = (
                "diagnostic"
                if "previous-budget" in row["case"] or "long-timeout" in row["case"]
                else "bounded_sufficiency"
            )
        completed = observed["request_completed"]
        if completed:
            row["outcome"] = "completed"
        else:
            text = block_text(directory, row["case"]) if not native else ""
            if "TimeoutError" in (observed.get("request_error") or ""):
                row["outcome"], row["failure_reason"] = "inconclusive_failure", "HTTP observation timeout"
            else:
                row["outcome"], row["failure_reason"] = failure_reason(
                    text + (observed.get("request_body") or "")
                )
            if row["outcome"] in {"instrumentation_failure", "unsupported_option"}:
                row["cohort"] = "diagnostic"
        try:
            estimate = current_estimate(forecast)
        except RuntimeError as error:
            row["current_estimate_error"] = str(error)
            row["current_refusal_matches_native"] = row["outcome"] == "unsupported_option"
            output.append(row)
            continue
        current = estimate.to_dict()
        row["current_estimate"] = current
        required = estimate.fit.required if estimate.fit else estimate.total
        previous = stored["fit"]["required"] if stored.get("fit") else stored["total"]
        row["frozen_allocation_replay_matches"] = required == previous
        row["fit_matches_native"] = (
            bool(estimate.fit.fits) == completed
            if completed or row["outcome"] == "memory_or_context_limit"
            else None
        )
        if forecast.get("bounded_budget_bytes"):
            row["fit_matches_native"] = None
        if observed["measurement_complete"] and observed["runtime_bytes"]:
            if observed["cache_defaults"]:
                expected_cache = estimate.fit.cache_bytes if estimate.fit else estimate.kv_cache.bytes
                row["non_kv_error_pct"] = (
                    (required - expected_cache) / (observed["runtime_bytes"] - observed["native_cache_bytes"])
                    - 1
                ) * 100
            else:
                row["total_error_pct"] = (required / observed["runtime_bytes"] - 1) * 100
                peak = max(observed["runtime_bytes"], observed["startup_bytes"] or 0)
                row["recommendation_covers_measured_peak"] = estimate.recommended_gpu_memory >= peak
        if forecast.get("unhinted") is not None and not native:
            unhinted = current_estimate(forecast, hardware=False)
            row["unhinted_estimate"] = unhinted.to_dict()
            if (
                estimate.fit
                and estimate.fit.gpus in {None, 1}
                and (completed or row["outcome"] == "memory_or_context_limit")
            ):
                row["unhinted_fit_matches_native"] = (
                    unhinted.recommended_gpu_memory <= estimate.fit.gpu_memory
                ) == completed
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    verification = read_json(args.directory / "job-verification.json")
    rows = []
    for path in sorted(args.directory.glob("*/candidate-predictions.json")):
        directory = path.parent
        if not (directory / "job.json").exists():
            continue
        if not any(
            (directory / p).exists()
            for p in [
                "evidence/summary.json",
                "collection-result.json",
                "public/aggregate.csv",
                "public/measurements.csv",
            ]
        ):
            if canceled := unstarted_cancellation(directory, verification):
                rows.extend(canceled)
                continue
            raise ValueError("Uncollected experiment: " + directory.name)
        rows.extend(evaluate(directory))
    primary = [r for r in rows if r["cohort"] == "prospective"]
    controlled = [r for r in primary if "total_error_pct" in r]
    defaults = [r for r in primary if "non_kv_error_pct" in r]
    fit_checks = [r for r in primary if r.get("fit_matches_native") is not None]
    bounded = [r for r in rows if r["cohort"] == "bounded_sufficiency"]
    result = {
        "scope": "Frozen forecasts and outcomes are retained. Current allocations are replayed separately; capacity policy now puts the whole allocation inside its headroom budget. Native default spare-KV reservations are excluded from total-error scores. Development data, instrumentation failures, unsupported options and timeout diagnostics are separate from prospective sizing accuracy.",
        "prospective_configurations": len(primary),
        "fit_checks": len(fit_checks),
        "fit_mismatches": [r["directory"] + "/" + r["case"] for r in fit_checks if not r["fit_matches_native"]],
        "unhinted_fit_mismatches": [
            r["directory"] + "/" + r["case"] for r in primary if r.get("unhinted_fit_matches_native") is False
        ],
        "controlled_allocation_accuracy": metrics([r["total_error_pct"] for r in controlled]),
        "default_non_kv_accuracy": metrics([r["non_kv_error_pct"] for r in defaults]),
        "frozen_allocations_replay": all(r.get("frozen_allocation_replay_matches") for r in primary),
        "recommendations_cover_controlled_peaks": all(
            r["recommendation_covers_measured_peak"] for r in controlled
        ),
        "bounded_recommendation_passed": bool(bounded)
        and all(
            r["request_completed"]
            and r["bounded_free_bytes"] <= r["frozen_estimate"]["recommended_gpu_memory"] + (2 << 20)
            for r in bounded
        ),
        "job_verification": verification["verified"],
        "observations": rows,
    }
    result["complete"] = (
        bool(primary)
        and not result["fit_mismatches"]
        and not result["unhinted_fit_mismatches"]
        and result["frozen_allocations_replay"]
        and result["recommendations_cover_controlled_peaks"]
        and result["bounded_recommendation_passed"]
        and result["job_verification"]
        and all(r.get("measurement_complete") or r["outcome"] == "memory_or_context_limit" for r in primary)
    )
    (args.directory / "sizing-evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
