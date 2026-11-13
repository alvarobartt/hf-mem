"""Evaluate scheduler scaling against fixed independent runtime measurements, offline."""

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from hf_mem import _overhead


def evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for reference in fixture["rows"]:
        overhead = _overhead.vllm(
            reference["config"],
            reference["moe"],
            reference["hybrid"],
            reference["tokens"],
            encoder_bytes=reference["encoder_bytes"],
            max_num_seqs=1,
        )
        measured = reference["native_runtime_footprint_bytes"]
        base = reference["estimated_weights_bytes"] + reference["estimated_kv_bytes"]
        measured_overhead = measured - reference["native_parameter_bytes"] - reference["native_cache_bytes"]
        rows.append(
            {
                "case": reference["case"]["name"],
                "model": reference["case"]["model"],
                "revision": reference["case"]["revision"],
                "original_total_error_pct": 100
                * ((base + reference["original_overhead_bytes"]) / measured - 1),
                "scheduler_total_error_pct": 100 * ((base + overhead.bytes) / measured - 1),
                "original_overhead_error_pct": 100
                * (reference["original_overhead_bytes"] / measured_overhead - 1),
                "scheduler_overhead_error_pct": 100 * (overhead.bytes / measured_overhead - 1),
                "scheduler_overhead_bytes": overhead.bytes,
                "native_residual_after_audited_parameters_and_cache_bytes": measured_overhead,
            }
        )
    return {
        "reference": fixture["source"],
        "reference_sha256": fixture["source_results_sha256"],
        "scope": fixture["scope"],
        "original_median_absolute_total_error_pct": median(abs(r["original_total_error_pct"]) for r in rows),
        "scheduler_median_absolute_total_error_pct": median(abs(r["scheduler_total_error_pct"]) for r in rows),
        "original_median_absolute_overhead_error_pct": median(
            abs(r["original_overhead_error_pct"]) for r in rows
        ),
        "scheduler_median_absolute_overhead_error_pct": median(
            abs(r["scheduler_overhead_error_pct"]) for r in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=Path(__file__).parents[2] / "tests/fixtures/overhead_scheduler.json"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(evaluate(json.loads(args.reference.read_text())), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
