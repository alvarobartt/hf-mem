"""Offline evaluation of measured TP1/2/4/8 and T4/H200 GGUF references."""

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from hf_mem import _overhead


def evaluate(reference_root: Path) -> dict[str, Any]:
    tp = json.loads((reference_root / "overhead_tensor_parallel.json").read_text())
    cpp = json.loads((reference_root / "overhead_llamacpp_hardware.json").read_text())
    single = _overhead.vllm(tp["config"], False, False, 8192, max_num_seqs=1)
    tp_rows = []
    for row in tp["observations"]:
        n = row["tensor_parallel_size"]
        fit = _overhead.fit(
            weights=tp["estimated_weights_bytes"],
            kv_fixed=0,
            kv_per_token=tp["estimated_cache_bytes"] / 32768,
            max_model_len=32768,
            context_length=32768,
            overhead=lambda count: (
                _overhead.vllm(
                    tp["config"], False, False, 8192, max_num_seqs=1, tensor_parallel_size=count
                ).bytes
            ),
            gpu_memory=f"{n}x139.8",
            utilization=0.92,
            context=single.context,
            kv_heads=8,
        )
        before = tp["estimated_weights_bytes"] + tp["estimated_cache_bytes"] + n * single.bytes
        actual = row["native_runtime_footprint_bytes"]
        tp_rows.append(
            {
                **row,
                "previous_required_bytes": before,
                "calibrated_required_bytes": fit.required,
                "calibrated_overhead_per_gpu_bytes": fit.overhead_per_gpu,
                "previous_total_error_pct": 100 * (before / actual - 1),
                "calibrated_total_error_pct": 100 * (fit.required / actual - 1),
                "previous_overhead_error_pct": 100 * (n * single.bytes / row["native_residual_bytes"] - 1),
                "calibrated_overhead_error_pct": 100
                * (n * (fit.overhead_per_gpu or 0) / row["native_residual_bytes"] - 1),
            }
        )
    cpp_rows = []
    for row in cpp["observations"]:
        o = _overhead.llama_cpp(gpu_memory=row["gpu_memory"])
        base = row["estimated_weights_bytes"] + row["estimated_cache_bytes"]
        actual = row["native_runtime_footprint_bytes"]
        residual = actual - row["native_model_bytes_rounded"] - row["native_cache_bytes_rounded"]
        cpp_rows.append(
            {
                **row,
                "calibrated_overhead_bytes": o.bytes,
                "previous_total_error_pct": 100 * ((base + row["original_overhead_bytes"]) / actual - 1),
                "calibrated_total_error_pct": 100 * ((base + o.bytes) / actual - 1),
                "previous_overhead_error_pct": 100 * (row["original_overhead_bytes"] / residual - 1),
                "calibrated_overhead_error_pct": 100 * (o.bytes / residual - 1),
            }
        )
    return {
        "scope": "Retrospective calibration evaluation, not an independent holdout or universal accuracy guarantee. TP uses one dense model and one scheduler/prefill configuration on H200; GGUF uses two models on T4/H200. Native cache pools include guard blocks/slots, while hf-mem estimates logical request capacity. NVML peaks are sampled lower bounds.",
        "tensor_parallel": tp_rows,
        "llamacpp_hardware": cpp_rows,
        "median_absolute_errors_pct": {
            kind: {
                metric: median(abs(row[metric]) for row in rows)
                for metric in [
                    "previous_total_error_pct",
                    "calibrated_total_error_pct",
                    "previous_overhead_error_pct",
                    "calibrated_overhead_error_pct",
                ]
            }
            for kind, rows in [("tensor_parallel", tp_rows), ("llamacpp_hardware", cpp_rows)]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path(__file__).parents[2] / "tests/fixtures")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(evaluate(args.reference_root), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
