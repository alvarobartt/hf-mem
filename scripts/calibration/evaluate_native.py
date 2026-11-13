"""Compare frozen forecasts against exact native configurations without conflating reserved KV with minimum KV."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def compare(root: Path) -> dict[str, Any]:
    path = root / "original-forecasts.json"
    original = json.loads(path.read_text())
    registration = json.loads((root / "registration.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != registration["forecasts_sha256"]:
        raise ValueError("Original frozen forecasts changed")
    experiments = {row["directory"]: row for row in registration["experiments"]}
    revised_path = root / "metadata-fix-forecasts.json"
    revised = (
        {x["gpus"]: x for x in json.loads(revised_path.read_text())["forecasts"]}
        if revised_path.exists()
        else {}
    )
    rows = []
    for forecast in original["forecasts"]:
        n = forecast["gpus"]
        directory = root / f"native-h200x{n}"
        spec_path = directory / "specification.json"
        if (
            hashlib.sha256(spec_path.read_bytes()).hexdigest()
            != experiments[directory.name]["specification_sha256"]
        ):
            raise ValueError("Native specification changed")
        spec = json.loads(spec_path.read_text())
        evidence = json.loads((directory / "evidence/summary.json").read_text())
        if not evidence["runtime_measurement_complete"]:
            raise ValueError("Native runtime/token/rank/sample gates failed")
        records = [json.loads(x) for x in (directory / "evidence/records.jsonl").read_text().splitlines()]
        controls = next(x["controls"] for x in records if x["kind"] == "resolved_controls")
        for key, expected in [
            ("context_length", forecast["context_length"]),
            ("tp_size", n),
            ("ep_size", n),
            ("attention_backend", "fa4"),
            ("dtype", "bfloat16"),
            ("page_size", 128),
            ("enable_multimodal", True),
        ]:
            if controls.get(key) != expected:
                raise ValueError(f"Resolved native control mismatch: {key}")
        if controls["max_running_requests"] != forecast["max_running_requests"]:
            raise ValueError("Requested scheduler controls differ")
        result = forecast["estimate"]
        args = forecast["arguments"]
        assert args["model_id"] == spec["model"] and args["revision"] == spec["revision"]
        assert args["max_model_len"] == spec["context_length"] and result["fit"]["gpus"] == n
        assert result["overhead"]["reference_engine"] == "sglang"
        predicted = result["fit"]["required"]
        kv = predicted - result["weights"]["bytes"] - n * result["fit"]["overhead_per_gpu"]
        # These exact configurations have no GQA replication or weight-tile padding.
        assert kv == result["kv_cache"]["bytes"]
        native = evidence["runtime_peak_bytes"]
        pool = evidence["native_reported_cache_bytes"]
        requests = evidence["result"]["requests"]
        row = {
            "gpus": n,
            "configured_context_length": forecast["context_length"],
            "requested_max_running_requests": forecast["max_running_requests"],
            "model": spec["model"],
            "revision": spec["revision"],
            "image": spec["image"],
            "job_id": json.loads((directory / "job.json").read_text())["job_id"],
            "frozen_required_bytes": predicted,
            "frozen_fit_result": result["fit"]["fits"],
            "frozen_weight_bytes": result["weights"]["bytes"],
            "frozen_minimum_kv_bytes": kv,
            "native_startup_peak_bytes": evidence["startup_peak_bytes"],
            "native_runtime_peak_bytes": native,
            "native_reserved_cache_bytes": pool,
            "native_noncache_bytes": native - pool,
            "frozen_noncache_bytes": predicted - kv,
            "raw_footprint_difference_pct": 100 * (predicted / native - 1),
            "noncache_footprint_difference_pct": 100 * ((predicted - kv) / (native - pool) - 1),
            "observed_deployment_fits": True,
            "largest_prompt_tokens": max(x["requested_prompt_tokens"] for x in requests),
            "generated_tokens_per_probe": 8,
            "probe_count": len(requests),
            "sample_count": evidence["sample_count"],
            "complete_runtime_evidence": True,
            "component_audit_complete": evidence["component_audits_complete"],
            "residual_overhead_error_pct": None,
            "native_mem_fraction_static": controls["mem_fraction_static"],
            "native_chunked_prefill_size": controls["chunked_prefill_size"],
            "native_swa_full_tokens_ratio": controls["swa_full_tokens_ratio"],
        }
        if n in revised:
            update = revised[n]
            assert update["arguments"] == args
            estimate = update["estimate"]
            row.update(
                retrospective_required_bytes=estimate["fit"]["required"],
                retrospective_minimum_kv_bytes=estimate["kv_cache"]["bytes"],
                retrospective_fit_result=estimate["fit"]["fits"],
                retrospective_layer_kinds=estimate["kv_cache"]["layers"],
            )
        rows.append(row)
    return {
        "scope": "Frozen forecasts preceded both Jobs. Exact requested image/model pin, TP=EP, FA4, BF16, page128 and multimodal-enabled controls were retained. Both native deployments completed all declared near-context-limit text probes. Raw aggregate GPU footprints subtract each empty baseline; stage peaks are not additive. Native automatic cache reservation is retained and differs from the estimator minimum workload KV. The non-cache comparison subtracts each side's own independently specified KV amount, not an inferred pure overhead score. Native custom expert weights are largely outside named parameters/buffers, so a residual-overhead error cannot be audited and is omitted. Metadata-fix forecasts are RETROSPECTIVE, computed after both outcomes, with no refitting of backend coefficients. Text probes do not validate maximum-image encoder peaks. NVML samples are lower bounds, and device queries within a sample are sequential.",
        "complete": True,
        "observations": rows,
        "original_source_sha256": original["source_sha256"],
        "registration_sha256": hashlib.sha256((root / "registration.json").read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.directory)
    (args.directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.directory / "comparison.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["observations"][0]))
        writer.writeheader()
        writer.writerows(report["observations"])
    for row in report["observations"]:
        print(
            json.dumps(
                {
                    "gpus": row["gpus"],
                    "frozen_GiB": row["frozen_required_bytes"] / 2**30,
                    "native_runtime_GiB": row["native_runtime_peak_bytes"] / 2**30,
                    "native_cache_GiB": row["native_reserved_cache_bytes"] / 2**30,
                    "raw_difference_pct": row["raw_footprint_difference_pct"],
                    "noncache_difference_pct": row["noncache_footprint_difference_pct"],
                    "retrospective_required_GiB": row.get("retrospective_required_bytes", 0) / 2**30,
                }
            )
        )


if __name__ == "__main__":
    main()
