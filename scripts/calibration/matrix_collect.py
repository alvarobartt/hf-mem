"""Audit matched vLLM/SGLang matrices on single GPUs and actual TP deployments."""

import argparse
import ast
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from collect import (
    TERMINAL,
    checksum,
    parse_native,
    parse_records,
    request_succeeded,
    sanitize,
    unique_records,
    write_csv,
)


def startup_settings(block: str) -> tuple[dict[str, Any], str] | None:
    """Hub transport wraps long native startup dictionaries at 16 KiB boundaries."""
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if "server_args={" not in line:
            continue
        pieces = [line.split("server_args=", 1)[1]]
        for offset in range(64):
            try:
                values = ast.literal_eval("".join(pieces))
            except (ValueError, SyntaxError):
                if index + offset + 1 >= len(lines):
                    break
                following = lines[index + offset + 1]
                if following.startswith(("HFMEM_", "[202")):
                    break
                pieces.append(following)
                continue
            if isinstance(values, dict):
                return values, hashlib.sha256("\n".join(lines[index : index + offset + 1]).encode()).hexdigest()
            break
    return None


def summarize(case: dict[str, Any], engine: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    count = case["tensor_parallel_size"]
    native = case["native"]
    baseline = case["baseline"]
    samples = case.get("samples", [])
    missing = []
    if len(baseline["devices"]) != count or baseline["processes"]:
        missing.append("device_baseline")
    if len(samples) != sum(v["sample_count"] for v in case["phase_summary"].values()):
        missing.append("sample_count")
    if not samples or any(len(s["devices"]) != count for s in samples):
        missing.append("sample_devices")
    if any(s["device_used_bytes"] != sum(d["device_used_bytes"] for d in s["devices"]) for s in samples):
        missing.append("sample_aggregate")
    kind = "worker_memory" if engine == "vllm" else "runner_memory"
    phases = (
        [
            "init_device",
            "load_model",
            "determine_available_memory",
            "initialize_from_config",
            "compile_or_warm_up_model",
        ]
        if engine == "vllm"
        else ["init_torch_distributed", "load_model", "alloc_memory_pool", "initialize", "init_cuda_graphs"]
    )
    observed = {e.get("rank") for e in native if e.get("kind") == kind}
    if observed != set(range(count)):
        missing.append("rank_set")
    ranks = []
    for rank in range(count):
        events = [e for e in native if e.get("kind") == kind and e.get("rank") == rank]
        if any(e.get("tensor_parallel_size") != count for e in events):
            missing.append(f"rank{rank}:topology")
        for phase in phases:
            selected = [e for e in events if e["phase"] == phase]
            if not selected or (engine == "vllm" and len(selected) != 1):
                missing.append(f"rank{rank}:{phase}")
        storage, cache = None, None
        if engine == "vllm":
            audits = [e for e in native if e.get("kind") == "weight_storage_audit" and e.get("rank") == rank]
            pools = [e for e in native if e.get("kind") == "cache_allocation" and e.get("worker_index") == rank]
            graphs = [e for e in native if e.get("kind") == "graph_allocation" and e.get("rank") == rank]
            if len(audits) == len(pools) == len(graphs) == 1:
                storage = audits[0]
                cache = pools[0]["planned_device_pool_bytes"]
            else:
                missing.append(f"rank{rank}:audit/pool/graph")
        else:
            loaded = [e for e in events if e["phase"] == "load_model" and e.get("model_storage")]
            allocated = [e for e in events if e["phase"] == "alloc_memory_pool" and e.get("cache")]
            initialized = [e for e in events if e["phase"] == "init_cuda_graphs"]
            if loaded and allocated and initialized:
                storage = loaded[-1]["model_storage"]
                # Use the final backed pool when the engine defers physical allocation.
                backed = [e for e in events if e.get("cache") and e["cache"].get("allocated_cache_bytes")]
                cache = backed[-1]["cache"]["allocated_cache_bytes"] if backed else None
            else:
                missing.append(f"rank{rank}:audit/pool/graph")
            for e in events:
                if e.get("context_length") != case["case"]["max_model_len"]:
                    missing.append(f"rank{rank}:context")
        if storage is None or cache is None:
            continue
        row = {
            "case": case["case"]["name"],
            "engine": engine,
            "rank": rank,
            "parameter_bytes": storage["parameter_storage_bytes"],
            "buffer_bytes": storage["buffer_storage_bytes"],
            "cache_bytes": cache,
        }
        for phase in phases:
            selected = [e for e in events if e["phase"] == phase]
            if selected:
                event = selected[-1]
                for key in [
                    "torch_allocated_bytes",
                    "torch_reserved_bytes",
                    "cuda_free_bytes",
                    "cuda_total_bytes",
                ]:
                    row[phase + "_" + key] = event.get(key)
                if event.get("cuda_total_bytes") is not None:
                    row[phase + "_outside_torch_reserved_bytes"] = (
                        event["cuda_total_bytes"] - event["cuda_free_bytes"] - event["torch_reserved_bytes"]
                    )
        ranks.append(row)
    resolved_sequences, resolved_prefill = None, None
    if engine == "vllm":
        plans = [e for e in native if e.get("kind") == "cache_plan"]
        if len(plans) != 1 or plans[0].get("worker_spec_count") != count:
            missing.append("native_plan")
        else:
            resolved_sequences = plans[0].get("max_num_seqs")
            resolved_prefill = plans[0].get("max_num_batched_tokens")
        if len(plans) == 1 and any(
            plans[0].get(key) != expected
            for key, expected in [
                (
                    "max_num_seqs",
                    None if case["case"].get("native_scheduler_defaults") else case["max_num_seqs"],
                ),
                ("max_model_len", case["case"]["max_model_len"]),
                (
                    "max_num_batched_tokens",
                    None
                    if case["case"].get("native_prefill_defaults")
                    else case["case"]["max_num_batched_tokens"],
                ),
            ]
            if expected is not None
        ):
            missing.append("native_plan_settings")
    else:
        settings = case.get("server_args", {})
        expected = {
            "tp_size": count,
            "context_length": case["case"]["max_model_len"],
            "chunked_prefill_size": case["case"].get("max_num_batched_tokens", 0),
            "max_running_requests": case["max_num_seqs"],
        }
        if case["case"].get("native_scheduler_defaults"):
            expected.pop("max_running_requests")
        if case["case"].get("native_prefill_defaults"):
            expected.pop("chunked_prefill_size")
        if case["case"].get("native_context_defaults"):
            # The startup argument remains null when omitted. The per-rank
            # model_config.context_len audit above verifies the resolved value.
            expected.pop("context_length")
        resolved_sequences = settings.get("max_running_requests")
        # SGLang may report null for the requested auto capacity. Its runtime
        # memory-pool plan independently records the capacity actually allocated.
        if resolved_sequences is None:
            capacities = {
                e["cache"]["memory_pool_config"].get("max_running_requests")
                for e in native
                if isinstance((e.get("cache") or {}).get("memory_pool_config"), dict)
            }
            if len(capacities) == 1:
                resolved_sequences = capacities.pop()
        resolved_prefill = settings.get("chunked_prefill_size")
        if any(settings.get(k) != v for k, v in expected.items()):
            missing.append("native_server_settings")
    if not isinstance(resolved_sequences, int) or resolved_sequences <= 0:
        missing.append("resolved_scheduler_capacity")
    if not isinstance(resolved_prefill, int) or resolved_prefill <= 0:
        missing.append("resolved_prefill_capacity")
    runtime = [s["device_used_bytes"] for s in samples if s["phase"] != "startup"]
    footprint = max(runtime) - baseline["device_used_bytes"] if runtime else None
    weights = sum(r["parameter_bytes"] for r in ranks)
    cache = sum(r["cache_bytes"] for r in ranks)
    if footprint is None or footprint < weights + cache:
        missing.append("runtime_below_audited_weights_and_cache")
    if engine == "sglang" and any((e.get("cache") or {}).get("post_capture_active") for e in native):
        if not any(
            e.get("phase") == "forward" and (e.get("forward") or {}).get("after_graph_capture") for e in native
        ):
            missing.append("deferred_pool_missing_post_capture_observation")
    row = {
        "case": case["case"]["name"],
        "model": case["case"]["model"],
        "revision": case["case"]["revision"],
        "engine": engine,
        "tensor_parallel_size": count,
        "max_num_seqs": case["max_num_seqs"],
        "max_model_len": case["case"]["max_model_len"],
        "prefill_tokens": case["case"].get("max_num_batched_tokens", 0),
        "resolved_max_num_seqs": resolved_sequences,
        "resolved_prefill_tokens": resolved_prefill,
        "native_scheduler_defaults": bool(case["case"].get("native_scheduler_defaults")),
        "native_prefill_defaults": bool(case["case"].get("native_prefill_defaults")),
        "request_count": len(case.get("requests", [case.get("request")])),
        "workload": (
            "mixed-text-image"
            if case["case"].get("image_probe", {}).get("after_text")
            else "image"
            if case["case"].get("image_probe")
            else "text"
        ),
        "request_completed": request_succeeded(case),
        "native_parameter_bytes": weights,
        "native_buffer_bytes": sum(r["buffer_bytes"] for r in ranks),
        "native_physical_cache_bytes": cache,
        "sampled_runtime_bytes": footprint,
        "native_residual_bytes": footprint - weights - cache if footprint is not None else None,
        "missing_evidence": sorted(set(missing)),
    }
    row["usable_for_accuracy"] = bool(row["request_completed"] and not missing and len(ranks) == count)
    return row, ranks


def export(text: str, directory: Path, token: str | None = None) -> dict[str, Any]:
    output = directory / "public"
    output.mkdir(exist_ok=True)
    text = sanitize(text, token)
    (output / "native.log").write_text(text)
    records, invalid = parse_records(text)
    records, duplicate_records = unique_records(records)
    cases = [r for r in records if r.get("kind") == "case" and r.get("mode") != "plan"]
    planning = [r for r in records if r.get("kind") == "case" and r.get("mode") == "plan"]
    blocks = list(
        dict.fromkeys(re.findall(r"HFMEM_LOG_START ([^\n]+)\n(.*?)\nHFMEM_LOG_END [^\n]+", text, re.S))
    )
    block_map = {name: block for name, block in blocks}
    runtime_blocks = [(name, block) for name, block in blocks if not name.endswith(" plan")]
    (output / "planning.jsonl").write_text("".join(json.dumps(r) + "\n" for r in planning))
    spec = json.loads((directory / "specification.json").read_text())
    engine = spec["engine"]
    summaries, ranks = [], []
    for case in cases:
        key = case["case"]["name"] + " " + case["mode"]
        block = block_map.get(key, "")
        recovered, bad = parse_native(block)
        invalid += bad
        case["native_event_recovery"] = {"driver_count": len(case["native"]), "raw_count": len(recovered)}
        case["native"] = recovered
        if engine == "sglang" and not any(case.get("server_args", {}).values()):
            # v0.5.23 flattens /server_info. Older drivers looked for a nested
            # server_args object. Recover the independently logged resolved startup
            # dict from this same trial; never synthesize values from expectations.
            if found := startup_settings(block):
                startup, digest = found
                case["server_args_recovery"] = {
                    "source": "matching native resolved startup server_args log",
                    "transport_lines_sha256": digest,
                    "original_readback": case.get("server_args"),
                }
                keys = case.get("server_args", {}) or dict.fromkeys(
                    ("tp_size", "context_length", "chunked_prefill_size", "max_running_requests")
                )
                case["server_args"] = {key: startup.get(key) for key in keys}
        row, rank_rows = summarize(case, engine)
        summaries.append(row)
        ranks.extend(rank_rows)
    with gzip.open(output / "samples.jsonl.gz", "wt") as file:
        for case in cases:
            for sample in case.pop("samples", []):
                file.write(json.dumps({"case": case["case"]["name"], **sample}) + "\n")
    (output / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cases))
    (output / "environment.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records if r.get("kind") == "environment")
    )
    write_csv(output / "aggregate.csv", summaries)
    write_csv(output / "ranks.csv", ranks)
    observed = {(r["case"], r["tensor_parallel_size"], r["max_num_seqs"]) for r in summaries}
    expected = {(t["case"]["name"], t["tensor_parallel_size"], t["max_num_seqs"]) for t in spec["trials"]}
    manifest = {
        "invalid_records": invalid,
        "duplicate_transport_records": duplicate_records,
        "environment_records": sum(r.get("kind") == "environment" for r in records),
        "start_records": sum(r.get("kind") == "start" for r in records),
        "missing_cases": sorted(expected - observed),
        "unexpected_cases": sorted(observed - expected),
        "complete": expected == observed
        and len(cases) == len(expected) == len(runtime_blocks)
        and sum(r.get("kind") == "environment" for r in records) == len(expected)
        and sum(r.get("kind") == "start" for r in records) == len(expected) + len(planning)
        and not invalid
        and all(r["usable_for_accuracy"] for r in summaries),
        "scope": "Matched engine matrix with actual TP ranks. Simultaneous aggregate sums sequential device reads in each NVML iteration. Peaks are sampled lower bounds; baseline is empty; stage peaks are not additive. Residual subtracts registered parameters and physical native pools, not broad weight-load deltas.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "specification.json").write_text(json.dumps(spec, indent=2) + "\n")
    executed = output / "executed"
    executed.mkdir(exist_ok=True)
    for name, digest in spec["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Executed source path escapes bundle")
        source_root = directory if (directory / name).exists() else directory / "executed"
        source = source_root / name
        if not source.resolve().is_relative_to(source_root.resolve()):
            raise ValueError("Executed source symlink escapes bundle")
        data = source.read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        (executed / name).write_bytes(data)
    for name in ["matrix_collect.py", "collect.py"]:
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    checksum(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, help="Audit a retained log offline instead of fetching a Job")
    args = parser.parse_args()
    if args.log_file is not None:
        data = args.log_file.read_bytes()
        if args.log_file.suffix == ".gz":
            data = gzip.decompress(data)
        print(json.dumps(export(data.decode(), args.directory)))
        return
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    job_id = json.loads((args.directory / "job.json").read_text())["job_id"]
    job = api.inspect_job(job_id=job_id)
    if job.status.stage not in TERMINAL:
        print(json.dumps({"stage": job.status.stage, "job_id": job_id}))
        return
    text = "\n".join(api.fetch_job_logs(job_id=job_id, tail=1000000))
    (args.directory / "private.log").write_text(text)
    manifest = export(text, args.directory, get_token())
    selected = "terminal API replay"
    live = args.directory / "live-private.log"
    live_digest = None
    if live.exists():
        live_text = live.read_text()
        live_digest = hashlib.sha256(live_text.encode()).hexdigest()
        if not manifest["complete"]:
            retained = export(live_text, args.directory, get_token())
            if retained["complete"] or len(retained["missing_cases"]) < len(manifest["missing_cases"]):
                manifest, selected = retained, "locally retained live event stream"
            else:
                manifest = export(text, args.directory, get_token())
    provenance = {
        "selected": selected,
        "terminal_replay_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "live_stream_sha256": live_digest,
        "scope": "No missing observations are synthesized. A locally retained stream may replace a truncated terminal replay only after the same independent rank/token/sample/control gates. Original traces remain local.",
    }
    (args.directory / "public/collection-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    checksum(args.directory / "public")
    print(json.dumps({"stage": job.status.stage, **manifest, "log_source": selected}))


if __name__ == "__main__":
    main()
