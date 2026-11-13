"""Regression checks for rejecting misleading calibration records and secret exports."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "calibration_collect", Path(__file__).parents[1] / "scripts/calibration/collect.py"
)
assert spec is not None and spec.loader is not None
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def case() -> dict[str, Any]:
    return {
        "kind": "case",
        "case": {"name": "dense", "model": "org/dense", "revision": "a" * 40},
        "mode": "single_sequence",
        "max_num_seqs": 1,
        "healthy": True,
        "request": {"success": True, "usage": {"prompt_tokens": 32760, "completion_tokens": 8}},
        "baseline": {"device_used_bytes": 100, "processes": []},
        "phase_summary": {
            "startup": {"peak_device_used_bytes": 900},
            "ready": {"peak_device_used_bytes": 600},
            "after_request": {"peak_device_used_bytes": 700},
        },
        "native": [
            {"kind": "cache_plan", "max_num_seqs": 1},
            {"kind": "cache_allocation", "planned_device_pool_bytes": 200},
            {"kind": "worker_memory", "model_memory_usage_bytes": 300},
        ],
    }


def test_transport_wrapped_records_and_truncation() -> None:
    record = json.dumps(case())
    parsed, invalid = collect.parse_records("HFMEM_RESULT " + record[:73] + "\n" + record[73:])
    assert parsed == [case()] and invalid == 0
    parsed, invalid = collect.parse_records('HFMEM_RESULT {"kind":"case"\nHFMEM_RESULT {"kind":"done"}')
    assert parsed == [{"kind": "done"}] and invalid == 1


def test_runtime_comparison_excludes_startup_peak_and_does_not_add_stage_peaks() -> None:
    row = collect.summarize(case(), "vllm")
    assert row["usable_for_runtime_comparison"]
    assert row["sampled_runtime_footprint_bytes"] == 600
    assert row["sampled_startup_footprint_bytes"] == 800
    assert row["residual_after_native_loading_and_cache_bytes"] == 100


@pytest.mark.parametrize(
    "failure", ["timeout", "abort", "no_decode", "wrong_prompt", "dirty_baseline", "foreign_stage"]
)
def test_unusable_observations_are_not_gold_runtime_comparisons(failure: str) -> None:
    record = case()
    if failure == "timeout":
        record["request"] = {"success": False, "error": "TimeoutError"}
    elif failure == "abort":
        record["request"]["finish_reason"] = {"type": "abort"}
    elif failure == "no_decode":
        record["request"]["usage"]["completion_tokens"] = 0
    elif failure == "wrong_prompt":
        record["request"]["usage"]["prompt_tokens"] = 10
    elif failure == "dirty_baseline":
        record["baseline"]["processes"] = [{"pid": 1}]
    else:
        record["native"][0]["max_num_seqs"] = 256
    row = collect.summarize(record, "vllm")
    assert not row["usable_for_runtime_comparison"]
    assert row["residual_after_native_loading_and_cache_bytes"] is None


def test_export_retains_failed_cases_and_sanitizes_both_logs_and_json(tmp_path: Path) -> None:
    token = "hf_" + "x" * 32
    record = case()
    record["request"] = {"success": False, "error": f"Bearer {token}"}
    source = tmp_path / "private.log"
    source.write_text("HFMEM_RESULT " + json.dumps(record))
    output = tmp_path / "public"
    manifest = collect.export(source, output, "vllm", token)
    assert manifest["case_records"] == 1 and manifest["usable_runtime_comparisons"] == 0
    assert json.loads((output / "failures.jsonl").read_text())["request"]["success"] is False
    for name in ["native.log", "results.jsonl", "failures.jsonl"]:
        assert token not in (output / name).read_text()
        assert "[REDACTED]" in (output / name).read_text()


def test_missing_raw_samples_are_detected_even_when_case_json_is_valid(tmp_path: Path) -> None:
    record = case()
    for phase in record["phase_summary"].values():
        phase["sample_count"] = 32
    source = tmp_path / "partial.log"
    source.write_text("HFMEM_RESULT " + json.dumps(record))
    manifest = collect.export(source, tmp_path / "public", "vllm")
    assert manifest["invalid_records"] == 0
    assert manifest["incomplete_timeseries_cases"][0]["expected_samples"] == 96
    assert manifest["incomplete_timeseries_cases"][0]["retained_samples"] == 0


def test_single_gpu_collector_rejects_tensor_parallel_aggregate() -> None:
    record = case()
    record["tensor_parallel_size"] = 2
    with pytest.raises(ValueError, match="final rank is not an aggregate"):
        collect.summarize(record, "vllm")


def test_tensor_parallel_aggregate_uses_same_sample_device_sum_and_requires_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    tp_collect = importlib.import_module("tp_collect")

    record = case()
    record["tensor_parallel_size"] = 2
    record["baseline"] = {
        "device_used_bytes": 200,
        "devices": [{"device_index": i, "device_used_bytes": 100, "processes": []} for i in range(2)],
    }
    record["samples"] = [
        {
            "phase": "request",
            "device_used_bytes": sum(values),
            "devices": [{"device_index": i, "device_used_bytes": used} for i, used in enumerate(values)],
        }
        for values in [(800, 400), (500, 700)]
    ]
    record["phase_summary"] = {"request": {"sample_count": 2}}
    record["native"] = [{"kind": "cache_plan", "worker_spec_count": 2, "max_num_seqs": 1}]
    for rank in range(2):
        record["native"].extend(
            [
                {
                    "kind": "weight_storage_audit",
                    "rank": rank,
                    "parameter_storage_bytes": 200,
                    "buffer_storage_bytes": 10,
                },
                {"kind": "cache_allocation", "worker_index": rank, "planned_device_pool_bytes": 100},
                {"kind": "graph_allocation", "rank": rank, "actual_graph_bytes": 10},
                *[
                    {"kind": "worker_memory", "rank": rank, "tensor_parallel_size": 2, "phase": phase}
                    for phase in [
                        "init_device",
                        "load_model",
                        "determine_available_memory",
                        "initialize_from_config",
                        "compile_or_warm_up_model",
                    ]
                ],
            ]
        )
    summary, ranks = tp_collect.summarize(record)
    assert summary["usable_for_accuracy"] and len(ranks) == 2
    assert summary["sampled_simultaneous_runtime_footprint_bytes"] == 1000
    assert summary["sum_individual_rank_peak_footprints_bytes"] == 1300
    assert summary["simultaneous_residual_after_parameters_and_cache_bytes"] == 400
    record["native"] = [
        event
        for event in record["native"]
        if not (event.get("kind") == "weight_storage_audit" and event.get("rank") == 1)
    ]
    assert not tp_collect.summarize(record)[0]["usable_for_accuracy"]


def test_adjacent_worker_json_on_one_line_preserves_both_ranks() -> None:
    first = {"kind": "worker_memory", "rank": 0, "phase": "initialize_from_config"}
    second = {**first, "rank": 1}
    events, invalid = collect.parse_native(
        "prefix HFMEM_NATIVE " + json.dumps(first) + "HFMEM_NATIVE " + json.dumps(second)
    )
    assert events == [first, second] and invalid == 0


def test_llamacpp_component_audit_excludes_sizing_pass_and_host_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    llamacpp_collect = importlib.import_module("llamacpp_collect")

    native = """load_tensors: CUDA0 model buffer size = 0.00 MiB
llama_kv_cache: CUDA0 KV buffer size = 0.00 MiB
sched_reserve: CUDA0 compute buffer size = 2.00 MiB
load_tensors: CPU_Mapped model buffer size = 5.00 MiB
load_tensors: CUDA0 model buffer size = 10.00 MiB
llama_kv_cache: CUDA0 KV buffer size = 20.00 MiB
llama_memory: CUDA0 RS buffer size = 3.00 MiB
sched_reserve: CUDA0 compute buffer size = 2.00 MiB
sched_reserve: CUDA0 compute buffer size = 4.00 MiB
"""
    result = llamacpp_collect.buffers(native)
    assert result["gpu_model_bytes"] == 10 * 2**20
    assert result["host_mapped_model_bytes"] == 5 * 2**20
    assert result["gpu_kv_bytes"] == 20 * 2**20
    assert result["gpu_recurrent_bytes"] == 3 * 2**20
    assert result["gpu_compute_bytes"] == 4 * 2**20
    record = {
        "healthy": True,
        "request": {
            "success": True,
            "tokens_evaluated": 32760,
            "tokens_predicted": 8,
            "stop_type": "limit",
            "truncated": True,
        },
    }
    assert llamacpp_collect.completed_request(record)
    record["request"]["tokens_evaluated"] = 32750
    assert not llamacpp_collect.completed_request(record)


def test_paid_replay_rejects_modified_sources_and_existing_job_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    replay = importlib.import_module("replay")
    bundle = tmp_path / "bundle"
    (bundle / "executed").mkdir(parents=True)
    (bundle / "executed/topology.py").write_text("print('modified')")
    (bundle / "specification.json").write_text(
        json.dumps(
            {
                "image": "org/image@sha256:" + "a" * 64,
                "engine": "vllm",
                "source_sha256": {"topology.py": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="Source hash mismatch"):
        replay.prepare(bundle)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "job.json").write_text('{"job_id":"existing-handle"}')
    monkeypatch.setattr(
        "sys.argv", ["replay.py", "--directory", str(bundle), "--output", str(output), "--launch"]
    )
    with pytest.raises(RuntimeError, match="Existing Job handle"):
        replay.main()


def test_repeated_workload_validation_requires_all_exact_requests() -> None:
    record = case()
    record["case"]["repeats"] = 3
    record["requested_prompt_tokens"] = 16376
    record["requested_completion_tokens"] = 8
    requests: list[dict[str, Any]] = [
        {"success": True, "usage": {"prompt_tokens": 16376, "completion_tokens": 8}} for _ in range(3)
    ]
    record["requests"] = requests
    assert collect.request_succeeded(record)
    requests[0]["usage"]["completion_tokens"] = 7
    assert not collect.request_succeeded(record)
    requests[0]["usage"]["completion_tokens"] = 8
    requests.pop()
    assert not collect.request_succeeded(record)


def test_replay_preserves_explicit_topology_and_scheduler_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    replay = importlib.import_module("replay")
    (tmp_path / "executed").mkdir()
    sources = {}
    for name in ["driver.py", "cases.json", "sitecustomize.py"]:
        data = b"[]" if name.endswith(".json") else b"# fixture\n"
        (tmp_path / "executed" / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    settings = {
        "engine": "vllm",
        "image": "org/image@sha256:" + "a" * 64,
        "source_sha256": sources,
        "tensor_parallel_size": 2,
        "max_num_seqs": 3,
    }
    (tmp_path / "specification.json").write_text(json.dumps(settings))
    _, command, env = replay.prepare(tmp_path)
    assert command.endswith("exec python3 -u /tmp/driver.py")
    assert env["HFMEM_GPU_INDICES"] == "0,1" and env["HFMEM_TP_SIZE"] == "2"
    assert env["HFMEM_MAX_NUM_SEQS"] == "3" and env["HFMEM_AUDIT_WEIGHTS"] == "1"
    del settings["tensor_parallel_size"]
    (tmp_path / "specification.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="Explicit positive topology"):
        replay.prepare(tmp_path)


def test_native_planning_stops_are_preserved_separately_from_inference(tmp_path: Path) -> None:
    planned = case()
    planned["mode"] = "plan"
    planned["healthy"] = False
    source = tmp_path / "planner.log"
    source.write_text("HFMEM_RESULT " + json.dumps(planned))
    out = tmp_path / "public"
    manifest = collect.export(source, out, "vllm")
    assert manifest["planning_records"] == 1 and manifest["case_records"] == 0
    assert json.loads((out / "planning.jsonl").read_text())["mode"] == "plan"
    assert (out / "failures.jsonl").read_text() == ""


def test_image_workload_requires_declared_token_range_and_full_decode() -> None:
    record = case()
    record["requested_prompt_tokens"] = None
    record["requested_completion_tokens"] = 8
    record["minimum_prompt_tokens"] = 16384
    record["case"]["max_model_len"] = 32768
    request: dict[str, Any] = {
        "success": True,
        "usage": {"prompt_tokens": 16400, "completion_tokens": 8},
    }
    record["request"] = request
    assert collect.request_succeeded(record)
    for prompt in [16383, 32761, None]:
        request["usage"] = {"prompt_tokens": prompt, "completion_tokens": 8}
        assert not collect.request_succeeded(record)
    record["request"]["usage"] = {"prompt_tokens": 16400, "completion_tokens": 7}
    assert not collect.request_succeeded(record)
    record["request"]["usage"]["completion_tokens"] = 8
    del record["minimum_prompt_tokens"]
    assert not collect.request_succeeded(record)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("native_defaults", [False, True])
def test_engine_matrix_rejects_missing_rank_stages_and_false_aggregate(
    engine: str, native_defaults: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    matrix = importlib.import_module("matrix_collect")
    record = case()
    record["case"].update(max_model_len=32768, max_num_batched_tokens=8192)
    record["tensor_parallel_size"] = 2
    record["baseline"] = {
        "device_used_bytes": 200,
        "processes": [],
        "devices": [{"device_index": i, "device_used_bytes": 100, "processes": []} for i in range(2)],
    }
    record["samples"] = [
        {
            "phase": "request",
            "device_used_bytes": 1200,
            "devices": [{"device_used_bytes": n} for n in values],
        }
        for values in [(800, 400), (500, 700)]
    ]
    record["phase_summary"] = {"request": {"sample_count": 2}}
    record["native"] = []
    record["server_args"] = {
        "tp_size": 2,
        "context_length": 32768,
        "chunked_prefill_size": 8192,
        "max_running_requests": 1,
    }
    if engine == "vllm":
        record["native"].append(
            {
                "kind": "cache_plan",
                "worker_spec_count": 2,
                "max_num_seqs": 1,
                "max_model_len": 32768,
                "max_num_batched_tokens": 8192,
            }
        )
    for rank in range(2):
        storage = {"parameter_storage_bytes": 200, "buffer_storage_bytes": 10}
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
        record["native"].extend(
            {
                "kind": "worker_memory" if engine == "vllm" else "runner_memory",
                "rank": rank,
                "tensor_parallel_size": 2,
                "phase": phase,
                "context_length": 32768,
                "model_storage": storage,
                "cache": {"allocated_cache_bytes": 100},
            }
            for phase in phases
        )
        if engine == "vllm":
            record["native"].extend(
                [
                    {"kind": "weight_storage_audit", "rank": rank, **storage},
                    {"kind": "cache_allocation", "worker_index": rank, "planned_device_pool_bytes": 100},
                    {"kind": "graph_allocation", "rank": rank},
                ]
            )
    if native_defaults:
        record["max_num_seqs"] = 0
        record["case"].update(
            max_num_batched_tokens=0, native_scheduler_defaults=True, native_prefill_defaults=True
        )
        if engine == "sglang":
            record["server_args"] = {**record["server_args"], "max_running_requests": None}
            for event in record["native"]:
                event["cache"]["memory_pool_config"] = {"max_running_requests": 1}
    result, ranks = matrix.summarize(record, engine)
    assert result["usable_for_accuracy"] and len(ranks) == 2
    assert result["sampled_runtime_bytes"] == 1000
    assert result["native_residual_bytes"] == 400
    assert result["resolved_max_num_seqs"] == 1 and result["resolved_prefill_tokens"] == 8192
    import copy

    unmapped = copy.deepcopy(record)
    for sample in unmapped["samples"]:
        sample["device_used_bytes"] = 220
        for device in sample["devices"]:
            device["device_used_bytes"] = 110
    rejected = matrix.summarize(unmapped, engine)[0]
    assert not rejected["usable_for_accuracy"]
    assert "runtime_below_audited_weights_and_cache" in rejected["missing_evidence"]
    record["samples"][0]["device_used_bytes"] = 1400
    assert not matrix.summarize(record, engine)[0]["usable_for_accuracy"]
    record["samples"][0]["device_used_bytes"] = 1200
    record["native"] = [
        e for e in record["native"] if not (e.get("rank") == 1 and e.get("phase") == "load_model")
    ]
    assert not matrix.summarize(record, engine)[0]["usable_for_accuracy"]


def test_sglang_wrapped_startup_provenance_is_recovered_without_inventing_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    matrix = importlib.import_module("matrix_collect")
    settings = {"model_path": "org/model", "tp_size": 4, "chunked_prefill_size": 8192}
    text = "[2026-11-12] server_args=" + repr(settings)
    wrapped = text[:39] + "\n" + text[39:64] + "\n" + text[64:]
    assert matrix.startup_settings(wrapped)[0] == settings
    assert matrix.startup_settings(wrapped[:-4]) is None
    assert matrix.startup_settings(wrapped[:-4] + "\n[2026-11-12] another stage") is None


@pytest.mark.parametrize(
    "fault", ["hash", "revision", "image", "path", "symlink", "required", "trials", "case"]
)
def test_matrix_replay_rejects_unverifiable_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    runner = importlib.import_module("matrix_run")
    sources = {
        name: hashlib.sha256(b"pass\n").hexdigest()
        for name in ["matrix.py", "driver.py", "sitecustomize.py", "trials.json", "cases-0.json"]
    }
    for name in sources:
        (tmp_path / name).write_bytes(b"pass\n")
    specification = {
        "engine": "vllm",
        "image": "vllm/image@sha256:" + "a" * 64,
        "trials": [{"case": {"revision": "a" * 40}}],
        "source_sha256": sources,
    }
    for name, value in [
        ("trials.json", specification["trials"]),
        ("cases-0.json", [specification["trials"][0]["case"]]),
    ]:
        data = json.dumps(value).encode()
        (tmp_path / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    (tmp_path / "specification.json").write_text(json.dumps(specification))
    assert runner.prepare(tmp_path)[0] == specification
    if fault == "hash":
        (tmp_path / "driver.py").write_bytes(b"changed")
    elif fault == "revision":
        specification["trials"][0]["case"]["revision"] = "main"
    elif fault == "image":
        specification["image"] = "vllm/image:latest"
    elif fault == "path":
        sources["../outside.py"] = sources["driver.py"]
    elif fault == "symlink":
        outside = tmp_path.parent / (tmp_path.name + "-outside.py")
        outside.write_bytes(b"pass\n")
        (tmp_path / "driver.py").unlink()
        (tmp_path / "driver.py").symlink_to(outside)
    elif fault in {"trials", "case"}:
        name = "trials.json" if fault == "trials" else "cases-0.json"
        data = b"[]"
        (tmp_path / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    else:
        del sources["driver.py"]
    (tmp_path / "specification.json").write_text(json.dumps(specification))
    with pytest.raises(ValueError):
        runner.prepare(tmp_path)


def test_native_only_publication_excludes_source_and_forecasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    publisher = importlib.import_module("publish_native")
    source = tmp_path / "trial"
    public = source / "public"
    public.mkdir(parents=True)
    (source / "job.json").write_text(json.dumps({"job_id": "job", "stage": "SCHEDULING"}))
    (public / "manifest.json").write_text(json.dumps({"complete": True}))
    (public / "native.log").write_text("native engine observation")
    for forbidden in ["driver.py", "candidate-predictions.json", "baseline-predictions.json"]:
        (public / forbidden).write_text("private candidate")
    output = tmp_path / "native-only"
    publisher.prepare(source, output, None)
    assert not any(
        (output / name).exists()
        for name in ["driver.py", "candidate-predictions.json", "baseline-predictions.json"]
    )
    assert (
        __import__("gzip").decompress((output / "native.log.gz").read_bytes()) == b"native engine observation"
    )


def test_exact_native_launcher_preserves_controls_and_rejects_mutated_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    native_run = importlib.import_module("native_run")

    (tmp_path / "hooks").mkdir()
    (tmp_path / "driver.py").write_text("pass\n")
    (tmp_path / "hooks/sitecustomize.py").write_text("pass\n")
    spec = {
        "model": "org/model",
        "revision": "a" * 40,
        "image": "image@sha256:" + "b" * 64,
        "gpus": 4,
        "flavor": "h200x4",
        "context_length": 262144,
        "server_arguments": [
            "--model-path",
            "org/model",
            "--revision",
            "a" * 40,
            "--tp",
            "4",
            "--ep",
            "4",
            "--context-length",
            "262144",
            "--attention-backend",
            "fa4",
            "--max-running-requests",
            "64",
            "--enable-multimodal",
        ],
        "source_sha256": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ["driver.py", "hooks/sitecustomize.py"]
        },
    }
    (tmp_path / "specification.json").write_text(json.dumps(spec))
    actual, command = native_run.prepare(tmp_path)
    assert actual == spec and command[:2] == ["bash", "-c"]
    assert "--max-total-tokens" not in actual["server_arguments"]
    assert "--disable-radix-cache" not in actual["server_arguments"]
    (tmp_path / "driver.py").write_text("mutated\n")
    with pytest.raises(ValueError, match="Source hash"):
        native_run.prepare(tmp_path)
    (tmp_path / "driver.py").write_text("pass\n")
    spec["server_arguments"][spec["server_arguments"].index("--ep") + 1] = "2"
    (tmp_path / "specification.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Mismatched native control"):
        native_run.prepare(tmp_path)


def test_native_collection_requires_full_trace_tokens_and_physical_weight_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/calibration"))
    native_collect = importlib.import_module("native_collect")

    spec = {"gpus": 2, "prompt_lengths": [16]}
    (tmp_path / "specification.json").write_text(json.dumps(spec))
    devices = [{"index": i, "used": 0, "total": 10 * 2**30, "processes": []} for i in range(2)]
    samples = [
        {"time": 1, "phase": "ready", "used": 4 * 2**30, "devices": [{**d, "used": 2 * 2**30} for d in devices]}
    ]
    request = {
        "success": True,
        "requested_prompt_tokens": 16,
        "meta_info": {"prompt_tokens": 16, "completion_tokens": 8},
    }
    result = {
        "kind": "result",
        "healthy": True,
        "requests": [request],
        "sample_count": 1,
        "baseline": {"used": 0, "devices": devices},
        "phase_summary": {"startup": {"peak_used": 4 * 2**30}, "ready": {"peak_used": 4 * 2**30}},
    }
    records: list[dict[str, Any]] = [
        result,
        {"kind": "samples", "offset": 0, "samples": samples},
        {"kind": "done"},
    ]
    native = []
    for rank in range(2):
        initial = {
            "kind": "runner_memory",
            "phase": "init_torch_distributed",
            "rank": rank,
            "tensor_parallel_size": 2,
            "cuda_device_index": rank,
            "torch_allocated_bytes": 0,
        }
        loaded = {
            **initial,
            "phase": "load_model",
            "torch_allocated_bytes": 2**30,
            "model_storage": {"parameter_storage_bytes": 2**30, "buffer_storage_bytes": 0},
        }
        cached = {**loaded, "phase": "init_cuda_graphs", "cache": {"allocated_cache_bytes": 10}}
        native.extend([initial, loaded, cached])
    path = tmp_path / "log.txt"

    def write() -> None:
        path.write_text(
            "".join("HFMEM_RESULT " + json.dumps(x) + "\n" for x in records)
            + "HFMEM_LOG_START native\n"
            + "".join("HFMEM_NATIVE " + json.dumps(x) + "\n" for x in native)
            + "HFMEM_LOG_END native\n"
        )

    write()
    observed = native_collect.collect(tmp_path, path)
    assert observed["runtime_measurement_complete"] and observed["component_audits_complete"]
    request["meta_info"]["completion_tokens"] = 2
    write()
    assert not native_collect.collect(tmp_path, path)["runtime_measurement_complete"]
    request["meta_info"]["completion_tokens"] = 8
    records[1]["offset"] = 128
    write()
    assert not native_collect.collect(tmp_path, path)["samples_complete"]
    records[1]["offset"] = 0
    # Plain tensor allocations held outside named Parameters must not be scored as overhead.
    native[1]["torch_allocated_bytes"] = 90 * 2**30
    write()
    observed = native_collect.collect(tmp_path, path)
    assert observed["runtime_measurement_complete"] and not observed["component_audits_complete"]
    assert "unattributed_runtime_residual_bytes" not in observed


def test_native_storage_audit_counts_packed_experts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    class Tensor:
        def __init__(self, pointer: int, size: int, cuda: bool = True) -> None:
            self.device = SimpleNamespace(type="cuda" if cuda else "cpu", index=0)
            self.dtype, self.shape = "int32", (size // 4,)
            self.pointer, self.size = pointer, size

        def untyped_storage(self) -> Any:
            return SimpleNamespace(data_ptr=lambda: self.pointer, nbytes=lambda: self.size)

    parameter = Tensor(1, 100)
    packed = Tensor(2, 200)
    cpu = Tensor(3, 300, False)
    empty = Tensor(0, 0)
    module = SimpleNamespace(_marlin={"w13": (packed, cpu, empty)}, alias=packed)
    module.cycle = module
    model = SimpleNamespace(
        named_parameters=lambda: [("weight", parameter)],
        named_buffers=lambda: [("alias", parameter)],
        named_modules=lambda: [("", module)],
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor)))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    spec = importlib.util.spec_from_file_location(
        "native_hooks_test", Path(__file__).parents[1] / "scripts/calibration/native_hooks.py"
    )
    assert spec is not None and spec.loader is not None
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    audit = hooks.storage(model)
    assert audit["model_storage_bytes"] == 300
    assert audit["parameter_storage_bytes"] == 100
    assert audit["buffer_storage_bytes"] == 0
    assert audit["unregistered_storage_bytes"] == 200
    assert len(audit["largest"]) == 2
    # Request/state pools are ordinary SGLang objects rather than nn.Modules.
    assert hooks.storage(module)["model_storage_bytes"] == 200


def test_transport_deduplication_preserves_distinct_observations() -> None:
    first = {"kind": "case", "time": 1, "healthy": True}
    later = {"kind": "case", "time": 2, "healthy": False}
    records, duplicates = collect.unique_records([first, dict(first), later])
    assert records == [first, later]
    assert duplicates == 1


@pytest.mark.parametrize("failure", [None, "sequence_count", "individual_decode"])
def test_concurrent_requests_require_all_sequences_and_tokens(failure: str | None) -> None:
    record = case()
    record["case"]["batch_size"] = 2
    record["requested_prompt_tokens"] = 65520
    record["requested_completion_tokens"] = 16
    record["request"].update(
        usage={"prompt_tokens": 65520, "completion_tokens": 16},
        sequence_count=2,
        sequence_usage=[{"prompt_tokens": 32760, "completion_tokens": 8} for _ in range(2)],
    )
    if failure == "sequence_count":
        record["request"]["sequence_count"] = 1
    elif failure == "individual_decode":
        record["request"]["sequence_usage"][1]["completion_tokens"] = 7
    assert collect.request_succeeded(record) is (failure is None)
