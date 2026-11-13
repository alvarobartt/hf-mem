import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from typing import Any

CURRENT_RANK: int | None = None


def emit(kind: str, **data: Any) -> None:
    print(
        "HFMEM_NATIVE "
        + json.dumps(
            {"kind": kind, "pid": os.getpid(), "rank": CURRENT_RANK, "time": time.time(), **data}, default=str
        ),
        flush=True,
    )


def snapshot(worker: Any, phase: str) -> None:
    import torch

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    data = {
        "phase": phase,
        "rank": worker.rank,
        "local_rank": worker.local_rank,
        "cuda_device_index": torch.cuda.current_device(),
        "tensor_parallel_size": worker.vllm_config.parallel_config.tensor_parallel_size,
        "max_num_seqs": worker.scheduler_config.max_num_seqs,
        "runner_type": type(worker.model_runner).__name__,
        "cudagraph_capture_sizes": worker.vllm_config.compilation_config.cudagraph_capture_sizes,
        "cudagraph_mode": str(worker.vllm_config.compilation_config.cudagraph_mode),
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cuda_free_bytes": free,
        "cuda_total_bytes": total,
        "model_memory_usage_bytes": int(getattr(worker.model_runner, "model_memory_usage", 0)),
    }
    for name in [
        "total_consumed",
        "peak_activation_memory",
        "cudagraph_memory_estimate",
        "available_kv_cache_memory_bytes",
        "requested_memory",
    ]:
        value = getattr(worker, name, None)
        if isinstance(value, (int, float)):
            data[name + "_bytes"] = int(value)
    emit("worker_memory", **data)


def patch(module: Any) -> None:
    if module.__name__ == "vllm.v1.core.kv_cache_planning":
        original = module.get_kv_cache_configs

        @functools.wraps(original)
        def plan(config: Any, specs: Any, available: Any, groups: Any) -> Any:
            from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder

            builder = get_kv_cache_config_builder(config)
            assert config.parallel_config.pipeline_parallel_size == 1, "This harness validates TP, not PP"
            assert len(specs) == config.parallel_config.tensor_parallel_size
            block_bytes = module._pool_bytes_per_block_from_builder(builder, config, groups)
            request_bytes = module._max_memory_usage_bytes_from_groups(builder, config, groups)
            blocks = (request_bytes + block_bytes - 1) // block_bytes + 1
            detail = []
            for group in groups:
                spec = group.kv_cache_spec
                detail.append(
                    {
                        "spec_type": type(spec).__name__,
                        "layers": list(group.layer_names),
                        "block_size": spec.block_size,
                        "page_size_bytes": spec.page_size_bytes,
                        "max_request_bytes": spec.max_memory_usage_bytes(config),
                        "host_resident": group.host_resident,
                    }
                )
            payload = {
                "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
                "worker_spec_count": len(specs),
                "required_blocks_including_null": blocks,
                "device_bytes_per_block": block_bytes,
                "native_request_cache_bytes": request_bytes,
                "groups": detail,
                "max_model_len": config.model_config.max_model_len,
                "max_num_seqs": config.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            }
            emit("cache_plan", **payload)
            if os.environ.get("HFMEM_PLAN_ONLY") == "1":
                raise RuntimeError("HFMEM_PLAN_COMPLETE")
            override = config.cache_config.num_gpu_blocks_override
            if override is not None:
                # Larger pools are valid for concurrent requests. Observe the
                # supplied control without preventing a native deployment.
                emit("cache_override", blocks=override, single_request_minimum=blocks)
            result = original(config, specs, available, groups)
            for index, item in enumerate(result):
                tensors = [
                    {
                        "size_bytes": tensor.size,
                        "layers": list(tensor.layers),
                        "offset": tensor.offset,
                        "host_resident": tensor.host_resident,
                    }
                    for tensor in item.kv_cache_tensors
                ]
                emit(
                    "cache_allocation",
                    worker_index=index,
                    num_blocks=item.num_blocks,
                    tensors=tensors,
                    planned_device_pool_bytes=item.num_blocks * block_bytes,
                    total_tensor_view_bytes=sum(tensor.size for tensor in item.kv_cache_tensors),
                    null_block_bytes=block_bytes,
                    requested_override=override,
                )
            return result

        module.get_kv_cache_configs = plan
    elif module.__name__ == "vllm.v1.worker.gpu_worker":
        for name in [
            "init_device",
            "load_model",
            "determine_available_memory",
            "initialize_from_config",
            "compile_or_warm_up_model",
        ]:
            original = getattr(module.Worker, name)

            def wrapper(
                self: Any, *args: Any, _original: Any = original, _name: str = name, **kwargs: Any
            ) -> Any:
                global CURRENT_RANK
                CURRENT_RANK = self.rank
                graph_original = None
                if _name == "compile_or_warm_up_model":
                    graph_original = self.model_runner.capture_model

                    def observed_capture(*capture_args: Any, **capture_kwargs: Any) -> Any:
                        amount = graph_original(*capture_args, **capture_kwargs)
                        emit("graph_allocation", actual_graph_bytes=amount)
                        return amount

                    self.model_runner.capture_model = observed_capture
                try:
                    result = _original(self, *args, **kwargs)
                finally:
                    if graph_original is not None:
                        self.model_runner.capture_model = graph_original
                snapshot(self, _name)
                if _name == "load_model" and (
                    os.environ.get("HFMEM_AUDIT_ONLY") == "1" or os.environ.get("HFMEM_AUDIT_WEIGHTS") == "1"
                ):
                    model = self.model_runner.get_model()
                    seen = set()
                    summary = {}
                    largest = []
                    for kind, iterator in [
                        ("parameters", model.named_parameters()),
                        ("buffers", model.named_buffers()),
                    ]:
                        for name, tensor in iterator:
                            if tensor.device.type != "cuda":
                                continue
                            storage = tensor.untyped_storage()
                            pointer = storage.data_ptr()
                            if pointer in seen:
                                continue
                            seen.add(pointer)
                            key = kind + "/" + str(tensor.dtype)
                            summary[key] = summary.get(key, 0) + storage.nbytes()
                            largest.append(
                                {
                                    "kind": kind,
                                    "name": name,
                                    "shape": list(tensor.shape),
                                    "dtype": str(tensor.dtype),
                                    "storage_bytes": storage.nbytes(),
                                }
                            )
                    emit(
                        "weight_storage_audit",
                        model=self.model_config.model,
                        summary=summary,
                        parameter_storage_bytes=sum(
                            v for k, v in summary.items() if k.startswith("parameters/")
                        ),
                        buffer_storage_bytes=sum(v for k, v in summary.items() if k.startswith("buffers/")),
                        largest=sorted(largest, key=lambda x: x["storage_bytes"], reverse=True)[:20],
                    )
                    if os.environ.get("HFMEM_AUDIT_ONLY") == "1":
                        raise RuntimeError("HFMEM_WEIGHT_AUDIT_COMPLETE")
                return result

            setattr(module.Worker, name, functools.wraps(original)(wrapper))
    elif module.__name__ == "vllm.v1.worker.gpu_model_runner":
        original = module.GPUModelRunner.capture_model

        @functools.wraps(original)
        def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, *args, **kwargs)
            emit("graph_allocation", actual_graph_bytes=result)
            return result

        module.GPUModelRunner.capture_model = capture


class Loader:
    def __init__(self, original: Any) -> None:
        self.original = original

    def create_module(self, spec: Any) -> Any:
        return self.original.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.original.exec_module(module)
        patch(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in {
            "vllm.v1.core.kv_cache_planning",
            "vllm.v1.worker.gpu_worker",
            "vllm.v1.worker.gpu_model_runner",
        }:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
