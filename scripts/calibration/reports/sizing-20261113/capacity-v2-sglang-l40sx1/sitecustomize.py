import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from typing import Any


def emit(kind: str, **data: Any) -> None:
    print(
        "HFMEM_NATIVE "
        + json.dumps({"kind": kind, "pid": os.getpid(), "time": time.time(), **data}, default=str),
        flush=True,
    )


def tensor_storage(model: Any) -> dict[str, Any]:
    seen: set[tuple[int, int]] = set()
    totals: dict[str, int] = {}
    largest = []
    for kind, iterator in [("parameters", model.named_parameters()), ("buffers", model.named_buffers())]:
        for name, tensor in iterator:
            if tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            label = kind + "/" + str(tensor.dtype)
            totals[label] = totals.get(label, 0) + storage.nbytes()
            largest.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "storage_bytes": storage.nbytes(),
                }
            )
    return {
        "summary": totals,
        "parameter_storage_bytes": sum(n for key, n in totals.items() if key.startswith("parameters/")),
        "buffer_storage_bytes": sum(n for key, n in totals.items() if key.startswith("buffers/")),
        "largest": sorted(largest, key=lambda item: item["storage_bytes"], reverse=True)[:10],
    }


def sum_sizes(value: Any) -> int:
    return sum(map(int, value)) if isinstance(value, (tuple, list)) else int(value)


def pool_geometry(pool: Any) -> dict[str, Any]:
    data = {"type": type(pool).__name__}
    for name in [
        "size",
        "size_swa",
        "page_size",
        "head_num",
        "head_dim",
        "layer_num",
        "start_layer",
        "end_layer",
        "tail_extra_slots",
        "dtype",
    ]:
        value = getattr(pool, name, None)
        if value is not None:
            data[name] = str(value) if name == "dtype" else value
    for name in ["full_kv_pool", "swa_kv_pool"]:
        child = getattr(pool, name, None)
        if child is not None:
            data[name] = pool_geometry(child)
    return data


def cache_info(runner: Any) -> dict[str, Any] | None:
    pool = getattr(runner, "token_to_kv_pool", None)
    if pool is None:
        return None
    size = pool.get_kv_size_bytes()
    post_capture = bool(getattr(pool, "post_capture_active", False))
    attention_bytes = int(pool.post_capture_backed_bytes) if post_capture else sum_sizes(size)
    state_pool = getattr(pool, "mamba_pool", None)
    if state_pool is None:
        state_pool = getattr(getattr(runner, "req_to_token_pool", None), "mamba_pool", None)
    state_bytes = int(state_pool.get_kv_size_bytes()) if state_pool is not None else 0
    config = getattr(runner, "memory_pool_config", None)
    try:
        msgspec = importlib.import_module("msgspec")
        config = msgspec.to_builtins(config) if config is not None else None
    except (TypeError, ImportError):
        config = str(config)
    return {
        "attention_cache_bytes": attention_bytes,
        "recurrent_state_bytes": state_bytes,
        "allocated_cache_bytes": attention_bytes + state_bytes,
        "native_reported_kv_sizes": size,
        "post_capture_active": post_capture,
        "memory_pool_config": config,
        "geometry": pool_geometry(pool),
        "mamba_pool_type": type(state_pool).__name__ if state_pool is not None else None,
        "mamba_pool_size": getattr(state_pool, "size", None),
    }


def snapshot(
    runner: Any,
    phase: str,
    include_storage: bool = False,
    synchronize: bool = True,
    forward_info: dict[str, Any] | None = None,
) -> None:
    torch = importlib.import_module("torch")
    parallel = importlib.import_module("sglang.srt.model_executor.model_runner").get_parallel()
    context = importlib.import_module("sglang.srt.model_executor.model_runner").get_context()
    if synchronize:
        torch.cuda.synchronize()
    data: dict[str, Any] = {
        "phase": phase,
        "rank": parallel.tp_rank,
        "tensor_parallel_size": parallel.tp_size,
        "gpu_id": runner.gpu_id,
        "cuda_device_index": torch.cuda.current_device(),
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    if synchronize:
        free, total = torch.cuda.mem_get_info()
        data.update(cuda_free_bytes=free, cuda_total_bytes=total)
    for name in [
        "max_total_num_tokens",
        "full_max_total_num_tokens",
        "swa_max_total_num_tokens",
        "max_running_requests",
        "page_size",
        "dtype",
        "graph_memory_usage",
        "decode_cuda_graph_capture_bs",
        "decode_cuda_graph_captured",
    ]:
        value = getattr(runner, name, None)
        if value is not None:
            data[name] = value if isinstance(value, (int, float, bool, list, dict)) else str(value)
    data["context_length"] = runner.model_config.context_len
    backend = getattr(runner, "attn_backend", None)
    data["attention_backend_types"] = {
        "wrapper": type(backend).__name__,
        "full": type(getattr(backend, "full_attn_backend", None)).__name__,
        "linear": type(getattr(backend, "linear_attn_backend", None)).__name__,
    }
    linear = getattr(backend, "linear_attn_backend", None)
    dispatcher = getattr(linear, "kernel_dispatcher", None)
    data["gdn_kernel_types"] = {
        "prefill": type(getattr(dispatcher, "extend_kernel", None)).__name__,
        "decode": type(getattr(dispatcher, "decode_kernel", None)).__name__,
    }
    if forward_info is not None:
        data["forward"] = forward_info
    resolved = context.resolved_server_args_dict()
    data["controls"] = {
        key: resolved.get(key)
        for key in [
            "context_length",
            "max_running_requests",
            "chunked_prefill_size",
            "tp_size",
            "max_total_tokens",
        ]
    }
    data["cache"] = cache_info(runner)
    if include_storage:
        data["model_storage"] = tensor_storage(runner.model)
    if hasattr(runner, "weight_load_mem_usage"):
        data["native_weight_load_delta_bytes"] = round(runner.weight_load_mem_usage * 2**30)
    emit("runner_memory", **data)


def patch(module: Any) -> None:
    for name in [
        "init_torch_distributed",
        "load_model",
        "initialize",
        "alloc_memory_pool",
        "init_cuda_graphs",
        "init_decode_cuda_graph",
        "init_prefill_cuda_graph",
    ]:
        original = getattr(module.ModelRunner, name)

        def wrapper(self: Any, *args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
            result = _original(self, *args, **kwargs)
            snapshot(self, _name, include_storage=_name in ["load_model", "initialize", "init_cuda_graphs"])
            if _name == "init_cuda_graphs":
                self._hfmem_observation_count = 0
                self._hfmem_after_graph_capture = True
            return result

        setattr(module.ModelRunner, name, functools.wraps(original)(wrapper))
    original_forward = module.ModelRunner.forward

    @functools.wraps(original_forward)
    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_forward(self, *args, **kwargs)
        count = getattr(self, "_hfmem_observation_count", 0)
        if (
            count < 64
            and getattr(self, "_hfmem_after_graph_capture", False)
            and getattr(self, "token_to_kv_pool", None) is not None
        ):
            self._hfmem_observation_count = count + 1
            batch = args[0] if args else kwargs.get("forward_batch")
            info = {"after_graph_capture": True, "mode": str(getattr(batch, "forward_mode", None))}
            for name in ["batch_size", "extend_num_tokens"]:
                value = getattr(batch, name, None)
                if isinstance(value, int):
                    info[name] = value
            lengths = getattr(batch, "seq_lens_cpu", None)
            if getattr(getattr(lengths, "device", None), "type", None) == "cpu":
                lengths = lengths.tolist()
            if isinstance(lengths, list) and all(isinstance(value, int) for value in lengths):
                info["sequence_lengths"] = lengths[:64]
            snapshot(self, "forward", synchronize=False, forward_info=info)
        return result

    module.ModelRunner.forward = forward


class Loader(importlib.abc.Loader):
    def __init__(self, original: Any) -> None:
        self.original = original

    def create_module(self, spec: Any) -> Any:
        return self.original.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.original.exec_module(module)
        patch(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "sglang.srt.model_executor.model_runner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
