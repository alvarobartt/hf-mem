"""Observational SGLang hooks tolerant of native API changes; never change controls."""

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


def sizes(value: Any) -> int:
    if isinstance(value, dict):
        return sum(sizes(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(sizes(x) for x in value)
    return int(value)


def storage(model: Any) -> dict[str, Any]:
    """Count unique CUDA storage, including packed tensors outside Parameters."""
    import torch

    seen: set[tuple[int, int]] = set()
    visited: set[int] = set()
    summary: dict[str, int] = {}
    largest = []

    def tensor(name: str, value: Any, kind: str) -> None:
        if value.device.type != "cuda":
            return
        data = value.untyped_storage()
        key = (value.device.index, data.data_ptr())
        if key in seen or not data.nbytes():
            return
        seen.add(key)
        label = kind + "/" + str(value.dtype)
        summary[label] = summary.get(label, 0) + data.nbytes()
        largest.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "storage_bytes": data.nbytes(),
            }
        )

    for kind, iterator in [
        ("parameters", model.named_parameters() if hasattr(model, "named_parameters") else []),
        ("buffers", model.named_buffers() if hasattr(model, "named_buffers") else []),
    ]:
        for name, value in iterator:
            tensor(name, value, kind)

    def visit(value: Any, name: str, depth: int = 0) -> None:
        if torch.is_tensor(value):
            tensor(name, value, "unregistered")
            return
        if depth > 8 or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, name + "." + str(key), depth + 1)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, name + "." + str(index), depth + 1)
        elif type(value).__module__.startswith(("sglang", "types")) and hasattr(value, "__dict__"):
            for key, child in vars(value).items():
                visit(child, name + "." + key, depth + 1)

    if hasattr(model, "named_modules"):
        for name, module in model.named_modules():
            for key, value in vars(module).items():
                if key not in {"_parameters", "_buffers", "_modules"}:
                    visit(value, name + "." + key)
    else:
        visit(model, "pool")
    return {
        "summary": summary,
        "parameter_storage_bytes": sum(v for k, v in summary.items() if k.startswith("parameters/")),
        "buffer_storage_bytes": sum(v for k, v in summary.items() if k.startswith("buffers/")),
        "unregistered_storage_bytes": sum(v for k, v in summary.items() if k.startswith("unregistered/")),
        "model_storage_bytes": sum(summary.values()),
        "largest": sorted(largest, key=lambda x: x["storage_bytes"], reverse=True)[:16],
    }


def geometry(pool: Any) -> dict[str, Any]:
    result = {"type": type(pool).__name__}
    for key in [
        "size",
        "size_swa",
        "page_size",
        "head_num",
        "head_dim",
        "k_head_dim",
        "v_head_dim",
        "layer_num",
        "start_layer",
        "end_layer",
        "dtype",
        "swa_layer_nums",
        "full_layer_nums",
    ]:
        value = getattr(pool, key, None)
        if isinstance(value, (int, float, str, list, tuple, bool)):
            result[key] = value
        elif value is not None:
            result[key] = str(value)
    for key in ["full_kv_pool", "swa_kv_pool"]:
        if getattr(pool, key, None) is not None:
            result[key] = geometry(getattr(pool, key))
    method = getattr(pool, "get_kv_size_bytes", None)
    if callable(method):
        value = method()
        result["native_size_bytes"] = value
        result["allocated_cache_bytes"] = sizes(value)
    return result


def snapshot(runner: Any, phase: str) -> None:
    try:
        import torch

        module = sys.modules[type(runner).__module__]
        rank = getattr(runner, "tp_rank", None)
        size = getattr(runner, "tp_size", None)
        if rank is None and callable(getattr(module, "get_parallel", None)):
            parallel = module.get_parallel()
            rank, size = parallel.tp_rank, parallel.tp_size
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info()
        data = dict(
            phase=phase,
            rank=rank,
            tensor_parallel_size=size,
            cuda_device_index=device,
            cuda_free_bytes=free,
            cuda_total_bytes=total,
            torch_allocated_bytes=torch.cuda.memory_allocated(),
            torch_reserved_bytes=torch.cuda.memory_reserved(),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        for key in [
            "max_total_num_tokens",
            "full_max_total_num_tokens",
            "swa_max_total_num_tokens",
            "max_running_requests",
            "page_size",
            "graph_memory_usage",
            "weight_load_mem_usage",
        ]:
            value = getattr(runner, key, None)
            if value is not None:
                data[key] = value
        pool = getattr(runner, "token_to_kv_pool", None)
        if pool is not None:
            data["cache"] = geometry(pool)
        requests = getattr(runner, "req_to_token_pool", None)
        if requests is not None and phase in {
            "alloc_memory_pool",
            "init_cuda_graphs",
            "init_cuda_graph_runner",
        }:
            data["request_pool_storage"] = storage(requests)
        backend = getattr(runner, "attn_backend", None)
        data["backend_type"] = type(backend).__name__
        model = getattr(runner, "model", None)
        if model is not None and phase in {
            "load_model",
            "initialize",
            "init_cuda_graphs",
            "init_cuda_graph_runner",
        }:
            data["model_storage"] = storage(model)
        emit("runner_memory", **data)
    except Exception as exc:
        # An observation failure must never make a native deployment fail.
        emit("audit_error", phase=phase, error=f"{type(exc).__name__}: {exc}")


def patch(module: Any) -> None:
    cls = getattr(module, "ModelRunner", None)
    if cls is None:
        return
    methods = []
    for name in [
        "init_torch_distributed",
        "load_model",
        "initialize",
        "alloc_memory_pool",
        "init_memory_pool",
        "init_cuda_graphs",
        "init_cuda_graph_runner",
    ]:
        original = getattr(cls, name, None)
        if not callable(original):
            continue

        def wrapped(self: Any, *args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
            result = _original(self, *args, **kwargs)
            snapshot(self, _name)
            return result

        setattr(cls, name, functools.wraps(original)(wrapped))
        methods.append(name)
    emit("hooks_installed", methods=methods, module=module.__name__)


class Loader(importlib.abc.Loader):
    def __init__(self, original: Any) -> None:
        self.original = original

    def create_module(self, spec: Any) -> Any:
        return self.original.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self.original.exec_module(module)
        try:
            patch(module)
        except Exception as exc:
            emit("hook_install_error", error=f"{type(exc).__name__}: {exc}")


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == "sglang.srt.model_executor.model_runner":
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec
        return None


sys.meta_path.insert(0, Finder())
