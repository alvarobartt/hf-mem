import argparse
import asyncio
import json
import sys
import time

from hf_mem import _overhead, _safetensors
from hf_mem._config import KV_CACHE_DTYPES
from hf_mem._gguf import CACHE_TYPES
from hf_mem._render import markdown, table
from hf_mem._source import HubError, Progress
from hf_mem._version import __version__
from hf_mem.estimate import FORMATS, arun


def _gpu_type(value: str) -> str:
    try:
        return _overhead.gpu_type_name(value)
    except RuntimeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class _Status:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        self.enabled = sys.stderr.isatty()
        self.last, self.frame = 0.0, 0

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if not self.enabled or (now - self.last < 0.08 and done < total):
            return
        self.last, self.frame = now, self.frame + 1
        width = 24
        filled = width * done // max(total, 1)
        bar = "━" * filled + "╺" + "─" * (width - filled - 1) if filled < width else "━" * width
        spinner = self.FRAMES[self.frame % len(self.FRAMES)]
        sys.stderr.write(f"\r\x1b[2K{spinner} Reading metadata {bar} {done}/{total} requests")
        sys.stderr.flush()

    def clear(self) -> None:
        if self.enabled:
            sys.stderr.write("\r\x1b[2K")
            sys.stderr.flush()


def main() -> None:
    _safetensors.POOL_MIN_FILES = 32
    parser = argparse.ArgumentParser(
        prog="hf-mem",
        description="Estimate the memory required to run a model from the Hugging Face Hub or a local path.",
    )
    parser.add_argument(
        "model", nargs="?", help="Hub model ID (`org/name`, `org/name@revision`, or URL) or local path."
    )
    parser.add_argument(
        "--model",
        "--model-path",
        "--hf-repo",
        "-m",
        "-hf",
        dest="model_option",
        help="Alias for the positional model ID or local path.",
    )
    parser.add_argument("--revision", help="Hub revision (branch, tag or commit), defaults to `main`.")
    parser.add_argument("--file", "--hf-file", "-hff", help="Single file, shard set or directory to estimate.")
    parser.add_argument(
        "--format",
        choices=FORMATS,
        help="Weights format to estimate, defaults to Safetensors, then GGUF, then the weight file sizes, then ONNX.",
    )
    context = parser.add_mutually_exclusive_group()
    context.add_argument(
        "--max-model-len",
        "--context-length",
        "--context-len",
        dest="max_model_len",
        type=int,
        help="Context tokens per sequence, defaults to the model config.",
    )
    context.add_argument(
        "--ctx-size", "-c", type=int, help="llama.cpp total context pool; divided by --parallel. Selects GGUF."
    )
    parser.add_argument(
        "--batch-size",
        "--parallel",
        "-np",
        dest="batch_size",
        type=int,
        default=1,
        help="Concurrent sequences in the KV cache, defaults to 1. llama.cpp --batch-size means tokens, not sequences.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="KV cache dtype, `auto` uses the model dtype (Safetensors) or F16 (GGUF). "
        f"Safetensors: {', '.join(KV_CACHE_DTYPES)}. GGUF: {', '.join(t.lower() for t in CACHE_TYPES)}. ONNX: set by the graph.",
    )
    parser.add_argument(
        "--gpu-memory",
        help="Memory per GPU: bare numbers or GiB are binary, GB is decimal, e.g. `8x141` or `8x139.8GiB`.",
    )
    parser.add_argument(
        "--engine",
        type=str.lower,
        choices=("vllm", "sglang"),
        default="vllm",
        help="Runtime reference for Safetensors overhead, defaults to vllm. GGUF uses llama.cpp.",
    )
    parser.add_argument(
        "--gpu-type",
        type=_gpu_type,
        choices=_overhead.GPU_TYPE_NAMES,
        help="GPU family for Safetensors context/backend/TP defaults. Without it, capacity is a coarse proxy. "
        "Optional; validates --gpu-memory against board capacities, allowing driver reservations.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of the GPU memory that can be used, defaults to 0.92 (as vLLM).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        "--chunked-prefill-size",
        dest="max_num_batched_tokens",
        type=int,
        help="Prefill token budget. vLLM defaults: 2048 below 70 GiB or on named A100, 8192 otherwise (16384 at 160 GiB). SGLang defaults: 2048 below 20 GiB, 4096 below 60 GiB, 8192 below 160 GiB, 16384 otherwise.",
    )
    parser.add_argument(
        "--max-num-seqs",
        "--max-running-requests",
        dest="max_num_seqs",
        type=int,
        help="Scheduler sequence capacity for Safetensors overhead, separate from --batch-size. "
        "vLLM defaults to 256 or 1024 on Hopper/Blackwell; SGLang uses 2048 request slots "
        "with separate hardware-dependent graph caps. Automatic cache reservations may be larger.",
    )
    parser.add_argument(
        "--disable-hybrid-kv-cache-manager",
        "--disable-hybrid-swa-memory",
        "--swa-full",
        "--full-sliding-window-cache",
        dest="full_sliding_window_cache",
        action="store_true",
        help="Allocate full-context cache for sliding-window layers (vLLM/SGLang/llama.cpp aliases), "
        "for backends without window-sized cache pools.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Skip identified multimodal encoder/projector weights and encoder activations; GGUF omits mmproj.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-mmproj",
        "--no-mmproj-auto",
        action="store_true",
        help="llama.cpp: omit the projector and its activations. Selects GGUF.",
    )
    parser.add_argument(
        "--cache-type-k", "-ctk", help="llama.cpp key-cache dtype; this estimator requires matching K/V types."
    )
    parser.add_argument(
        "--cache-type-v",
        "-ctv",
        help="llama.cpp value-cache dtype; this estimator requires matching K/V types.",
    )
    parser.add_argument("--hf-token", help="Hub token, defaults to `HF_TOKEN` or the `hf auth login` token.")
    parser.add_argument("--color", action="store_true", help="Enable color in terminal tables.")
    parser.add_argument(
        "-o",
        "--output",
        choices=("table", "json", "markdown"),
        default="table",
        help="Output format, defaults to `table`.",
    )
    parser.add_argument("--version", action="version", version=f"hf-mem {__version__}")
    args = parser.parse_args()
    if args.model and args.model_option and args.model != args.model_option:
        parser.error("Provide the model once, either positionally or with --model-path/--hf-repo")
    args.model = args.model or args.model_option
    if not args.model:
        parser.error("A model ID or local path is required")
    if args.no_mmproj:
        if args.format not in (None, "gguf"):
            parser.error("--no-mmproj is a GGUF option; use --language-model-only for Safetensors")
        args.format, args.language_model_only = "gguf", True
    if args.ctx_size is not None:
        if args.format not in (None, "gguf"):
            parser.error("--ctx-size is a llama.cpp total-pool option; use --max-model-len for other formats")
        if args.ctx_size < 1 or args.batch_size < 1 or args.ctx_size % args.batch_size:
            parser.error("--ctx-size must be positive and divisible by --parallel/--batch-size")
        args.format, args.max_model_len = "gguf", args.ctx_size // args.batch_size
    if args.cache_type_k is not None or args.cache_type_v is not None:
        if args.format not in (None, "gguf"):
            parser.error("--cache-type-k/--cache-type-v are llama.cpp GGUF options")
        if args.cache_type_k is None or args.cache_type_v is None:
            parser.error(
                "Provide both --cache-type-k and --cache-type-v; only matching K/V types are supported"
            )
        if args.cache_type_k.lower() != args.cache_type_v.lower():
            parser.error(
                "Separate K/V cache dtypes are not supported; provide matching --cache-type-k/--cache-type-v"
            )
        if args.kv_cache_dtype.lower() not in ("auto", args.cache_type_k.lower()):
            parser.error("--kv-cache-dtype conflicts with --cache-type-k/--cache-type-v")
        args.format, args.kv_cache_dtype = "gguf", args.cache_type_k.lower()

    if args.batch_size < 1 or any(
        v is not None and v < 1 for v in (args.max_model_len, args.max_num_batched_tokens, args.max_num_seqs)
    ):
        parser.error(
            "`--batch-size`, `--max-model-len`, `--max-num-batched-tokens` and `--max-num-seqs` must be positive"
        )
    if args.max_num_seqs is not None and args.max_num_seqs < args.batch_size:
        parser.error("`--max-num-seqs` must be at least `--batch-size`")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("`--gpu-memory-utilization` must be in (0, 1]")

    status = _Status()
    try:
        result = asyncio.run(
            arun(
                args.model,
                revision=args.revision,
                file=args.file,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                kv_cache_dtype=args.kv_cache_dtype,
                hf_token=args.hf_token,
                gpu_memory=args.gpu_memory,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
                full_sliding_window_cache=args.full_sliding_window_cache,
                engine=args.engine,
                gpu_type=args.gpu_type,
                text_only=args.text_only,
                language_model_only=args.language_model_only,
                progress=Progress(status),
                format=args.format,
            )
        )
    except (RuntimeError, HubError) as e:
        status.clear()
        parser.exit(1, f"hf-mem: error: {e}\n")
    except KeyboardInterrupt:
        status.clear()
        parser.exit(130)
    status.clear()

    if args.output == "json":
        print(json.dumps(result.to_dict()))
        return
    for warning in result.warnings:
        print(f"hf-mem: warning: {warning}", file=sys.stderr)
    print(markdown(result) if args.output == "markdown" else table(result, color=args.color))
