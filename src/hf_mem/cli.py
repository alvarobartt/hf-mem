import argparse
import asyncio
import json
import sys
import time

from hf_mem import _safetensors
from hf_mem._config import KV_CACHE_DTYPES
from hf_mem._gguf import KV_DTYPES
from hf_mem._render import markdown, table
from hf_mem._source import HubError, Progress
from hf_mem._version import __version__
from hf_mem.estimate import FORMATS, arun


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
    parser.add_argument("model", help="Hub model ID (`org/name`, `org/name@revision`, or URL) or local path.")
    parser.add_argument("--revision", help="Hub revision (branch, tag or commit), defaults to `main`.")
    parser.add_argument("--file", help="Single file, shard set or directory to estimate.")
    parser.add_argument(
        "--format",
        choices=FORMATS,
        help="Weights format to estimate, defaults to Safetensors, then GGUF, then the weight file sizes, then ONNX.",
    )
    parser.add_argument("--max-model-len", type=int, help="Context length, defaults to the model config.")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Concurrent sequences in the KV cache, defaults to 1."
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="KV cache dtype, `auto` uses the model dtype (Safetensors) or F16 (GGUF). "
        f"Safetensors: {', '.join(KV_CACHE_DTYPES)}. GGUF: {', '.join(KV_DTYPES)}. ONNX: set by the graph.",
    )
    parser.add_argument(
        "--gpu-memory", help="GPU memory in GiB, e.g. `80` or `8x80`, to check whether it fits."
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of the GPU memory that can be used, defaults to 0.92 (as vLLM).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help="Tokens per batch as in vLLM, defaults to 8192 (16384 with `--gpu-memory` of 160 GiB or more, 2048 under 70 GiB), use 2048 for A100.",
    )
    parser.add_argument("--hf-token", help="Hub token, defaults to `HF_TOKEN` or the `hf auth login` token.")
    parser.add_argument("--json", action="store_true", help="Print JSON.")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table.")
    parser.add_argument("--version", action="version", version=f"hf-mem {__version__}")
    args = parser.parse_args()

    if args.batch_size < 1 or any(
        v is not None and v < 1 for v in (args.max_model_len, args.max_num_batched_tokens)
    ):
        parser.error("`--batch-size`, `--max-model-len` and `--max-num-batched-tokens` must be positive")
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

    if args.json:
        print(json.dumps(result.to_dict()))
        return
    for warning in result.warnings:
        print(f"hf-mem: warning: {warning}", file=sys.stderr)
    print(markdown(result) if args.markdown else table(result))
