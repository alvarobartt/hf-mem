import argparse
import asyncio
import json
import warnings

import httpx2

from hf_mem.gguf.print import print_gguf_files_report, print_gguf_report
from hf_mem.gguf.types import GGUFDtype
from hf_mem.run import Result, arun
from hf_mem.safetensors.kv_cache import KV_CACHE_DTYPE_CHOICES
from hf_mem.safetensors.print import print_safetensors_report

VALID_KV_CACHE_DTYPES = set(KV_CACHE_DTYPE_CHOICES) | set(GGUFDtype.__members__.keys())


def _print_result(result: Result) -> None:
    if result.safetensors is not None:
        print_safetensors_report(
            model_id=result.model_id,
            revision=result.revision,
            metadata=result.safetensors,
            kv_cache=result.kv_cache_metadata,
            moe=result.moe_metadata,
        )
        return

    if result.gguf_files is not None:
        if result.filename is not None:
            print_gguf_report(
                model_id=result.model_id,
                filename=result.filename,
                revision=result.revision,
                metadata=result.gguf_files[result.filename],
                kv_cache=result.gguf_files[result.filename].kv_cache,
            )
        else:
            print_gguf_files_report(
                model_id=result.model_id,
                revision=result.revision,
                gguf_files=result.gguf_files,
                memory=result.memory,  # type: ignore[arg-type]
                kv_cache=result.kv_cache,  # type: ignore[arg-type]
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", required=True, help="Model ID on the Hugging Face Hub")
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision on the Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        required=False,
        help="The Hugging Face Hub token to use when sending requests to the Hub. If not provided it will be retrieved from either the `HF_TOKEN` environment variable or from the `.cache/huggingface/token` file if applicable.",
    )

    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Also estimate the KV cache for generative models (and the experts breakdown for MoEs).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        # NOTE: https://docs.vllm.ai/en/stable/configuration/engine_args/#-max-model-len
        help="Model context length (prompt and output). If unspecified, will be automatically derived from the model config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to help estimate the required RAM for caching when running the inference. Defaults to 1.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        # NOTE: https://docs.vllm.ai/en/stable/cli/serve/#-kv-cache-dtype
        help=f"Data type for the KV cache storage. If `auto` is specified, it will use the default model dtype specified in the `config.json` (if available) or F16 for GGUF files. Despite the FP8 data types having different formats, all those take 1 byte, meaning that the calculation would lead to the same results. Valid values are {KV_CACHE_DTYPE_CHOICES} for safetensors files and {['auto'] + list(GGUFDtype.__members__.keys())} for GGUF files. Defaults to `auto`.",
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Whether to provide the output as a JSON instead of printed as table.",
    )
    parser.add_argument(
        "--ignore-table-width",
        action="store_true",
        help="Deprecated: table width now auto-expands when needed, so this flag is ignored.",
    )
    parser.add_argument(
        "--gguf-file",
        type=str,
        default=None,
        help="Specific GGUF file to estimate. If not provided, all GGUF files found in the repo will be estimated. The path within the repo is required (e.g. Q4_K_M/model-Q4_K_M.gguf).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-component dtype and parameter breakdowns in JSON output (`--json-output` only).",
    )
    args = parser.parse_args()
    warnings.formatwarning = lambda message, *_, **__: f"hf-mem: warning: {message}\n"

    if args.ignore_table_width:
        warnings.warn(
            "`--ignore-table-width` is deprecated and has no effect; table width auto-expands when needed by default."
        )

    if args.kv_cache_dtype not in VALID_KV_CACHE_DTYPES:
        parser.error(
            f"--kv-cache-dtype={args.kv_cache_dtype} not recognized. Valid options: {VALID_KV_CACHE_DTYPES}."
        )

    try:
        result = asyncio.run(
            arun(
                model_id=args.model_id,
                revision=args.revision,
                hf_token=args.hf_token,
                experimental=args.experimental,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                kv_cache_dtype=args.kv_cache_dtype,
                gguf_file=args.gguf_file,
                details=args.details,
            )
        )
    except (RuntimeError, httpx2.HTTPError) as e:
        parser.exit(1, f"hf-mem: error: {e}\n")

    if args.json_output:
        print(json.dumps(result.to_json()))
    else:
        _print_result(result)
