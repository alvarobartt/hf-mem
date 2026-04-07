import argparse
import asyncio
import json
import warnings

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

    parser.add_argument("--model-id", required=False, default=None, help="Model ID on the Hugging Face Hub")
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
        help="Whether to enable the experimental KV Cache estimation or not. Only applies to `...ForCausalLM` and `...ForConditionalGeneration` models from Transformers.",
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
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Show the full quantization estimates table in the hardware fitness report. Without this flag, only the minimum quantization required to fit the model is shown.",
    )

    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help=(
            "Hardware profile to check fitness against. Options: "
            "'local' or 'auto' for GPU auto-detection, "
            "a catalog name (e.g. 'a100-80gb', 'rtx-4090'), "
            "a cloud instance (e.g. 'aws:p4d.24xlarge', 'azure:standard_nc24ads_a100_v4'), "
            "a multiplier (e.g. '4x a100-80gb'), "
            "or a bare VRAM size (e.g. '24'). "
            "Use --list-hardware to see all built-in profiles."
        ),
    )
    parser.add_argument(
        "--hardware-file",
        type=str,
        default=None,
        help='Path to a JSON file describing custom hardware. Schema: {"name": "...", "gpus": [{"name": "...", "vram_gb": N}]}.',
    )
    parser.add_argument(
        "--list-hardware",
        action="store_true",
        help="List all built-in hardware profiles and exit.",
    )

    args = parser.parse_args()

    if args.list_hardware:
        from hf_mem.hardware.print import print_catalog

        print_catalog()
        return

    if args.model_id is None:
        parser.error("--model-id is required (unless using --list-hardware)")

    if args.experimental:
        warnings.warn(
            "`--experimental` is set, which means that models with an architecture as `...ForCausalLM` and `...ForConditionalGeneration` will include estimations for the KV Cache as well. You can also provide the args `--max-model-len` and `--batch-size` as part of the estimation. Note that enabling `--experimental` means that the output will be different both when displayed and when dumped as JSON with `--json-output`, so bear that in mind."
        )

    if args.ignore_table_width:
        warnings.warn(
            "`--ignore-table-width` is deprecated and has no effect; table width auto-expands when needed by default."
        )

    if args.kv_cache_dtype not in VALID_KV_CACHE_DTYPES:
        raise RuntimeError(
            f"--kv-cache-dtype={args.kv_cache_dtype} not recognized. Valid options: {VALID_KV_CACHE_DTYPES}."
        )

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

    # Hardware fitness check (when --hardware or --hardware-file is provided)
    fitness = None
    if args.hardware or args.hardware_file:
        from hf_mem.hardware.fitness import check_fitness
        from hf_mem.hardware.print import print_fitness_report
        from hf_mem.hardware.resolve import resolve_hardware

        profile = resolve_hardware(hardware=args.hardware, hardware_file=args.hardware_file)
        if profile is not None:
            fitness = check_fitness(result, profile)

    if args.json_output:
        output = result.to_json()
        if fitness is not None:
            output["fitness"] = fitness.to_json()
        print(json.dumps(output))
    else:
        _print_result(result)
        if fitness is not None:
            print()
            print_fitness_report(fitness, extended=args.extended)
