import argparse
import asyncio
import json
import os
import struct
import warnings
from dataclasses import asdict
from functools import reduce
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

from hf_mem.metadata import parse_safetensors_metadata
from hf_mem.print import print_report, print_report_for_gguf
from hf_mem.types import TorchDtypes, get_safetensors_dtype_bytes, torch_dtype_to_safetensors_dtype

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))
MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))


# NOTE: Return type-hint set to `Any`, but it will only be a JSON-compatible object
async def get_json_file(client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


async def fetch_safetensors_metadata(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    headers = {"Range": f"bytes=0-{MAX_METADATA_SIZE}", **(headers or {})}
    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata = response.read()
    # NOTE: Parse the first 8 bytes as a little-endian uint64 (size of the metadata)
    metadata_size = struct.unpack("<Q", metadata[:8])[0]

    if metadata_size < MAX_METADATA_SIZE:
        metadata = metadata[8 : metadata_size + 8]
        return json.loads(metadata)

    # NOTE: Given that by default we just fetch the first 100_000 bytes, if the content is larger
    # then we simply fetch the remainder again
    metadata = metadata[8 : MAX_METADATA_SIZE + 8]
    headers["Range"] = f"bytes={MAX_METADATA_SIZE + 1}-{metadata_size + 7}"

    response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    metadata += response.read()
    return json.loads(metadata)


async def fetch_modules_and_dense_metadata(
    client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]]
) -> Dict[str, Any]:
    dense_metadata = {}

    modules = await get_json_file(client=client, url=f"{url}/modules.json", headers=headers)
    paths = [
        module.get("path")
        for module in modules
        if "type" in module and module.get("type") == "sentence_transformers.models.Dense" and "path" in module
    ]

    for path in paths:
        # NOTE: It's "safe" to assume that if there's a `Dense` module defined in `modules.json`, it contains
        # Safetensors weights and if so, it's a single `model.safetensors` file as the sharding has a default on
        # ~5Gb per file, and usually the extra `Dense` layers are not larger than that (usually not even close).
        dense_metadata[path] = await fetch_safetensors_metadata(
            client=client, url=f"{url}/{path}/model.safetensors", headers=headers
        )

    return dense_metadata


async def run(
    model_id: str,
    revision: str,
    # START_KV_CACHE_ARGS
    experimental: bool = False,
    max_model_len: int | None = None,
    batch_size: int = 1,
    kv_cache_dtype: str | None = None,
    # END_KV_CACHE_ARGS
    json_output: bool = False,
    ignore_table_width: bool = False,
) -> Dict[str, Any] | None:
    headers = {"User-Agent": f"hf-mem/0.4; id={uuid4()}; model_id={model_id}; revision={revision}"}
    # NOTE: Read from `HF_TOKEN` if provided, then fallback to reading from `$HF_HOME/token`
    if token := os.getenv("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    elif "Authorization" not in headers:
        path = os.getenv("HF_HOME", ".cache/huggingface")
        filename = (
            os.path.join(os.path.expanduser("~"), path, "token")
            if not os.path.isabs(path)
            else os.path.join(path, "token")
        )

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                headers["Authorization"] = f"Bearer {f.read().strip()}"

    client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=MAX_CONCURRENCY,
            max_connections=MAX_CONCURRENCY,
        ),
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        # NOTE: HTTP/2 for header-compression and connection multiplexing
        http2=True,
        follow_redirects=True,
    )

    # TODO: `recursive=true` shouldn't really be required unless it's a Diffusers
    # models... I don't think this adds extra latency anyway
    url = f"https://huggingface.co/api/models/{model_id}/tree/{revision}?recursive=true"
    files = await get_json_file(client=client, url=url, headers=headers)
    file_paths = [f["path"] for f in files if f.get("path") and f.get("type") == "file"]

    if "model.safetensors" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors"
        raw_metadata = await fetch_safetensors_metadata(client=client, url=url, headers=headers)

        if "config_sentence_transformers.json" in file_paths:
            dense_metadata = (
                {}
                if "modules.json" not in file_paths
                else await fetch_modules_and_dense_metadata(
                    client=client, url=f"https://huggingface.co/{model_id}/resolve/{revision}", headers=headers
                )
            )

            raw_metadata = {"0_Transformer": raw_metadata, **dense_metadata}
        else:
            # NOTE: If the model is a transformers model, then we simply set the component name to `Transformer`, to
            # make sure that we provide the expected input to the `parse_safetensors_metadata`
            raw_metadata = {"Transformer": raw_metadata}

        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    elif "model.safetensors.index.json" in file_paths:
        # TODO: We could eventually skip this request in favour of a greedy approach on trying to pull all the
        # files following the formatting `model-00000-of-00000.safetensors`
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors.index.json"
        files_index = await get_json_file(client=client, url=url, headers=headers)

        urls = {
            f"https://huggingface.co/{model_id}/resolve/{revision}/{f}"
            for f in set(files_index["weight_map"].values())
        }

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def fetch_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await fetch_safetensors_metadata(client=client, url=url, headers=headers)

        tasks = [asyncio.create_task(fetch_with_semaphore(url)) for url in urls]
        metadata_list: List[Dict[str, Any]] = await asyncio.gather(*tasks, return_exceptions=False)

        raw_metadata = reduce(lambda acc, metadata: acc | metadata, metadata_list, {})

        if "config_sentence_transformers.json" in file_paths:
            dense_metadata = (
                {}
                if "modules.json" not in file_paths
                else await fetch_modules_and_dense_metadata(
                    client=client, url=f"https://huggingface.co/{model_id}/resolve/{revision}", headers=headers
                )
            )

            raw_metadata = {"0_Transformer": raw_metadata, **dense_metadata}
        else:
            # NOTE: If the model is a transformers model, then we simply set the component name to `Transformer`, to
            # make sure that we provide the expected input to the `parse_safetensors_metadata`
            raw_metadata = {"Transformer": raw_metadata}

        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    elif "model_index.json" in file_paths:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model_index.json"
        files_index = await get_json_file(client=client, url=url, headers=headers)
        paths = {k for k, _ in files_index.items() if not k.startswith("_")}

        path_urls: Dict[str, List[str]] = {}
        for path in paths:
            if f"{path}/diffusion_pytorch_model.safetensors" in file_paths:
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/diffusion_pytorch_model.safetensors"
                ]
            elif f"{path}/model.safetensors" in file_paths:
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/model.safetensors"
                ]
            elif f"{path}/diffusion_pytorch_model.safetensors.index.json" in file_paths:
                url = f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/diffusion_pytorch_model.safetensors.index.json"
                files_index = await get_json_file(client=client, url=url, headers=headers)
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/{f}"
                    for f in set(files_index["weight_map"].values())
                ]
            elif f"{path}/model.safetensors.index.json" in file_paths:
                url = (
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/model.safetensors.index.json"
                )
                files_index = await get_json_file(client=client, url=url, headers=headers)
                path_urls[path] = [
                    f"https://huggingface.co/{model_id}/resolve/{revision}/{path}/{f}"
                    for f in set(files_index["weight_map"].values())
                ]

        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        async def fetch_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await fetch_safetensors_metadata(client=client, url=url, headers=headers)

        # NOTE: Given that we need to fetch the Safetensors metadata for multiple components on Diffusers models,
        # to speed the download up and not block (await) the for-loop, we instead create all the tasks within a
        # for-loop then we await for those outside
        _tasks = {}
        for path, urls in path_urls.items():
            _tasks[path] = [asyncio.create_task(fetch_with_semaphore(url)) for url in urls]
        await asyncio.gather(*[task for tasks in _tasks.values() for task in tasks], return_exceptions=False)

        raw_metadata = {}
        for path, tasks in _tasks.items():
            metadata_list = [task.result() for task in tasks]
            raw_metadata[path] = reduce(lambda acc, metadata: acc | metadata, metadata_list, {})

        metadata = parse_safetensors_metadata(raw_metadata=raw_metadata)
    else:
        # Check for GGUF files
        gguf_files = {
            f["path"]: f["size"]
            for f in files
            if f.get("path", "").endswith(".gguf") and f.get("size") is not None
        }

        if gguf_files:
            print_report_for_gguf(
                model_id=model_id,
                revision=revision,
                gguf_files=gguf_files,
                ignore_table_width=ignore_table_width,
            )
            return
        else:
            raise RuntimeError(
                "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json`, OR `.gguf` FILES HAVE BEEN FOUND"
            )

    cache_size = None
    if experimental:
        # NOTE: In theory, `config.json` should always be present, but checking beforehand just in case
        if "config.json" in file_paths:
            url = f"https://huggingface.co/{model_id}/resolve/{revision}/config.json"
            config: Dict[str, Any] = await get_json_file(client, url, headers)

            if "architectures" not in config or (
                "architectures" in config
                and not any(
                    arch.__contains__("ForCausalLM") or arch.__contains__("ForConditionalGeneration")
                    for arch in config["architectures"]
                )
            ):
                warnings.warn(
                    "`--experimental` was provided, but either `config.json` doesn't have the `architectures` key meaning that the model architecture cannot be inferred, or rather that it's neither `...ForCausalLM` not `...ForConditionalGeneration`, meaning that the KV Cache estimation might not apply. If that's the case, then remove the `--experimental` flag from the command to suppress this warning."
                )
            else:
                if (
                    any(arch.__contains__("ForConditionalGeneration") for arch in config["architectures"])
                    and "text_config" in config
                ):
                    warnings.warn(
                        f"Given that `--model-id={model_id}` is a `...ForConditionalGeneration` model, then the configuration from `config.json` will be retrieved from the key `text_config` instead."
                    )
                    config = config["text_config"]

                if max_model_len is None:
                    max_model_len = config.get(
                        "max_position_embeddings",
                        config.get("n_positions", config.get("max_seq_len", max_model_len)),
                    )

                if max_model_len is None:
                    warnings.warn(
                        f"Either the `--max-model-len` was not set, is not available in `config.json` with the any of the keys: `max_position_embeddings`, `n_positions`, or `max_seq_len` (in that order of priority), or both; so the memory required to fit the context length cannot be estimated."
                    )

                if not all(k in config for k in {"hidden_size", "num_hidden_layers", "num_attention_heads"}):  # type: ignore
                    warnings.warn(
                        f"`config.json` doesn't contain all the keys `hidden_size`, `num_hidden_layers`, and `num_attention_heads`, but only {config.keys()}."  # type: ignore
                    )

                if kv_cache_dtype in {"fp8_e5m2", "fp8_e4m3"}:
                    cache_dtype = kv_cache_dtype.upper().replace("FP8", "F8")
                elif kv_cache_dtype in {"fp8", "fp8_ds_mla", "fp8_inc"}:
                    # NOTE: Default to `F8_E4M3` for the calculations, given that all those take 1 byte, but only F8_E5M2
                    # or `F8_E4M3` are supported in Safetensors, whilst `FP8_DS_MLA` (DeepSeek MLA) and `FP8_INC` (Intel HPUs)
                    # are not; and `F8_E4M3` is supported on both CUDA and AMD, hence seems a reasonable default
                    warnings.warn(
                        f"--kv-cache-dtype={kv_cache_dtype}` has been provided, but given that none of those matches an actual Safetensors dtype since it should be any of `F8_E5M2` or `F8_E4M3`, the `--kv-cache-dtype` will default to `F8_E4M3` instead, which implies that the calculations are the same given that both dtypes take 1 byte despite the quantization scheme of it, or the hardware compatibility; so the estimations should be accurate enough."
                    )
                    cache_dtype = "F8_E4M3"
                elif kv_cache_dtype == "bfloat16":
                    cache_dtype = "BF16"
                elif "quantization_config" in config and "quant_method" in config["quantization_config"]:
                    _quantization_config = config["quantization_config"]
                    _quant_method = _quantization_config["quant_method"]

                    if _quant_method != "fp8":  # NOTE: e.g., compressed-tensors for `moonshotai/Kimi-K2.5`
                        raise RuntimeError(
                            f"Provided `--kv-cache-dtype=auto` (or unset) and given that `config.json` contains the following `quantization_config={_quantization_config}` with a `quant_method` different than `fp8` i.e., `{_quant_method}`, which is not supported; you should enforce the `--kv-cache-dtype` value to whatever quantization precision it's using, if applicable.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
                        )

                    _fmt = _quantization_config.get("fmt", _quantization_config.get("format", None))
                    if _fmt:
                        if not _fmt.startswith("float8_"):
                            _fmt = f"float8_{_fmt}"

                        if _fmt not in TorchDtypes.__args__:
                            raise RuntimeError(
                                f"Provided `--kv-cache-dtype=auto` (or unset) and given that `config.json` contains the following `quantization_config={_quantization_config}` with a `fmt` (or `format`) value of `{_fmt}` that's not supported (should be any of {TorchDtypes.__args__}), you might need to set `--kv-cache-dtype=fp8` to enforce the dtype instead of pulling it from the `config.json`.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
                            )

                        cache_dtype = torch_dtype_to_safetensors_dtype(_fmt)
                    else:
                        # NOTE: If `quant_method` in `quantization_config` is set to `fp8` and `fmt` is not set, then
                        # we get the most used `F8_*` Safetensors dtype to map the `quant_method=fp8` to an actual Safetensors
                        # dtype, as `F8` is not a valid dtype neither on PyTorch nor on Safetensors, as we need to append
                        # the scheme / format.
                        # SAFETY: As per the snippets above, if `_fmt` is None we assume that `_quant_method=fp8`
                        cache_dtype = max(
                            (
                                l := [
                                    d
                                    for c in metadata.components.values()
                                    for d in c.dtypes.keys()
                                    if d in {"F8_E5M2", "F8_E4M3"}
                                ]
                            ),
                            key=l.count,
                            default=None,
                        )

                        # TODO: Not sure if we should default to `F8_E4M3` as a reasonable default as when `FP8`,
                        # `FP8_DS_MLA` or `FP8_INC` are provided... to prevent raising an exception
                        if not cache_dtype:
                            raise RuntimeError(
                                f"The `config.json` file for `--model-id={model_id}` contains `quantization_config={_quantization_config}` but the `quant_method=fp8` whereas any tensor in the model weights is set to any of `F8_E4M3` nor `F8_E5M2`, which means that the `F8_` format for the Safetensors dtype cannot be inferred; so you might need to set `--kv-cache-dtype=fp8` to enforce the dtype instead of pulling it from the `config.json`.\nAs KV cache estimation is still experimental, as that might not be the case for your model, then feel free to open an issue at https://github.com/alvarobartt/hf-mem with a report and eventually what solution you would like to see implemented."
                            )
                elif _cache_dtype := config.get("torch_dtype", None):
                    cache_dtype = torch_dtype_to_safetensors_dtype(_cache_dtype)
                elif _cache_dtype := config.get("dtype", None):
                    cache_dtype = torch_dtype_to_safetensors_dtype(_cache_dtype)
                else:
                    raise RuntimeError(
                        f"Provided `--kv-cache-dtype={kv_cache_dtype}` but it needs to be any of `auto`, `bfloat16`, `fp8`, `fp8_ds_mla`, `fp8_e4m3`, `fp8_e5m2` or `fp8_inc`. If `--kv-cache-dtype=auto` (or unset), then the `config.json` should either contain the `torch_dtype` or `dtype` fields set; or if quantized, then `quantization_config` needs to be set and contain the key `quant_method` with value `fp8` (as none of `fp32`, `fp16` or `bf16` is considered within the `quantization_config`), and optionally also contain `fmt` set to any valid FP8 format as `float8_e4m3` or `float8_e4m3fn`."
                    )

                # Reference: https://gist.github.com/alvarobartt/1097ca1b07c66fd71470937d599c2072
                cache_size = (
                    # NOTE: 2 because it applies to both key and value projections
                    2
                    * config.get("num_hidden_layers")  # type: ignore
                    # NOTE: `num_key_value_heads` defaults to `num_attention_heads` in MHA
                    * config.get("num_key_value_heads", config.get("num_attention_heads"))  # type: ignore
                    * (config.get("hidden_size") // config.get("num_attention_heads"))  # type: ignore
                    * max_model_len
                    * get_safetensors_dtype_bytes(cache_dtype)
                )

                if batch_size:
                    cache_size *= batch_size

    if json_output:
        out = {"model_id": model_id, "revision": revision, **asdict(metadata)}
        if experimental and cache_size:
            out["max_model_len"] = max_model_len
            out["batch_size"] = batch_size
            out["cache_size"] = cache_size
            out["cache_dtype"] = cache_dtype  # type: ignore
        print(json.dumps(out))
    else:
        # TODO: Use a `KvCache` dataclass instead and make sure that the JSON output is aligned
        if experimental and cache_size:
            print_report(
                model_id=model_id,
                revision=revision,
                metadata=metadata,
                cache={
                    "max_model_len": max_model_len,
                    "cache_size": cache_size,
                    "batch_size": batch_size,
                    "cache_dtype": cache_dtype,  # type: ignore
                },
                ignore_table_width=ignore_table_width,
            )
        else:
            print_report(
                model_id=model_id,
                revision=revision,
                metadata=metadata,
                ignore_table_width=ignore_table_width,
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
        "--experimental",
        action="store_true",
        help="Whether to enable the experimental KV Cache estimation or not. Only applies to `...ForCausalLM` and `...ForConditionalGeneration` models from Transformers.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        # Reference: https://docs.vllm.ai/en/stable/configuration/engine_args/#-max-model-len
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
        choices={"auto", "bfloat16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"},
        help="Data type for the KV cache storage. If `auto` is specified, it will use the default model dtype specified in the `config.json` (if available). Despite the FP8 data types having different formats, all those take 1 byte, meaning that the calculation would lead to the same results. Defaults to `auto`.",
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Whether to provide the output as a JSON instead of printed as table.",
    )
    parser.add_argument(
        "--ignore-table-width",
        action="store_true",
        help="Whether to ignore the maximum recommended table width, in case the `--model-id` and/or `--revision` cause a row overflow when printing those.",
    )

    args = parser.parse_args()

    if args.experimental:
        warnings.warn(
            "`--experimental` is set, which means that models with an architecture as `...ForCausalLM` and `...ForConditionalGeneration` will include estimations for the KV Cache as well. You can also provide the args `--max-model-len` and `--batch-size` as part of the estimation. Note that enabling `--experimental` means that the output will be different both when displayed and when dumped as JSON with `--json-output`, so bear that in mind."
        )

    asyncio.run(
        run(
            model_id=args.model_id,
            revision=args.revision,
            # NOTE: Below are the arguments that affect the KV cache estimation
            experimental=args.experimental,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            kv_cache_dtype=args.kv_cache_dtype,
            # NOTE: Below are the arguments that affect the output format
            json_output=args.json_output,
            ignore_table_width=args.ignore_table_width,
        )
    )
