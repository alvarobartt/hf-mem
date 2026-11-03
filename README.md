<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/hf-mem.png" />

---

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors) and [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, CUDA graphs and the CUDA context), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers) and GGUF models.

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hub model ID (optionally with `@revision`), a Hub URL, or a local path.

### Transformers

```bash
uvx hf-mem zai-org/GLM-5.3-Flash
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/transformers.png" />

For generative models (LLMs, VLMs, omni and speech models) the KV cache is estimated for the context length in `config.json` and a single sequence, which can be changed with `--max-model-len` and `--batch-size`. The KV cache dtype defaults to `auto`, which as in vLLM is the model dtype unless the `quantization_config` declares a quantized KV cache, and can be set with `--kv-cache-dtype` (e.g. `fp8`).

```bash
uvx hf-mem Qwen/Qwen3.8-27B --max-model-len 32768 --batch-size 4
```

### Diffusers

```bash
uvx hf-mem Qwen/Qwen-Image-2.1
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/diffusers.png" />

### Sentence Transformers

```bash
uvx hf-mem google/embeddinggemma-2
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/sentence-transformers.png" />

### GGUF

GGUF repositories list every file (skipping `mmproj` and `imatrix` files), and `--file` estimates a single file, which can also be a directory or any shard of a sharded file.

```bash
uvx hf-mem LiquidAI/LFM3-1.7B-GGUF
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/gguf.png" />

```bash
uvx hf-mem unsloth/Qwen3.8-27B-GGUF --file Qwen3.8-27B-UD-Q4_K_M.gguf
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/gguf-file.png" />

`--file` also selects a single `*.safetensors` file or shard set, e.g. on ComfyUI repositories that ship multiple checkpoints.

### Does it fit?

`--gpu-memory` takes the memory per GPU in GiB (e.g. `80`, or `8x80` for 8 GPUs) and reports whether the model fits, how many GPUs it needs, and the longest context that fits, using 90% of the memory as vLLM does by default (`--gpu-memory-utilization`).

```bash
uvx hf-mem meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 --gpu-memory 24
```

### Local models and the cache

Local directories and files work the same way, and models already downloaded to the Hugging Face cache are read from disk when unchanged on the Hub, or only from the cache with `HF_HUB_OFFLINE=1`.

```bash
uvx hf-mem ./path/to/model
HF_HUB_OFFLINE=1 uvx hf-mem Qwen/Qwen3-8B
```

Gated and private models need a token, read from `--hf-token`, `HF_TOKEN`, or the token saved by `hf auth login`.

### JSON and Markdown

`--json` prints the estimate in bytes, and `--markdown` prints a table to share in model cards, issues or pull requests.

```bash
uvx hf-mem Qwen/Qwen3-8B --max-model-len 32768 --json
```

```json
{
  "hf_mem": "1.0.0",
  "model": "Qwen/Qwen3-8B",
  "revision": "main",
  "format": "safetensors",
  "total": 23934363648,
  "weights": {"bytes": 16381470720, "params": 8190735360, "dtypes": {"BF16": {"bytes": 16381470720, "params": 8190735360}}},
  "kv_cache": {"bytes": 4831838208, "dtype": "BF16", "max_model_len": 32768, "batch_size": 1, "layers": {"full": 36}},
  "overhead": {"bytes": 2721054720, "activations": 1647312896, "cuda_graphs": 268435456, "runtime": 805306368},
  "warnings": []
}
```

- `total` is the sum of `weights`, `kv_cache` and `overhead` in bytes; `kv_cache` and `overhead` are `null` for non-generative models.
- `components` (per Diffusers component or file), `moe` (base model and experts breakdown) and `fit` (with `--gpu-memory`) are added when they apply.
- `format` is `safetensors`, `gguf`, or `file-sizes` for repositories with only `.bin`, `.pt`, `.ckpt` or `.nemo` weights.
- GGUF listings return a `files` list with the `file`, `total`, `weights`, `kv_cache` and `overhead` of each file instead.

### Python

```python
from hf_mem import run

result = run("Qwen/Qwen3-8B", max_model_len=32768)
print(result.total, result.weights.bytes, result.kv_cache.bytes, result.overhead.bytes)
# 23934363648 16381470720 4831838208 2721054720
print(result.to_dict())
```

`run` takes the same options as the CLI (`revision`, `file`, `max_model_len`, `batch_size`, `kv_cache_dtype`, `hf_token`, `gpu_memory` and `gpu_memory_utilization`), and `await arun(...)` can be used within an async application. Both return an `Estimate`, or a `Listing` with one `Estimate` per file for GGUF repositories.

### Agents

Coding agents can use `uvx hf-mem <model> --json` to answer questions such as how much memory a model needs or whether it fits on a given GPU (adding `--gpu-memory`), choosing a single `--file` for GGUF repositories since only one file is loaded for inference.

### Extensions

`hf-mem` can also be added as an extension to the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) as `hf mem ...`:

```bash
hf extensions add alvarobartt/hf-mem
```

## How it works

- **Weights** are the tensor sizes from the Safetensors headers (or the GGUF tensor infos), per dtype and per component, so they match what is loaded in memory.
- **KV cache** follows vLLM's allocation for Safetensors models: full attention layers hold `max_model_len` tokens, sliding window layers `sliding_window + 4096` tokens, Multi-head Latent Attention (MLA) caches the compressed latent, and the state of Mamba, linear attention and convolution layers is a constant per sequence. GGUF follows llama.cpp, i.e. sliding window layers hold `sliding_window + 512` tokens and the recurrent state is stored in FP32.
- **Overhead** follows vLLM's memory profiling for Safetensors models (activations for the sampler logits and 4096 tokens, CUDA graphs, CUDA context and non-torch memory) and llama.cpp's compute buffer and CUDA context for GGUF models.

The estimates have been validated on NVIDIA A100 GPUs against vLLM, SGLang, llama.cpp, Diffusers and Sentence Transformers, e.g. the KV cache per 32K-token request matches vLLM exactly for Llama 3.1, Qwen3, Qwen3 MoE, DeepSeek-V2 (MLA), Gemma 3 (sliding window), Qwen3.5 (linear attention) and Nemotron-H (Mamba), and the total is within 2% of the measured footprint for 7 of the 9 models tested. The vision encoder of VLMs and the linear attention kernels are not profiled, which added up to 3 GiB more on Qwen2.5-VL and Qwen3.5.

## Contributing

`hf-mem` is intentionally lightweight, simple and opinionated, so fixes and small improvements are welcome, but please open an issue before any non-trivial change, avoid new dependencies or abstractions, and keep the changes minimal and focused. `hf-mem` is maintained by humans, so please use AI assistance thoughtfully, similarly to the [Ghostty AI Policy](https://github.com/ghostty-org/ghostty/blob/main/AI_POLICY.md).

```bash
uv run --group test pytest
uv run --group lint pre-commit run --all-files
```

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [vLLM - Memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
