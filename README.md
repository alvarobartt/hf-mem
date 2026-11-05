<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/6079c94fcfeb6a8feb79a0c84883cc99fd08d915/hf-mem.png" />

---

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ONNX](https://onnx.ai) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers), GGUF and ONNX models (without the ONNX Runtime overhead).

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hub model ID (optionally with `@revision`, and with `:<quantization>` for GGUF), a Hub URL, or a local path.

### Transformers

```bash
uvx hf-mem zai-org/GLM-5.3-Flash
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/6079c94fcfeb6a8feb79a0c84883cc99fd08d915/transformers.png" />

For generative models (LLMs, VLMs, omni and speech models) the KV cache is estimated for the context length in `config.json` and a single sequence, which can be changed with `--max-model-len` and `--batch-size`. The KV cache dtype defaults to `auto`, which as in vLLM is the model dtype unless the `quantization_config` declares a quantized KV cache, and can be set with `--kv-cache-dtype` (e.g. `fp8`).

```bash
uvx hf-mem Qwen/Qwen3.8-27B --max-model-len 32768 --batch-size 4
```

### Diffusers

```bash
uvx hf-mem Qwen/Qwen-Image-2.1
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/6079c94fcfeb6a8feb79a0c84883cc99fd08d915/diffusers.png" />

### Sentence Transformers

```bash
uvx hf-mem google/embeddinggemma-2
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/6079c94fcfeb6a8feb79a0c84883cc99fd08d915/sentence-transformers.png" />

### GGUF

GGUF repositories list every file, and a quantization tag as in llama.cpp (`<model>:Q4_K_M`, case-insensitive) or `--file` (a tag, file, directory or any shard of a sharded file) estimates a single one. The `mmproj` projector of vision and audio models is included as llama.cpp does, picking the one with the same quantization, or else Q8_0, F16, BF16 or F32, together with the compute buffer of its encoders in the overhead.

```bash
uvx hf-mem LiquidAI/LFM3-1.7B-GGUF
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/6079c94fcfeb6a8feb79a0c84883cc99fd08d915/gguf.png" />

```bash
uvx hf-mem unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/6079c94fcfeb6a8feb79a0c84883cc99fd08d915/gguf-file.png" />

`--file` also selects a single `*.safetensors` file or shard set, e.g. on ComfyUI repositories that ship multiple checkpoints.

### ONNX

ONNX repositories list every variant, grouping the files loaded together (e.g. `encoder_model` and `decoder_model_merged`, or `vision_encoder`, `embed_tokens` and `decoder_model_merged` in [Transformers.js](https://github.com/huggingface/transformers.js) exports) with their external data, and `--file` estimates a single variant, file or directory (e.g. an [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) folder).

```bash
uvx hf-mem onnx-community/Qwen3-0.6B-ONNX --file onnx/model_q4f16.onnx
```

The weights are the initializers in the ONNX graph, and the KV cache comes from the `past_*` graph inputs, i.e. the dtype and the cached heads per token as exported, with the context length from `config.json`. ONNX files over 16 MiB with the weights inline are estimated from their file sizes, reading only the graph inputs at the end of the file, and the ONNX Runtime overhead is not estimated.

### Picking a format

Safetensors weights are estimated first, then GGUF, and finally ONNX, while PyTorch pickle weights (`.bin`, `.pt`, `.ckpt` or `.nemo`) are not supported, as they cannot be read without loading them. `--format safetensors`, `--format gguf` or `--format onnx` picks one, and a warning mentions the GGUF or ONNX files that were found next to the Safetensors weights.

### Does it fit?

`--gpu-memory` takes the memory per GPU in GiB (e.g. `80`, or `8x80` for 8 GPUs) and adds to the summary the memory required on the given GPUs out of the usable memory, whether the model fits or how many GPUs it needs, and the longest context that fits on the given GPUs, using 92% of the memory as vLLM does by default (`--gpu-memory-utilization`). With tensor parallelism, the KV heads (and the MLA latent) that are replicated across GPUs are taken into account. Weights that the inference engine keeps in CPU memory (e.g. the DeepSeek-V4.1 Engram tables) are shown apart as `CPU MEMORY`, as they need that much free system memory instead. vLLM batches fewer tokens on A100s than on other GPUs with 80 GiB, so use `--max-num-batched-tokens 2048` there.

```bash
uvx hf-mem meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 --gpu-memory 24
```

### Local models and the cache

Local directories and files work the same way (including `hf download --local-dir` folders, shown with the downloaded commit), and models already downloaded to the Hugging Face cache are read from disk when unchanged on the Hub, or only from the cache with `HF_HUB_OFFLINE=1`.

```bash
uvx hf-mem ./path/to/model
HF_HUB_OFFLINE=1 uvx hf-mem Qwen/Qwen3-8B
```

Gated and private models need a token, read from `--hf-token`, `HF_TOKEN`, or the token saved by `hf auth login`.

### JSON and Markdown

`-o json` (or `--output json`) prints the estimate in bytes, and `-o markdown` prints a table to share in model cards, issues or pull requests.

```bash
uvx hf-mem Qwen/Qwen3-8B --max-model-len 32768 -o json
```

```json
{
  "version": "hf-mem@1.0.0",
  "model": "Qwen/Qwen3-8B",
  "revision": "main",
  "file": null,
  "format": "safetensors",
  "total": 27343113879,
  "weights": {"bytes": 16381470720, "params": 8190735360, "dtypes": {"BF16": {"bytes": 16381470720, "params": 8190735360}}},
  "offloaded": null,
  "components": null,
  "kv_cache": {"bytes": 4831838208, "dtype": "BF16", "max_model_len": 32768, "batch_size": 1, "layers": {"full": 36}},
  "overhead": {"bytes": 6129804951, "activations": 3371389470, "encoder": 0, "graphs": 739780853, "runtime": 2018634628},
  "moe": null,
  "fit": null,
  "warnings": []
}
```

- `total` is the sum of `weights`, `kv_cache` and `overhead` in bytes, and the `overhead` parts are described in [How it works](#how-it-works); `kv_cache` and `overhead` are `null` for non-generative models.
- `file`, `offloaded` (weights in CPU memory, not in `total`), `components` (per Diffusers component, file or GGUF `mmproj`), `moe` (base model and experts breakdown) and `fit` (with `--gpu-memory`) are `null` when they do not apply, and `params` is `null` when the weights come from file sizes (ONNX files with the weights inline).
- `fit` has the `gpu_memory` and `usable` memory per GPU, the memory `required` on the given GPUs (with the overhead and the replicated KV cache per GPU), the GPUs `needed`, whether it `fits`, and the longest `max_model_len` that fits.
- `format` is `safetensors`, `gguf` or `onnx`.
- GGUF and ONNX repositories with several files return a `files` list with the `file`, `total`, `weights`, `kv_cache` and `overhead` of each file instead.

### Python

```python
from hf_mem import run

result = run("Qwen/Qwen3-8B", max_model_len=32768)
print(result.total, result.weights.bytes, result.kv_cache.bytes, result.overhead.bytes)
# 27343113879 16381470720 4831838208 6129804951
print(result.to_dict())
```

`run` takes the same options as the CLI (`revision`, `file`, `format`, `max_model_len`, `batch_size`, `kv_cache_dtype`, `hf_token`, `gpu_memory`, `gpu_memory_utilization` and `max_num_batched_tokens`), and `await arun(...)` can be used within an async application. Both return an `Estimate`, or a `Listing` with one `Estimate` per file for GGUF and ONNX repositories.

### Agents

Coding agents can use `uvx hf-mem <model> -o json` to answer questions such as how much memory a model needs or whether it fits on a given GPU (adding `--gpu-memory`), as described in the [`hf-mem` skill](skills/hf-mem/SKILL.md).

### Extensions

`hf-mem` can also be added as an extension to the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) as `hf mem ...`:

```bash
hf extensions add alvarobartt/hf-mem
```

## How it works

`hf-mem` only reads the metadata of the weight files (the Safetensors headers, the GGUF metadata or the ONNX graph) with HTTP Range requests, and adds up the memory an inference engine needs on the GPU:

- **Weights**: the exact size of every tensor, per dtype and component. Weights that the engine keeps in CPU memory are reported apart, as vLLM does with the DeepSeek-V4.1 Engram tables and llama.cpp with the token embeddings.
- **KV cache**: what the engine allocates for `--max-model-len` tokens and `--batch-size` sequences, following the model's attention: full attention caches every token, sliding window layers only the window, latent and compressed attention (e.g. DeepSeek, GLM) a smaller record per token, and linear attention or Mamba layers a fixed state per sequence.
- **Overhead**: the memory used on top of that, i.e. the activations of a forward pass, the vision or audio encoder for the largest input, the CUDA graphs, and the runtime (the CUDA context and library workspaces).

The KV cache and the overhead follow vLLM for Safetensors models and llama.cpp for GGUF models, as validated on NVIDIA T4 to H200 GPUs with vLLM 0.33, SGLang 0.5 and llama.cpp on 50+ models: the KV cache matches vLLM exactly for most architectures (within 3% for DeepSeek-V4.1, Kimi Linear and Qwen3.8-Flash-Next), the overhead is within 10% of vLLM's on median, and the vision and audio encoders of GGUF models are within 5% of llama.cpp's.

## Limitations

- The estimates follow vLLM and llama.cpp with their default settings, so other engines (e.g. SGLang, which reserves a fixed share of the GPU instead) and non-default settings use different amounts of memory.
- ONNX models have no overhead estimate, and GGUF and ONNX listings estimate each file from its size, so a single file (`:<quantization>` or `--file`) gives the most accurate numbers.
- vLLM profiles the vision encoder of some models (e.g. Qwen2.5-VL) with a video when it has more tokens than an image, which can reserve a few GiB more (`hf-mem` warns about it).
- PyTorch pickle weights (`.bin`, `.pt`, `.ckpt` or `.nemo`) are not supported, and encoder-decoder models (e.g. Whisper) have no KV cache estimate.
- AMD and NVIDIA Blackwell GPUs go through the same accounting in vLLM, but have not been validated.

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [ONNX - Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [vLLM - Memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
