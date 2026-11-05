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

`--gpu-memory` takes the memory per GPU in GiB (e.g. `80`, or `8x80` for 8 GPUs) and adds to the summary the memory required on the given GPUs out of the usable memory, whether the model fits or how many GPUs it needs, and the longest context that fits on the given GPUs, using 92% of the memory as vLLM does by default (`--gpu-memory-utilization`). With tensor parallelism, the KV heads (and the MLA latent) that are replicated across GPUs are taken into account.

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

Coding agents can use `uvx hf-mem <model> -o json` to answer questions such as how much memory a model needs or whether it fits on a given GPU (adding `--gpu-memory`), choosing a single `--file` for GGUF and ONNX repositories since only one file is loaded for inference.

### Extensions

`hf-mem` can also be added as an extension to the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) as `hf mem ...`:

```bash
hf extensions add alvarobartt/hf-mem
```

## How it works

- **Weights** are the tensor sizes from the Safetensors headers (or the GGUF tensor infos), per dtype and per component, so they match what is loaded in memory. Weights the engine keeps in CPU memory are shown apart and not in the GPU total: the Engram tables of DeepSeek-V4.1 and the n-gram embeddings of Qwen3.8-Flash-Next, which vLLM keeps in pinned host memory, and the token embeddings, which llama.cpp keeps on the CPU (with a GPU copy when tied to the output). With `--gpu-memory`, the MXFP4 experts are padded per GPU to vLLM's kernel tiles (e.g. +33% for DeepSeek-V4.1 on 8 GPUs).
- **KV cache** follows vLLM's allocation for Safetensors models: full attention layers hold `max_model_len` tokens, sliding window layers `sliding_window` plus two batches of tokens in flight, Multi-head Latent Attention (MLA) caches the compressed latent (plus the FP8 indexer keys of DeepSeek-V3.2 and GLM-5 on the layers with their own indexer), DeepSeek-V4.1 caches its compressed and sliding window records as vLLM does, and the state of Mamba, linear attention and convolution layers is a constant per sequence. GGUF follows llama.cpp, i.e. sliding window layers hold `sliding_window + 512` tokens and the recurrent state is stored in FP32.
- **Overhead** is the memory an inference engine needs on top of the weights and the KV cache, per GPU, modeled on vLLM's memory profiling for Safetensors models and on llama.cpp's compute buffer and device context for GGUF models:
  - `activations`: the peak of the dummy forward pass over the batched tokens and of the sampler over the vocabulary.
  - `encoder`: the vision or audio encoder activations for the largest image or audio clip, from `config.json` and the processor configs, or from the `mmproj` metadata for GGUF, as llama.cpp grows it up to the largest image (4096 tokens for dynamic resolution encoders, `image_size` otherwise) and 30 seconds of audio.
  - `graphs`: the CUDA (or HIP) graphs captured for decoding, as estimated by vLLM before sizing the KV cache.
  - `runtime`: the device context, which vLLM keeps outside `--gpu-memory-utilization`, plus the attention, BLAS and collective communication workspaces.

  vLLM picks its batching by GPU memory, not architecture: 16384 batched tokens on GPUs with 160 GiB or more (e.g. B200, MI300X), 8192 tokens on GPUs with 70 GiB or more (e.g. H100, H200), and 2048 tokens with 256 sequences on smaller GPUs and on A100s, which are excluded by name. hf-mem follows `--gpu-memory`, or `--max-num-batched-tokens` (e.g. 2048 on A100). With the same batching, only the graphs differ between architectures (up to 0.5 GiB more on Hopper than on Ampere).

The estimates have been validated with vLLM 0.33 and SGLang 0.5 on NVIDIA T4, L4, A10G, A100, RTX PRO 6000 and H200 GPUs (up to 8 GPUs), with llama.cpp, and with Diffusers and Sentence Transformers, on 40+ LLMs, MoEs, hybrid, vision, audio and omni models. AMD GPUs go through the same memory profiling, batching and graph capture in vLLM on ROCm, so the same estimate applies, but it has not been validated there and the device context and workspaces may differ.

- The KV cache per request matches vLLM exactly for Llama 3, Qwen3, Qwen3 MoE, DeepSeek-V2 (MLA), GLM-5.3 (MLA with sparse indexers), Gemma 3 and Gemma 4 (sliding window), gpt-oss, Qwen3.5 (linear attention) and Nemotron-H (Mamba), and within 3% for DeepSeek-V4.1 (compressed attention), Kimi Linear (KDA) and Qwen3.8-Flash-Next, from 32K to 512K tokens on 8 x H200, and the KV cache per token matches SGLang.
- The overhead has a median error of 10% against vLLM and lies between vLLM's reservation and SGLang's actual usage for most models (SGLang reserves a fixed share of the GPU instead), e.g. the longest context for Llama 3.1 8B on an L4 is 29,878 tokens against 30,048 in vLLM.
- vLLM profiles the vision encoder with the largest video when it has more tokens than an image (e.g. Qwen2.5-VL reserves 3 to 7 GiB more at 32K to 64K tokens), and hf-mem warns about it.
- The `mmproj` encoder buffer matches llama.cpp with the largest image within 5% for Gemma 3, Qwen2.5-VL, LFM3-VL, LFM2.5-VL, SmolVLM2, InternVL3, Pixtral and Ultravox, which can be up to 2x what llama.cpp reserves at startup.
- The weights per GPU match vLLM within 5% for DeepSeek-V4.1 (with 189 GiB of Engram tables in host memory), GLM-5.3, Qwen3.8-Flash-Next, Kimi Linear and gpt-oss, and the token embeddings in CPU memory match llama.cpp exactly. SGLang loads the gpt-oss experts in BF16 on Hopper GPUs instead.

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [ONNX - Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [vLLM - Memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
