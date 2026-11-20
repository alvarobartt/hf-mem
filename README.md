![hf-mem](https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/hf-mem.png)

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ONNX](https://onnx.ai) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers), GGUF and ONNX models (without the ONNX Runtime overhead).

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hugging Face Hub model ID (optionally with `@revision`, and with `:<quantization>` for GGUF), a URL, or a local path. Use `--gpu-memory` to check whether the model fits and `--help` for all options.

### Transformers

```console
$ uvx hf-mem zai-org/GLM-5.3-Flash
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│             INFERENCE MEMORY ESTIMATE FOR             │
│      https://hf.co/zai-org/GLM-5.3-Flash @ main       │
│        w/ max-model-len=1048576, batch-size=1         │
├─────────────────┬─────────────────────────────────────┤
│ GPU CAPACITY    │ 361.67 GiB (388.34 GB) ON ONE GPU   │
│ RECOMMENDED     │ ███████████████████████████████████ │
├─────────────────┴─────────────────────────────────────┤
│            MODEL (9.67B PARAMS, 15.46 GiB)            │
├─────────────────┬─────────────────────────────────────┤
│ BF16            │ 12.90 / 361.67 GiB                  │
│ 6.93B PARAMS    │ █▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────┤
│ F8_E4M3         │ 2.55 / 361.67 GiB                   │
│ 2.74B PARAMS    │ ▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────┤
│ F32             │ 0.00 / 361.67 GiB                   │
│ 295.52K PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────┤
│       EXPERTS (288 x 1.08B PARAMS, 290.32 GiB)        │
├─────────────────┬─────────────────────────────────────┤
│ F8_E4M3         │ 290.25 / 361.67 GiB                 │
│ 311.65B PARAMS  │ ████████████████████████████▏░░░░░░ │
├─────────────────┼─────────────────────────────────────┤
│ F32             │ 0.07 / 361.67 GiB                   │
│ 0 PARAMS        │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────┤
│         KV CACHE (1048576 TOKENS, 16.81 GiB)          │
├─────────────────┬─────────────────────────────────────┤
│ BF16            │ 16.81 / 361.67 GiB                  │
│ 1048576 TOKENS  │ █▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────┤
│                  OVERHEAD (2.91 GiB)                  │
├─────────────────┬─────────────────────────────────────┤
│ ACTIVATIONS     │ 1.46 / 361.67 GiB                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────┤
│ ENCODER         │ 0.05 / 361.67 GiB                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────┤
│ GRAPHS          │ 0.49 / 361.67 GiB                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────┤
│ RUNTIME         │ 0.92 / 361.67 GiB                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴─────────────────────────────────────┘
```

Generative models default to the context length in `config.json` and one sequence. Set `--max-model-len`, `--batch-size` and `--kv-cache-dtype` (e.g. `fp8`) to estimate your workload; automatic cache precision follows the model config.

```bash
uvx hf-mem Qwen/Qwen3.8-27B --max-model-len 32768 --batch-size 4
```

### Diffusers

```bash
uvx hf-mem Qwen/Qwen-Image-2.1
```

### Sentence Transformers

```bash
uvx hf-mem google/embeddinggemma-2
```

Text encoders, rerankers and Sentence Transformers include warmup memory. Use `--max-num-batched-tokens` for the warmup token budget and `--max-model-len` for the sequence length.

```bash
uvx hf-mem BAAI/bge-base-en-v1.5 --max-num-batched-tokens 2048 --max-model-len 512 --gpu-memory 24
```

For unquantized Safetensors text encoders, `--dtype float16`, `--dtype bfloat16` or `--dtype float32` sets weight and activation precision. Encoder fit checks assume one replica per GPU.

### GGUF

GGUF repositories list the available quantizations. Select one with `<model>:Q4_K_M` or `--file`, which also accepts a file, directory or shard. Vision and audio models include the `mmproj` projector and encoder buffers.

```bash
uvx hf-mem LiquidAI/LFM3-1.7B-GGUF
```

```bash
uvx hf-mem unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

`--file` also selects a single `*.safetensors` file or shard set, e.g. on ComfyUI repositories that ship multiple checkpoints.

### ONNX

ONNX repositories list the available variants, grouping graphs loaded together with their external data. Use `--file` to select a variant, file or directory.

```bash
uvx hf-mem onnx-community/Qwen3-0.6B-ONNX --file onnx/model_q4f16.onnx
```

Weights and cache precision come from the graph, with the context length from `config.json`. Large files with inline weights use file sizes for the weight estimate. ONNX Runtime overhead is not included.

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

All sizes in JSON are bytes; `recommended_gpu_memory` gives the suggested GPU capacity.

### Python

```python
from hf_mem import run

result = run("Qwen/Qwen3-8B", max_model_len=32768)
print(result.recommended_gpu_memory)
print(result.to_dict())
```

`run` accepts the CLI options as keyword arguments. Use `await arun(...)` in async applications. Both return an `Estimate`, or a `Listing` for repositories with several GGUF or ONNX variants.

### Extensions

`hf-mem` can also be added as an extension to the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) as `hf mem ...`:

```bash
hf extensions add alvarobartt/hf-mem
```

## Agents

Coding agents can use `uvx hf-mem <model> -o json` to answer questions such as how much memory a model needs or whether it fits on a given GPU (adding `--gpu-memory`), as described in the [`hf-mem` skill](skills/hf-mem/SKILL.md).

## How it works

`hf-mem` only reads the metadata of the weight files (the Safetensors headers, the GGUF metadata or the ONNX graph) with HTTP Range requests, and adds up the memory an inference engine needs on the GPU:

- **Weights**: model tensors at the selected precision. Weights offloaded to CPU memory are reported separately.
- **KV cache**: memory for the context length and batch size, accounting for full attention, sliding windows and recurrent state.
- **Overhead**: activations, vision/audio encoders, persistent buffers, CUDA graphs and runtime workspaces.

Estimates depend on the engine and workload settings. Kernels and other runtime optimizations can change memory use.

## Limitations

- The fit check models memory for the stated workload; engine compatibility and automatic spare-cache reservations are separate. Different backends and non-default optimizations can change the footprint.
- ONNX models have no overhead estimate, and GGUF and ONNX listings estimate each file from its size, so a single file (`:<quantization>` or `--file`) gives the most accurate numbers.
- Vision and audio memory depends on the input size, so actual use can differ from the estimate.
- PyTorch pickle weights (`.bin`, `.pt`, `.ckpt` or `.nemo`) are not supported, and encoder-decoder models (e.g. Whisper) have no KV cache estimate.

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [ONNX - Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [vLLM - Memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
