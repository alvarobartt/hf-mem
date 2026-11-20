![hf-mem](https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/hf-mem.png)

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ONNX](https://onnx.ai) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers), GGUF and ONNX models (without the ONNX Runtime overhead).

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hugging Face Hub model ID (optionally with `@revision`, and with `:<quantization>` for GGUF), a URL, or a local path. Use `--gpu-memory` to check whether the model fits and `--help` for all options.

### Transformers

```console
$ uvx hf-mem zai-org/GLM-5.3-Flash
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
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

```console
$ uvx hf-mem Qwen/Qwen3.8-27B --max-model-len 32768 --batch-size 4
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│            INFERENCE MEMORY ESTIMATE FOR            │
│        https://hf.co/Qwen/Qwen3.8-27B @ main        │
│        w/ max-model-len=32768, batch-size=4         │
├─────────────────┬───────────────────────────────────┤
│ GPU CAPACITY    │ 73.71 GiB (79.14 GB) ON ONE GPU   │
│ RECOMMENDED     │ █████████████████████████████████ │
├─────────────────┴───────────────────────────────────┤
│          MODEL (27.36B PARAMS, 50.96 GiB)           │
├─────────────────┬───────────────────────────────────┤
│ BF16            │ 50.96 / 73.71 GiB                 │
│ 27.36B PARAMS   │ ██████████████████████▉░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────┤
│         KV CACHE (131072 TOKENS, 9.19 GiB)          │
├─────────────────┬───────────────────────────────────┤
│ BF16            │ 9.19 / 73.71 GiB                  │
│ 131072 TOKENS   │ ████▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────┤
│                 OVERHEAD (6.19 GiB)                 │
├─────────────────┬───────────────────────────────────┤
│ ACTIVATIONS     │ 2.65 / 73.71 GiB                  │
│                 │ █▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────┤
│ ENCODER         │ 2.42 / 73.71 GiB                  │
│                 │ █▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────┤
│ GRAPHS          │ 0.39 / 73.71 GiB                  │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────┤
│ RUNTIME         │ 0.74 / 73.71 GiB                  │
│                 │ ▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴───────────────────────────────────┘
```

### Diffusers

```console
$ uvx hf-mem Qwen/Qwen-Image-2.1
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│         INFERENCE MEMORY ESTIMATE FOR          │
│    https://hf.co/Qwen/Qwen-Image-2.1 @ main    │
├─────────────────┬──────────────────────────────┤
│ TOTAL MEMORY    │ 30.84 GiB (16.22B PARAMS)    │
│                 │ ████████████████████████████ │
├─────────────────┴──────────────────────────────┤
│     TEXT_ENCODER (8.77B PARAMS, 16.33 GiB)     │
├─────────────────┬──────────────────────────────┤
│ BF16            │ 16.33 / 30.84 GiB            │
│ 8.77B PARAMS    │ ██████████████▉░░░░░░░░░░░░░ │
├─────────────────┴──────────────────────────────┤
│     TRANSFORMER (7.12B PARAMS, 13.25 GiB)      │
├─────────────────┬──────────────────────────────┤
│ BF16            │ 13.25 / 30.84 GiB            │
│ 7.12B PARAMS    │ ████████████░░░░░░░░░░░░░░░░ │
├─────────────────┴──────────────────────────────┤
│         VAE (337.74M PARAMS, 1.26 GiB)         │
├─────────────────┬──────────────────────────────┤
│ F32             │ 1.26 / 30.84 GiB             │
│ 337.74M PARAMS  │ █▏░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴──────────────────────────────┘
```

### Sentence Transformers

```console
$ uvx hf-mem google/embeddinggemma-2
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│           INFERENCE MEMORY ESTIMATE FOR            │
│    https://hf.co/google/embeddinggemma-2 @ main    │
├─────────────────┬──────────────────────────────────┤
│ TOTAL MEMORY    │ 1.39 GiB (744.37M PARAMS)        │
│                 │ ████████████████████████████████ │
├─────────────────┴──────────────────────────────────┤
│          MODEL (744.37M PARAMS, 1.39 GiB)          │
├─────────────────┬──────────────────────────────────┤
│ BF16            │ 1.39 / 1.39 GiB                  │
│ 744.37M PARAMS  │ ████████████████████████████████ │
└─────────────────┴──────────────────────────────────┘
```

Text encoders, rerankers and Sentence Transformers include warmup memory. Use `--max-num-batched-tokens` for the warmup token budget and `--max-model-len` for the sequence length.

```console
$ uvx hf-mem BAAI/bge-base-en-v1.5 --max-num-batched-tokens 2048 --max-model-len 512 --gpu-memory 24
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│               INFERENCE MEMORY ESTIMATE FOR                │
│         https://hf.co/BAAI/bge-base-en-v1.5 @ main         │
├─────────────────┬──────────────────────────────────────────┤
│ GPU CAPACITY    │ 1.11 GiB (1.20 GB) ON ONE GPU            │
│ RECOMMENDED     │ ████████████████████████████████████████ │
├─────────────────┼──────────────────────────────────────────┤
│ GPU MEMORY      │ 1.11 / 24.00 GiB [5%] ON 1 x 24.00 GiB   │
│ FITS            │ █▉░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴──────────────────────────────────────────┤
│              MODEL (109.48M PARAMS, 0.41 GiB)              │
├─────────────────┬──────────────────────────────────────────┤
│ F32             │ 0.41 / 1.11 GiB                          │
│ 109.48M PARAMS  │ ██████████████▋░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼──────────────────────────────────────────┤
│ I64             │ 0.00 / 1.11 GiB                          │
│ 512 PARAMS      │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴──────────────────────────────────────────┤
│                    OVERHEAD (0.60 GiB)                     │
├─────────────────┬──────────────────────────────────────────┤
│ ACTIVATIONS     │ 0.28 / 1.11 GiB                          │
│                 │ █████████▉░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼──────────────────────────────────────────┤
│ RUNTIME         │ 0.32 / 1.11 GiB                          │
│                 │ ███████████▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴──────────────────────────────────────────┘
```

For unquantized Safetensors text encoders, `--dtype float16`, `--dtype bfloat16` or `--dtype float32` sets weight and activation precision. Encoder fit checks assume one replica per GPU.

### GGUF

GGUF repositories list the available quantizations. Select one with `<model>:Q4_K_M` or `--file`, which also accepts a file, directory or shard. Vision and audio models include the `mmproj` projector and encoder buffers.

```console
$ uvx hf-mem ggml-org/LFM3-1.7B-GGUF
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                                    INFERENCE MEMORY ESTIMATE FOR                                     │
│                             https://hf.co/ggml-org/LFM3-1.7B-GGUF @ main                             │
│                                w/ max-model-len=262144, batch-size=1                                 │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                FILES                                                 │
├───────────────────────┬──────────────────────────────────────────────────────────────────────────────┤
│                       │ 7.42 GiB GPU CAPACITY (3.22 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-BF16.gguf   │ ████████████████████████████████████████████████████████████████████████████ │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                       │ 4.97 GiB GPU CAPACITY (1.02 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-Q4_K_M.gguf │ ██████████████████████████████████████████████████▉░░░░░░░░░░░░░░░░░░░░░░░░░ │
├───────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                       │ 5.74 GiB GPU CAPACITY (1.71 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-Q8_0.gguf   │ ██████████████████████████████████████████████████████████▉░░░░░░░░░░░░░░░░░ │
└───────────────────────┴──────────────────────────────────────────────────────────────────────────────┘
```

```console
$ uvx hf-mem ggml-org/LFM3-1.7B-GGUF:Q4_K_M
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                    INFERENCE MEMORY ESTIMATE FOR                    │
│            https://hf.co/ggml-org/LFM3-1.7B-GGUF @ main             │
│                     FOR `LFM3-1.7B-Q4_K_M.gguf`                     │
│                w/ max-model-len=262144, batch-size=1                │
├─────────────────┬───────────────────────────────────────────────────┤
│ GPU CAPACITY    │ 4.97 GiB (5.34 GB) ON ONE GPU                     │
│ RECOMMENDED     │ █████████████████████████████████████████████████ │
├─────────────────┼───────────────────────────────────────────────────┤
│ CPU MEMORY      │ 0.20 GiB (262.14M PARAMS OFFLOADED, NOT ON GPU)   │
│ HOST MEMORY     │ ██▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                   MODEL (1.73B PARAMS, 1.02 GiB)                    │
├─────────────────┬───────────────────────────────────────────────────┤
│ Q4_K            │ 0.67 / 4.97 GiB                                   │
│ 1.27B PARAMS    │ ██████▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q6_K            │ 0.35 / 4.97 GiB                                   │
│ 456.13M PARAMS  │ ███▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ F32             │ 0.00 / 4.97 GiB                                   │
│ 387.90K PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                 KV CACHE (262144 TOKENS, 3.01 GiB)                  │
├─────────────────┬───────────────────────────────────────────────────┤
│ F16             │ 3.01 / 4.97 GiB                                   │
│ 262144 TOKENS   │ █████████████████████████████▋░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                         OVERHEAD (0.45 GiB)                         │
├─────────────────┬───────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 0.32 / 4.97 GiB                                   │
│                 │ ███▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ RUNTIME         │ 0.13 / 4.97 GiB                                   │
│                 │ █▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴───────────────────────────────────────────────────┘
```

`--file` also selects a single `*.safetensors` file or shard set, e.g. on ComfyUI repositories that ship multiple checkpoints.

### ONNX

ONNX repositories list the available variants, grouping graphs loaded together with their external data. Use `--file` to select a variant, file or directory.

```console
$ uvx hf-mem onnx-community/Qwen3-0.6B-ONNX --file onnx/model_q4f16.onnx
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.1.1 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│               INFERENCE MEMORY ESTIMATE FOR               │
│    https://hf.co/onnx-community/Qwen3-0.6B-ONNX @ main    │
│                FOR `onnx/model_q4f16.onnx`                │
│           w/ max-model-len=40960, batch-size=1            │
├─────────────────┬─────────────────────────────────────────┤
│ TOTAL MEMORY    │ 4.91 GiB (FROM FILE SIZES + KV CACHE)   │
│                 │ ███████████████████████████████████████ │
├─────────────────┴─────────────────────────────────────────┤
│                     MODEL (0.53 GiB)                      │
├─────────────────┬─────────────────────────────────────────┤
│ FILE SIZE       │ 0.53 / 4.91 GiB                         │
│                 │ ████▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────┤
│             KV CACHE (40960 TOKENS, 4.38 GiB)             │
├─────────────────┬─────────────────────────────────────────┤
│ F16             │ 4.38 / 4.91 GiB                         │
│ 40960 TOKENS    │ ██████████████████████████████████▊░░░░ │
└─────────────────┴─────────────────────────────────────────┘
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
print(result.total, result.weights.bytes, result.kv_cache.bytes, result.overhead.bytes)
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
