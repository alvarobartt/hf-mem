# hf-mem

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ONNX](https://onnx.ai) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers), GGUF and ONNX models (without the ONNX Runtime overhead).

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hugging Face Hub model ID (optionally with `@revision`, and with `:<quantization>` for GGUF), an URL, or a local path.

### Transformers

```console
$ uvx hf-mem zai-org/GLM-5.3-Flash
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
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

For generative models (LLMs, VLMs, omni and speech models) the KV cache is estimated for the context length in `config.json` and a single sequence, which can be changed with `--max-model-len` and `--batch-size`. The KV cache dtype defaults to `auto`, using model precision unless the `quantization_config` declares a quantized KV cache, and can be set with `--kv-cache-dtype` (e.g. `fp8`).

```console
$ uvx hf-mem Qwen/Qwen3.8-27B --max-model-len 32768 --batch-size 4
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
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
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
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
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
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

```bash
uvx hf-mem BAAI/bge-base-en-v1.5 --max-num-batched-tokens 2048 --max-model-len 512 --gpu-memory 24
```

For unquantized Safetensors text encoders, `--dtype float16`, `--dtype bfloat16` or `--dtype float32` sets weight and activation precision. Encoder fit checks assume one replica per GPU.

### GGUF

GGUF repositories list every file, and a quantization tag as in llama.cpp (`<model>:Q4_K_M`, case-insensitive) or `--file` (a tag, file, directory or any shard of a sharded file) estimates a single one. The `mmproj` projector of vision and audio models is included as llama.cpp does, picking the one with the same quantization, or else Q8_0, F16, BF16 or F32, together with the compute buffer of its encoders in the overhead.

```console
$ uvx hf-mem LiquidAI/LFM3-1.7B-GGUF
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                                     INFERENCE MEMORY ESTIMATE FOR                                      │
│                              https://hf.co/LiquidAI/LFM3-1.7B-GGUF @ main                              │
│                                 w/ max-model-len=262144, batch-size=1                                  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                 FILES                                                  │
├─────────────────────────┬──────────────────────────────────────────────────────────────────────────────┤
│                         │ 7.42 GiB GPU CAPACITY (3.22 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-BF16.gguf     │ ████████████████████████████████████████████████████████████████████████████ │
├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                         │ 7.42 GiB GPU CAPACITY (3.22 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-F16.gguf      │ ████████████████████████████████████████████████████████████████████████████ │
├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                         │ 5.16 GiB GPU CAPACITY (1.18 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-NVFP4.gguf    │ ████████████████████████████████████████████████████▉░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                         │ 4.97 GiB GPU CAPACITY (1.02 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-Q4_K_M.gguf   │ ██████████████████████████████████████████████████▉░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                         │ 5.14 GiB GPU CAPACITY (1.16 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-Q5_K_M.gguf   │ ████████████████████████████████████████████████████▋░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                         │ 5.31 GiB GPU CAPACITY (1.32 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-Q6_K.gguf     │ ██████████████████████████████████████████████████████▍░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                         │ 5.74 GiB GPU CAPACITY (1.71 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-Q8_0.gguf     │ ██████████████████████████████████████████████████████████▉░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│                         │ 4.92 GiB GPU CAPACITY (0.97 GiB WEIGHTS + 3.01 GiB KV + 0.45 GiB OVERHEAD)   │
│ LFM3-1.7B-QAD-Q4_0.gguf │ ██████████████████████████████████████████████████▍░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────┴──────────────────────────────────────────────────────────────────────────────┘
```

```console
$ uvx hf-mem unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                   INFERENCE MEMORY ESTIMATE FOR                   │
│           https://hf.co/unsloth/Qwen3.8-27B-GGUF @ main           │
│                 FOR `Qwen3.8-27B-UD-Q4_K_M.gguf`                  │
│               w/ max-model-len=262144, batch-size=1               │
├─────────────────┬─────────────────────────────────────────────────┤
│ GPU CAPACITY    │ 36.74 GiB (39.45 GB) ON ONE GPU                 │
│ RECOMMENDED     │ ███████████████████████████████████████████████ │
├─────────────────┼─────────────────────────────────────────────────┤
│ CPU MEMORY      │ 0.67 GiB (1.27B PARAMS OFFLOADED, NOT ON GPU)   │
│ HOST MEMORY     │ ▉░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────┤
│                 MODEL (26.05B PARAMS, 14.75 GiB)                  │
├─────────────────┬─────────────────────────────────────────────────┤
│ IQ4_XS          │ 5.30 / 36.74 GiB                                │
│ 10.72B PARAMS   │ ██████▊░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ Q5_K            │ 4.87 / 36.74 GiB                                │
│ 7.61B PARAMS    │ ██████▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ Q4_K            │ 2.26 / 36.74 GiB                                │
│ 4.31B PARAMS    │ ██▉░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ Q6_K            │ 1.68 / 36.74 GiB                                │
│ 2.20B PARAMS    │ ██▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ IQ4_NL          │ 0.40 / 36.74 GiB                                │
│ 765.46M PARAMS  │ ▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ Q8_0            │ 0.08 / 36.74 GiB                                │
│ 81.26M PARAMS   │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ Q3_K            │ 0.07 / 36.74 GiB                                │
│ 178.26M PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ IQ3_S           │ 0.07 / 36.74 GiB                                │
│ 178.26M PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ F32             │ 0.01 / 36.74 GiB                                │
│ 2.67M PARAMS    │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────┤
│                 MMPROJ (460.73M PARAMS, 0.86 GiB)                 │
├─────────────────┬─────────────────────────────────────────────────┤
│ F16             │ 0.85 / 36.74 GiB                                │
│ 457.67M PARAMS  │ █▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ F32             │ 0.01 / 36.74 GiB                                │
│ 3.06M PARAMS    │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────┤
│                KV CACHE (262144 TOKENS, 16.15 GiB)                │
├─────────────────┬─────────────────────────────────────────────────┤
│ F16             │ 16.15 / 36.74 GiB                               │
│ 262144 TOKENS   │ ████████████████████▋░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────┤
│                        OVERHEAD (1.31 GiB)                        │
├─────────────────┬─────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 0.36 / 36.74 GiB                                │
│                 │ ▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ ENCODER         │ 0.47 / 36.74 GiB                                │
│                 │ ▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────┤
│ RUNTIME         │ 0.47 / 36.74 GiB                                │
│                 │ ▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴─────────────────────────────────────────────────┘
```

`--file` also selects a single `*.safetensors` file or shard set, e.g. on ComfyUI repositories that ship multiple checkpoints.

### ONNX

ONNX repositories list every variant, grouping the files loaded together (e.g. `encoder_model` and `decoder_model_merged`, or `vision_encoder`, `embed_tokens` and `decoder_model_merged` in [Transformers.js](https://github.com/huggingface/transformers.js) exports) with their external data, and `--file` estimates a single variant, file or directory (e.g. an [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) folder).

```console
$ uvx hf-mem onnx-community/Qwen3-0.6B-ONNX --file onnx/model_q4f16.onnx
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
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

The weights are the initializers in the ONNX graph, and the KV cache comes from the `past_*`, `cache.*` or `cache_params.*` graph inputs, i.e. the dtype and the cached heads per token as exported, with the context length from `config.json`. This includes Transformers' cache rename in [#50585](https://github.com/huggingface/transformers/pull/50585). ONNX files over 16 MiB with the weights inline are estimated from their file sizes, reading only the graph inputs at the end of the file, and the ONNX Runtime overhead is not estimated.

### Local models and the cache

Local directories and files work the same way (including `hf download --local-dir` folders, shown with the downloaded commit), and models already downloaded to the Hugging Face cache are read from disk when unchanged on the Hub, or only from the cache with `HF_HUB_OFFLINE=1`.

```bash
uvx hf-mem ./path/to/model
HF_HUB_OFFLINE=1 uvx hf-mem Qwen/Qwen3-8B
```

Gated and private models need a token, read from `--hf-token`, `HF_TOKEN`, or the token saved by `hf auth login`.

### Deployment sizing

Use `--gpu-memory` to check whether a model fits. Bare sizes and `GiB` are binary; `GB` is decimal: `8x141` means eight GPUs with 141 GiB each. The recommendation includes startup memory and a 90% allocation budget, adjustable with `--gpu-memory-utilization`.

```bash
uvx hf-mem thinkingmachines/Inkling-2-Small --engine sglang --gpu-memory 8x141 --context-length 1048576
```

`--gpu-type` is optional. Supported profiles are `nvidia-t4`, `nvidia-a10g`, `nvidia-l4`, `nvidia-l40s`, `nvidia-a100`, `nvidia-h100`, `nvidia-h200` and `nvidia-rtx-pro-6000`. When provided, `--gpu-memory` must match a supported capacity for that GPU, allowing for driver reservations.

Use `--engine sglang` for SGLang estimates. Safetensors default to vLLM, except Inkling 2 defaults to SGLang. GGUF uses llama.cpp.

For SGLang, `--swa-full-tokens-ratio` and `--mem-fraction-static` match the server's cache split and weights/cache budget. The latter is separate from `--gpu-memory-utilization`, which controls overall headroom.

Use `--language-model-only` to exclude identified vision/audio encoders and projectors when deploying a multimodal model for language tasks only.

| Meaning | Options |
| --- | --- |
| Model | positional ID, `--model`, `--model-path`, `--hf-repo` |
| Context per sequence | `--max-model-len`, `--context-length`, `--context-len` |
| Cached sequences | `--batch-size`, `--parallel`, `-np` |
| Admitted requests | `--max-num-seqs`, `--max-running-requests` |
| Prefill tokens | `--max-num-batched-tokens`, `--chunked-prefill-size` |
| Full sliding-window storage | `--disable-hybrid-kv-cache-manager`, `--disable-hybrid-swa-memory`, `--swa-full`, `--full-sliding-window-cache` |

Match `--max-num-seqs` to the server's admission limit; `--batch-size` is the number of sequences in the workload.

For llama.cpp, `--ctx-size`/`-c` sets the total context pool, divided by `--parallel`, and selects GGUF. `--no-mmproj` omits the projector. `--cache-type-k` and `--cache-type-v` must match. hf-mem's `--batch-size` counts sequences; llama.cpp's counts tokens.

### JSON and Markdown

`-o json` (or `--output json`) prints the estimate in bytes, and `-o markdown` prints a table to share in model cards, issues or pull requests.

```bash
uvx hf-mem Qwen/Qwen3-8B --max-model-len 32768 -o json
```

`recommended_gpu_memory` is the suggested GPU capacity, `total` is the modeled allocation, and `fit` checks the capacity supplied with `--gpu-memory`. All sizes in JSON are bytes.

### Python

```python
from hf_mem import run

result = run("Qwen/Qwen3-8B", max_model_len=32768)
print(result.recommended_gpu_memory)
print(result.total, result.weights.bytes, result.kv_cache.bytes, result.overhead.bytes)
print(result.to_dict())
```

`run` accepts the CLI options as keyword arguments. Use `await arun(...)` in async applications. Both return an `Estimate`, or a `Listing` for repositories with several GGUF or ONNX variants.

### Agents

Coding agents can use `uvx hf-mem <model> -o json` to answer questions such as how much memory a model needs or whether it fits on a given GPU (adding `--gpu-memory`), as described in the [`hf-mem` skill](skills/hf-mem/SKILL.md).

### Extensions

`hf-mem` can also be added as an extension to the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) as `hf mem ...`:

```bash
hf extensions add alvarobartt/hf-mem
```

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
