# hf-mem

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ONNX](https://onnx.ai) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers), GGUF and ONNX models (without the ONNX Runtime overhead).

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hugging Face Hub model ID (optionally with `@revision`, and with `:<quantization>` for GGUF), an URL, or a local path.

### Transformers

```console
$ uvx hf-mem zai-org/GLM-5.3-Flash
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                     INFERENCE MEMORY ESTIMATE FOR                     │
│              https://hf.co/zai-org/GLM-5.3-Flash @ main               │
│                w/ max-model-len=1048576, batch-size=1                 │
├─────────────────┬─────────────────────────────────────────────────────┤
│ TOTAL MEMORY    │ 329.36 GiB (321.34B PARAMS + KV CACHE + OVERHEAD)   │
│ REQUIREMENTS    │ ███████████████████████████████████████████████████ │
├─────────────────┴─────────────────────────────────────────────────────┤
│                    MODEL (9.67B PARAMS, 15.46 GiB)                    │
├─────────────────┬─────────────────────────────────────────────────────┤
│ BF16            │ 12.90 / 329.36 GiB                                  │
│ 6.93B PARAMS    │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ F8_E4M3         │ 2.55 / 329.36 GiB                                   │
│ 2.74B PARAMS    │ ▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ F32             │ 0.00 / 329.36 GiB                                   │
│ 462.94K PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────────┤
│               EXPERTS (288 x 1.08B PARAMS, 290.32 GiB)                │
├─────────────────┬─────────────────────────────────────────────────────┤
│ F8_E4M3         │ 290.25 / 329.36 GiB                                 │
│ 311.65B PARAMS  │ █████████████████████████████████████████████░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ F32             │ 0.07 / 329.36 GiB                                   │
│ 19.02M PARAMS   │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────────┤
│                 KV CACHE (1048576 TOKENS, 16.80 GiB)                  │
├─────────────────┬─────────────────────────────────────────────────────┤
│ BF16            │ 16.80 / 329.36 GiB                                  │
│ 1048576 TOKENS  │ ██▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────────┤
│                          OVERHEAD (6.78 GiB)                          │
├─────────────────┬─────────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 3.17 / 329.36 GiB                                   │
│                 │ ▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ ENCODER         │ 0.03 / 329.36 GiB                                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ GRAPHS          │ 1.69 / 329.36 GiB                                   │
│                 │ ▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ RUNTIME         │ 1.88 / 329.36 GiB                                   │
│                 │ ▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴─────────────────────────────────────────────────────┘
```

For generative models (LLMs, VLMs, omni and speech models) the KV cache is estimated for the context length in `config.json` and a single sequence, which can be changed with `--max-model-len` and `--batch-size`. The KV cache dtype defaults to `auto`, which as in vLLM is the model dtype unless the `quantization_config` declares a quantized KV cache, and can be set with `--kv-cache-dtype` (e.g. `fp8`).

Both `max_position_embeddings` and `model_max_length` are supported, as are `torch_dtype` and `dtype`. Attention settings can use the older global fields or Transformers' sparse `per_layer_config` mapping. Local attention layers (including Inkling-2's `local_layer_ids`) and shared KV layers are accounted for in the cache estimate.

```console
$ uvx hf-mem Qwen/Qwen3.8-27B --max-model-len 32768 --batch-size 4
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                    INFERENCE MEMORY ESTIMATE FOR                    │
│                https://hf.co/Qwen/Qwen3.8-27B @ main                │
│                w/ max-model-len=32768, batch-size=4                 │
├─────────────────┬───────────────────────────────────────────────────┤
│ TOTAL MEMORY    │ 70.15 GiB (27.78B PARAMS + KV CACHE + OVERHEAD)   │
│ REQUIREMENTS    │ █████████████████████████████████████████████████ │
├─────────────────┴───────────────────────────────────────────────────┤
│                  MODEL (27.78B PARAMS, 51.75 GiB)                   │
├─────────────────┬───────────────────────────────────────────────────┤
│ BF16            │ 51.75 / 70.15 GiB                                 │
│ 27.78B PARAMS   │ ████████████████████████████████████▏░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                 KV CACHE (131072 TOKENS, 9.19 GiB)                  │
├─────────────────┬───────────────────────────────────────────────────┤
│ BF16            │ 9.19 / 70.15 GiB                                  │
│ 131072 TOKENS   │ ██████▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                         OVERHEAD (9.22 GiB)                         │
├─────────────────┬───────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 4.38 / 70.15 GiB                                  │
│                 │ ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ ENCODER         │ 1.61 / 70.15 GiB                                  │
│                 │ █▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ GRAPHS          │ 1.34 / 70.15 GiB                                  │
│                 │ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ RUNTIME         │ 1.88 / 70.15 GiB                                  │
│                 │ █▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴───────────────────────────────────────────────────┘
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
│ REQUIREMENTS    │ ████████████████████████████ │
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
│ REQUIREMENTS    │ ████████████████████████████████ │
├─────────────────┼──────────────────────────────────┤
│ BF16            │ 1.39 / 1.39 GiB                  │
│ 744.37M PARAMS  │ ████████████████████████████████ │
└─────────────────┴──────────────────────────────────┘
```

### GGUF

GGUF repositories list every file, and a quantization tag as in llama.cpp (`<model>:Q4_K_M`, case-insensitive) or `--file` (a tag, file, directory or any shard of a sharded file) estimates a single one. The `mmproj` projector of vision and audio models is included as llama.cpp does, picking the one with the same quantization, or else Q8_0, F16, BF16 or F32, together with the compute buffer of its encoders in the overhead.

```console
$ uvx hf-mem LiquidAI/LFM3-1.7B-GGUF
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                               INFERENCE MEMORY ESTIMATE FOR                               │
│                       https://hf.co/LiquidAI/LFM3-1.7B-GGUF @ main                        │
│                           w/ max-model-len=262144, batch-size=1                           │
├───────────────────────────────────────────────────────────────────────────────────────────┤
│                                           FILES                                           │
├─────────────────────────┬─────────────────────────────────────────────────────────────────┤
│                         │ 6.85 GiB (3.22 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-BF16.gguf     │ ███████████████████████████████████████████████████████████████ │
├─────────────────────────┼─────────────────────────────────────────────────────────────────┤
│                         │ 6.85 GiB (3.22 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-F16.gguf      │ ███████████████████████████████████████████████████████████████ │
├─────────────────────────┼─────────────────────────────────────────────────────────────────┤
│                         │ 4.81 GiB (1.18 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-NVFP4.gguf    │ ████████████████████████████████████████████▎░░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼─────────────────────────────────────────────────────────────────┤
│                         │ 4.64 GiB (1.02 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-Q4_K_M.gguf   │ ██████████████████████████████████████████▊░░░░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼─────────────────────────────────────────────────────────────────┤
│                         │ 4.79 GiB (1.16 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-Q5_K_M.gguf   │ ████████████████████████████████████████████▏░░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼─────────────────────────────────────────────────────────────────┤
│                         │ 4.95 GiB (1.32 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-Q6_K.gguf     │ █████████████████████████████████████████████▌░░░░░░░░░░░░░░░░░ │
├─────────────────────────┼─────────────────────────────────────────────────────────────────┤
│                         │ 5.34 GiB (1.71 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-Q8_0.gguf     │ █████████████████████████████████████████████████▏░░░░░░░░░░░░░ │
├─────────────────────────┼─────────────────────────────────────────────────────────────────┤
│                         │ 4.60 GiB (0.97 GiB WEIGHTS + 3.01 GiB KV + 0.62 GiB OVERHEAD)   │
│ LFM3-1.7B-QAD-Q4_0.gguf │ ██████████████████████████████████████████▎░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────┴─────────────────────────────────────────────────────────────────┘
```

```console
$ uvx hf-mem unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                    INFERENCE MEMORY ESTIMATE FOR                    │
│            https://hf.co/unsloth/Qwen3.8-27B-GGUF @ main            │
│                  FOR `Qwen3.8-27B-UD-Q4_K_M.gguf`                   │
│                w/ max-model-len=262144, batch-size=1                │
├─────────────────┬───────────────────────────────────────────────────┤
│ TOTAL MEMORY    │ 32.76 GiB (27.78B PARAMS + KV CACHE + OVERHEAD)   │
│ REQUIREMENTS    │ █████████████████████████████████████████████████ │
├─────────────────┼───────────────────────────────────────────────────┤
│ CPU MEMORY      │ 0.67 GiB (1.27B PARAMS OFFLOADED, NOT ON GPU)     │
│ HOST MEMORY     │ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                  MODEL (26.05B PARAMS, 14.66 GiB)                   │
├─────────────────┬───────────────────────────────────────────────────┤
│ Q5_K            │ 4.50 / 32.76 GiB                                  │
│ 7.03B PARAMS    │ ██████▊░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ IQ4_XS          │ 4.43 / 32.76 GiB                                  │
│ 8.96B PARAMS    │ ██████▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q4_K            │ 3.25 / 32.76 GiB                                  │
│ 6.20B PARAMS    │ ████▉░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q6_K            │ 1.69 / 32.76 GiB                                  │
│ 2.21B PARAMS    │ ██▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ IQ4_NL          │ 0.31 / 32.76 GiB                                  │
│ 587.20M PARAMS  │ ▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q3_K            │ 0.25 / 32.76 GiB                                  │
│ 623.90M PARAMS  │ ▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ IQ3_S           │ 0.14 / 32.76 GiB                                  │
│ 356.52M PARAMS  │ ▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q8_0            │ 0.08 / 32.76 GiB                                  │
│ 76.02M PARAMS   │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ F32             │ 0.01 / 32.76 GiB                                  │
│ 2.67M PARAMS    │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                  MMPROJ (460.73M PARAMS, 0.86 GiB)                  │
├─────────────────┬───────────────────────────────────────────────────┤
│ F16             │ 0.85 / 32.76 GiB                                  │
│ 457.67M PARAMS  │ █▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ F32             │ 0.01 / 32.76 GiB                                  │
│ 3.06M PARAMS    │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                 KV CACHE (262144 TOKENS, 16.15 GiB)                 │
├─────────────────┬───────────────────────────────────────────────────┤
│ F16             │ 16.15 / 32.76 GiB                                 │
│ 262144 TOKENS   │ ████████████████████████▏░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                         OVERHEAD (1.09 GiB)                         │
├─────────────────┬───────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 0.15 / 32.76 GiB                                  │
│                 │ ▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ ENCODER         │ 0.47 / 32.76 GiB                                  │
│                 │ ▊░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ RUNTIME         │ 0.47 / 32.76 GiB                                  │
│                 │ ▊░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴───────────────────────────────────────────────────┘
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
│ REQUIREMENTS    │ ███████████████████████████████████████ │
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
- Kimi K3.1 defaults to vLLM's `fp8_ds_mla` cache layout on Hopper. Use `--kv-cache-dtype fp8` for the per-tensor FP8 layout used by FlashInfer on Blackwell.
- AMD and NVIDIA Blackwell GPUs go through the same accounting in vLLM, but have not been validated.

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [ONNX - Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [vLLM - Memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
