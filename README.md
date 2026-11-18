# hf-mem

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ONNX](https://onnx.ai) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers), GGUF and ONNX models (without the ONNX Runtime overhead).

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hugging Face Hub model ID (optionally with `@revision`, and with `:<quantization>` for GGUF), an URL, or a local path.

For generative models, **GPU CAPACITY** is the recommended physical capacity with budget headroom and an initialization allowance. **TOTAL MEMORY / ALLOCATIONS** shows weights, workload cache and modeled overhead. A model ID alone estimates one sequence at the configured context; neither `--gpu-type` nor `--gpu-memory` is required. Serving engines can reserve spare VRAM for extra caches, so their observed usage can exceed the workload estimate.

### Transformers

```console
$ uvx hf-mem zai-org/GLM-5.3-Flash
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                     INFERENCE MEMORY ESTIMATE FOR                     │
│              https://hf.co/zai-org/GLM-5.3-Flash @ main               │
│                w/ max-model-len=1048576, batch-size=1                 │
│                 VLLM REFERENCE, 90% ALLOCATION BUDGET                 │
├─────────────────┬─────────────────────────────────────────────────────┤
│ GPU CAPACITY    │ 361.67 GiB (388.34 GB) ON ONE GPU                   │
│ RECOMMENDED     │ ███████████████████████████████████████████████████ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ TOTAL MEMORY    │ 325.50 GiB (321.32B PARAMS + KV CACHE + OVERHEAD)   │
│ ALLOCATIONS     │ ███████████████████████████████████████████████████ │
├─────────────────┴─────────────────────────────────────────────────────┤
│                    MODEL (9.67B PARAMS, 15.46 GiB)                    │
├─────────────────┬─────────────────────────────────────────────────────┤
│ BF16            │ 12.90 / 325.50 GiB                                  │
│ 6.93B PARAMS    │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ F8_E4M3         │ 2.55 / 325.50 GiB                                   │
│ 2.74B PARAMS    │ ▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ F32             │ 0.00 / 325.50 GiB                                   │
│ 295.52K PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────────┤
│               EXPERTS (288 x 1.08B PARAMS, 290.32 GiB)                │
├─────────────────┬─────────────────────────────────────────────────────┤
│ F8_E4M3         │ 290.25 / 325.50 GiB                                 │
│ 311.65B PARAMS  │ █████████████████████████████████████████████▌░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ F32             │ 0.07 / 325.50 GiB                                   │
│ 0 PARAMS        │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────────┤
│                 KV CACHE (1048576 TOKENS, 16.81 GiB)                  │
├─────────────────┬─────────────────────────────────────────────────────┤
│ BF16            │ 16.81 / 325.50 GiB                                  │
│ 1048576 TOKENS  │ ██▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴─────────────────────────────────────────────────────┤
│                          OVERHEAD (2.91 GiB)                          │
├─────────────────┬─────────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 1.46 / 325.50 GiB                                   │
│                 │ ▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ ENCODER         │ 0.05 / 325.50 GiB                                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ GRAPHS          │ 0.49 / 325.50 GiB                                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼─────────────────────────────────────────────────────┤
│ RUNTIME         │ 0.92 / 325.50 GiB                                   │
│                 │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────┴─────────────────────────────────────────────────────┘
```

For generative models (LLMs, VLMs, omni and speech models) the KV cache is estimated for the context length in `config.json` and a single sequence, which can be changed with `--max-model-len` and `--batch-size`. The KV cache dtype defaults to `auto`, using model precision unless the `quantization_config` declares a quantized KV cache, and can be set with `--kv-cache-dtype` (e.g. `fp8`).

```console
$ uvx hf-mem Qwen/Qwen3.8-27B --max-model-len 32768 --batch-size 4
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                    INFERENCE MEMORY ESTIMATE FOR                    │
│                https://hf.co/Qwen/Qwen3.8-27B @ main                │
│                w/ max-model-len=32768, batch-size=4                 │
│                VLLM REFERENCE, 90% ALLOCATION BUDGET                │
├─────────────────┬───────────────────────────────────────────────────┤
│ GPU CAPACITY    │ 73.71 GiB (79.14 GB) ON ONE GPU                   │
│ RECOMMENDED     │ █████████████████████████████████████████████████ │
├─────────────────┼───────────────────────────────────────────────────┤
│ TOTAL MEMORY    │ 66.34 GiB (27.36B PARAMS + KV CACHE + OVERHEAD)   │
│ ALLOCATIONS     │ █████████████████████████████████████████████████ │
├─────────────────┴───────────────────────────────────────────────────┤
│                  MODEL (27.36B PARAMS, 50.96 GiB)                   │
├─────────────────┬───────────────────────────────────────────────────┤
│ BF16            │ 50.96 / 66.34 GiB                                 │
│ 27.36B PARAMS   │ █████████████████████████████████████▋░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                 KV CACHE (131072 TOKENS, 9.19 GiB)                  │
├─────────────────┬───────────────────────────────────────────────────┤
│ BF16            │ 9.19 / 66.34 GiB                                  │
│ 131072 TOKENS   │ ██████▊░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                         OVERHEAD (6.19 GiB)                         │
├─────────────────┬───────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 2.65 / 66.34 GiB                                  │
│                 │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ ENCODER         │ 2.42 / 66.34 GiB                                  │
│                 │ █▊░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ GRAPHS          │ 0.39 / 66.34 GiB                                  │
│                 │ ▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ RUNTIME         │ 0.74 / 66.34 GiB                                  │
│                 │ ▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
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
│ ALLOCATIONS     │ ████████████████████████████ │
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
│ ALLOCATIONS     │ ████████████████████████████████ │
├─────────────────┴──────────────────────────────────┤
│          MODEL (744.37M PARAMS, 1.39 GiB)          │
├─────────────────┬──────────────────────────────────┤
│ BF16            │ 1.39 / 1.39 GiB                  │
│ 744.37M PARAMS  │ ████████████████████████████████ │
└─────────────────┴──────────────────────────────────┘
```

Text encoders, rerankers and Sentence Transformers include a warmup estimate over `--max-num-batched-tokens` tokens, using `--max-model-len` as the sequence length (otherwise the Sentence Transformers or model configuration). The batch can contain several sequences. No KV cache or decoding graphs are added.

```bash
uvx hf-mem BAAI/bge-base-en-v1.5 --max-num-batched-tokens 2048 --max-model-len 512 --gpu-memory 24
```

By default, weights keep their stored dtypes, including mixed precision and quantization buffers. Activations use the configured dtype or the dominant floating-point weight dtype. For unquantized Safetensors text encoders, `--dtype float16`, `--dtype bfloat16` or `--dtype float32` explicitly casts floating-point parameters and activations; integer tensors remain unchanged. Other model types reject this option. Warmup assumes memory-efficient attention in half precision, and eager attention for MPNet, DeBERTa, configurations selecting it, and F32 activations, as memory-efficient attention kernels need half precision. Fit checks assume one replica per GPU.

To estimate a Text Embeddings Inference deployment, use its configured `--dtype` (`float16` or `float32`) and the actual number of warmup tokens, which may be smaller than its batch limit:

```bash
uvx hf-mem BAAI/bge-base-en-v1.5 --dtype float16 --max-num-batched-tokens 2048 --max-model-len 512 --gpu-memory 24
```

Changing `--dtype` changes weight and activation precision, and `float32` runs eager attention; it does not change the token budget or sequence length. Loading allows for a temporary tensor in its stored dtype alongside the converted weights, so F32 checkpoints loaded in F16 can peak while loading rather than at warmup (`loading` in the overhead). Engine padding and workspaces can still change observed memory use.

FP32 decoder checkpoints use a 16-bit CUDA serving reference for floating-point weights and automatic KV cache precision. The estimate labels that reference F16; an engine may choose BF16 with the same memory width. Encoder defaults preserve their stored precision, and quantized checkpoints keep their declared representation.

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
┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬ hf-mem v1.0.0 ┐
├┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┤
│                    INFERENCE MEMORY ESTIMATE FOR                    │
│            https://hf.co/unsloth/Qwen3.8-27B-GGUF @ main            │
│                  FOR `Qwen3.8-27B-UD-Q4_K_M.gguf`                   │
│                w/ max-model-len=262144, batch-size=1                │
│             LLAMA.CPP REFERENCE, 90% ALLOCATION BUDGET              │
├─────────────────┬───────────────────────────────────────────────────┤
│ GPU CAPACITY    │ 36.74 GiB (39.45 GB) ON ONE GPU                   │
│ RECOMMENDED     │ █████████████████████████████████████████████████ │
├─────────────────┼───────────────────────────────────────────────────┤
│ TOTAL MEMORY    │ 33.07 GiB (27.78B PARAMS + KV CACHE + OVERHEAD)   │
│ ALLOCATIONS     │ █████████████████████████████████████████████████ │
├─────────────────┼───────────────────────────────────────────────────┤
│ CPU MEMORY      │ 0.67 GiB (1.27B PARAMS OFFLOADED, NOT ON GPU)     │
│ HOST MEMORY     │ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                  MODEL (26.05B PARAMS, 14.75 GiB)                   │
├─────────────────┬───────────────────────────────────────────────────┤
│ IQ4_XS          │ 5.30 / 33.07 GiB                                  │
│ 10.72B PARAMS   │ ███████▉░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q5_K            │ 4.87 / 33.07 GiB                                  │
│ 7.61B PARAMS    │ ███████▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q4_K            │ 2.26 / 33.07 GiB                                  │
│ 4.31B PARAMS    │ ███▍░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q6_K            │ 1.68 / 33.07 GiB                                  │
│ 2.20B PARAMS    │ ██▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ IQ4_NL          │ 0.40 / 33.07 GiB                                  │
│ 765.46M PARAMS  │ ▋░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q8_0            │ 0.08 / 33.07 GiB                                  │
│ 81.26M PARAMS   │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ Q3_K            │ 0.07 / 33.07 GiB                                  │
│ 178.26M PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ IQ3_S           │ 0.07 / 33.07 GiB                                  │
│ 178.26M PARAMS  │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ F32             │ 0.01 / 33.07 GiB                                  │
│ 2.67M PARAMS    │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                  MMPROJ (460.73M PARAMS, 0.86 GiB)                  │
├─────────────────┬───────────────────────────────────────────────────┤
│ F16             │ 0.85 / 33.07 GiB                                  │
│ 457.67M PARAMS  │ █▎░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ F32             │ 0.01 / 33.07 GiB                                  │
│ 3.06M PARAMS    │ ▏░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                 KV CACHE (262144 TOKENS, 16.15 GiB)                 │
├─────────────────┬───────────────────────────────────────────────────┤
│ F16             │ 16.15 / 33.07 GiB                                 │
│ 262144 TOKENS   │ ███████████████████████▉░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┴───────────────────────────────────────────────────┤
│                         OVERHEAD (1.31 GiB)                         │
├─────────────────┬───────────────────────────────────────────────────┤
│ ACTIVATIONS     │ 0.36 / 33.07 GiB                                  │
│                 │ ▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ ENCODER         │ 0.47 / 33.07 GiB                                  │
│                 │ ▊░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├─────────────────┼───────────────────────────────────────────────────┤
│ RUNTIME         │ 0.47 / 33.07 GiB                                  │
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
│ ALLOCATIONS     │ ███████████████████████████████████████ │
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

Use `--gpu-memory` to check a deployment. Bare sizes and `GiB` are binary; `GB` is decimal. For example, `8x141` uses 141 GiB per GPU, while `8x141GB` means 141 billion bytes per GPU. The default allocation budget is 90%, with the initialization allowance included in the recommendation. Decoder initialization uses at least 1 GiB of workspace per GPU, sharing the modeled activation reserve. This is an allowance for temporary workspaces rather than a claim that an engine always allocates that amount.

```bash
uvx hf-mem thinkingmachines/Inkling-2-Small --engine sglang --gpu-type nvidia-h200 --gpu-memory 8x141 --context-length 1048576
```

Optional GPU profiles are `nvidia-t4`, `nvidia-a10g`, `nvidia-l4`, `nvidia-l40s`, `nvidia-a100`, `nvidia-h100`, `nvidia-h200` and `nvidia-rtx-pro-6000`. Unprefixed names remain aliases. A named profile checks capacity against board variants, allowing driver/ECC reservations: H200 accepts its 141 label or a measured `139.8GiB`, and rejects `80`. It does not query hardware or identify partitions. Without a profile, capacity selects a coarse workspace reference.

`--engine sglang` selects SGLang scheduling/workspace defaults; Safetensors otherwise use the vLLM reference, except Inkling 2 uses its SGLang reference. GGUF uses llama.cpp. These profiles share model geometry and approximate runtime allowances rather than reproducing an engine's allocator or every optimization.

For encoder-free deployment, `--language-model-only` removes identified encoder/projector tensors and encoder activations. Unknown tensor namespaces remain counted. Engine support for this mode varies by architecture. The legacy `--text-only` option keeps the weights and omits encoder activations only.

| Meaning | Options |
| --- | --- |
| Model | positional ID, `--model`, `--model-path`, `--hf-repo` |
| Context per sequence | `--max-model-len`, `--context-length`, `--context-len` |
| Cached sequences | `--batch-size`, `--parallel`, `-np` |
| Scheduler capacity | `--max-num-seqs`, `--max-running-requests` |
| Prefill tokens | `--max-num-batched-tokens`, `--chunked-prefill-size` |
| Full sliding-window storage | `--disable-hybrid-kv-cache-manager`, `--disable-hybrid-swa-memory`, `--swa-full`, `--full-sliding-window-cache` |

llama.cpp's `--ctx-size`/`-c` is the total context pool, divided by `--parallel`, and selects GGUF. `--no-mmproj`/`--no-mmproj-auto` omit its projector; `--cache-type-k`/`--cache-type-v` require matching types. Native llama.cpp `--batch-size` counts tokens, while hf-mem's counts sequences. SGLang's `--mem-fraction-static` has different semantics from the overall fit budget and is not an alias. Full-window aliases model storage, not every behavioral effect of disabling an engine's cache manager.

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
  "total": 22911041288,
  "recommended_gpu_memory": 26068644020,
  "gpu_memory_utilization": 0.9,
  "startup_reserve_per_gpu": 550738330,
  "weights": {"bytes": 16381470720, "params": 8190735360, "dtypes": {"BF16": {"bytes": 16381470720, "params": 8190735360}}},
  "offloaded": null,
  "components": null,
  "kv_cache": {"bytes": 4831838208, "dtype": "BF16", "max_model_len": 32768, "batch_size": 1, "layers": {"full": 36}},
  "overhead": {"bytes": 1697732360, "activations": 523003494, "encoder": 0, "graphs": 326472826, "runtime": 848256040, "buffers": 0, "loading": 0, "max_num_seqs": 256, "reference_engine": "vllm"},
  "moe": null,
  "fit": null,
  "warnings": []
}
```

- `recommended_gpu_memory` is the physical GPU capacity recommendation in bytes, including initialization and the allocation budget. With `--gpu-memory NxSIZE`, it is the aggregate for that topology.
- `startup_reserve_per_gpu`, when present, is initialization storage beyond the modeled runtime allocations. `fit.startup_required` includes it in the fit decision.
- `total` is the sum of `weights`, `kv_cache` and `overhead` in bytes, and the `overhead` parts are described in [How it works](#how-it-works); `kv_cache` is `null` for non-generative models; text encoders include warmup `overhead`, while other non-generative models have no overhead estimate.
- `file`, `offloaded` (weights in CPU memory, not in `total`), `components` (per Diffusers component, file or GGUF `mmproj`), `moe` (base model and experts breakdown) and `fit` (with `--gpu-memory`) are `null` when they do not apply, and `params` is `null` when the weights come from file sizes (ONNX files with the weights inline).
- For multiple GPUs, `fit.required` includes topology replication and per-GPU overhead; `total` remains the component reference sum.
- `fit` has the `gpu_memory` and `usable` memory per GPU, the memory `required` on the given GPUs (with the overhead and the replicated KV cache per GPU), the GPUs `needed`, whether it `fits`, and the longest `max_model_len` that fits.
- `format` is `safetensors`, `gguf` or `onnx`.
- GGUF and ONNX repositories with several files return a `files` list with the `file`, `total`, `weights`, `kv_cache` and `overhead` of each file instead.

### Python

```python
from hf_mem import run

result = run("Qwen/Qwen3-8B", max_model_len=32768)
print(result.recommended_gpu_memory)
print(result.total, result.weights.bytes, result.kv_cache.bytes, result.overhead.bytes)
print(result.to_dict())
```

`run` takes the same options as the CLI (`revision`, `file`, `format`, `max_model_len`, `batch_size`, `kv_cache_dtype`, `hf_token`, `gpu_memory`, `gpu_memory_utilization`, `max_num_batched_tokens`, `max_num_seqs`, `dtype`, `engine`, `gpu_type`, `language_model_only` and `full_sliding_window_cache`), and `await arun(...)` can be used within an async application. Both return an `Estimate`, or a `Listing` with one `Estimate` per file for GGUF and ONNX repositories.

### Agents

Coding agents can use `uvx hf-mem <model> -o json` to answer questions such as how much memory a model needs or whether it fits on a given GPU (adding `--gpu-memory`), as described in the [`hf-mem` skill](skills/hf-mem/SKILL.md).

### Extensions

`hf-mem` can also be added as an extension to the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) as `hf mem ...`:

```bash
hf extensions add alvarobartt/hf-mem
```

## How it works

`hf-mem` only reads the metadata of the weight files (the Safetensors headers, the GGUF metadata or the ONNX graph) with HTTP Range requests, and adds up the memory an inference engine needs on the GPU:

- **Weights**: tensor storage at the selected serving precision, per dtype and component. Weights that the engine keeps in CPU memory are reported apart, as vLLM does with the DeepSeek-V4.1 Engram tables and llama.cpp with the token embeddings.
- **KV cache**: what the engine allocates for `--max-model-len` tokens and `--batch-size` sequences, following the model's attention: full attention caches every token, sliding window layers only the window, latent and compressed attention (e.g. DeepSeek, GLM) a smaller record per token, and linear attention or Mamba layers a fixed state per sequence.
- **Overhead**: the memory used on top of that, i.e. the activations of a forward pass, the vision or audio encoder for the largest input, the CUDA graphs, and the runtime (the CUDA context and library workspaces).

The reference estimates use shared weight/cache geometry and workspace allowances, with optional engine and hardware defaults. Fitting recomputes sharded and replicated storage at each GPU count and context candidate. Runtime overhead and initialization can vary with kernels, graph capture and other optimizations.

## Limitations

- The fit check models memory for the stated workload; engine compatibility and automatic spare-cache reservations are separate. Different backends and non-default optimizations can change the footprint.
- ONNX models have no overhead estimate, and GGUF and ONNX listings estimate each file from its size, so a single file (`:<quantization>` or `--file`) gives the most accurate numbers.
- vLLM profiles the vision encoder of some models (e.g. Qwen2.5-VL) with a video when it has more tokens than an image, which can reserve a few GiB more (`hf-mem` warns about it).
- PyTorch pickle weights (`.bin`, `.pt`, `.ckpt` or `.nemo`) are not supported, and encoder-decoder models (e.g. Whisper) have no KV cache estimate.
- Kimi K3.1 defaults to vLLM's `fp8_ds_mla` cache layout on Hopper. Use `--kv-cache-dtype fp8` for the per-tensor FP8 layout used by FlashInfer on Blackwell.
- A capacity-only profile cannot identify GPU architecture or backend selection. AMD and partitioned GPU configurations have no dedicated profile.

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [ONNX - Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [vLLM - Memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
