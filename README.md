# hf-mem

`hf-mem` estimates the memory required to run any model from the Hugging Face Hub, or from a local path, without downloading the weights. It reads the [Safetensors](https://github.com/huggingface/safetensors), [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and [ONNX](https://onnx.ai) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests), and only depends on [`httpx2`](https://github.com/pydantic/httpx2).

The estimate covers the model weights, the KV cache for generative models, and the runtime overhead of the inference engine (activations, encoder, graphs and runtime), for [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), [Sentence Transformers](https://github.com/huggingface/sentence-transformers), GGUF and ONNX models (without the ONNX Runtime overhead).

## Usage

Run it with [`uv`](https://github.com/astral-sh/uv) as `uvx hf-mem <model>`, where `<model>` is a Hugging Face Hub model ID (optionally with `@revision`, and with `:<quantization>` for GGUF), an URL, or a local path.

For generative models, **GPU CAPACITY** is the physical memory recommendation, including budget headroom. **TOTAL MEMORY / ALLOCATIONS** is the estimated weights, workload cache and runtime allocation before that headroom. With only a model ID, the workload is one sequence at the configured model context, using an automatically selected capacity profile. Safetensors use vLLM by default, except Inkling 2 uses its calibrated SGLang reference. Add `--engine sglang` for other SGLang deployments; GGUF uses llama.cpp automatically. Neither `--gpu-type` nor `--gpu-memory` is required.

Serving engines can reserve additional free memory for cache, so their observed usage can exceed the recommendation without establishing a larger minimum. A successful fit means the stated workload fits the modeled budget; it does not predict the engine’s whole-device cache reservation. Set `--gpu-memory` for a specific deployment and compare CUDA-reported capacity in the same units. Models without a runtime-overhead estimate show allocations only.

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

For generative models (LLMs, VLMs, omni and speech models) the KV cache is estimated for the context length in `config.json` and a single sequence, which can be changed with `--max-model-len` and `--batch-size`. The KV cache dtype defaults to `auto`, which as in vLLM is the model dtype unless the `quantization_config` declares a quantized KV cache, and can be set with `--kv-cache-dtype` (e.g. `fp8`).

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
  "recommended_gpu_memory": 25456712543,
  "gpu_memory_utilization": 0.9,
  "weights": {"bytes": 16381470720, "params": 8190735360, "dtypes": {"BF16": {"bytes": 16381470720, "params": 8190735360}}},
  "offloaded": null,
  "components": null,
  "kv_cache": {"bytes": 4831838208, "dtype": "BF16", "max_model_len": 32768, "batch_size": 1, "layers": {"full": 36}},
  "overhead": {"bytes": 1697732360, "activations": 523003494, "encoder": 0, "graphs": 326472826, "runtime": 848256040, "max_num_seqs": 256, "reference_engine": "vllm"},
  "moe": null,
  "fit": null,
  "warnings": []
}
```

- `recommended_gpu_memory` is the recommended physical capacity in bytes, including the `gpu_memory_utilization` budget (0.90 by default). With `--gpu-memory NxSIZE` it uses the aggregate allocation on the requested topology; otherwise it is a single-GPU reference. It is `null` when runtime overhead is not modeled.
- `total` is the sum of `weights`, `kv_cache` and `overhead` in bytes, and the `overhead` parts are described in [How it works](#how-it-works); `kv_cache` and `overhead` are `null` for non-generative models.
- `file`, `offloaded` (weights in CPU memory, not in `total`), `components` (per Diffusers component, file or GGUF `mmproj`), `moe` (base model and experts breakdown) and `fit` (with `--gpu-memory`) are `null` when they do not apply, and `params` is `null` when the weights come from file sizes (ONNX files with the weights inline).
- `startup_reserve_per_gpu`, when present, is an initialization allowance included in the capacity recommendation and `fit.startup_required`, separately from steady-state `total`.
- `fit` has the `gpu_memory` and `usable` memory per GPU, the memory `required` on the given GPUs (with the overhead and the replicated KV cache per GPU), the GPUs `needed`, whether it `fits`, and the longest `max_model_len` that fits. For Safetensors, `fit.overhead_per_gpu` records the per-rank overhead applied to the requested topology; `total` and `overhead` retain the single-GPU reference breakdown.
- `format` is `safetensors`, `gguf` or `onnx`.
- GGUF and ONNX repositories with several files return a `files` list with the `file`, `recommended_gpu_memory`, `gpu_memory_utilization`, `total`, `weights`, `kv_cache` and `overhead` of each file instead.

### Python

```python
from hf_mem import run

result = run("Qwen/Qwen3-8B", max_model_len=32768)
print(result.recommended_gpu_memory)  # Physical GPU capacity, including budget headroom.
print(result.total, result.weights.bytes, result.kv_cache.bytes, result.overhead.bytes)
# 22911041288 16381470720 4831838208 1697732360
print(result.to_dict())
```

`run` takes the same options as the CLI (`revision`, `file`, `format`, `max_model_len`, `batch_size`, `kv_cache_dtype`, `hf_token`, `gpu_memory`, `gpu_memory_utilization`, `max_num_batched_tokens`, `max_num_seqs` and `full_sliding_window_cache`), and `await arun(...)` can be used within an async application. Both return an `Estimate`, or a `Listing` with one `Estimate` per file for GGUF and ONNX repositories.

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

Safetensors use vLLM as the default runtime reference, except Inkling 2 automatically uses SGLang. Select `--engine sglang` for other SGLang deployments. GGUF uses llama.cpp. Give `--gpu-type` and `--gpu-memory` when the deployment hardware is known: capacity alone cannot distinguish A100, H100 and Blackwell backend defaults. Named profiles are `nvidia-t4`, `nvidia-a10g`, `nvidia-l4`, `nvidia-l40s`, `nvidia-a100`, `nvidia-h100`, `nvidia-h200` and `nvidia-rtx-pro-6000`; H100 currently uses a capacity-based initialization reserve. Without hardware information, the estimator selects the smallest capacity class that can hold the workload using that class’s prefill and scheduler defaults. This is a capacity-based reference, rather than GPU autodetection; a named GPU refines backend and interconnect assumptions.

```bash
hf-mem meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 \
  --engine sglang --gpu-type nvidia-h200 --gpu-memory 2x141
```

For a language-model-only deployment of a multimodal checkpoint, add `--language-model-only`. It excludes identified encoder/projector tensors and encoder activations. The vLLM capacity recommendation separately includes an initialization allowance based on omitted encoder bytes: cold startup can exceed the decoder’s steady-state footprint. This is an allowance, not a claim that those tensors remain loaded. The pinned SGLang standalone mode supports only specific architectures; unsupported launches are rejected, as they are by the engine. Unknown encoder namespaces remain counted with a warning; this avoids guessing which decoder tensors an engine will skip. The older `--text-only` option remains accepted for a text request on a server that still loads the multimodal weights; it only omits encoder activations. The default includes a maximum admissible image/audio reserve. Paired measurements show that the same server can use substantially different memory for text and large images, so the workload is part of the estimate.

`--batch-size` counts concurrent cached sequences. `--max-num-seqs` describes scheduler capacity separately. Prefill, request-slot and graph-capture defaults depend on the engine, GPU and topology; graph workspaces stop growing proportionally once native capture limits are reached. `--max-num-batched-tokens` and `--max-num-seqs` override the modeled defaults when the server uses custom settings. Automatic whole-device cache reservations are different from the memory needed for the requested workload.

With `--gpu-memory NxSIZE`, fitting recomputes overhead and cache geometry for each tensor-parallel topology and context candidate. FFN and attention projections shard; context, graph pools and communication reserves remain local to each rank. Compare aggregate runtime allocations to `fit.required`; `fit.startup_required`, when present, adds the per-rank initialization allowance for fitting. The capacity recommendation places that combined requirement inside the utilization budget; `fit.overhead_per_gpu` records the applied rank overhead. The top-level component breakdown is the single-GPU reference.

Both `--engine` and `--gpu-type` are optional. Bare GPU-memory values and a `GiB` suffix mean binary GiB; an explicit `GB` suffix means decimal gigabytes. When a GPU type is provided, the capacity is checked before reading model metadata. Named A100 profiles accept the 40/80 variants and H100 accepts 80/94/96 (including GH200). The H200 reference capacity is 141, matching NVIDIA's board label; a real CUDA total may be lower (139.80 GiB in our pinned runs). Validation therefore accepts driver/ECC reservations: up to 2% below nominal for HBM/RTX boards and 10% for the tested ECC-enabled GDDR6 families. An H200 paired with `8x80` is rejected; `8x141` and a measured `8x139.8GiB` are accepted. Supplied capacity is never silently replaced, and `--gpu-memory-utilization` separately limits the usable fraction. Old unprefixed GPU names remain input aliases; output uses `nvidia-*` names. This catalog check does not query the deployment or validate a MIG/vGPU partition.

The CLI accepts equivalent engine argument names for the controls it estimates:

| Meaning | vLLM / hf-mem | SGLang | llama.cpp |
| --- | --- | --- | --- |
| Model ID/path | positional model or `--model` | `--model-path` | `-m`, `--hf-repo`, `-hf` |
| Context per sequence | `--max-model-len` | `--context-length`, `--context-len` | `--ctx-size`, `-c` specify the **total** pool, divided by `--parallel` |
| Concurrent sequences in the KV estimate | `--batch-size` (hf-mem) | same hf-mem control | `--parallel`, `-np` |
| Scheduler capacity | `--max-num-seqs` | `--max-running-requests` | no direct alias |
| Prefill token budget | `--max-num-batched-tokens` | `--chunked-prefill-size` | no direct alias |
| Full-context storage for sliding-window layers | `--disable-hybrid-kv-cache-manager` | `--disable-hybrid-swa-memory` | `--swa-full` |
| Skip multimodal encoder weights/activations | `--language-model-only` | `--language-model-only` (supported architectures only) | `--no-mmproj`, `--no-mmproj-auto` |
| KV dtype | `--kv-cache-dtype` | `--kv-cache-dtype` | matching `--cache-type-k` / `--cache-type-v` (`-ctk` / `-ctv`) |

The old `--full-sliding-window-cache` spelling remains an alias. Its new engine spellings change the cache estimate and the context fit; `--swa-full` also works on GGUF. These controls model sliding-window storage, not every recurrent-state allocator change caused by disabling a hybrid manager. `--ctx-size` must be positive and divisible by the number of parallel sequences, and cannot be combined with a per-sequence context option. llama.cpp's context and projector/cache-type options select GGUF. `--hf-file`/`-hff` alias `--file`; model selection uses hf-mem's model-ID/revision/file syntax, without llama.cpp's `:QUANT` shorthand or automatic Q4 selection.

Similar names are not always equivalent. SGLang's `--mem-fraction-static` budgets weights plus the static KV pool, unlike the overall `--gpu-memory-utilization` fit budget; it is not an alias. SGLang's `--language-only` participates in encoder disaggregation, so use the standalone `--language-model-only` mode here. vLLM's `--disable-sliding-window` changes attention/context behavior, so it is not a full-cache alias. llama.cpp's native `--batch-size` and `--ubatch-size` count tokens, while hf-mem's `--batch-size` counts sequences; there is no `--ubatch-size` alias for the fixed GGUF compute reserve. Separate K/V cache dtypes and native automatic values such as `--chunked-prefill-size -1` are not currently modeled and are rejected rather than treated as equivalent settings.

For example, these two Safetensors commands set the same context, scheduler capacity and prefill budget using the calibrated Inkling SGLang reference:

```bash
uv run hf-mem thinkingmachines/Inkling-2-Small --gpu-type nvidia-h200 --gpu-memory 8x141 --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 8192
uv run hf-mem --model-path thinkingmachines/Inkling-2-Small --engine sglang --gpu-type nvidia-h200 --gpu-memory 8x141 --context-length 32768 --max-running-requests 16 --chunked-prefill-size 8192
```

Argument mappings follow the [vLLM reference](https://docs.vllm.ai/en/stable/configuration/engine_args/), [SGLang reference](https://docs.sglang.io/docs/advanced_features/server_arguments), and [llama.cpp parser](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp). Board labels follow [NVIDIA's H200 specifications](https://www.nvidia.com/en-gb/data-center/h200/) and [GPU variants](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-gpus.html). The examples use the catalog label; use the actual CUDA-reported GiB capacity for a deployment-specific fit budget.

Configured tied embedding/output tables are counted once when checkpoint headers prove matching dense geometry. The ordinary Qwen3.5 reference skips serialized `mtp.*` draft weights, as both engines do; speculative decoding needs additional memory. SGLang Gemma3 uses full-context storage for local-attention layers. Its GDN models use dedicated recurrent-state slots instead of vLLM's shared attention/state pages: an explicit scheduler capacity models the dedicated pool with radix caching disabled. Automatic recurrent/radix cache reservations can be larger. `--full-sliding-window-cache` remains available for other backends that store full-context local layers.

The calibration compares pinned vLLM and SGLang images, immutable model revisions, exact token counts, native parameter storage, physical cache pools, initialization/graph stages and sampled aggregate use across every TP rank. Failed or incomplete requests remain diagnostics. Retrospective calibration and fresh frozen-forecast validation are reported separately in [calibration methods and results](scripts/calibration/README.md); raw evidence from earlier published runs remains in the [validation bucket](https://huggingface.co/buckets/alvarobartt/hf-mem-validation).

For GGUF, the CUDA runtime reserve follows capacity classes: 0.13 GiB through 16 GiB, 0.24 GiB through 24 GiB, 0.47 GiB in the middle, and 0.60 GiB from 120 GiB. Without a hardware hint, the estimator selects the smallest fitting class. Decoder compute scales with context, hidden/FFN dimensions and llama.cpp’s reference microbatch size instead of using a fixed 0.15 GiB. These are calibrated defaults, not GPU autodetection.

The final independent 14-configuration check reduces median absolute total-memory error from 7.37% to 1.95% and residual-overhead error from 40.15% to 5.55%. The largest final total error is 6.35%. A broader 150-configuration retrospective replay covers both engines, seven GPU families and actual TP1/2/4/8; its 0.81% median total error includes calibration data. See the [versioned results and limitations](scripts/calibration/reports/crossengine-20261112/summary.json) for per-engine metrics and exclusions. These are minimum-workload estimates against pinned engine versions, distinct from automatic whole-device cache reservations.

The subsequent [capacity and Inkling validation](scripts/calibration/README.md#capacity-recommendations-and-defaults-2026-11-13) separates recommended physical capacity from runtime allocations, audits native packed weights and convolution pools, and checks default and explicit workloads against all three engines. Fresh Inkling TP=EP4/8 allocation errors are −0.70% and −0.20%. All controlled recommendations cover sampled startup/runtime peaks; a constrained-memory llama.cpp run also completes at the recommended budget while the previous budget fails. The versioned report retains failed and unsupported trials as well as positive results.

The earlier held-out L4 gap motivated the expanded cross-engine calibration. New checks also cover native scheduler defaults, PCIe and NVLink tensor parallelism, expanded attention projections, MoE and recurrent-state models. The calibration report retains earlier misses and distinguishes minimum logical cache from native pool padding and reservation.

## Limitations

- The overhead profiles use pinned vLLM, SGLang and llama.cpp reference settings. Backend selection, custom graph captures, speculative decoding, parallelism beyond TP, allocator versions and non-default optimization settings can still change memory use. Both vLLM and SGLang can reserve most available VRAM for cache; that reservation is not a single-request requirement.
- ONNX models have no overhead estimate, and GGUF and ONNX listings estimate each file from its size, so a single file (`:<quantization>` or `--file`) gives the most accurate numbers.
- vLLM profiles the vision encoder of some models (e.g. Qwen2.5-VL) with a video when it has more tokens than an image, which can reserve a few GiB more (`hf-mem` warns about it).
- Vision-model overhead includes the largest-image encoder reserve unless `--language-model-only` (or legacy `--text-only`) is supplied. Hybrid cache alignment and state checkpoints can also depend on the backend: the held-out H200 FlashInfer GDN pools are 5.6–9.5% larger than the logical cache estimates, including reserved blocks. These cache differences are separate from overhead.
- PyTorch pickle weights (`.bin`, `.pt`, `.ckpt` or `.nemo`) are not supported, and encoder-decoder models (e.g. Whisper) have no KV cache estimate.
- Kimi K3.1 defaults to vLLM's `fp8_ds_mla` cache layout on Hopper. Use `--kv-cache-dtype fp8` for the per-tensor FP8 layout used by FlashInfer on Blackwell.
- AMD has not been validated. Blackwell validation covers RTX PRO 6000; it does not establish accuracy on every Blackwell instance, MIG partition or multi-node topology.

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [ONNX - Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [vLLM - Memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
