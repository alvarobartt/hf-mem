---
name: hf-mem
description: CLI to estimate the required memory to load either Safetensors or GGUF model weights for inference from the Hugging Face Hub
license: mit
---

# hf-mem

Estimates inference memory (model weights + optional KV cache) for models on the Hugging Face Hub using HTTP Range requests — no weights are downloaded.

## When to use

- User asks how much VRAM or memory a model needs to run
- User wants to know if a model fits on their GPU or a given instance
- User references a Hugging Face model ID or URL and asks about inference requirements

## Requirements

- `uv` installed (for `uvx`)
- `HF_TOKEN` env var or `--hf-token` flag (gated/private models only)

## Safetensors models

Auto-detected when the repo contains `model.safetensors`, `model.safetensors.index.json`, or `model_index.json`. Covers Transformers, Diffusers, and Sentence Transformers — no extra flags needed.

```bash
uvx hf-mem --model-id <org/model>
```

## GGUF models

Auto-detected when the repo contains only `.gguf` files. When both Safetensors and GGUF files coexist, pass `--gguf-file` to target a specific file. Any shard path works for sharded models.

```bash
uvx hf-mem --model-id <org/model> --gguf-file <path-in-repo>
```

## KV cache estimation (`--experimental`)

Adds KV cache memory on top of weights. Applies to LLMs (`...ForCausalLM`), VLMs (`...ForConditionalGeneration`), and GGUF models. Reads `max_model_len` from `config.json` by default; override with `--max-model-len`. KV cache dtype defaults to `auto` (reads `torch_dtype`/`dtype` from `config.json`, or the FP8 quantization format if applicable; for GGUF `auto` = `F16`).

```bash
uvx hf-mem --model-id <org/model> [--gguf-file <path>] \
  --experimental [--max-model-len N] [--batch-size N] \
  [--kv-cache-dtype auto|bfloat16|fp8|fp8_e4m3|fp8_e5m2]
```

## Examples

```bash
# Transformers
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2

# Diffusers
uvx hf-mem --model-id Qwen/Qwen-Image

# Sentence Transformers
uvx hf-mem --model-id google/embeddinggemma-300m

# LLM with KV cache
uvx hf-mem --model-id mistralai/Mistral-7B-v0.1 --experimental

# GGUF with KV cache (sharded)
uvx hf-mem --model-id unsloth/Qwen3.5-397B-A17B-GGUF \
  --gguf-file Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf \
  --experimental
```

## Errors

- **HTTP 401** — model is gated or private; provide `HF_TOKEN` or `--hf-token`
- **HTTP 404** — model ID not found on the Hub
- **RuntimeError** — no supported weight format found, or `--gguf-file` path doesn't match any file in the repo
