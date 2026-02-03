---
name: hf-mem
description: CLI to estimate the required VRAM to load Safetensors models for inference from the Hugging Face Hub (Transformers, Diffusers and Sentence Transformers)
license: mit
---

# hf-mem

## What it does

Estimates inference memory requirements for models on the Hugging Face Hub via Safetensors metadata with HTTP Range requests.

## Requirements

- `uv` package manager (for `uvx` command)
- `HF_TOKEN` environment variable (only for gated/private models)

## When to use

- User asks about model VRAM/memory needs
- User wants to check if a model fits in their GPU
- User provides a Hugging Face model URL or model ID

## Usage

```bash
uvx hf-mem --model-id <org/model-name>
```

Or add `--experimental` since `hf-mem` 0.4.3 to include KV cache estimations for LLMs and VLMs too.

### Examples

- `uvx hf-mem --model-id black-forest-labs/FLUX.1-dev`
- `uvx hf-mem --model-id mistralai/Mistral-7B-v0.1 --experimental`

## When it fails

- HTTP 401, if the model is gated/private, meaning you need to set `HF_TOKEN` with read access to it.
- HTTP 404, if the provided `--model-id` is not available on the Hugging Face Hub.
- RuntimeError, if none of `model.safetensors`, `model.safetensors.index.json`, or `model_index.json` is available.
