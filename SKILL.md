---
name: hf-mem
description: CLI to estimate the required memory to load either Safetensors or GGUF model weights for inference from the Hugging Face Hub
license: mit
---

# hf-mem

## What it does

Estimates inference memory requirements for models on the Hugging Face Hub via Safetensors and/or GGUF metadata with HTTP Range requests.

Note that GGUF support is only available v0.5.0 onwards.

## Requirements

- `uv` package manager (for `uvx` command)
- `HF_TOKEN` environment variable (only for gated/private models), or `--hf-token` argument

## When to use

- User asks about model VRAM/memory needs or requirements
- User wants to check if a model fits in their GPU or on a given instance by spec
- User provides a Hugging Face model URL or model ID and asks about inference requirements
- If the user wants to estimate the memory for a GGUF repository, then try to use the `--gguf-file` with the path to a file in the Hub repository; preferably the first file when the model weights are sharded.

## Usage

```bash
uvx hf-mem --model-id <org/model-name>
```

Or add `--experimental` since `hf-mem` 0.4.3 to include KV cache estimations for LLMs and VLMs too; use `hf-mem` 0.5.0 onwards to also benefit from the `--experimental` flag for GGUF model weights too.

### Examples

- `uvx hf-mem --model-id black-forest-labs/FLUX.1-dev`
- `uvx hf-mem --model-id mistralai/Mistral-7B-v0.1 --experimental`
- `uvx hf-mem --model-id unsloth/Qwen3.5-397B-A17B-GGUF --experimental --gguf-file Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf`

## When it fails

- HTTP 401, if the model is gated/private, meaning you need to set `HF_TOKEN` with read access to it.
- HTTP 404, if the provided `--model-id` is not available on the Hugging Face Hub.
- RuntimeError, if none of `model.safetensors`, `model.safetensors.index.json`, or `model_index.json` is available; neither any `*.gguf` file
