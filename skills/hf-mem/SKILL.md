---
name: hf-mem
description: CLI to estimate the required VRAM to load Safetensors models for inference from the Hugging Face Hub (Transformers, Diffusers and Sentence Transformers)
license: mit
---

# hf-mem

## What it does

Estimates inference memory requirements for Hugging Face models without downloading the full weights. Fetches Safetensors metadata via HTTP Range requests.

## Requirements

- `uv` package manager (for `uvx` command)
- `HF_TOKEN` environment variable (only for gated/private models)

## When to use

- User asks about model VRAM/memory needs
- User wants to check if a model fits their GPU
- User provides a HuggingFace model URL or model ID

## Usage

```bash
uvx hf-mem --model-id <org/model-name>
```

### Examples

```bash
# Text model
uvx hf-mem --model-id mistralai/Mistral-7B-v0.1

# Image generation model
uvx hf-mem --model-id black-forest-labs/FLUX.1-dev
```

## Common errors

| Error | Solution |
|-------|----------|
| 401 Unauthorized | Set `HF_TOKEN` env var or accept model license on huggingface.co |
| 404 Not Found | Verify exact model ID on huggingface.co |
| No safetensors found | Model uses GGUF/bin format - check file size directly |

## References

- Repository: https://github.com/alvarobartt/hf-mem
- Agent Skills: https://github.com/anthropics/skills
