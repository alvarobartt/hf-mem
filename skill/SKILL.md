---
name: hf-mem
description: Estimate HuggingFace model memory requirements using hf-mem CLI. Use when user asks about model VRAM/memory needs, wants to check if a model fits their GPU, needs to compare model sizes, or provides a HuggingFace model URL. Triggers on "how much memory", "will it fit on my GPU", "check memory for", "VRAM requirements", or any huggingface.co URL.
---

# HuggingFace Model Memory Estimator

Estimate inference memory requirements for any HuggingFace model without downloading the full weights. Uses HTTP Range requests to fetch Safetensors metadata efficiently.

## Prerequisites

### Check if uv is installed
```bash
uv --version
```

### Install uv (if not installed)
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### For gated models (optional)
Set HuggingFace token as environment variable:
```bash
export HF_TOKEN=hf_xxxxx  # Linux/macOS
setx HF_TOKEN "hf_xxxxx"  # Windows (permanent)
```

## CLI Usage

### Basic Command
```bash
uvx hf-mem --model-id <org/model-name>
```

### CLI Options
| Option | Description |
|--------|-------------|
| `--model-id` | HuggingFace model ID (e.g., `mistralai/Mistral-7B-v0.1`) |
| `--revision` | Git revision (branch, tag, commit) - defaults to `main` |
| `--help` | Show help message |

### Examples
```bash
# Text model
uvx hf-mem --model-id mistralai/Mistral-7B-v0.1

# Image generation model
uvx hf-mem --model-id black-forest-labs/FLUX.1-dev

# Specific revision
uvx hf-mem --model-id meta-llama/Llama-2-7b --revision main
```

## Workflow

1. **Parse model ID from user input**
   - HuggingFace URL: `https://huggingface.co/org/model` -> extract `org/model`
   - Direct model ID: use as provided
   - Model name only: search for common orgs (meta-llama, mistralai, Qwen, etc.)

2. **Run hf-mem CLI**
   ```bash
   PYTHONIOENCODING=utf-8 uvx hf-mem --model-id <model-id>
   ```

3. **Check Unsloth variant** (often has optimized/quantized versions)
   ```bash
   PYTHONIOENCODING=utf-8 uvx hf-mem --model-id unsloth/<model-name>
   ```

4. **Parse and present results** in structured format with GPU recommendations

## Output Format Template

Present results like this:

```markdown
**[Model Name]** ([Organization]):

| Component | Memory (BF16) | Parameters |
|-----------|---------------|------------|
| Total | XX.XX GB | X.XXB |
| Transformer | XX.XX GB | X.XXB |
| Text Encoder | XX.XX GB | X.XXB |
| VAE | X.XX GB | XXM |

**GPU Compatibility (user's GPU or common GPUs):**
- Full BF16: Requires XX GB VRAM
- FP8/INT8 (~50%): ~XX GB
- Q4 GGUF (~30%): ~XX GB - [fits/doesn't fit] on [GPU]

**Recommendation:** [Specific advice based on user's GPU if known]
```

## GPU VRAM Reference

| VRAM | Max BF16 | Max Q4 GGUF | Example GPUs |
|------|----------|-------------|--------------|
| 4GB | ~2B | ~6B | GTX 1650, RTX 3050 |
| 6GB | ~3B | ~10B | RTX 2060, RTX 3060 |
| 8GB | ~4B | ~13B | RTX 3070, RTX 4060 |
| 12GB | ~6B | ~20B | RTX 3080, RTX 4070 Ti |
| 16GB | ~8B | ~30B | RTX 4080, A4000 |
| 24GB | ~12B | ~45B | RTX 4090, A5000 |
| 48GB | ~24B | ~80B | A6000, dual GPU |
| 80GB | ~40B | ~130B | A100, H100 |

**Memory formulas:**
- BF16/FP16: params * 2 bytes
- FP8/INT8: params * 1 byte (~50% of BF16)
- Q4 GGUF: params * 0.5-0.6 bytes (~30% of BF16)

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Gated/private model | 1. Set `HF_TOKEN` env var, 2. Accept license at huggingface.co |
| 404 Not Found | Wrong model ID | Verify exact repo name on huggingface.co |
| No safetensors found | Model uses GGUF/pickle/bin | File size = memory needed; check repo files directly |
| Connection error | Network issue | Check internet connection |

## Limitations

- Only works with models containing Safetensors weights
- GGUF models: check file size directly (file size ≈ memory needed)
- Pickle/bin models: not supported, estimate from parameter count
- Does not account for KV cache, activation memory, or batch size

## References

- Repository: https://github.com/alvarobartt/hf-mem
- Safetensors metadata: https://huggingface.co/docs/safetensors/metadata_parsing
- uv package manager: https://github.com/astral-sh/uv
