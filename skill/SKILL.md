---
name: hf-mem
description: Estimate HuggingFace model memory requirements. Use when user asks about model VRAM/memory needs, wants to check if a model fits their GPU, or needs to compare model sizes. Triggers on questions like "how much memory does X need", "will X fit on my GPU", "check memory for model X".
---

# HuggingFace Model Memory Estimator

Estimate inference memory requirements for HuggingFace models using hf-mem CLI.

## Quick Start

```bash
uvx hf-mem --model-id <org/model-name>
```

## Workflow

1. **Parse model ID** from user input:
   - URL format: `https://huggingface.co/org/model` -> `org/model`
   - Direct ID: use as-is

2. **Run memory estimate**:
   ```bash
   PYTHONIOENCODING=utf-8 uvx hf-mem --model-id <model-id>
   ```

3. **Check Unsloth version** (often has optimized variants):
   ```bash
   PYTHONIOENCODING=utf-8 uvx hf-mem --model-id unsloth/<model-name>
   ```

4. **Present results** in table format with GPU compatibility assessment.

## Output Format

```
**Model-Name** (Organization):

| Component | Memory (BF16) | Parameters |
|-----------|---------------|------------|
| Total     | X.XX GB       | X.XXB      |

**GPU Compatibility:**
- 6GB (RTX 2060): [assessment]
- 8GB (RTX 3070): [assessment]
- 12GB (RTX 3080): [assessment]
```

## GPU VRAM Guidelines

| VRAM | Q4 GGUF Capacity | Example GPUs |
|------|------------------|--------------|
| 6GB  | ~10B params      | RTX 2060, 3060 |
| 8GB  | ~13B params      | RTX 3070, 4060 |
| 12GB | ~20B params      | RTX 3080, 4070 |
| 16GB | ~30B params      | RTX 4080 |
| 24GB | ~45B params      | RTX 4090, A10 |

Q4 GGUF is approximately 30% of BF16 memory.

## Error Handling

| Error | Meaning | Solution |
|-------|---------|----------|
| 401 Unauthorized | Gated model | Set HF_TOKEN or accept license on HuggingFace |
| 404 Not Found | Wrong model ID | Verify exact repo name on huggingface.co |
| No safetensors | Different format | Check file sizes directly (GGUF = file size) |

## Environment

- Requires `uv` package manager
- HF_TOKEN env variable for gated models
- Only works with safetensors models (not GGUF)
