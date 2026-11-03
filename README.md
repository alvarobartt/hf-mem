<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/hf-mem.png" />

---

> [!WARNING]
> `hf-mem` is still experimental and therefore subject to major changes across releases, so please keep in mind that breaking changes may occur until v1.0.0.

`hf-mem` is a CLI to estimate inference memory requirements for Hugging Face models, written in Python. `hf-mem` is lightweight, only depends on [`httpx2`](https://github.com/pydantic/httpx2), as it pulls the [Safetensors](https://github.com/huggingface/safetensors) and / or [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests). It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements to run any model from the Hugging Face Hub, including [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers) and [Sentence Transformers](https://github.com/huggingface/sentence-transformers) models, or really any model as long as it contains any of [Safetensors](https://github.com/huggingface/safetensors) or [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) weights. Repositories without a standard layout (e.g. `consolidated-*.safetensors` or ComfyUI single-file checkpoints) are estimated per `*.safetensors` file.

Read more information about `hf-mem` in [this short-form post](https://alvarobartt.com/hf-mem), but note it's not up-to-date as it was written in January 2026.

## Usage

### CLI (Recommended)

#### Transformers

```bash
uvx hf-mem --model-id zai-org/GLM-5.3-Flash
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/transformers.png" />

#### Diffusers

```bash
uvx hf-mem --model-id Qwen/Qwen-Image-2.1
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/diffusers.png" />

#### Sentence Transformers

```bash
uvx hf-mem --model-id google/embeddinggemma-2
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/sentence-transformers.png" />

### Python

You can also run it programmatically with Python as:

```python
from hf_mem import run

result = run(model_id="zai-org/GLM-5.3-Flash", experimental=True)
print(result)
# Result(model_id='zai-org/GLM-5.3-Flash', revision='main', filename=None, memory=328326771576, kv_cache=14763950080, total_memory=343090721656, details=False)
```

If you're already inside an async application, use `arun(...)` instead:

```python
from hf_mem import arun

result = await arun(model_id="zai-org/GLM-5.3-Flash", experimental=True)
print(result)
# Result(model_id='zai-org/GLM-5.3-Flash', revision='main', filename=None, memory=328326771576, kv_cache=14763950080, total_memory=343090721656, details=False)
```

## Experimental

By enabling the `--experimental` flag, you can enable the KV Cache memory estimation for generative models (LLMs, VLMs, omni and speech models), even including a custom `--max-model-len` (defaults to the `config.json` default), `--batch-size` (defaults to 1), and the `--kv-cache-dtype` (defaults to `auto` which, same as vLLM, uses the model data type set in `config.json` under `torch_dtype` or `dtype`, unless the `quantization_config` declares a quantized KV cache).

The KV cache estimation accounts for the attention layout of the model: Multi-head Latent Attention (MLA) caches the compressed latent, hybrid models only cache the full and sliding attention layers (linear attention, convolution or Mamba layers don't grow with the context length), and the text decoder is resolved from nested configurations such as `text_config` or `llm_config`.

Additionally, for Mixture of Experts (MoEs), the `--experimental` flag also adds a weights-only breakdown of the base model and all the experts combined.

```bash
uvx hf-mem --model-id zai-org/GLM-5.3-Flash --experimental
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/experimental.png" />

## GGUF

If the repository contains GGUF model weights, those will be listed by default (only if there are no Safetensors weights, otherwise the GGUFs will be ignored) and the memory will be estimated for each one of those, skipping `mmproj` and `imatrix` files; whereas if a specific file (or a directory containing a sharded file) is provided via `--gguf-file`, then the memory estimation will be targeted for that given file instead.

```bash
uvx hf-mem --model-id LiquidAI/LFM3-1.7B-GGUF --experimental
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/gguf.png" />

Or if you want to only get the estimation on a given file:

```bash
uvx hf-mem --model-id unsloth/Qwen3.8-27B-GGUF --gguf-file Qwen3.8-27B-UD-Q4_K_M.gguf --experimental
```

<img src="https://huggingface.co/datasets/alvarobartt/hf-mem/resolve/main/gguf-file.png" />

## Skills

Optionally, you can add `hf-mem` as an agent skill, which allows the underlying coding agent to discover and use it when provided as a [`SKILL.md`](skills/hf-mem/SKILL.md), e.g., `.claude/skills/hf-mem/SKILL.md`.

More information can be found at [Anthropic Agent Skills and how to use them](https://github.com/anthropics/skills).

## Extensions

Optionally, you can also add `hf-mem` as an extension to the Hugging Face Hub CLI under `hf` as `hf mem ...`. First you need to install `hf` as explained in [Hugging Face CLI - "Getting started"](https://huggingface.co/docs/huggingface_hub/en/guides/cli). To add `hf-mem` as an extension, all you need to do is run:

```bash
hf extensions add alvarobartt/hf-mem
```
More information can be found at [Hugging Face Hub CLI - "How Python extensions are installed"](https://huggingface.co/docs/huggingface_hub/en/guides/cli-extensions#how-python-extensions-are-installed).

Once installed, you can use `hf-mem` via `hf` as `hf mem ...` along with the rest of Hugging Face extensions you have installed (which you can list via `hf extensions list`).

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
