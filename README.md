<img src="https://github.com/user-attachments/assets/509a8244-8a91-4051-b337-41b7b2fe0e2f" />

---

> [!WARNING]
> `hf-mem` is still experimental and therefore subject to major changes across releases, so please keep in mind that breaking changes may occur until v1.0.0.

`hf-mem` is a CLI to estimate inference memory requirements for Hugging Face models, written in Python. `hf-mem` is lightweight, only depends on `httpx`, as it pulls the [Safetensors](https://github.com/huggingface/safetensors) and / or [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests). It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements to run any model from the Hugging Face Hub, including [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers) and [Sentence Transformers](https://github.com/huggingface/sentence-transformers) models, or really any model as long as it contains any of [Safetensors](https://github.com/huggingface/safetensors) or [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) weights.

Read more information about `hf-mem` in [this short-form post](https://alvarobartt.com/hf-mem), but note it's not up-to-date as it was written in January 2026.

## Usage

### CLI (Recommended)

#### Transformers

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2
```

<img src="https://github.com/user-attachments/assets/545be630-4485-41ac-ba8d-2aedbbed8835" />

#### Diffusers

```bash
uvx hf-mem --model-id Qwen/Qwen-Image
```

<img src="https://github.com/user-attachments/assets/7a260369-26b2-48e1-a97b-f18cc72f7475" />

#### Sentence Transformers

```bash
uvx hf-mem --model-id google/embeddinggemma-300m
```

<img src="https://github.com/user-attachments/assets/aef1ddd8-96c2-4944-83e2-57171ff6ac7a" />

### Python

You can also run it programmatically with Python as:

```python
from hf_mem import run

result = run(model_id="MiniMaxAI/MiniMax-M2", experimental=True)
print(result)
# Result(model_id='MiniMaxAI/MiniMax-M2', revision='main', filename=None, memory=230121630720, kv_cache=24964497408, total_memory=255086128128, details=False)
```

If you're already inside an async application, use `arun(...)` instead:

```python
from hf_mem import arun

result = await arun(model_id="MiniMaxAI/MiniMax-M2", experimental=True)
print(result)
# Result(model_id='MiniMaxAI/MiniMax-M2', revision='main', filename=None, memory=230121630720, kv_cache=24964497408, total_memory=255086128128, details=False)
```

## Experimental

By enabling the `--experimental` flag, you can enable the KV Cache memory estimation for LLMs (`...ForCausalLM`) and VLMs (`...ForConditionalGeneration`), even including a custom `--max-model-len` (defaults to the `config.json` default), `--batch-size` (defaults to 1), and the `--kv-cache-dtype` (defaults to `auto` which means it uses the default data type set in `config.json` under `torch_dtype` or `dtype`, or rather from `quantization_config` when applicable).

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2 --experimental
```

<img src="https://github.com/user-attachments/assets/ec0ef39d-0323-4616-bba5-6a18ffee211c" />

## GGUF

If the repository contains GGUF model weights, those will be listed by default (only if there are no Safetensors weights, otherwise the GGUFs will be ignored) and the memory will be estimated for each one of those; whereas if a specific file is provided, then the memory estimation will be targeted for that given file instead.

```bash
uvx hf-mem --model-id TheBloke/deepseek-llm-7B-chat-GGUF --experimental
```

<img src="https://github.com/user-attachments/assets/b5514d35-f9c4-4a07-a719-a0185ff1dd9f" />

Or if you want to only get the estimation on a given file:

```bash
uvx hf-mem --model-id TheBloke/deepseek-llm-7B-chat-GGUF --gguf-file deepseek-llm-7b-chat.Q2_K.gguf --experimental
```

<img src="https://github.com/user-attachments/assets/e32ad635-05e5-4b33-b35b-3e689215dedd" />

## Skills

Optionally, you can add `hf-mem` as an agent skill, which allows the underlying coding agent to discover and use it when provided as a [`SKILL.md`](skills/hf-mem/SKILL.md), e.g., `.claude/skills/hf-mem/SKILL.md`.

More information can be found at [Anthropic Agent Skills and how to use them](https://github.com/anthropics/skills).

## Extensions

Optionally, you can also add `hf-mem` as an extension to the Hugging Face Hub CLI, so as to unify all the Hugging Face CLIs around the `hf ...` entrypoint. To add `hf-mem` as an extension, all you need to do is run `pip install huggingface_hub --upgrade` and then `hf extensions add alvarobartt/hf-mem`, which will install `hf-mem` to be used as `hf mem ...` along with the rest of Hugging Face extensions you have installed, that you can list via `hf extensions list`. More information can be found in [the Hugging Face Hub CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli-extensions#how-python-extensions-are-installed).

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [GGUF File format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF on the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
