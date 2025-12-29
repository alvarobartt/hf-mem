# `hf-mem`

> [!WARNING]
> `hf-mem` works, but is still experimental. Ideally there should be a minor release anytime soon making things a bit more stable (and aesthetic).

`hf-mem` is a CLI to estimate inference memory requirements for Hugging Face models, written in Python. `hf-mem` is lightweight, only depends on `httpx`. It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements to run any model from the Hugging Face Hub, including Transformers, Diffusers and Sentence Transformers models, as well as any model that contains [Safetensors](https://github.com/huggingface/safetensors) compatible weights.

## Usage

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2
```

<img width="1008" height="724" alt="Screenshot 2025-12-29 at 17 09 16" src="https://github.com/user-attachments/assets/1ac40819-5c08-4cfa-bb09-7e332e7467fc" />

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
