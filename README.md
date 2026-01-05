<img width="1280" height="640" alt="hf-mem" src="https://github.com/user-attachments/assets/0775a4c1-0750-401e-bfc8-dfb988691e51" />

`hf-mem` is an experimental CLI to estimate inference memory requirements for Hugging Face models, written in Python. `hf-mem` is lightweight, only depends on `httpx`. It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements to run any model from the Hugging Face Hub, including Transformers, Diffusers and Sentence Transformers models, as well as any model that contains [Safetensors](https://github.com/huggingface/safetensors) compatible weights.

Read more information about `hf-mem` in [this short-form post](https://alvarobartt.com/hf-mem).

## Usage

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2
```

<img src="https://github.com/user-attachments/assets/530f8b14-a415-4fd6-9054-bcd81cafae09" />

```bash
uvx hf-mem --model-id Qwen/Qwen-Image
```

<img src="https://github.com/user-attachments/assets/cd4234ec-bdcc-4db4-8b01-0ac9b5cd390c" />

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
