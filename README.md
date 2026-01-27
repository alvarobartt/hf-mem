<img src="https://github.com/user-attachments/assets/509a8244-8a91-4051-b337-41b7b2fe0e2f" />

---

> [!WARNING]
> `hf-mem` is still experimental and therefore subject to major changes across releases, so please keep in mind that breaking changes may occur until v1.0.0.

`hf-mem` is a CLI to estimate inference memory requirements for Hugging Face models, written in Python.

`hf-mem` is lightweight, only depends on `httpx`, as it pulls the Safetensors metadata via HTTP Range requests. It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements to run any model from the Hugging Face Hub, including Transformers, Diffusers and Sentence Transformers models, as well as any model that contains [Safetensors](https://github.com/huggingface/safetensors) compatible weights.

```console
$ uvx hf-mem --help
usage: hf-mem [-h] --model-id MODEL_ID [--revision REVISION] [--experimental]
              [--max-model-len MAX_MODEL_LEN] [--batch-size BATCH_SIZE]
              [--json-output] [--ignore-table-width]

options:
  -h, --help            show this help message and exit
  --model-id MODEL_ID   Model ID on the Hugging Face Hub
  --revision REVISION   Model revision on the Hugging Face Hub
  --experimental        Whether to enable the experimental KV Cache estimation
                        or not. Only applies to `...ForCausalLM` and
                        `...ForConditionalGeneration` models from
                        Transformers.
  --max-model-len MAX_MODEL_LEN
                        Model context length (prompt and output). If
                        unspecified, will be automatically derived from the
                        model config.
  --batch-size BATCH_SIZE
                        Batch size to help estimate the required RAM for
                        caching when running the inference. Defaults to 1.
  --json-output         Whether to provide the output as a JSON instead of
                        printed as table.
  --ignore-table-width  Whether to ignore the maximum recommended table width,
                        in case the `--model-id` and/or `--revision` cause a
                        row overflow when printing those.
```

Read more information about `hf-mem` in [this short-form post](https://alvarobartt.com/hf-mem).

## Usage

### Transformers

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2
```

<img src="https://github.com/user-attachments/assets/530f8b14-a415-4fd6-9054-bcd81cafae09" />

### Diffusers

```bash
uvx hf-mem --model-id Qwen/Qwen-Image
```

<img src="https://github.com/user-attachments/assets/cd4234ec-bdcc-4db4-8b01-0ac9b5cd390c" />

### Sentence Transformers

```bash
uvx hf-mem --model-id google/embeddinggemma-300m
```

<img src="https://github.com/user-attachments/assets/2844582f-6207-415a-bc6c-27569a5eb262" />

## Experimental

By enabling the `--experimental` flag, you can enable the KV Cache memory estimation for LLMs and VLMs, even including a custom `--max-model-len` (defaults to the `config.json` default) and `--batch-size` (defaults to 1).

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2 --experimental --max-model-len 65536 --batch-size 2
```

<img src="https://github.com/user-attachments/assets/233a174b-9907-4639-9165-365bd8077de4" />

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
