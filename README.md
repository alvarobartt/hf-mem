<img src="https://github.com/user-attachments/assets/509a8244-8a91-4051-b337-41b7b2fe0e2f" />

---

> [!WARNING]
> `hf-mem` is still experimental and therefore subject to major changes across releases, so please keep in mind that breaking changes may occur until v1.0.0.

`hf-mem` is a CLI to estimate inference memory requirements for Hugging Face models, written in Python.

`hf-mem` is lightweight, only depends on `httpx`, as it pulls the [Safetensors](https://github.com/huggingface/safetensors) metadata via [HTTP Range requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Range_requests). It's recommended to run with [`uv`](https://github.com/astral-sh/uv) for a better experience.

`hf-mem` lets you estimate the inference requirements to run any model from the Hugging Face Hub, including [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers) and [Sentence Transformers](https://github.com/huggingface/sentence-transformers) models, as well as any model that contains [Safetensors](https://github.com/huggingface/safetensors) compatible weights.

```console
$ uvx hf-mem --help
usage: hf-mem [-h] --model-id MODEL_ID [--revision REVISION] [--experimental]
              [--max-model-len MAX_MODEL_LEN] [--batch-size BATCH_SIZE]
              [--kv-cache-dtype {fp8,fp8_e5m2,fp8_e4m3,bfloat16,fp8_ds_mla,auto,fp8_inc}]
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
  --kv-cache-dtype {fp8,fp8_e5m2,fp8_e4m3,bfloat16,fp8_ds_mla,auto,fp8_inc}
                        Data type for the KV cache storage. If `auto` is
                        specified, it will use the default model dtype
                        specified in the `config.json` (if available). Despite
                        the FP8 data types having different formats, all those
                        take 1 byte, meaning that the calculation would lead
                        to the same results. Defaults to `auto`.
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

By enabling the `--experimental` flag, you can enable the KV Cache memory estimation for LLMs (`...ForCausalLM`) and VLMs (`...ForConditionalGeneration`), even including a custom `--max-model-len` (defaults to the `config.json` default), `--batch-size` (defaults to 1), and the `--kv-cache-dtype` (defaults to `auto` which means it uses the default data type set in `config.json` under `torch_dtype` or `dtype`, or rather from `quantization_config` when applicable).

```bash
uvx hf-mem --model-id MiniMaxAI/MiniMax-M2 --experimental
```

<img src="https://github.com/user-attachments/assets/247113cf-59a7-4f76-a8df-735e292558a0" />

## References

- [Safetensors Metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [usgraphics - TR-100 Machine Report](https://github.com/usgraphics/usgc-machine-report)
