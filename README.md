# hf-mem

[![Crates.io](https://img.shields.io/crates/v/hf-mem.svg)](https://crates.io/crates/hf-mem)

> CLI to estimate inference memory requirements from the Hugging Face Hub

## Install

```console
$ cargo install hf-mem
```

## Usage

```console
$ hf-mem --help
CLI to estimate inference memory requirements from the Hugging Face Hub

Usage: hf-mem [OPTIONS] --model-id <MODEL_ID>

Options:
  -m, --model-id <MODEL_ID>  ID of the model on the Hugging Face Hub
  -r, --revision <REVISION>  Revision of the model on the Hugging Face Hub [default: main]
  -t, --token <TOKEN>        Hugging Face Hub token with read access over the provided model ID, optional
  -h, --help                 Print help
  -V, --version              Print version
```

## Features

- Fast and light CLI with a single installable binary
- Fetches just the required bytes from the `safetensors` files on the Hugging Face
Hub that contain the metadata
- Provides an estimation based on the count of the parameters on the different
dtypes
- Supports both sharded i.e. `model-00000-of-00000.safetensors` and not sharded i.e.
`model.safetensors` files

## What's next?

- [ ] Add tracing and progress bars when fetching from the Hub
- [ ] Support other file types as e.g. `gguf`
- [ ] Read metadata from local files if existing, instead of just fetching from
the Hub every single time
- [ ] Add more flags to support estimations assuming quantization, extended context
lengths, any added memory overhead, etc.

## License

This project is licensed under either of the following licenses, at your option:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
