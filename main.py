import json
import logging
import math
import os
import struct
import sys
from typing import Any, Dict
from urllib.request import Request, urlopen

logging.basicConfig(
    format="[%(levelname)s] @ %(filename)s:%(lineno)d :: %(message)s",
    level=logging.INFO,
)

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000

REQUEST_TIMEOUT = 30.0


def fetch_safetensors_metadata(url: str) -> Dict[str, Any]:
    headers = {"Range": f"bytes=0-{MAX_METADATA_SIZE}"}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        if response.status not in {200, 206}:
            raise RuntimeError(f"REQUEST FAILED WITH {response.status}")

        metadata = response.read()
        # NOTE: Parse the first 8 bytes as a little-endian uint64 (size of the metadata)
        metadata_size = struct.unpack("<Q", metadata[:8])[0]

        if metadata_size < MAX_METADATA_SIZE:
            metadata = metadata[8 : metadata_size + 8]
            return json.loads(metadata)

        # NOTE: Given that by default we just fetch the first 100_000 bytes, if the content is larger
        # then we simply fetch the remainder again
        metadata = metadata[8 : MAX_METADATA_SIZE + 8]
        headers["Range"] = f"bytes={MAX_METADATA_SIZE}-{metadata_size + 7}"

        request = Request(url, headers=headers, method="GET")
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            if response.status not in {200, 206}:
                raise RuntimeError(f"REQUEST FAILED WITH {response.status}")

            metadata += response.read()
            return json.loads(metadata)


if __name__ == "__main__":
    model_id = "google-bert/bert-base-uncased"
    url = f"https://huggingface.co/api/models/{model_id}/tree/main"

    headers = {}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        if response.status != 200:
            raise RuntimeError(f"REQUEST FAILED WITH {response.status}")

        files = [
            f["path"]
            for f in json.loads(response.read())
            if f.get("path", None) is not None
        ]

    if "model.safetensors" in files:
        url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors"
        metadata = fetch_safetensors_metadata(url=url)
    elif "model.safetensors.index.json" in files:
        sys.exit(1)
    elif "model_index.json" in files:
        sys.exit(1)
    else:
        sys.exit(1)

    ppdt = {}
    for key, value in metadata.items():
        if key in {"__metadata__"}:
            continue
        if value["dtype"] not in ppdt:
            ppdt[value["dtype"]] = 0
        ppdt[value["dtype"]] += math.prod(value["shape"])

    for dtype, count in ppdt.items():
        logging.info(f"DTYPE={dtype}, COUNT={count}")

        match dtype:
            case "F32":
                dtype_b = 4
            case _:
                logging.error(f"DTYPE={dtype} NOT HANDLED")
                sys.exit(2)

        bytes_count = count * dtype_b
        logging.info(f"VRAM={bytes_count} bytes")
        logging.info(f"VRAM={bytes_count / 1024**2:.2f} megabytes")

    sys.exit(0)
