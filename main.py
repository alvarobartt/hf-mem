import json
import logging
import math
import os
import struct
from urllib.request import Request, urlopen

logging.basicConfig(
    format="[%(levelname)s] @ %(filename)s:%(lineno)d w/ `%(message)s`",
    level=logging.INFO,
)

if __name__ == "__main__":
    headers = {"Range": "bytes=0-100000"}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors",
        headers=headers,
        method="GET",
    )

    with urlopen(request) as response:
        if response.status not in {200, 206}:
            raise RuntimeError(f"REQUEST FAILED WITH {response.status}")

        metadata = response.read()
        metadata_size = struct.unpack("<Q", metadata[:8])[0]

        metadata = metadata[8 : metadata_size + 8]
        metadata = json.loads(metadata)

        ppdt = {}
        for key, value in metadata.items():
            if key in {"__metadata__"}:
                continue
            if value["dtype"] not in ppdt:
                ppdt[value["dtype"]] = 0
            ppdt[value["dtype"]] += math.prod(value["shape"])

        for dtype, count in ppdt.items():
            logging.info(f"DTYPE={dtype}, COUNT={count}")
            logging.info(f"VRAM={(count * 4) // 1024**2} MB")
