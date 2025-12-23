import argparse
import json
import logging
import math
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from typing import Any, Dict
from urllib.request import Request, urlopen

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata
# can indeed be larger than that
MAX_METADATA_SIZE = 100_000
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 30.0))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))


logging.basicConfig(
    format="[%(levelname)s] @ %(filename)s:%(lineno)d :: %(message)s",
    level=logging.INFO,
)


def fetch_safetensors_metadata(url: str) -> Dict[str, Any]:
    headers = {"Range": f"bytes=0-{MAX_METADATA_SIZE}"}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        if response.status not in {200, 206}:
            logging.error(f"REQUEST FAILED WITH {response.status}")
            sys.exit(3)

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
                logging.error(f"REQUEST FAILED WITH {response.status}")
                sys.exit(3)

            metadata += response.read()
            return json.loads(metadata)


def main(model_id: str, revision: str) -> None:
    url = f"https://huggingface.co/api/models/{model_id}/tree/{revision}"

    headers = {}
    if token := os.getenv("HF_TOKEN", None):
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        if response.status != 200:
            logging.error(f"REQUEST FAILED WITH {response.status}")
            sys.exit(3)

        files = [
            f["path"]
            for f in json.loads(response.read())
            if f.get("path", None) is not None
        ]

    if "model.safetensors" in files:
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors"
        metadata = fetch_safetensors_metadata(url=url)
    elif "model.safetensors.index.json" in files:
        # TODO: We could eventually skip this request in favour of a greedy approach on trying to pull all the
        # files following the formatting `model-00000-of-00000.safetensors`
        url = f"https://huggingface.co/{model_id}/resolve/{revision}/model.safetensors.index.json"
        request = Request(url, headers=headers, method="GET")
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:
                logging.error(f"REQUEST FAILED WITH {response.status}")
                sys.exit(3)

            urls = {
                f"https://huggingface.co/{model_id}/resolve/{revision}/{f}"
                for _, f in json.loads(response.read())["weight_map"].items()
            }

        metadatas = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(fetch_safetensors_metadata, url): url for url in urls
            }
            for future in as_completed(futures):
                try:
                    metadatas.append(future.result())
                except Exception as e:
                    logging.error(f'ANY OF THE FUTURES FAILED "{e}"')
                    sys.exit(4)

        metadata = reduce(lambda acc, metadata: acc | metadata, metadatas, {})
    elif "model_index.json" in files:
        logging.warning("model_index.json NOT SUPPORTED YET")
        sys.exit(1)
    else:
        logging.error(
            "NONE OF `model.safetensors`, `model.safetensors.index.json`, `model_index.json` HAS BEEN FOUND"
        )
        sys.exit(1)

    ppdt = {}
    for key, value in metadata.items():
        if key in {"__metadata__"}:
            continue
        if value["dtype"] not in ppdt:
            ppdt[value["dtype"]] = 0
        ppdt[value["dtype"]] += math.prod(value["shape"])

    total = 0
    for dtype, count in ppdt.items():
        logging.info(f"DTYPE={dtype}, COUNT={count}\n")

        match dtype:
            case "F64" | "I64" | "U64":
                dtype_b = 8
            case "F32" | "I32" | "U32":
                dtype_b = 4
            case "F16" | "BF16" | "I16" | "U16":
                dtype_b = 2
            case "F8_E5M2" | "F8_E4M3" | "I8" | "U8":
                dtype_b = 1
            case _:
                logging.error(f"DTYPE={dtype} NOT HANDLED")
                sys.exit(2)

        bytes_count = count * dtype_b
        total += bytes_count
        logging.info(f"VRAM={bytes_count} (IN BYTES)")
        logging.info(f"VRAM={bytes_count / (1024**2):.2f} (IN MEGABYTES)")
        logging.info(f"VRAM={bytes_count / (1024**3):.2f} (IN GIGABYTES)\n")

    logging.info(f"TOTAL VRAM={total} (IN BYTES)")
    logging.info(f"TOTAL VRAM={total / (1024**2):.2f} (IN MEGABYTES)")
    logging.info(f"TOTAL VRAM={total / (1024**3):.2f} (IN GIGABYTES)")

    sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", required=True, help="Model ID on the Hugging Face Hub"
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision on the Hugging Face Hub",
    )

    args = parser.parse_args()
    main(model_id=args.model_id, revision=args.revision)
