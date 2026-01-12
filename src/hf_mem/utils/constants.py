import os

# NOTE: Defines the bytes that will be fetched per safetensors file, but the metadata can indeed be larger than that.
MAX_METADATA_SIZE = 100_000

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 10.0))

MAX_CONCURRENCY = int(os.getenv("MAX_WORKERS", min(32, (os.cpu_count() or 1) + 4)))