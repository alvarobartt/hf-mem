import math
from dataclasses import dataclass
from typing import Any

GIB = 1 << 30
MIB = 1 << 20
LARGE_GPU = 70 * GIB
LARGEST_GPU = 160 * GIB
# llama.cpp grows the mmproj compute buffer up to the largest image: 4096 merged tokens for dynamic resolution
# encoders (unless `image_max_pixels` is lower, or llama.cpp caps the projector), `image_size` otherwise, and 30 s
# of audio.
CLIP_MAX_TOKENS = {"pixtral": 1024, "lfm2": 256}
CLIP_DYNAMIC_TOKENS = 4096
CLIP_AUDIO_POSITIONS = 1500

# Fitted on vLLM 0.33 memory profiles (A100 for "small", H200 for "large"), see the README.
_PROFILES = {
    "small": {
        "vocab": 5004,
        "inter": 6.6,
        "hidden": 77.2,
        "act": 0,
        "graph": 0.16 * GIB,
        "graph_layer": 5.4 * MIB,
        "graph_moe": 5.6 * MIB,
        "graph_hybrid": 0,
        "graph_lh": 0,
        "graph_width": 6174,
        "non_torch": 0.27 * GIB,
        "context": 0.83 * GIB,
    },
    "large": {
        "vocab": 12702,
        "inter": 0,
        "hidden": 13.2,
        "act": 0.93 * GIB,
        "graph": 0.29 * GIB,
        "graph_layer": 0,
        "graph_moe": 12.9 * MIB,
        "graph_hybrid": 7.6 * MIB,
        "graph_lh": 248.8,
        "graph_width": 23908,
        "non_torch": 0.87 * GIB,
        "context": 1.01 * GIB,
    },
}
_GATED = ("silu", "swiglu", "swish")


@dataclass
class Overhead:
    activations: int
    encoder: int
    graphs: int
    runtime: int
    context: int

    @property
    def bytes(self) -> int:
        return self.activations + self.encoder + self.graphs + self.runtime

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes": self.bytes,
            "activations": self.activations,
            "encoder": self.encoder,
            "graphs": self.graphs,
            "runtime": self.runtime,
        }


def batched_tokens(gpu_memory: str | None, override: int | None = None) -> int:
    if override is not None:
        return override
    size = parse_gpu_memory(gpu_memory)[1] if gpu_memory is not None else LARGE_GPU
    return 16384 if size >= LARGEST_GPU else 8192 if size >= LARGE_GPU else 2048


def _tower_bytes(tower: dict[str, Any], positions: int) -> int:
    hidden = tower.get("hidden_size") or tower.get("d_model") or tower.get("embed_dim") or 0
    inter = tower.get("intermediate_size") or tower.get("encoder_ffn_dim") or 4 * hidden
    gated = any(a in str(tower.get("hidden_act") or tower.get("activation_function") or "") for a in _GATED)
    return positions * 2 * (4 * hidden + (3 if gated else 2) * inter)


def _image_patches(config: dict[str, Any], vision: dict[str, Any], processor: dict[str, Any]) -> int:
    patch = vision.get("patch_size") or processor.get("patch_size") or 14
    size: dict[str, Any] = processor["size"] if isinstance(processor.get("size"), dict) else {}
    pixels = processor.get("max_pixels") or (
        size.get("longest_edge") if (size.get("longest_edge") or 0) > 65536 else None
    )
    if pixels:
        return pixels // patch**2
    if soft := processor.get("max_soft_tokens") or config.get("vision_soft_tokens_per_image"):
        return soft * (processor.get("pooling_kernel_size") or vision.get("pooling_kernel_size") or 1) ** 2
    side = vision.get("image_size") or config.get("force_image_size") or size.get("height") or 224
    tiles = 1
    if dynamic := config.get("max_dynamic_patch"):
        tiles = dynamic + bool(config.get("use_thumbnail"))
    elif (ratio := str(config.get("vision_aspect_ratio") or "")).startswith("anyres_max_"):
        tiles = int(ratio.rsplit("_", 1)[1]) + 1
    elif (tile := (processor.get("max_image_size") or {}).get("longest_edge")) and size.get("longest_edge"):
        tiles = (size["longest_edge"] // tile) ** 2 + 1
    elif hd := processor.get("dynamic_hd"):
        tiles = hd + 1
    return tiles * (side // patch) ** 2


def _root(config: dict[str, Any]) -> dict[str, Any]:
    thinker = config.get("thinker_config")
    return thinker if isinstance(thinker, dict) else config


def encoder(config: dict[str, Any], processor: dict[str, Any]) -> int:
    root = _root(config)
    vision, audio = root.get("vision_config"), root.get("audio_config")
    largest = 0
    if isinstance(vision, dict):
        largest = _tower_bytes(vision, _image_patches(root, vision, processor))
    if isinstance(audio, dict):
        largest = max(largest, _tower_bytes(audio, audio.get("max_source_positions") or 1500))
    return largest


def supports_video(config: dict[str, Any], files: Any) -> bool:
    vision = _root(config).get("vision_config")
    vision = vision if isinstance(vision, dict) else {}
    return bool(vision) and (
        "video_preprocessor_config.json" in files or bool(vision.get("temporal_patch_size"))
    )


def vllm(config: dict[str, Any], moe: bool, hybrid: bool, tokens: int, encoder_bytes: int = 0) -> Overhead:
    p = _PROFILES["large" if tokens >= 8192 else "small"]
    layers = config.get("num_hidden_layers") or 0
    hidden = config.get("hidden_size") or 0
    vocab = config.get("vocab_size") or 32000
    intermediate = config.get("intermediate_size") or config.get("ffn_hidden_size") or 4 * hidden
    if moe and config.get("moe_intermediate_size"):
        intermediate = config["moe_intermediate_size"] * (config.get("num_experts_per_tok") or 1)
    activations = p["vocab"] * vocab + tokens * (p["inter"] * intermediate + p["hidden"] * hidden) + p["act"]
    graphs = (
        p["graph"]
        + layers
        * (p["graph_layer"] + moe * p["graph_moe"] + hybrid * p["graph_hybrid"] + p["graph_lh"] * hidden)
        + p["graph_width"] * (hidden + intermediate)
    )
    context = int(p["context"])
    return Overhead(int(activations), encoder_bytes, int(graphs), int(p["non_torch"]) + context, context)


def _clip_bytes(positions: int, hidden: int, ffn: int | None, gated: bool) -> int:
    return int((18 if gated else 7.2) * positions * max(hidden, ffn or 4 * hidden))


def mmproj_encoder(metadata: dict[str, Any]) -> int:
    total = 0
    if hidden := metadata.get("clip.vision.embedding_length"):
        patch = metadata.get("clip.vision.patch_size") or 14
        projector = metadata.get("clip.projector_type") or metadata.get("clip.vision.projector_type")
        window = metadata.get("clip.vision.n_wa_pattern")
        merge = metadata.get("clip.vision.spatial_merge_size") or metadata.get(
            "clip.vision.projector.scale_factor"
        )
        if projector in CLIP_MAX_TOKENS:
            patches = CLIP_MAX_TOKENS[projector] * (merge or 1) ** 2
        elif (
            window or "clip.vision.spatial_merge_size" in metadata or "clip.vision.image_max_pixels" in metadata
        ):
            patches = CLIP_DYNAMIC_TOKENS * (merge or 2) ** 2
            if max_pixels := metadata.get("clip.vision.image_max_pixels"):
                patches = min(patches, max_pixels // patch**2)
        else:
            patches = ((metadata.get("clip.vision.image_size") or 224) // patch) ** 2
        ffn, gated = metadata.get("clip.vision.feed_forward_length"), bool(metadata.get("clip.use_silu"))
        total += _clip_bytes(patches, hidden, ffn, gated) + (2 * patches**2 if window else 0)
    if hidden := metadata.get("clip.audio.embedding_length"):
        total += _clip_bytes(
            CLIP_AUDIO_POSITIONS, hidden, metadata.get("clip.audio.feed_forward_length"), False
        )
    return total


def llama_cpp(projector: dict[str, Any] | None = None) -> Overhead:
    encoder = mmproj_encoder(projector) if projector else 0
    return Overhead(int(0.15 * GIB), encoder, 0, int(0.47 * GIB), int(0.47 * GIB))


@dataclass
class Fit:
    gpu_memory: int
    usable: int
    required: int
    gpus: int | None
    needed: int | None
    max_model_len: int

    @property
    def fits(self) -> bool:
        return self.needed is not None and self.needed <= (self.gpus or 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_memory": self.gpu_memory,
            "usable": self.usable,
            "required": self.required,
            "gpus": self.gpus,
            "needed": self.needed,
            "fits": self.fits,
            "max_model_len": self.max_model_len,
        }


def parse_gpu_memory(value: str) -> tuple[int | None, int]:
    count, sep, size = value.lower().removesuffix("gib").removesuffix("gb").rpartition("x")
    try:
        gpus, memory = (int(count) if sep else None), int(float(size) * GIB)
    except (ValueError, OverflowError):
        gpus = memory = 0
    if (gpus is not None and gpus < 1) or memory <= 0:
        raise RuntimeError(f"`--gpu-memory={value}` must look like `80` or `8x80` (GiB per GPU).")
    return gpus, memory


def fit(
    weights: int,
    kv_fixed: int,
    kv_per_token: float,
    max_model_len: int,
    context_length: int,
    overhead: int,
    gpu_memory: str,
    utilization: float,
    kv_heads: int | None = None,
    context: int = 0,
) -> Fit:
    gpus, size = parse_gpu_memory(gpu_memory)
    usable = min(size * utilization, size - context) - (overhead - context)

    def per_token(n: int) -> float:
        return kv_per_token * (max(1.0, n / kv_heads) if kv_heads else 1.0)

    needed = next(
        (
            n
            for n in (1, 2, 4, 8, 16, 32, 64)
            if weights + kv_fixed + per_token(n) * max_model_len <= n * usable
        ),
        None,
    )
    n = gpus or 1
    budget = n * usable - weights - kv_fixed
    longest = math.floor(budget / per_token(n)) if kv_per_token > 0 and budget > 0 else 0
    required = weights + kv_fixed + per_token(n) * max_model_len + n * overhead
    return Fit(
        size,
        int(min(size * utilization, size - context) + context),
        int(required),
        gpus,
        needed,
        min(longest, context_length),
    )
