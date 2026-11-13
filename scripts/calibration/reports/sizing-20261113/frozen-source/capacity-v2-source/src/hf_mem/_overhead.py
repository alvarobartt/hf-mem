import math
from collections.abc import Callable
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

# Reference capacity, measured initialization reserve, NVLink-class TP defaults,
# and Hopper FA3 defaults. Unmeasured initialization falls back to capacity.
GPU_TYPES: dict[str, tuple[float, float | None, bool, bool]] = {
    "t4": (16, 0.14, False, False),
    "a10g": (24, 0.32, False, False),
    "l4": (24, 0.25, False, False),
    "l40s": (48, 0.52, False, False),
    "a100": (80, 0.52, True, False),
    "h100": (80, None, True, True),
    "h200": (141, 0.70, True, True),
    "rtx-pro-6000": (96, 0.70, False, False),
}


# Nominal board capacities, with multiple variants where the family is ambiguous.
# HBM/RTX boards allow 2% for driver reservations. ECC-enabled GDDR6 boards
# allow 10%: measured T4/A10G/L4/L40S totals are ~92-93% of the board labels.
GPU_CAPACITIES = {
    "t4": (16,),
    "a10g": (24,),
    "l4": (24,),
    "l40s": (48,),
    "a100": (40, 80),
    "h100": (80, 94, 96),
    "h200": (141,),
    "rtx-pro-6000": (96,),
}
GPU_TYPE_NAMES = tuple("nvidia-" + name for name in GPU_TYPES)


def normalize_gpu_type(value: str) -> str:
    name = value.lower().removeprefix("nvidia-")
    if name not in GPU_TYPES:
        raise RuntimeError(f"`--gpu-type` must be one of {', '.join(GPU_TYPE_NAMES)}.")
    return name


def gpu_type_name(value: str) -> str:
    return "nvidia-" + normalize_gpu_type(value)


def validate_gpu_memory(gpu_memory: str, gpu_type: str) -> None:
    name = normalize_gpu_type(gpu_type)
    size = parse_gpu_memory(gpu_memory)[1] / GIB
    capacities = GPU_CAPACITIES[name]
    fraction = 0.90 if name in {"t4", "a10g", "l4", "l40s"} else 0.98
    if not any(capacity * fraction <= size <= capacity for capacity in capacities):
        expected = " or ".join(str(capacity) for capacity in capacities)
        raise RuntimeError(
            f"`--gpu-memory={gpu_memory}` does not match {gpu_type_name(name)} "
            f"(nominal capacity {expected}; --gpu-memory uses GiB per GPU). "
            "Driver-reported totals within the board family's ECC/reservation range are accepted. "
            "Use --gpu-memory-utilization to limit the usable fraction."
        )


def hardware(gpu_memory: str | None, gpu_type: str | None) -> tuple[float, bool, bool]:
    size = parse_gpu_memory(gpu_memory)[1] / GIB if gpu_memory is not None else 80.0
    context = None
    fast, hopper = size >= 70, size >= 120
    if gpu_type is not None:
        gpu_type = normalize_gpu_type(gpu_type)
        capacity, context, fast, hopper = GPU_TYPES[gpu_type]
        if gpu_memory is None:
            size = capacity
    if context is None:
        points = [(16, 0.14), (24, 0.29), (48, 0.52), (80, 0.52), (140, 0.70)]
        context = points[-1][1]
        if size <= points[0][0]:
            context = points[0][1]
        for (left, low), (right, high) in zip(points, points[1:]):
            if left <= size <= right:
                context = low + (high - low) * (size - left) / (right - left)
                break
    return context * GIB, fast, hopper


def default_sequences(gpu_memory: str | None, gpu_type: str | None, engine: str) -> int:
    if engine == "sglang":
        # Native auto request pools start at 2048; graph workspaces saturate
        # separately. Automatic whole-device cache pools can raise this to 4096.
        return 2048
    name = normalize_gpu_type(gpu_type) if gpu_type is not None else None
    return 1024 if hardware(gpu_memory, name)[2] or name == "rtx-pro-6000" else 256


# Fitted on vLLM 0.33 memory profiles (A100 for "small", H200 for "large"), see the README.
_PROFILES = {
    "small": {
        "seqs": 256,
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
        "seqs": 1024,
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
    max_num_seqs: int | None = None
    reference_engine: str | None = None
    gpu_type: str | None = None

    @property
    def bytes(self) -> int:
        return self.activations + self.encoder + self.graphs + self.runtime

    def to_dict(self) -> dict[str, Any]:
        out = {
            "bytes": self.bytes,
            "activations": self.activations,
            "encoder": self.encoder,
            "graphs": self.graphs,
            "runtime": self.runtime,
        }
        if self.max_num_seqs is not None:
            out["max_num_seqs"] = self.max_num_seqs
        if self.reference_engine is not None:
            out["reference_engine"] = self.reference_engine
        if self.gpu_type is not None:
            out["gpu_type"] = self.gpu_type
        return out


def _capacity_gib(gpu_memory: str | None, gpu_type: str | None) -> float:
    if gpu_memory is not None:
        return parse_gpu_memory(gpu_memory)[1] / GIB
    return GPU_TYPES[normalize_gpu_type(gpu_type)][0] if gpu_type is not None else 80.0


def graph_capacity(gpu_memory: str | None, gpu_type: str | None, ranks: int, engine: str) -> int:
    if engine == "vllm":
        return 256
    size = _capacity_gib(gpu_memory, gpu_type)
    if size < 20:
        return 8
    if size < 35:
        return 48 if ranks < 4 else 160
    if size < 60:
        return 32 if ranks < 4 else 160
    return 256 if ranks < 4 and size < 160 else 512


def batched_tokens(
    gpu_memory: str | None, override: int | None = None, gpu_type: str | None = None, engine: str = "vllm"
) -> int:
    if gpu_type is not None:
        gpu_type = normalize_gpu_type(gpu_type)
    if override is not None:
        return override
    if engine == "sglang":
        size = _capacity_gib(gpu_memory, gpu_type)
        return 2048 if size < 20 else 4096 if size < 60 else 8192 if size < 160 else 16384
    if gpu_type == "a100":
        return 2048
    if gpu_type is not None:
        size = GPU_TYPES[gpu_type][0] * GIB
        return 16384 if size >= LARGEST_GPU else 8192 if size >= LARGE_GPU else 2048
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


def encoder(
    config: dict[str, Any], processor: dict[str, Any], context_length: int | None = None, engine: str = "vllm"
) -> int:
    root = _root(config)
    vision, audio = root.get("vision_config"), root.get("audio_config")
    largest = 0
    if isinstance(vision, dict):
        positions = _image_patches(root, vision, processor)
        # Qwen-style spatial merging maps merge_size**2 encoder patches to one
        # cached text token. An image larger than the entire context is inadmissible.
        merge = vision.get("spatial_merge_size") or processor.get("merge_size")
        if context_length is not None and isinstance(merge, int) and merge > 0:
            positions = min(positions, context_length * merge**2)
        # Paired long-text/large-image runs in both engines retain roughly
        # 1.5x the old tower buffer estimate. Audio has its independent layout.
        largest = int(_tower_bytes(vision, positions) * 1.5)
    if isinstance(audio, dict):
        largest = max(largest, _tower_bytes(audio, audio.get("max_source_positions") or 1500))
    return largest


def supports_video(config: dict[str, Any], files: Any) -> bool:
    vision = _root(config).get("vision_config")
    vision = vision if isinstance(vision, dict) else {}
    return bool(vision) and (
        "video_preprocessor_config.json" in files or bool(vision.get("temporal_patch_size"))
    )


def vllm(
    config: dict[str, Any],
    moe: bool,
    hybrid: bool,
    tokens: int,
    encoder_bytes: int = 0,
    max_num_seqs: int | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory: str | None = None,
    gpu_type: str | None = None,
    engine: str = "vllm",
    batch_size: int = 1,
    context_length: int = 0,
) -> Overhead:
    if gpu_type is not None:
        gpu_type = normalize_gpu_type(gpu_type)
    layers = config.get("num_hidden_layers") or 0
    hidden = config.get("hidden_size") or 0
    vocab = config.get("vocab_size") or 32000
    intermediate = config.get("intermediate_size") or config.get("ffn_hidden_size") or 4 * hidden
    if moe and config.get("moe_intermediate_size"):
        intermediate = config["moe_intermediate_size"] * (config.get("num_experts_per_tok") or 1)

    def graph_bytes(profile: dict[str, float]) -> float:
        return (
            profile["graph"]
            + layers
            * (
                profile["graph_layer"]
                + moe * profile["graph_moe"]
                + hybrid * profile["graph_hybrid"]
                + profile["graph_lh"] * hidden
            )
            + profile["graph_width"] * (hidden + intermediate)
        )

    # Native stage audits distinguish driver/library initialization, prefill
    # workspace, and replicated communication buffers. These are empirical
    # defaults, not replicas of either allocator. Both engines share geometry.
    context, fast, hopper = hardware(gpu_memory, gpu_type)
    n = tensor_parallel_size
    capacity = max_num_seqs if max_num_seqs is not None else default_sequences(gpu_memory, gpu_type, engine)
    # Default decode capture saturates at 512 tokens in vLLM (twice
    # scheduler capacity) and 256 requests in SGLang. Larger scheduler
    # ceilings do not retain a proportional vocabulary-sized workspace.
    bounded_capacity = min(
        tokens, max(batch_size, min(capacity, graph_capacity(gpu_memory, gpu_type, n, engine)))
    )
    fraction = min(1.0, max(0.0, (tokens - 2048) / (8192 - 2048)))

    def interpolate(key: str) -> float:
        small, large = _PROFILES["small"], _PROFILES["large"]
        left, right = small[key] / small["seqs"], large[key] / large["seqs"]
        return (left + fraction * (right - left)) * bounded_capacity

    def graph_pool(profile: dict[str, float]) -> float:
        return graph_bytes(profile) * (0.05 + 0.95 * bounded_capacity / profile["seqs"])

    left, right = graph_pool(_PROFILES["small"]), graph_pool(_PROFILES["large"])
    graphs = left + fraction * (right - left)
    if engine == "sglang" and fast and not hopper and n > 1:
        # Native FlashInfer decode capture retains collective buffers that
        # grow with the capture batch; FA3 follows its separate graph path.
        graphs += 0.35 * GIB * bounded_capacity / 256 * min(1.0, (config.get("head_dim") or 128) / 128)
    head_dim = config.get("head_dim") or hidden // max(1, config.get("num_attention_heads") or 1)
    head_factor = min(1.0, (head_dim or 128) / 128)
    if engine == "vllm" and not hopper:
        # Native small-tier captures retain less shared pool storage than the
        # original profiling reserve. Keep the low-capacity floor intact.
        graphs -= 0.14 * GIB * bounded_capacity / 256 * (1 - fraction)
    # SGLang retains a float32 shared logits buffer sized by decode capture,
    # not by its independently much larger request-index pool.
    logits = 4 * vocab * bounded_capacity
    # Residual streams follow hidden_size; attention projections can be wider
    # (e.g. explicit head_dim). Only the local projection width shards in TP.
    query_width = (config.get("num_attention_heads") or 0) * (config.get("head_dim") or 0)
    activation_width = (hidden + max(hidden, query_width / n)) / 2
    gdn = hybrid and bool(config.get("linear_num_value_heads"))
    # Full-attention FA3 does not describe the separately dispatched GDN
    # prefill kernel. Its empirical activation reserve follows the linear path.
    if engine == "sglang" and (not hopper or gdn):
        activations = 0.9 * GIB + tokens * (32 * activation_width + 2.6 * intermediate / n) + logits
    else:
        small = 2048 * (6.6 * intermediate / n + 24 * activation_width)
        workspace = ((0.55 if engine == "vllm" else 0.70) + 0.35 / n) if hopper else 0.72 / n
        if engine == "sglang":
            workspace *= min(1.0, hidden / 4096)
        large = workspace * GIB + 8192 * 13.2 * activation_width
        if tokens <= 2048:
            activations = tokens / 2048 * small
        elif tokens >= 8192:
            activations = large + (tokens - 8192) * 13.2 * activation_width
        else:
            activations = small + fraction * (large - small)
        activations += logits
    if engine == "vllm":
        communication = 1.0 if hopper else 0.65 if fast else 0.1
    else:
        # Ampere/FlashInfer TP4 retains larger collective workspaces than
        # TP2 in native stage audits. Hopper follows its separate FA3 path.
        communication = (1.35 if n >= 4 and not hopper else 1.0) if fast else 0.5
    communication = communication * GIB if n > 1 else 0
    if engine == "vllm" and hopper and n > 1:
        communication += 0.5 * GIB * bounded_capacity / 256
    non_torch = interpolate("non_torch")
    if not hopper and gpu_type != "rtx-pro-6000":
        non_torch *= head_factor
    if moe and n > 1:
        active = config.get("num_experts_per_tok") or 1
        if engine == "vllm":
            # Separate native MoE communicator initialization (~0.5 GiB)
            # and Hopper fused-collective library buffers (~0.5 GiB).
            communication += (1.0 if hopper else 0.5 if fast else 0.1) * GIB
            if not hopper:
                activations += tokens * 4 * hidden * active
        else:
            communication += ((0.9 + 0.6 * max(0.0, 1 - 2 / n)) if hopper else 0.15) * GIB
            activations += tokens * 4 * hidden * active
    if gdn:
        if engine == "sglang" and not hopper:
            # Triton prefill's key-side scratch has a different head domain
            # from the recurrent value states, which are already in the cache.
            heads = config.get("linear_num_key_heads") or config["linear_num_value_heads"]
            width = config.get("linear_key_head_dim") or 0
        else:
            heads = config["linear_num_value_heads"]
            width = config.get("linear_value_head_dim") or 0
        local_width = math.ceil(heads / n) * width
        scratch = tokens * local_width * (24 if engine == "vllm" else 96)
        if (
            engine == "sglang"
            and hopper
            and config.get("linear_key_head_dim") == 128
            and config.get("linear_value_head_dim") == 128
            and config.get("mamba_ssm_dtype") == "float32"
            and not config.get("linear_attn_config")
            and tokens <= 32768
        ):
            # SM90's automatically selected fused GDN prefill does not retain
            # the token-sized Triton intermediates. This empirical reserve is
            # bounded separately from the token-dependent hidden/output buffers.
            scratch = min(scratch, 0.5 * GIB)
        activations += scratch
        if n > 1 and not moe:
            if engine == "vllm":
                communication += (0.5 if hopper else 0.1 if fast else 0) * GIB
            elif hopper:
                # Retained fused-collective workspace observed outside Torch.
                # MoE already carries a larger reserve; do not add both.
                communication += (0.5 + 0.5 * max(0.0, 1 - 2 / n)) * GIB
    return Overhead(
        int(activations),
        encoder_bytes,
        int(max(0, graphs)),
        int(non_torch + context + communication + (capacity + 1) * context_length * 4 * (engine == "sglang")),
        int(context),
        capacity,
        engine,
        gpu_type_name(gpu_type) if gpu_type is not None else None,
    )


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


def llama_cpp(projector: dict[str, Any] | None = None, gpu_memory: str | None = None) -> Overhead:
    encoder = mmproj_encoder(projector) if projector else 0
    runtime = 0.47
    if gpu_memory is not None:
        size = parse_gpu_memory(gpu_memory)[1]
        # Matched CUDA b12316 inference: T4 residual runtime ~0.13 GiB, H200 ~0.60.
        # Capacity is a coarse hardware proxy; retain the existing middle/unknown tier.
        runtime = 0.13 if size <= 16 * GIB else 0.60 if size >= 120 * GIB else runtime
    return Overhead(int(0.15 * GIB), encoder, 0, int(runtime * GIB), int(runtime * GIB))


@dataclass
class Fit:
    gpu_memory: int
    usable: int
    required: int
    gpus: int | None
    needed: int | None
    max_model_len: int
    overhead_per_gpu: int | None = None

    @property
    def fits(self) -> bool:
        return self.required <= (self.gpus or 1) * self.usable

    def to_dict(self) -> dict[str, Any]:
        out = {
            "gpu_memory": self.gpu_memory,
            "usable": self.usable,
            "required": self.required,
            "gpus": self.gpus,
            "needed": self.needed,
            "fits": self.fits,
            "max_model_len": self.max_model_len,
        }
        if self.overhead_per_gpu is not None:
            out["overhead_per_gpu"] = self.overhead_per_gpu
        return out


def mxfp4_padding(experts: int, intermediate: int, hidden: int) -> Callable[[int], int]:
    # vLLM rounds the MXFP4 expert shards up to its kernel tiles (128 intermediate, 256 hidden on Hopper)

    def extra(n: int) -> int:
        shard = max(1, intermediate // n)
        factor = (-(-shard // 128) * 128 / shard) * (-(-hidden // 256) * 256 / hidden)
        return int(experts * (factor - 1))

    return extra


def parse_gpu_memory(value: str) -> tuple[int | None, int]:
    value_lower = value.strip().lower()
    unit = 10**9 if value_lower.endswith("gb") and not value_lower.endswith("gib") else GIB
    count, sep, size = value_lower.removesuffix("gib").removesuffix("gb").rpartition("x")
    try:
        gpus, memory = (int(count) if sep else None), int(float(size) * unit)
    except (ValueError, OverflowError):
        gpus = memory = 0
    if (gpus is not None and gpus < 1) or memory <= 0:
        raise RuntimeError(f"`--gpu-memory={value}` must look like `80` or `8x80` (GiB per GPU).")
    return gpus, memory


def recommended_memory(allocation: int, utilization: float, context: int = 0) -> int:
    """Physical capacity that satisfies the same budget as ``fit``.

    The CUDA context is outside the engine budget; all allocations must still
    fit in physical memory. ``context`` is summed across the selected ranks.
    """
    return max(allocation, math.ceil((allocation - context) / utilization))


def fit(
    weights: int,
    kv_fixed: int,
    kv_per_token: float,
    max_model_len: int,
    context_length: int,
    overhead: int | Callable[[int], int],
    gpu_memory: str,
    utilization: float,
    kv_heads: int | None = None,
    context: int = 0,
    padding: Callable[[int], int] | None = None,
    kv_bytes: Callable[[int, int], int] | None = None,
    overhead_at_context: Callable[[int, int], int] | None = None,
) -> Fit:
    gpus, size = parse_gpu_memory(gpu_memory)

    def rank_overhead(n: int, length: int = max_model_len) -> int:
        if overhead_at_context is not None:
            return overhead_at_context(n, length)
        return overhead if isinstance(overhead, int) else overhead(n)

    def usable(n: int, length: int = max_model_len) -> float:
        return min(size * utilization, size - context) - (rank_overhead(n, length) - context)

    base = weights

    def cache(length: int, n: int) -> float:
        replication = max(1.0, n / kv_heads) if kv_heads else 1.0
        if kv_bytes is not None:
            return kv_bytes(length, n)
        return kv_fixed + kv_per_token * length * replication

    def padded(n: int) -> int:
        return base + (padding(n) if padding else 0)

    needed = next(
        (
            n
            for n in sorted({1, 2, 4, 8, 16, 32, 64, gpus or 1})
            if padded(n) + cache(max_model_len, n) <= n * usable(n)
        ),
        None,
    )
    n = gpus or 1
    weights = padded(n)
    if kv_bytes is not None or overhead_at_context is not None:
        # Search the actual allocation: cache pages, sliding windows and recurrent state make it non-linear.
        low, high = 0, context_length
        while low < high:
            middle = (low + high + 1) // 2
            if weights + cache(middle, n) <= n * usable(n, middle):
                low = middle
            else:
                high = middle - 1
        longest = low
    else:
        budget = n * usable(n) - weights - kv_fixed
        per_token = kv_per_token * (max(1.0, n / kv_heads) if kv_heads else 1.0)
        longest = math.floor(budget / per_token) if per_token > 0 and budget > 0 else 0
    required = weights + cache(max_model_len, n) + n * rank_overhead(n)
    return Fit(
        size,
        int(min(size * utilization, size - context) + context),
        int(required),
        gpus,
        needed,
        min(longest, context_length),
        rank_overhead(n) if callable(overhead) else None,
    )
