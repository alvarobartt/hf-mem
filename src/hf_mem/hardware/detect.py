import subprocess

from hf_mem.hardware.types import GPU, HardwareProfile


def detect_local_gpus() -> HardwareProfile:
    """Detect NVIDIA GPUs via nvidia-smi. Raises RuntimeError on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "nvidia-smi not found — cannot auto-detect GPUs. "
            "Specify hardware manually with `--hardware <profile>`. "
            "Run `hf-mem --list-hardware` to see available profiles."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out after 10s. Is the NVIDIA driver responsive?")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"nvidia-smi failed (exit code {result.returncode}): {stderr}. "
            "Specify hardware manually with `--hardware <profile>`."
        )

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        try:
            mem_mib = int(parts[1].strip())
        except ValueError:
            continue
        gpus.append(GPU(name=name, vram_bytes=mem_mib * 1024 * 1024))

    if not gpus:
        raise RuntimeError(
            "nvidia-smi returned no GPUs. "
            "Specify hardware manually with `--hardware <profile>`. "
            "Run `hf-mem --list-hardware` to see available profiles."
        )

    profile_name = f"{len(gpus)}x {gpus[0].name}" if len(gpus) > 1 else gpus[0].name
    return HardwareProfile(name=profile_name, gpus=gpus, source="local")
