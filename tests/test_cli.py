import json
import re
import sys
from pathlib import Path

import pytest
from helpers import llama_config, llama_tensors, write_gguf, write_json, write_safetensors
from test_gguf import ARCH, TENSORS

from hf_mem import __version__
from hf_mem._render import _bar, _where
from hf_mem.cli import main


@pytest.fixture
def model(tmp_path: Path) -> str:
    write_safetensors(str(tmp_path / "model.safetensors"), llama_tensors())
    write_json(str(tmp_path / "config.json"), llama_config())
    return str(tmp_path)


def cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int | str | None, str, str]:
    monkeypatch.setattr(sys, "argv", ["hf-mem", *args])
    try:
        main()
        code: int | str | None = 0
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out, out.err


def test_table(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, err = cli(monkeypatch, capsys, model)
    assert code == 0
    assert "TOTAL MEMORY" in out and "KV CACHE (1024 TOKENS" in out and "MODEL (" in out
    assert f"hf-mem v{__version__}" in out
    assert "\x1b[" not in out and err == ""
    assert len({len(line) for line in out.splitlines()}) == 1


def test_table_without_context_length(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    config = llama_config()
    del config["max_position_embeddings"]
    write_json(str(Path(model) / "config.json"), config)

    code, out, err = cli(monkeypatch, capsys, model)

    assert code == 0
    assert "MODEL (" in out and "BF16" in out and "F32" in out
    assert "KV CACHE (" not in out
    assert "KV cache estimation skipped, as the context length is not in `config.json`" in err
    assert len({len(line) for line in out.splitlines()}) == 1


@pytest.mark.parametrize(
    "tty,color,no_color,expected",
    [
        (True, False, False, False),
        (True, True, False, True),
        (True, True, True, False),
        (False, True, False, False),
    ],
)
def test_terminal_color_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model: str,
    tty: bool,
    color: bool,
    no_color: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)
    monkeypatch.delenv("NO_COLOR", raising=False)
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")

    code, out, _ = cli(monkeypatch, capsys, model, *(["--color"] if color else []))

    assert code == 0
    assert ("\x1b[" in out) is expected
    assert "TOTAL MEMORY" in out


def test_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "-o", "json", "--max-model-len", "10", "--batch-size", "2")
    data = json.loads(out)
    assert code == 0
    keys = [
        "version",
        "model",
        "revision",
        "file",
        "format",
        "total",
        "recommended_gpu_memory",
        "gpu_memory_utilization",
        "weights",
        "offloaded",
        "components",
        "kv_cache",
        "overhead",
        "moe",
        "fit",
        "warnings",
    ]
    assert list(data) == keys
    assert data["version"] == f"hf-mem@{__version__}"
    assert data["kv_cache"]["max_model_len"] == 10 and data["kv_cache"]["batch_size"] == 2
    assert data["total"] == data["weights"]["bytes"] + data["kv_cache"]["bytes"] + data["overhead"]["bytes"]


def test_gpu_fit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "2x4", "-o", "json")
    fit = json.loads(out)["fit"]
    assert code == 0 and fit["gpus"] == 2 and fit["needed"] == 1 and fit["fits"]
    assert fit["max_model_len"] == 1024
    usable = fit["usable"] / (1 << 30)
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "4")
    assert code == 0 and "GPU MEMORY" in out and f"/ {usable:.2f} GiB [" in out and "│ FITS " in out
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "2x4")
    assert "FITS ON 1 GPU" in out and f"/ {2 * usable:.2f} GiB [" in out
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "2x4", "--output", "markdown")
    assert "| Overhead (activations, graphs, runtime) |" in out
    assert "Fits on 2 x 4.00 GiB, needs 1 GPU, up to 1024 tokens." in out
    code, out, _ = cli(monkeypatch, capsys, model, "--gpu-memory", "0.2")
    assert "⚠ NEEDS 2 GPUS" in out or "⚠ DOES NOT FIT" in out
    code, _, err = cli(monkeypatch, capsys, model, "--gpu-memory", "lots")
    assert code == 1 and "--gpu-memory" in err


def test_markdown(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "--output", "markdown")
    assert code == 0
    assert "| **Total** |" in out and "| KV cache (BF16, 1024 tokens x 1) |" in out


def test_removed_flags(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str) -> None:
    for flag in (
        "--model-id",
        "--experimental",
        "--json",
        "--markdown",
        "--json-output",
        "--details",
        "--gguf-file",
    ):
        code, _, err = cli(monkeypatch, capsys, model, flag, "x")
        assert code == 2 and "unrecognized arguments" in err


def test_gguf_listing_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for quant in ("Q4_K_M", "Q8_0"):
        write_gguf(str(tmp_path / f"m-{quant}.gguf"), ARCH, TENSORS)
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path), "--file", "m-Q8_0.gguf", "-o", "json")
    assert code == 0 and json.loads(out)["file"] == "m-Q8_0.gguf"
    code, out, _ = cli(monkeypatch, capsys, str(tmp_path))
    assert code == 0 and "FILES" in out and "m-Q4_K_M.gguf" in out
    assert "GiB WEIGHTS +" in out and "GiB OVERHEAD)" in out


def test_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, _, err = cli(monkeypatch, capsys, str(tmp_path / "missing"))
    assert code == 1 and err.startswith("hf-mem: error:")
    code, _, err = cli(monkeypatch, capsys)
    assert code == 2 and "required" in err
    code, _, _ = cli(monkeypatch, capsys, str(tmp_path), "--batch-size", "0")
    assert code == 2


def test_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = cli(monkeypatch, capsys, "--version")
    assert code == 0 and out.strip() == f"hf-mem {__version__}"


def test_titles_shorten_paths_and_commits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _where(str(tmp_path / "cwd" / "m"), "a" * 40) == "m @ aaaaaaa"
    assert _where(str(tmp_path / "home" / "m"), None) == "~/m"
    assert _where("https://hf.co/org/model", "main") == "https://hf.co/org/model @ main"
    long = _where("https://hf.co/org/" + "x" * 200, None)
    assert len(long) == 72 and long.startswith("...") and long.endswith("x")


def test_version_matches_pyproject() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    match = re.search(r'^version = "(.+)"$', pyproject, re.M)
    assert match is not None and match[1] == __version__


def test_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"\0" * 1000)
    (tmp_path / "optimizer.pt").write_bytes(b"\0" * 10)
    code, _, err = cli(monkeypatch, capsys, str(tmp_path))
    assert code == 1 and "only has PyTorch pickle weights (e.g. `pytorch_model.bin`)" in err
    code, _, err = cli(monkeypatch, capsys, str(tmp_path / "missing"))
    assert code == 1 and "does not exist" in err
    code, _, err = cli(monkeypatch, capsys, "not-a-model")
    assert code == 1 and "is not a Hub model ID (`org/name`)" in err


def test_bars_show_small_fractions() -> None:
    assert _bar(0.0, 4) == "░░░░"
    assert _bar(0.001, 4) == "▏░░░"
    assert _bar(0.5, 4) == "██░░"
    assert _bar(0.3, 4) == "█▎░░"
    assert _bar(1.0, 4) == "████"
    assert all(len(_bar(f / 100, 7)) == 7 for f in range(101))


def test_scheduler_capacity_is_separate_from_requested_cache_batch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, "-o", "json", "--max-num-batched-tokens", "8192")
    assert code == 0
    default = json.loads(out)
    code, out, _ = cli(
        monkeypatch, capsys, model, "-o", "json", "--max-num-batched-tokens", "8192", "--max-num-seqs", "1"
    )
    one = json.loads(out)
    assert code == 0 and one["overhead"]["max_num_seqs"] == 1
    assert one["kv_cache"] == default["kv_cache"]
    assert one["weights"] == default["weights"]
    assert one["total"] < default["total"]
    code, _, err = cli(monkeypatch, capsys, model, "--batch-size", "2", "--max-num-seqs", "1")
    assert code == 2 and "at least" in err


def test_full_window_option_changes_cache_and_fit_without_changing_attention_or_overhead(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    write_json(
        str(Path(model) / "config.json"),
        llama_config(
            max_position_embeddings=2000000,
            sliding_window=128,
            layer_types=["sliding_attention"] * 4,
        ),
    )
    args = (model, "-o", "json", "--gpu-memory", "2", "--max-model-len", "2000000", "--max-num-seqs", "1")
    code, out, _ = cli(monkeypatch, capsys, *args)
    assert code == 0
    windowed = json.loads(out)
    code, out, _ = cli(monkeypatch, capsys, *args, "--full-sliding-window-cache")
    assert code == 0
    full = json.loads(out)
    assert full["kv_cache"]["full_sliding_window_cache"]
    assert full["kv_cache"]["bytes"] > windowed["kv_cache"]["bytes"]
    assert full["kv_cache"]["layers"] == windowed["kv_cache"]["layers"]
    assert full["overhead"] == windowed["overhead"]
    assert full["fit"]["max_model_len"] < windowed["fit"]["max_model_len"]


@pytest.mark.parametrize("flag", ["--max-model-len", "--context-length", "--context-len"])
def test_context_aliases(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str, flag: str
) -> None:
    code, out, _ = cli(monkeypatch, capsys, model, flag, "128", "-o", "json")
    assert code == 0 and json.loads(out)["kv_cache"]["max_model_len"] == 128


@pytest.mark.parametrize("flag", ["--model", "--model-path", "--hf-repo", "-m", "-hf"])
def test_model_aliases(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str, flag: str
) -> None:
    code, out, _ = cli(monkeypatch, capsys, flag, model, "-o", "json")
    assert code == 0 and json.loads(out)["format"] == "safetensors"


def test_sglang_controls_and_prefixed_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    common = (model, "--engine", "sglang", "--gpu-memory", "8x141", "--gpu-type", "nvidia-h200", "-o", "json")
    code, out, _ = cli(monkeypatch, capsys, *common, "--max-num-batched-tokens", "4096", "--max-num-seqs", "16")
    assert code == 0
    canonical = json.loads(out)
    code, out, _ = cli(
        monkeypatch, capsys, *common, "--chunked-prefill-size", "4096", "--max-running-requests", "16"
    )
    assert code == 0 and json.loads(out) == canonical
    assert canonical["overhead"]["gpu_type"] == "nvidia-h200"
    code, _, err = cli(
        monkeypatch, capsys, "missing/model", "--gpu-type", "nvidia-h200", "--gpu-memory", "8x80"
    )
    assert code == 1 and "does not match nvidia-h200" in err


@pytest.mark.parametrize(
    "flag", ["--disable-hybrid-kv-cache-manager", "--disable-hybrid-swa-memory", "--full-sliding-window-cache"]
)
def test_engine_full_window_aliases(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str, flag: str
) -> None:
    write_json(
        str(Path(model) / "config.json"), llama_config(sliding_window=32, layer_types=["sliding_attention"] * 4)
    )
    code, out, _ = cli(monkeypatch, capsys, model, flag, "-o", "json")
    assert code == 0 and json.loads(out)["kv_cache"]["full_sliding_window_cache"]


def test_llamacpp_context_pool_parallel_cache_and_full_swa(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = str(tmp_path / "model.gguf")
    metadata = {
        **ARCH,
        "llama.attention.sliding_window": 32,
        "llama.attention.sliding_window_pattern": [True] * int(ARCH["llama.block_count"]),
    }
    write_gguf(path, metadata, TENSORS)
    code, out, _ = cli(
        monkeypatch,
        capsys,
        path,
        "--ctx-size",
        "4096",
        "--parallel",
        "2",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--swa-full",
        "-o",
        "json",
    )
    assert code == 0
    data = json.loads(out)
    assert data["kv_cache"]["max_model_len"] == 2048 and data["kv_cache"]["batch_size"] == 2
    assert data["kv_cache"]["full_sliding_window_cache"]
    code, out, _ = cli(
        monkeypatch,
        capsys,
        path,
        "--format",
        "gguf",
        "--max-model-len",
        "2048",
        "--batch-size",
        "2",
        "--kv-cache-dtype",
        "f16",
        "--full-sliding-window-cache",
        "-o",
        "json",
    )
    assert code == 0 and json.loads(out) == data
    for extra in [
        ("--ctx-size", "4097", "--parallel", "2"),
        ("--cache-type-k", "f16", "--cache-type-v", "q8_0"),
        ("--cache-type-k", "f16"),
    ]:
        code, _, err = cli(monkeypatch, capsys, path, *extra)
        assert code == 2 and "error" in err


def test_language_model_only_cli_changes_loaded_weights(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    write_safetensors(
        str(Path(model) / "model.safetensors"), {**llama_tensors(), "visual.weight": ("F32", [256])}
    )
    write_json(str(Path(model) / "config.json"), {**llama_config(), "vision_config": {"hidden_size": 256}})
    code, out, _ = cli(monkeypatch, capsys, model, "--language-model-only", "-o", "json")
    assert code == 0
    lm = json.loads(out)
    code, out, _ = cli(monkeypatch, capsys, model, "--text-only", "-o", "json")
    assert code == 0 and json.loads(out)["weights"]["bytes"] - lm["weights"]["bytes"] == 256 * 4
    assert lm["workload"] == "language-model-only"


@pytest.mark.parametrize("flag", ["--language-model-only", "--no-mmproj", "--no-mmproj-auto"])
def test_projector_skip_aliases(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, flag: str
) -> None:
    write_gguf(str(tmp_path / "model-F16.gguf"), ARCH, TENSORS)
    write_gguf(str(tmp_path / "mmproj-model-F16.gguf"), {"clip.vision.embedding_length": 256}, TENSORS)
    code, out, _ = cli(
        monkeypatch, capsys, "--hf-repo", str(tmp_path), "--hf-file", "model-F16.gguf", flag, "-o", "json"
    )
    assert code == 0
    data = json.loads(out)
    assert data["format"] == "gguf" and data["workload"] == "language-model-only"
    assert data["components"] is None and data["overhead"]["encoder"] == 0
