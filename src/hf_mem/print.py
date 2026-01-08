import re
import warnings
from typing import Dict, Literal, Optional

from hf_mem.metadata import SafetensorsMetadata

MIN_NAME_LEN = 5
MAX_NAME_LEN = 14
MIN_DATA_LEN = 20
MAX_DATA_LEN = 64
BORDERS_AND_PADDING = 7

BOX = {
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "ht": "─",
    "vt": "│",
    "tsep": "┬",
    "bsep": "┴",
    "lm": "├",
    "rm": "┤",
    "mm": "┼",
}


def _print_with_color(content: str) -> None:
    print(f"\x1b[38;2;244;183;63m{content}\x1b[0m")


def _print_header(current_len: int, name_len: int = MAX_NAME_LEN) -> None:
    length = current_len + name_len + BORDERS_AND_PADDING
    top = BOX["tl"] + (BOX["tsep"] * (length - 2)) + BOX["tr"]
    _print_with_color(top)

    bottom = BOX["lm"] + (BOX["bsep"] * (length - 2)) + BOX["rm"]
    _print_with_color(bottom)


def _print_centered(text: str, current_len: int, name_len: int = MAX_NAME_LEN) -> None:
    max_len = current_len + name_len - BORDERS_AND_PADDING
    total_width = max_len + 12
    text_len = len(text)
    pad_left = (total_width - text_len) // 2
    pad_right = total_width - text_len - pad_left
    _print_with_color(f"{BOX['vt']}{' ' * pad_left}{text}{' ' * pad_right}{BOX['vt']}")


def _print_divider(
    current_len: int,
    side: Optional[Literal["top", "top-continue", "bottom", "bottom-continue"]] = None,
    name_len: int = MAX_NAME_LEN,
) -> None:
    match side:
        case "top":
            left, mid, right = BOX["lm"], BOX["tsep"], BOX["rm"]
        case "top-continue":
            left, mid, right = BOX["lm"], BOX["bsep"], BOX["rm"]
        case "bottom":
            left, mid, right = BOX["bl"], BOX["bsep"], BOX["br"]
        case "bottom-continue":
            left, mid, right = BOX["lm"], BOX["bsep"], BOX["rm"]
        case _:
            left, mid, right = BOX["lm"], BOX["mm"], BOX["rm"]

    name_col_inner = name_len + 2
    data_col_inner = current_len + 1

    line = left
    line += BOX["ht"] * name_col_inner
    line += mid
    line += BOX["ht"] * data_col_inner
    line += right
    _print_with_color(line)


def _format_name(name: str, name_len: int = MAX_NAME_LEN) -> str:
    if len(name) < MIN_NAME_LEN:
        return f"{name:<{MIN_NAME_LEN}}"
    if len(name) > name_len:
        return name[: name_len - 3] + "..."
    return f"{name:<{name_len}}"


def _print_row(name: str, text: str, current_len: int, name_len: int = MAX_NAME_LEN) -> None:
    name_fmt = _format_name(name, name_len)
    data_fmt = f"{str(text):<{current_len}}"
    _print_with_color(f"{BOX['vt']} {name_fmt} {BOX['vt']} {data_fmt} {BOX['vt']}")


def _make_bar(used: float, total: float, width: int) -> str:
    if total <= 0:
        return "░" * width
    frac = min(max(used / total, 0.0), 1.0)
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _format_short_number(n: float) -> str:
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0:
            return f"{int(n)}" if unit == "" else f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def _bytes_to_gb(nbytes: int, use_decimal: bool = False) -> float:
    """Convert bytes to gigabytes.

    Args:
        nbytes: Number of bytes
        use_decimal: If True, use GB (1e9), else use GiB (1024^3)
    """
    return nbytes / 1e9 if use_decimal else nbytes / (1024**3)


def print_report(
    model_id: str,
    revision: str,
    metadata: SafetensorsMetadata,
    ignore_table_width: bool = False,
) -> None:
    rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
        "TOTAL MEMORY",
        "REQUIREMENTS",
    ]

    for name, nested_metadata in metadata.components.items():
        if len(metadata.components) > 1:
            rows.append(name)

        for dtype, dtype_metadata in nested_metadata.dtypes.items():
            rows.append(f"{dtype} {dtype_metadata.param_count} {dtype_metadata.bytes_count}")

    max_len = max(len(str(r)) for r in rows)

    if max_len > MAX_DATA_LEN and ignore_table_width is False:
        warnings.warn(
            f"Given that the provided `--model-id {model_id}` (with `--revision {revision}`) is longer than {MAX_DATA_LEN} characters, the table width will be expanded to fit the provided values within their row, but it might lead to unexpected table views. If you'd like to ignore the limit, then provide the `--ignore-table-width` flag to ignore the {MAX_DATA_LEN} width limit, to simply accommodate to whatever the longest text length is."
        )

    current_len = min(max_len, MAX_DATA_LEN) if ignore_table_width is False else max_len

    _print_header(current_len)
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len)
    _print_divider(current_len + 1, "top")

    total_text = (
        f"{_bytes_to_gb(metadata.bytes_count):.2f} GB ({_format_short_number(metadata.param_count)} params)"
    )
    _print_row("TOTAL MEMORY", total_text, current_len)

    total_bar = _make_bar(metadata.bytes_count, metadata.bytes_count, current_len)
    _print_row("REQUIREMENTS", total_bar, current_len)

    for key, value in metadata.components.items():
        if len(metadata.components) > 1:
            _print_divider(current_len + 1, "top-continue")
            _print_centered(
                f"{key.upper()} ({_bytes_to_gb(value.bytes_count):.2f} GB)",
                current_len,
            )

            _print_divider(current_len + 1, "top")
        else:
            _print_divider(current_len + 1)

        max_length = max(
            [
                len(f"{_format_short_number(dtype_metadata.param_count)} PARAMS")
                for _, dtype_metadata in value.dtypes.items()
            ]
        )
        for idx, (dtype, dtype_metadata) in enumerate(value.dtypes.items()):
            gb_text = (
                f"{_bytes_to_gb(dtype_metadata.bytes_count):.2f} / {_bytes_to_gb(metadata.bytes_count):.2f} GB"
            )
            _print_row(
                dtype.upper() + " " * (max_length - len(dtype)),
                gb_text,
                current_len,
            )

            bar = _make_bar(
                _bytes_to_gb(dtype_metadata.bytes_count),
                _bytes_to_gb(metadata.bytes_count),
                current_len,
            )
            _print_row(
                f"{_format_short_number(dtype_metadata.param_count)} PARAMS",
                bar,
                current_len,
            )

            if idx < len(value.dtypes) - 1:
                _print_divider(current_len + 1)

    _print_divider(current_len + 1, "bottom")


def _group_gguf_files(gguf_files: Dict[str, int]) -> Dict[str, int]:
    """Group sharded GGUF files by model variant and sum their sizes.

    Files like 'BF16/model-00001-of-00010.gguf' are grouped together.
    For example, 'BF16/model-00001-of-00010.gguf' and 'BF16/model-00002-of-00010.gguf'
    are grouped under 'BF16/model.gguf'.
    Single files like 'model-Q4_K_M.gguf' remain as-is.
    Supports various shard formats: -1-of-2.gguf, -001-of-002.gguf, -00001-of-00010.gguf, etc.
    """
    grouped: Dict[str, int] = {}
    shard_pattern = re.compile(r"-\d+-of-\d+\.gguf$")

    for path, size in gguf_files.items():
        if shard_pattern.search(path):
            base = shard_pattern.sub(".gguf", path)
            grouped[base] = grouped.get(base, 0) + size
        else:
            grouped[path] = size

    return grouped


def print_report_for_gguf(
    model_id: str,
    revision: str,
    gguf_files: Dict[str, int],
    ignore_table_width: bool = False,
) -> None:
    """Print VRAM report for GGUF models.

    Args:
        model_id: HuggingFace model ID
        revision: Model revision
        gguf_files: Dict mapping filename to file size in bytes
        ignore_table_width: Whether to ignore max table width
    """
    grouped_files = _group_gguf_files(gguf_files)

    max_name_len = max(len(filename) for filename in grouped_files.keys())

    rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
    ]

    max_len = 0
    for r in rows:
        max_len = max(max_len, len(str(r)))

    if max_len > MAX_DATA_LEN and ignore_table_width is False:
        warnings.warn(
            f"Given that the provided `--model-id {model_id}` (with `--revision {revision}`) is longer than {MAX_DATA_LEN} characters, the table width will be expanded to fit the provided values within their row, but it might lead to unexpected table views. If you'd like to ignore the limit, then provide the `--ignore-table-width` flag to ignore the {MAX_DATA_LEN} width limit, to simply accommodate to whatever the longest text length is."
        )

    current_len = min(max_len, MAX_DATA_LEN) if ignore_table_width is False else max_len

    _print_header(current_len, max_name_len)
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len, max_name_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len, max_name_len)
    _print_divider(current_len + 1, "top", max_name_len)

    for i, (filename, size_bytes) in enumerate(grouped_files.items()):
        file_gb = _bytes_to_gb(size_bytes, use_decimal=True)
        _print_row(filename, f"{file_gb:.2f} GB", current_len, max_name_len)

        if i < len(grouped_files) - 1:
            _print_divider(current_len + 1, name_len=max_name_len)

    _print_divider(current_len + 1, "bottom", max_name_len)
