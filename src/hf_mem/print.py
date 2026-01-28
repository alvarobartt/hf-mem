import warnings
from typing import Any, Dict, Literal, Optional

from hf_mem.metadata import SafetensorsMetadata

MIN_NAME_LEN = 5
MAX_NAME_LEN = 14
MIN_DATA_LEN = 20
MAX_DATA_LEN = 64
BORDERS_AND_PADDING = 4

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


def _print_header(current_len: int) -> None:
    length = current_len + 2 * BORDERS_AND_PADDING + 2
    top = BOX["tl"] + (BOX["tsep"] * (length - 2)) + BOX["tr"]
    _print_with_color(top)

    bottom = BOX["lm"] + (BOX["bsep"] * (length - 2)) + BOX["rm"]
    _print_with_color(bottom)


def _print_centered(text: str, current_len: int) -> None:
    total_width = current_len + 2 * BORDERS_AND_PADDING
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


def _format_name(name: str, max_len: int = MAX_NAME_LEN) -> str:
    if len(name) < MIN_NAME_LEN:
        return f"{name:<{MIN_NAME_LEN}}"
    if len(name) > max_len:
        return name[: max_len - 3] + "..."
    return f"{name:<{max_len}}"


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


def _bytes_to_gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def print_report(
    model_id: str,
    revision: str,
    metadata: SafetensorsMetadata,
    cache: Optional[Dict[str, Any]] = None,
    ignore_table_width: bool = False,
) -> None:
    rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
        f"w/ max-model-len={cache['max_model_len']}, batch-size={cache['batch_size']}"
        if cache is not None
        else "",
        "TOTAL MEMORY",
        "REQUIREMENTS",
    ]

    if cache:
        rows.append(f"{_bytes_to_gb(metadata.bytes_count + cache['cache_size']):.2f} GB")

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
    data_col_width = current_len + 2 * BORDERS_AND_PADDING - MAX_NAME_LEN - 5

    _print_header(current_len)
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len)
    if cache:
        _print_centered(
            f"w/ max-model-len={cache['max_model_len']}, batch-size={cache['batch_size']}",
            current_len,
        )
    _print_divider(data_col_width + 1, "top")

    if cache:
        combined_total = metadata.bytes_count + cache["cache_size"]
        total_text = f"{_bytes_to_gb(combined_total):.2f} GB ({_format_short_number(metadata.param_count)} PARAMS + KV CACHE)"
        total_bar = _make_bar(combined_total, combined_total, data_col_width)
        _print_row("TOTAL MEMORY", total_text, data_col_width)
        _print_row("REQUIREMENTS", total_bar, data_col_width)
    else:
        model_text = (
            f"{_bytes_to_gb(metadata.bytes_count):.2f} GB ({_format_short_number(metadata.param_count)} PARAMS)"
        )
        model_bar = _make_bar(metadata.bytes_count, metadata.bytes_count, data_col_width)
        _print_row("TOTAL MEMORY", model_text, data_col_width)
        _print_row("REQUIREMENTS", model_bar, data_col_width)

    for key, value in metadata.components.items():
        if len(metadata.components) > 1:
            _print_divider(data_col_width + 1, "top-continue")
            _print_centered(
                f"{key.upper()} ({_format_short_number(value.param_count)} PARAMS, {_bytes_to_gb(value.bytes_count):.2f} GB)",
                current_len,
            )
            _print_divider(data_col_width + 1, "top")
        elif cache:
            _print_divider(data_col_width + 1, "top-continue")
            _print_centered(
                f"MODEL ({_format_short_number(value.param_count)} PARAMS, {_bytes_to_gb(value.bytes_count):.2f} GB)",
                current_len,
            )
            _print_divider(data_col_width + 1, "top")
        else:
            _print_divider(data_col_width + 1)

        max_length = max(
            [
                len(f"{_format_short_number(dtype_metadata.param_count)} PARAMS")
                for _, dtype_metadata in value.dtypes.items()
            ]
        )
        for idx, (dtype, dtype_metadata) in enumerate(value.dtypes.items()):
            gb_text = f"{_bytes_to_gb(dtype_metadata.bytes_count):.2f} / {_bytes_to_gb(metadata.bytes_count if not cache else combined_total):.2f} GB"  # type: ignore
            _print_row(
                dtype.upper() + " " * (max_length - len(dtype)),
                gb_text,
                data_col_width,
            )

            bar = _make_bar(
                _bytes_to_gb(dtype_metadata.bytes_count),
                _bytes_to_gb(metadata.bytes_count if not cache else combined_total),  # type: ignore
                data_col_width,
            )
            _print_row(
                f"{_format_short_number(dtype_metadata.param_count)} PARAMS",
                bar,
                data_col_width,
            )

            if idx < len(value.dtypes) - 1:
                _print_divider(data_col_width + 1)

    if cache:
        _print_divider(data_col_width + 1, "top-continue")
        _print_centered(
            f"KV CACHE ({cache['max_model_len'] * cache['batch_size']} TOKENS, {_bytes_to_gb(cache['cache_size']):.2f} GB)",
            current_len,
        )
        _print_divider(data_col_width + 1, "top")

        kv_text = f"{_bytes_to_gb(cache['cache_size']):.2f} / {_bytes_to_gb(combined_total):.2f} GB"  # type: ignore
        _print_row(
            cache["cache_dtype"].upper() + " " * (max_length - len(cache["cache_dtype"])),  # type: ignore
            kv_text,
            data_col_width,
        )

        kv_bar = _make_bar(cache["cache_size"], combined_total, data_col_width)  # type: ignore
        _print_row(
            f"{cache['max_model_len'] * cache['batch_size']} TOKENS",
            kv_bar,
            data_col_width,
        )

    _print_divider(data_col_width + 1, "bottom")
