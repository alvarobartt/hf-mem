import math
import warnings
from typing import Any, Dict, Literal, Optional

MIN_NAME_LEN = 5
MAX_NAME_LEN = 13
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


def _print_header(current_len: int) -> None:
    length = current_len + MAX_NAME_LEN + BORDERS_AND_PADDING
    top = BOX["tl"] + (BOX["tsep"] * (length - 2)) + BOX["tr"]
    _print_with_color(top)

    bottom = BOX["lm"] + (BOX["bsep"] * (length - 2)) + BOX["rm"]
    _print_with_color(bottom)


def _print_centered(text: str, current_len: int) -> None:
    max_len = current_len + MAX_NAME_LEN - BORDERS_AND_PADDING
    total_width = max_len + 12
    text_len = len(text)
    pad_left = (total_width - text_len) // 2
    pad_right = total_width - text_len - pad_left
    _print_with_color(f"{BOX['vt']}{' ' * pad_left}{text}{' ' * pad_right}{BOX['vt']}")


def _print_divider(
    current_len: int,
    side: Optional[Literal["top", "top-continue", "bottom", "bottom-continue"]] = None,
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

    name_col_inner = MAX_NAME_LEN + 2
    data_col_inner = current_len + 1

    line = left
    line += BOX["ht"] * name_col_inner
    line += mid
    line += BOX["ht"] * data_col_inner
    line += right
    _print_with_color(line)


def _format_name(name: str) -> str:
    if len(name) < MIN_NAME_LEN:
        return f"{name:<{MIN_NAME_LEN}}"
    if len(name) > MAX_NAME_LEN:
        return name[: MAX_NAME_LEN - 3] + "..."
    return f"{name:<{MAX_NAME_LEN}}"


def _print_row(name: str, text: str, current_len: int) -> None:
    name_fmt = _format_name(name)
    data_fmt = f"{str(text):<{current_len}}"
    _print_with_color(f"{BOX['vt']} {name_fmt} {BOX['vt']} {data_fmt} {BOX['vt']}")


def _make_bar(used: float, total: float, width: int) -> str:
    if total <= 0:
        return "░" * width
    frac = min(max(used / total, 0.0), 1.0)
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _format_short_number(n: float) -> Optional[str]:
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0:
            return f"{int(n)}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000.0


def _bytes_to_gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def print_report_for_transformers(
    model_id: str,
    revision: str,
    metadata: Dict[str, Any],
    ignore_table_width: bool = False,
) -> None:
    ppdt = {}
    for key, value in metadata.items():
        if key in {"__metadata__"}:
            continue
        if value["dtype"] not in ppdt:
            ppdt[value["dtype"]] = (0, 0)

        match value["dtype"]:
            case "F64" | "I64" | "U64":
                dtype_b = 8
            case "F32" | "I32" | "U32":
                dtype_b = 4
            case "F16" | "BF16" | "I16" | "U16":
                dtype_b = 2
            case "F8_E5M2" | "F8_E4M3" | "I8" | "U8":
                dtype_b = 1
            case _:
                raise RuntimeError(f"DTYPE={value['dtype']} NOT HANDLED")

        current_shape = math.prod(value["shape"])
        current_shape_bytes = current_shape * dtype_b

        ppdt[value["dtype"]] = (
            ppdt[value["dtype"]][0] + current_shape,
            ppdt[value["dtype"]][1] + current_shape_bytes,
        )

    rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
        "TOTAL MEMORY",
        "REQUIREMENTS",
    ]
    for dt, (params, nbytes) in ppdt.items():
        rows.append(f"{dt} {params} {nbytes}")

    max_len = 0
    for r in rows:
        max_len = max(max_len, len(str(r)))

    if max_len > MAX_DATA_LEN and ignore_table_width is False:
        warnings.warn(
            f"Given that the provided `--model-id {model_id}` (with `--revision {revision}`) is longer than {MAX_DATA_LEN} characters, the table width will be expanded to fit the provided values within their row, but it might lead to unexpected table views. If you'd like to ignore the limit, then provide the `--ignore-table-width` flag to ignore the {MAX_DATA_LEN} width limit, to simply accommodate to whatever the longest text length is."
        )

    current_len = min(max_len, MAX_DATA_LEN) if ignore_table_width is False else max_len

    total_bytes = sum(nbytes for _, nbytes in ppdt.values())
    total_params = sum(params for params, _ in ppdt.values())
    total_gb = _bytes_to_gb(total_bytes)

    _print_header(current_len)
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len)
    _print_divider(current_len + 1, "top")

    total_text = f"{_bytes_to_gb(total_bytes):.2f} GB ({_format_short_number(total_params)} params)"
    _print_row("TOTAL MEMORY", total_text, current_len)

    total_bar = _make_bar(total_bytes, total_bytes, current_len)
    _print_row("REQUIREMENTS", total_bar, current_len)
    _print_divider(current_len + 1)

    max_length = max([
        len(f"{_format_short_number(params)} PARAMS") for params, _ in ppdt.values()
    ])
    for i, (dtype, (params, nbytes)) in enumerate(ppdt.items()):
        dtype_name = dtype.upper()
        dtype_gb = _bytes_to_gb(nbytes)

        gb_text = f"{dtype_gb:.1f} / {total_gb:.1f} GB"
        _print_row(
            dtype_name + " " * (max_length - len(dtype_name)),
            gb_text,
            current_len,
        )

        bar = _make_bar(dtype_gb, total_gb, current_len)
        _print_row(f"{_format_short_number(params)} PARAMS", bar, current_len)

        if i < len(ppdt) - 1:
            _print_divider(current_len + 1)

    _print_divider(current_len + 1, "bottom")


def print_report_for_diffusers(
    model_id: str,
    revision: str,
    metadata: Dict[str, Dict[str, Any]],
    ignore_table_width: bool = False,
) -> None:
    components_ppdt: Dict[str, Dict[str, tuple[int, int]]] = {}
    total_bytes = 0
    total_params = 0

    for path, path_metadata in metadata.items():
        ppdt: Dict[str, tuple[int, int]] = {}
        for key, value in path_metadata.items():
            if key in {"__metadata__"}:
                continue

            dtype = value["dtype"]
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
                    raise RuntimeError(f"DTYPE={dtype} NOT HANDLED")

            current_shape = math.prod(value["shape"])
            current_shape_bytes = current_shape * dtype_b

            if dtype not in ppdt:
                ppdt[dtype] = (0, 0)
            ppdt[dtype] = (
                ppdt[dtype][0] + current_shape,
                ppdt[dtype][1] + current_shape_bytes,
            )

        components_ppdt[path] = ppdt
        for params, nbytes in ppdt.values():
            total_params += params
            total_bytes += nbytes

    rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
        "TOTAL MEMORY",
        "REQUIREMENTS",
    ]
    for path, ppdt in components_ppdt.items():
        rows.append(path)
        for dt, (params, nbytes) in ppdt.items():
            rows.append(f"{dt} {params} {nbytes}")

    max_len = 0
    for r in rows:
        max_len = max(max_len, len(str(r)))

    if max_len > MAX_DATA_LEN and ignore_table_width is False:
        warnings.warn(
            f"Given that the provided `--model-id {model_id}` (with `--revision {revision}`) is longer than {MAX_DATA_LEN} characters, the table width will be expanded to fit the provided values within their row, but it might lead to unexpected table views. If you'd like to ignore the limit, then provide the `--ignore-table-width` flag to ignore the {MAX_DATA_LEN} width limit, to simply accommodate to whatever the longest text length is."
        )

    current_len = min(max_len, MAX_DATA_LEN) if ignore_table_width is False else max_len

    _print_header(current_len)
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len)
    _print_divider(current_len + 1, "top")

    total_text = f"{_bytes_to_gb(total_bytes):.2f} GB ({_format_short_number(total_params)} params)"
    _print_row("TOTAL MEMORY", total_text, current_len)

    total_bar = _make_bar(total_bytes, total_bytes, current_len)
    _print_row("REQUIREMENTS", total_bar, current_len)

    for path, ppdt in components_ppdt.items():
        _print_divider(current_len + 1, "top-continue")

        path_bytes = sum(nbytes for _, nbytes in ppdt.values())
        path_gb = _bytes_to_gb(path_bytes)

        _print_centered(f"{path.upper()} ({path_gb:.2f} GB)", current_len)
        _print_divider(current_len + 1, "top")

        max_length = max([
            len(f"{_format_short_number(params)} PARAMS") for params, _ in ppdt.values()
        ])
        for i, (dtype, (params, nbytes)) in enumerate(ppdt.items()):
            dtype_name = dtype.upper()
            dtype_gb = _bytes_to_gb(nbytes)

            gb_text = f"{dtype_gb:.1f} / {_bytes_to_gb(total_bytes):.1f} GB"
            _print_row(
                dtype_name + " " * (max_length - len(dtype_name)),
                gb_text,
                current_len,
            )

            bar = _make_bar(dtype_gb, _bytes_to_gb(total_bytes), current_len)
            _print_row(f"{_format_short_number(params)} PARAMS", bar, current_len)

            if i < len(ppdt) - 1:
                _print_divider(current_len + 1, "bottom-continue")

    _print_divider(current_len + 1, "bottom")
