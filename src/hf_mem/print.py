MIN_NAME_LEN = 5
MAX_NAME_LEN = 13
MIN_DATA_LEN = 20
MAX_DATA_LEN = 32
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


def print_color(content: str) -> None:
    print(f"\x1b[38;2;244;183;63m{content}\x1b[0m")


def _max_length(strings):
    max_len = 0
    for s in strings:
        max_len = max(max_len, len(str(s)))
    return min(max_len, MAX_DATA_LEN)


def _set_current_len(model_id, stats):
    rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"`{model_id}`",
        "TOTAL MEMORY",
        "MEMORY REQUIREMENTS",
    ]
    for dt, (params, nbytes) in stats.items():
        rows.append(f"{dt} {params} {nbytes}")
    return _max_length(rows)


def _print_header(current_len):
    length = current_len + MAX_NAME_LEN + BORDERS_AND_PADDING
    top = BOX["tl"] + (BOX["tsep"] * (length - 2)) + BOX["tr"]
    bottom = BOX["lm"] + (BOX["bsep"] * (length - 2)) + BOX["rm"]
    print_color(top)
    print_color(bottom)


def _print_centered(text, current_len):
    max_len = current_len + MAX_NAME_LEN - BORDERS_AND_PADDING
    total_width = max_len + 12
    text_len = len(text)
    pad_left = (total_width - text_len) // 2
    pad_right = total_width - text_len - pad_left
    print_color(f"{BOX['vt']}{' ' * pad_left}{text}{' ' * pad_right}{BOX['vt']}")


def _print_divider(current_len, side=None):
    if side == "top":
        left, mid, right = BOX["lm"], BOX["tsep"], BOX["rm"]
    elif side == "bottom":
        left, mid, right = BOX["bl"], BOX["bsep"], BOX["br"]
    else:
        left, mid, right = BOX["lm"], BOX["mm"], BOX["rm"]

    name_col_inner = MAX_NAME_LEN + 2
    data_col_inner = current_len + 1

    line = left
    line += BOX["ht"] * name_col_inner
    line += mid
    line += BOX["ht"] * data_col_inner
    line += right
    print_color(line)


def _format_name(name):
    name = str(name)
    if len(name) < MIN_NAME_LEN:
        return f"{name:<{MIN_NAME_LEN}}"
    if len(name) > MAX_NAME_LEN:
        return name[: MAX_NAME_LEN - 3] + "..."
    return f"{name:<{MAX_NAME_LEN}}"


def _print_row(name, text, current_len):
    name_fmt = _format_name(name)
    data_fmt = f"{str(text):<{current_len}}"
    print_color(f"{BOX['vt']} {name_fmt} {BOX['vt']} {data_fmt} {BOX['vt']}")


def _make_bar(used, total, width):
    if total <= 0:
        return "░" * width
    frac = min(max(used / total, 0.0), 1.0)
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _format_short_number(n):
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0 or unit == "T":
            if unit == "":
                return f"{int(n)}"
            return f"{n:.1f}{unit}"
        n /= 1000.0


def bytes_to_gb(nbytes):
    return nbytes / (1024**3)


def print_report(model_id, stats):
    current_len = _set_current_len(model_id, stats)
    bar_width = current_len

    total_bytes = sum(nbytes for _, nbytes in stats.values())
    total_params = sum(params for params, _ in stats.values())
    total_gb = bytes_to_gb(total_bytes)

    _print_header(current_len)
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"`{model_id}`", current_len)
    _print_divider(current_len + 1, "top")

    total_text = f"{bytes_to_gb(total_bytes):.2f} GB ({_format_short_number(total_params)} params)"
    _print_row("MEMORY", total_text, current_len)

    total_bar = _make_bar(total_bytes, total_bytes, bar_width)
    _print_row("REQUIREMENTS", total_bar, current_len)
    _print_divider(current_len + 1)

    max_dtype_length = len("FLOAT16")
    for i, (dtype, (params, nbytes)) in enumerate(stats.items()):
        dtype_name = dtype.upper()
        dtype_gb = bytes_to_gb(nbytes)

        gb_text = f"{dtype_gb:.1f} / {total_gb:.1f} GB"
        _print_row(
            dtype_name + " " * (max_dtype_length - len(dtype_name)),
            gb_text,
            current_len,
        )

        bar = _make_bar(dtype_gb, total_gb, bar_width)
        _print_row(f"{_format_short_number(params)} PARAMS", bar, current_len)

        if i < len(stats) - 1:
            _print_divider(current_len + 1)

    _print_divider(current_len + 1, "bottom")
