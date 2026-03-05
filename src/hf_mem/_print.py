from typing import Literal

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
    side: Literal["top", "top-continue", "mid", "bottom", "bottom-continue"] | None = None,
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
        case "mid":
            left, mid, right = BOX["lm"], BOX["ht"], BOX["rm"]
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


def _bytes_to_gib(nbytes: int) -> float:
    return nbytes / (1024**3)
