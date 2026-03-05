import warnings
from typing import Any, Dict, Optional

from hf_mem._print import (
    BORDERS_AND_PADDING,
    MAX_DATA_LEN,
    MAX_NAME_LEN,
    _bytes_to_gib,
    _format_short_number,
    _make_bar,
    _print_centered,
    _print_divider,
    _print_header,
    _print_row,
    _print_with_color,
)
from hf_mem._version import __version__
from hf_mem.safetensors.metadata import SafetensorsMetadata


def print_safetensors_report(
    model_id: str,
    revision: str,
    metadata: SafetensorsMetadata,
    cache: Optional[Dict[str, Any]] = None,
    ignore_table_width: bool = False,
) -> None:
    combined_total = metadata.bytes_count + cache["cache_size"] if cache else metadata.bytes_count

    centered_rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
    ]
    if cache:
        centered_rows.append(f"w/ max-model-len={cache['max_model_len']}, batch-size={cache['batch_size']}")
    for name, nested_metadata in metadata.components.items():
        if len(metadata.components) > 1:
            centered_rows.append(
                f"{name.upper()} ({_format_short_number(nested_metadata.param_count)} PARAMS, {_bytes_to_gib(nested_metadata.bytes_count):.2f} GiB)"
            )
        elif cache:
            centered_rows.append(
                f"MODEL ({_format_short_number(nested_metadata.param_count)} PARAMS, {_bytes_to_gib(nested_metadata.bytes_count):.2f} GiB)"
            )
    if cache:
        centered_rows.append(
            f"KV CACHE ({cache['max_model_len'] * cache['batch_size']} TOKENS, {_bytes_to_gib(cache['cache_size']):.2f} GiB)"
        )

    data_rows = []
    if cache:
        data_rows.append(
            f"{_bytes_to_gib(combined_total):.2f} GiB ({_format_short_number(metadata.param_count)} PARAMS + KV CACHE)"
        )
    else:
        data_rows.append(
            f"{_bytes_to_gib(metadata.bytes_count):.2f} GiB ({_format_short_number(metadata.param_count)} PARAMS)"
        )
    for _, nested_metadata in metadata.components.items():
        for dtype, dtype_metadata in nested_metadata.dtypes.items():
            data_rows.append(
                f"{_bytes_to_gib(dtype_metadata.bytes_count):.2f} / {_bytes_to_gib(combined_total):.2f} GiB"
            )
    if cache:
        data_rows.append(f"{_bytes_to_gib(cache['cache_size']):.2f} / {_bytes_to_gib(combined_total):.2f} GiB")

    max_centered_len = max(len(r) for r in centered_rows)
    max_data_len = max(len(r) for r in data_rows)

    min_width_for_data = MAX_NAME_LEN + max_data_len + 5
    max_len = max(max_centered_len, min_width_for_data)

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

    _print_row("VERSION", f"hf-mem {__version__}", data_col_width)
    _print_divider(data_col_width + 1)

    if cache:
        total_text = f"{_bytes_to_gib(combined_total):.2f} GiB ({_format_short_number(metadata.param_count)} PARAMS + KV CACHE)"
        total_bar = _make_bar(combined_total, combined_total, data_col_width)
        _print_row("TOTAL MEMORY", total_text, data_col_width)
        _print_row("REQUIREMENTS", total_bar, data_col_width)
    else:
        model_text = f"{_bytes_to_gib(metadata.bytes_count):.2f} GiB ({_format_short_number(metadata.param_count)} PARAMS)"
        model_bar = _make_bar(metadata.bytes_count, metadata.bytes_count, data_col_width)
        _print_row("TOTAL MEMORY", model_text, data_col_width)
        _print_row("REQUIREMENTS", model_bar, data_col_width)

    max_length = 0
    for key, value in metadata.components.items():
        if len(metadata.components) > 1:
            _print_divider(data_col_width + 1, "top-continue")
            _print_centered(
                f"{key.upper()} ({_format_short_number(value.param_count)} PARAMS, {_bytes_to_gib(value.bytes_count):.2f} GiB)",
                current_len,
            )
            _print_divider(data_col_width + 1, "top")
        elif cache:
            _print_divider(data_col_width + 1, "top-continue")
            _print_centered(
                f"MODEL ({_format_short_number(value.param_count)} PARAMS, {_bytes_to_gib(value.bytes_count):.2f} GiB)",
                current_len,
            )
            _print_divider(data_col_width + 1, "top")
        else:
            _print_divider(data_col_width + 1)

        max_length = max(
            len(f"{_format_short_number(dtype_metadata.param_count)} PARAMS")
            for _, dtype_metadata in value.dtypes.items()
        )
        for idx, (dtype, dtype_metadata) in enumerate(value.dtypes.items()):
            gib_text = (
                f"{_bytes_to_gib(dtype_metadata.bytes_count):.2f} / {_bytes_to_gib(combined_total):.2f} GiB"
            )
            # NOTE: dtype is a string key in safetensors (e.g. "F16", "BF16")
            _print_row(
                dtype.upper() + " " * (max_length - len(dtype)),
                gib_text,
                data_col_width,
            )

            bar = _make_bar(
                _bytes_to_gib(dtype_metadata.bytes_count),
                _bytes_to_gib(combined_total),
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
            f"KV CACHE ({cache['max_model_len'] * cache['batch_size']} TOKENS, {_bytes_to_gib(cache['cache_size']):.2f} GiB)",
            current_len,
        )
        _print_divider(data_col_width + 1, "top")

        kv_text = f"{_bytes_to_gib(cache['cache_size']):.2f} / {_bytes_to_gib(combined_total):.2f} GiB"
        _print_row(
            cache["cache_dtype"].upper() + " " * (max_length - len(cache["cache_dtype"])),  # type: ignore
            kv_text,
            data_col_width,
        )

        kv_bar = _make_bar(cache["cache_size"], combined_total, data_col_width)
        _print_row(
            f"{cache['max_model_len'] * cache['batch_size']} TOKENS",
            kv_bar,
            data_col_width,
        )

    _print_divider(data_col_width + 1, "bottom")
