import warnings
from typing import Dict

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
)
from hf_mem._types import KvCache
from hf_mem._version import __version__
from hf_mem.gguf.metadata import GGUFMetadata


def print_gguf_report(
    model_id: str,
    filename: str,
    revision: str,
    metadata: GGUFMetadata,
    kv_cache: KvCache | None = None,
    ignore_table_width: bool = False,
) -> None:
    combined_total = metadata.bytes_count + kv_cache.cache_size if kv_cache else metadata.bytes_count

    centered_rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
        f"FOR FILE {filename}",
    ]
    if kv_cache:
        centered_rows.append(f"w/ max-model-len={kv_cache.max_model_len}, batch-size={kv_cache.batch_size}")
    for name, nested_metadata in metadata.components.items():
        if len(metadata.components) > 1:
            centered_rows.append(
                f"{name.upper()} ({_format_short_number(nested_metadata.param_count)} PARAMS, {_bytes_to_gib(nested_metadata.bytes_count):.2f} GiB)"
            )
        elif kv_cache:
            centered_rows.append(
                f"MODEL ({_format_short_number(nested_metadata.param_count)} PARAMS, {_bytes_to_gib(nested_metadata.bytes_count):.2f} GiB)"
            )
    if kv_cache:
        centered_rows.append(
            f"KV CACHE ({kv_cache.max_model_len * kv_cache.batch_size} TOKENS, {_bytes_to_gib(kv_cache.cache_size):.2f} GiB)"
        )

    data_rows = []
    if kv_cache:
        data_rows.append(
            f"{_bytes_to_gib(combined_total):.2f} GiB ({_format_short_number(metadata.param_count)} PARAMS + KV CACHE)"
        )
    else:
        data_rows.append(
            f"{_bytes_to_gib(metadata.bytes_count):.2f} GiB ({_format_short_number(metadata.param_count)} PARAMS)"
        )
    for _, nested_metadata in metadata.components.items():
        for _, dtype_metadata in nested_metadata.dtypes.items():
            data_rows.append(
                f"{_bytes_to_gib(dtype_metadata.bytes_count):.2f} / {_bytes_to_gib(combined_total):.2f} GiB"
            )
    if kv_cache:
        data_rows.append(f"{_bytes_to_gib(kv_cache.cache_size):.2f} / {_bytes_to_gib(combined_total):.2f} GiB")

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
    _print_centered(f"FOR FILE {filename}", current_len)
    if kv_cache:
        _print_centered(
            f"w/ max-model-len={kv_cache.max_model_len}, batch-size={kv_cache.batch_size}",
            current_len,
        )
    _print_divider(data_col_width + 1, "top")

    _print_row("VERSION", f"hf-mem {__version__}", data_col_width)
    _print_divider(data_col_width + 1)

    if kv_cache:
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
        elif kv_cache:
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
            _print_row(
                dtype.name.upper() + " " * (max_length - len(dtype.name)),
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

    if kv_cache:
        _print_divider(data_col_width + 1, "top-continue")
        _print_centered(
            f"KV CACHE ({kv_cache.max_model_len * kv_cache.batch_size} TOKENS, {_bytes_to_gib(kv_cache.cache_size):.2f} GiB)",
            current_len,
        )
        _print_divider(data_col_width + 1, "top")

        kv_text = f"{_bytes_to_gib(kv_cache.cache_size):.2f} / {_bytes_to_gib(combined_total):.2f} GiB"
        _print_row(
            kv_cache.cache_dtype.upper() + " " * (max_length - len(kv_cache.cache_dtype)),  # type: ignore[operator]
            kv_text,
            data_col_width,
        )
        kv_bar = _make_bar(kv_cache.cache_size, combined_total, data_col_width)
        _print_row(
            f"{kv_cache.max_model_len * kv_cache.batch_size} TOKENS",
            kv_bar,
            data_col_width,
        )

    _print_divider(data_col_width + 1, "bottom")


def print_gguf_files_report(
    model_id: str,
    revision: str,
    gguf_files: Dict[str, GGUFMetadata],
    memory: Dict[str, int],
    kv_cache: Dict[str, int] | None = None,
    ignore_table_width: bool = False,
) -> None:
    # NOTE: All GGUF files in the same repo share the same architecture, so param count and
    # KV cache config are taken from the first entry as representative for all
    first_meta = next(iter(gguf_files.values()))
    kv_cache_config: KvCache | None = first_meta.kv_cache

    # NOTE: Build the rows that determine the minimum required table width
    centered_rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
    ]
    if kv_cache_config is not None:
        centered_rows.append(
            f"w/ max-model-len={kv_cache_config.max_model_len}, batch-size={kv_cache_config.batch_size}"
        )

    # NOTE: Pre-compute GiB strings to measure max data column width
    param_text = _format_short_number(first_meta.param_count)
    file_gib_rows = []
    for filename, meta in gguf_files.items():
        transformer = meta.components.get("Transformer")
        if transformer:
            total_bytes = memory[filename] + (kv_cache[filename] if kv_cache else 0)
            suffix = " (WEIGHTS + KV CACHE)" if kv_cache else ""
            file_gib_rows.append((filename, f"{_bytes_to_gib(total_bytes):.2f} GiB{suffix}"))

    params_data = f"{param_text} PARAMS"
    kv_data = (
        f"{_bytes_to_gib(kv_cache_config.cache_size):.2f} GiB ({kv_cache_config.max_model_len * kv_cache_config.batch_size} TOKENS)"
        if kv_cache_config
        else None
    )

    max_name_length = min(max(len(fn) for fn in gguf_files.keys()), MAX_DATA_LEN)
    all_data_texts = [params_data] + ([kv_data] if kv_data else []) + [gib for _, gib in file_gib_rows]
    max_data_text_len = max(len(t) for t in all_data_texts)

    min_width_for_data = max_name_length + max_data_text_len + 5
    max_centered_len = max(len(r) for r in centered_rows)
    max_len = max(max_centered_len, min_width_for_data)

    if max_len > MAX_DATA_LEN and ignore_table_width is False:
        warnings.warn(
            f"Given that the provided `--model-id {model_id}` (with `--revision {revision}`) is longer than {MAX_DATA_LEN} characters, the table width will be expanded to fit the provided values within their row, but it might lead to unexpected table views. If you'd like to ignore the limit, then provide the `--ignore-table-width` flag to ignore the {MAX_DATA_LEN} width limit, to simply accommodate to whatever the longest text length is."
        )

    current_len = min(max_len, MAX_DATA_LEN) if ignore_table_width is False else max_len
    data_col_width = current_len + 2 * BORDERS_AND_PADDING - max_name_length - 5

    _print_header(current_len)
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len)
    if kv_cache_config is not None:
        _print_centered(
            f"w/ max-model-len={kv_cache_config.max_model_len}, batch-size={kv_cache_config.batch_size}",
            current_len,
        )
    _print_divider(data_col_width + 1, "top", name_len=max_name_length)
    _print_row("VERSION", f"hf-mem {__version__}", data_col_width, name_len=max_name_length)
    _print_divider(data_col_width + 1, name_len=max_name_length)

    # NOTE: Show param count and (optionally) KV cache config once before the file listing,
    # so each file row only needs to show the byte total without repeating the same info
    _print_row("PARAMS", params_data, data_col_width, name_len=max_name_length)
    if kv_data is not None:
        _print_divider(data_col_width + 1, name_len=max_name_length)
        _print_row("KV CACHE", kv_data, data_col_width, name_len=max_name_length)

    # NOTE: "top" divider acts as a visual section break between the header metadata rows
    # above and the per-file listing below
    _print_divider(data_col_width + 1, "top", name_len=max_name_length)

    for i, (filename, gib_text) in enumerate(file_gib_rows):
        _print_row(
            filename + " " * (max_name_length - len(filename)),
            gib_text,
            data_col_width,
            name_len=max_name_length,
        )
        if i < len(file_gib_rows) - 1:
            _print_divider(data_col_width + 1, name_len=max_name_length)
        else:
            _print_divider(data_col_width + 1, "bottom", name_len=max_name_length)
