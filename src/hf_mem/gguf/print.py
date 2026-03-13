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
    _print_full_divider,
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
) -> None:
    combined_total = metadata.bytes_count + kv_cache.cache_size if kv_cache else metadata.bytes_count

    centered_rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
        f"FOR `{filename}`",
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

    if max_len > MAX_DATA_LEN:
        warnings.warn(
            f"Given that the provided `--model-id {model_id}` (with `--revision {revision}`) is longer than {MAX_DATA_LEN} characters, the table width will be expanded to fit the provided values within their row, but it might lead to unexpected table views."
        )

    current_len = max_len
    data_col_width = current_len + 2 * BORDERS_AND_PADDING - MAX_NAME_LEN - 5

    _print_header(current_len, badge=f"hf-mem v{__version__}")
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len)
    _print_centered(f"FOR `{filename}`", current_len)
    if kv_cache:
        _print_centered(
            f"w/ max-model-len={kv_cache.max_model_len}, batch-size={kv_cache.batch_size}",
            current_len,
        )
    _print_divider(data_col_width + 1, "top")

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
) -> None:
    kv_cache_config = next((meta.kv_cache for meta in gguf_files.values() if meta.kv_cache is not None), None)

    centered_rows = [
        "INFERENCE MEMORY ESTIMATE FOR",
        f"https://hf.co/{model_id} @ {revision}",
    ]
    if kv_cache_config is not None:
        centered_rows.append(
            f"w/ max-model-len={kv_cache_config.max_model_len}, batch-size={kv_cache_config.batch_size}"
        )

    file_rows = []
    for filename, meta in gguf_files.items():
        transformer = meta.components.get("Transformer")
        if not transformer:
            continue

        weights_bytes = memory[filename]
        kv_cache_bytes = kv_cache.get(filename, 0) if kv_cache else 0
        total_bytes = weights_bytes + kv_cache_bytes
        details = f"{_bytes_to_gib(total_bytes):.2f} GiB"
        if kv_cache_bytes:
            details += (
                f" ({_bytes_to_gib(weights_bytes):.2f} GiB WEIGHTS + "
                f"{_bytes_to_gib(kv_cache_bytes):.2f} GiB KV)"
            )

        file_rows.append((filename, details, total_bytes))

    max_total_bytes = max(total_bytes for _, _, total_bytes in file_rows)

    max_name_length = min(max(len(fn) for fn in gguf_files.keys()), MAX_DATA_LEN)
    all_data_texts = [details for _, details, _ in file_rows]
    max_data_text_len = max(len(t) for t in all_data_texts)

    min_width_for_data = max_name_length + max_data_text_len + 5
    max_centered_len = max(max(len(r) for r in centered_rows), len("FILES"))
    max_len = max(max_centered_len, min_width_for_data)

    if max_len > MAX_DATA_LEN:
        warnings.warn(
            f"Given that the provided `--model-id {model_id}` (with `--revision {revision}`) is longer than {MAX_DATA_LEN} characters, the table width will be expanded to fit the provided values within their row, but it might lead to unexpected table views."
        )

    current_len = max_len
    data_col_width = current_len + 2 * BORDERS_AND_PADDING - max_name_length - 5

    _print_header(current_len, badge=f"hf-mem v{__version__}")
    _print_centered("INFERENCE MEMORY ESTIMATE FOR", current_len)
    _print_centered(f"https://hf.co/{model_id} @ {revision}", current_len)
    if kv_cache_config is not None:
        _print_centered(
            f"w/ max-model-len={kv_cache_config.max_model_len}, batch-size={kv_cache_config.batch_size}",
            current_len,
        )
    _print_full_divider(current_len, "top")
    _print_centered("FILES", current_len)
    _print_divider(data_col_width + 1, "top", name_len=max_name_length)

    for i, (filename, details, total_bytes) in enumerate(file_rows):
        _print_row(" " * max_name_length, details, data_col_width, name_len=max_name_length)
        _print_row(
            filename,
            _make_bar(total_bytes, max_total_bytes, data_col_width),
            data_col_width,
            name_len=max_name_length,
        )
        if i < len(file_rows) - 1:
            _print_divider(data_col_width + 1, name_len=max_name_length)
        else:
            _print_divider(data_col_width + 1, "bottom", name_len=max_name_length)
