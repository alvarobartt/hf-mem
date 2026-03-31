from hf_mem._print import (
    BORDERS_AND_PADDING,
    MAX_DATA_LEN,
    MAX_NAME_LEN,
    _bytes_to_gib,
    _make_bar,
    _print_centered,
    _print_divider,
    _print_full_divider,
    _print_header,
    _print_row,
    _print_with_color,
)
from hf_mem._version import __version__
from hf_mem.hardware.catalog import list_profiles
from hf_mem.hardware.types import FitnessResult


def print_fitness_report(fitness: FitnessResult, *, extended: bool = False) -> None:
    profile = fitness.profile
    total_vram_gib = _bytes_to_gib(profile.total_vram_bytes)
    gpu_count = len(profile.gpus)

    # Build centered header rows
    centered_rows = [
        "HARDWARE FITNESS CHECK",
        profile.name,
    ]
    if gpu_count > 1:
        per_gpu_gib = _bytes_to_gib(profile.gpus[0].vram_bytes)
        centered_rows.append(f"{gpu_count}x GPUs, {per_gpu_gib:.0f} GiB each, {total_vram_gib:.0f} GiB total")
    else:
        centered_rows.append(f"{total_vram_gib:.0f} GiB VRAM")

    # Build data rows to determine data column width
    model_gib = _bytes_to_gib(fitness.model_memory)
    total_gib = _bytes_to_gib(fitness.total_memory)
    headroom_gib = _bytes_to_gib(abs(fitness.headroom_bytes))
    headroom_sign = "" if fitness.headroom_bytes >= 0 else "-"

    status_text = "FITS" if fitness.fits else f"DOES NOT FIT (needs {total_gib:.2f} GiB)"

    data_rows = [
        status_text,
        f"{model_gib:.2f} GiB (weights)",
        f"{total_gib:.2f} GiB",
        f"{headroom_sign}{headroom_gib:.2f} GiB ({fitness.headroom_pct * 100:.1f}%)",
    ]
    for q in fitness.quantization_options:
        est_gib = _bytes_to_gib(q.estimated_total_bytes)
        label = "FITS" if q.fits else "NO"
        if q.gpu_count_needed > 1 and q.fits:
            label = f"FITS ({q.gpu_count_needed} GPUs)"
        data_rows.append(f"{est_gib:.2f} GiB  {label}")

    # Compute widths (same pattern as safetensors/print.py)
    max_centered_len = max(len(r) for r in centered_rows)
    max_data_len = max(len(r) for r in data_rows)
    min_width_for_data = MAX_NAME_LEN + max_data_len + 5
    current_len = max(max_centered_len, min_width_for_data, MAX_DATA_LEN)
    data_col_width = current_len + 2 * BORDERS_AND_PADDING - MAX_NAME_LEN - 5

    # Header
    _print_header(current_len, badge=f"hf-mem {__version__}")
    for row in centered_rows:
        _print_centered(row, current_len)

    # Status section
    _print_divider(data_col_width + 1, side="top")
    _print_row("STATUS", status_text, data_col_width)
    pct = min(fitness.total_memory / profile.total_vram_bytes * 100, 999) if profile.total_vram_bytes > 0 else 0
    bar_width = data_col_width - len(f" {pct:.0f}%")
    bar = _make_bar(fitness.total_memory, profile.total_vram_bytes, bar_width)
    _print_row("UTILIZATION", f"{bar} {pct:.0f}%", data_col_width)

    _print_divider(data_col_width + 1)
    _print_row("MODEL", f"{model_gib:.2f} GiB (weights)", data_col_width)
    if fitness.total_memory != fitness.model_memory:
        kv_gib = _bytes_to_gib(fitness.total_memory - fitness.model_memory)
        _print_row("KV CACHE", f"{kv_gib:.2f} GiB", data_col_width)
    _print_row("TOTAL", f"{total_gib:.2f} GiB", data_col_width)
    _print_row(
        "HEADROOM", f"{headroom_sign}{headroom_gib:.2f} GiB ({fitness.headroom_pct * 100:.1f}%)", data_col_width
    )

    if fitness.gpu_count_needed > 1:
        _print_row("MIN GPUs", f"{fitness.gpu_count_needed}x {profile.gpus[0].name}", data_col_width)

    # Quantization estimates
    if fitness.quantization_options:
        if extended:
            _print_divider(data_col_width + 1, side="top-continue")
            _print_centered("QUANTIZATION ESTIMATES", current_len)
            _print_divider(data_col_width + 1, side="top")

            # Find max lengths for alignment within the quant section
            max_qname_len = max(len(q.name) for q in fitness.quantization_options)
            bpw_labels = [f"{q.bits_per_weight}b" for q in fitness.quantization_options]
            max_bpw_len = max(len(l) for l in bpw_labels)

            for i, q in enumerate(fitness.quantization_options):
                est_gib = _bytes_to_gib(q.estimated_total_bytes)
                label = "FITS" if q.fits else "NO"
                if q.gpu_count_needed > 1 and q.fits:
                    label = f"FITS ({q.gpu_count_needed} GPUs)"

                padded_name = q.name + " " * (max_qname_len - len(q.name))
                _print_row(padded_name, f"{est_gib:.2f} GiB  {label}", data_col_width)

                bar = _make_bar(q.estimated_total_bytes, profile.total_vram_bytes, data_col_width)
                bpw_label = bpw_labels[i] + " " * (max_bpw_len - len(bpw_labels[i]))
                _print_row(bpw_label, bar, data_col_width)

                if i < len(fitness.quantization_options) - 1:
                    _print_divider(data_col_width + 1)
        else:
            # Compact mode: show only the minimum quantization required to fit
            # (first entry that fits = highest quality / least aggressive quantization)
            min_quant = next((q for q in fitness.quantization_options if q.fits), None)
            if min_quant is not None:
                _print_divider(data_col_width + 1)
                est_gib = _bytes_to_gib(min_quant.estimated_total_bytes)
                gpu_note = f" ({min_quant.gpu_count_needed} GPUs)" if min_quant.gpu_count_needed > 1 else ""
                _print_row(
                    "MIN QUANT",
                    f"{min_quant.name} ({min_quant.bits_per_weight}b) — {est_gib:.2f} GiB{gpu_note}",
                    data_col_width,
                )
                bar = _make_bar(min_quant.estimated_total_bytes, profile.total_vram_bytes, data_col_width)
                _print_row(f"{min_quant.bits_per_weight}b", bar, data_col_width)
            else:
                _print_divider(data_col_width + 1)
                most_aggressive = min(fitness.quantization_options, key=lambda q: q.estimated_total_bytes)
                est_gib = _bytes_to_gib(most_aggressive.estimated_total_bytes)
                _print_row(
                    "MIN QUANT",
                    f"Does not fit even at {most_aggressive.name} ({est_gib:.2f} GiB)",
                    data_col_width,
                )

    # Notes
    if fitness.notes:
        _print_full_divider(current_len, side="top-continue")
        for note in fitness.notes:
            _print_centered(note, current_len)

    # Bottom border
    if fitness.notes:
        _print_full_divider(current_len, side="bottom")
    else:
        _print_divider(data_col_width + 1, side="bottom")


def print_catalog() -> None:
    """Print the built-in hardware catalog as a table."""
    profiles = list_profiles()

    rows = []
    for key, p in profiles.items():
        gpu_count = len(p.gpus)
        total_gib = _bytes_to_gib(p.total_vram_bytes)
        gpu_desc = f"{gpu_count}x {p.gpus[0].name}" if gpu_count > 1 else p.gpus[0].name
        rows.append((key, gpu_desc, f"{total_gib:.0f} GiB", p.source))

    key_width = max(len(r[0]) for r in rows)
    gpu_width = max(len(r[1]) for r in rows)
    vram_width = max(len(r[2]) for r in rows)
    source_width = max(len(r[3]) for r in rows)

    header_key = "PROFILE".ljust(key_width)
    header_gpu = "GPU".ljust(gpu_width)
    header_vram = "VRAM".ljust(vram_width)
    header_source = "SOURCE".ljust(source_width)

    _print_with_color(f"  {header_key}  {header_gpu}  {header_vram}  {header_source}")
    _print_with_color(f"  {'─' * key_width}  {'─' * gpu_width}  {'─' * vram_width}  {'─' * source_width}")

    for key, gpu_desc, vram, source in rows:
        k = key.ljust(key_width)
        g = gpu_desc.ljust(gpu_width)
        v = vram.ljust(vram_width)
        s = source.ljust(source_width)
        _print_with_color(f"  {k}  {g}  {v}  {s}")
