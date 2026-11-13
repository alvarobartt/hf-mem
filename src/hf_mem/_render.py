import os
import re
import sys

from hf_mem._version import __version__
from hf_mem._weights import Weights
from hf_mem.estimate import Estimate, KVCache, Listing

NAME_WIDTH = 15
MAX_FILE_WIDTH = 64
MAX_TITLE_WIDTH = 72

Item = tuple[str, str, str, float]
Section = tuple[str | None, list[Item]]


def gib(n: int) -> str:
    return f"{n / 1024**3:.2f}"


def short(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{int(n)}" if unit == "" else f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def _bar(fraction: float, width: int) -> str:
    eighths = round(min(max(fraction, 0.0), 1.0) * width * 8)
    full, rest = divmod(max(eighths, 1) if fraction > 0 else eighths, 8)
    bar = "█" * full + ("▏▎▍▌▋▊▉"[rest - 1] if rest else "")
    return bar + "░" * (width - len(bar))


def _fit(name: str, width: int) -> str:
    return (name[: width - 3] + "...") if len(name) > width else f"{name:<{width}}"


def _draw(titles: list[str], sections: list[Section], name_width: int) -> list[str]:
    texts = [t for _, items in sections for _, t, _, _ in items]
    inner = max(
        max(map(len, titles + [s for s, _ in sections if s])) + 8, name_width + max(map(len, texts)) + 7
    )
    data = inner - name_width - 5
    badge = f" hf-mem v{__version__} "
    lines: list[str] = ["┌" + "┬" * (inner - len(badge)) + badge + "┐", "├" + "┴" * inner + "┤"]
    lines += [f"│{t:^{inner}}│" for t in titles]

    def divider(left: str, mid: str, right: str) -> str:
        return left + "─" * (name_width + 2) + mid + "─" * (data + 2) + right

    for index, (title, items) in enumerate(sections):
        if title is not None:
            if index:
                lines.append(divider("├", "┴", "┤"))
            else:
                lines.append("├" + "─" * inner + "┤")
            lines.append(f"│{title:^{inner}}│")
            lines.append(divider("├", "┬", "┤"))
        elif index == 0:
            lines.append(divider("├", "┬", "┤"))
        else:
            lines.append(divider("├", "┼", "┤"))
        for i, (top, text, bottom, fraction) in enumerate(items):
            if i:
                lines.append(divider("├", "┼", "┤"))
            lines.append(f"│ {_fit(top, name_width)} │ {text:<{data}} │")
            lines.append(f"│ {_fit(bottom, name_width)} │ {_bar(fraction, data)} │")
    lines.append(divider("└", "┴", "┘"))
    return lines


def _shorten(text: str, width: int = MAX_TITLE_WIDTH) -> str:
    return text if len(text) <= width else "..." + text[3 - width :]


def _where(url: str, revision: str | None) -> str:
    if os.path.isabs(url):
        home = os.path.expanduser("~")
        try:
            relative = os.path.relpath(url)
        except ValueError:
            relative = ".."
        if not relative.startswith(".."):
            url = relative
        elif url.startswith(home + os.sep):
            url = "~" + url[len(home) :]
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = revision[:7]
    return _shorten(f"{url} @ {revision}" if revision else url)


def _titles(url: str, revision: str | None, file: str | None, kv: KVCache | None) -> list[str]:
    titles = ["INFERENCE MEMORY ESTIMATE FOR", _where(url, revision)]
    if file:
        titles.append(_shorten(f"FOR `{file}`"))
    if kv is not None:
        titles.append(f"w/ max-model-len={kv.max_model_len}, batch-size={kv.batch_size}")
    return titles


def _dtype_items(weights: Weights, total: int) -> list[Item]:
    width = max(len(f"{short(w.params)} PARAMS") for _, w in weights.sorted_dtypes())
    return [
        (
            f"{dtype:<{width}}",
            f"{gib(w.bytes)} / {gib(total)} GiB",
            f"{short(w.params)} PARAMS",
            w.bytes / total,
        )
        for dtype, w in weights.sorted_dtypes()
    ]


def table(estimate: Estimate | Listing, color: bool = False) -> str:
    if isinstance(estimate, Listing):
        return _listing(estimate, color)
    total, kv, weights = estimate.total or 1, estimate.kv_cache, estimate.weights
    offloaded = estimate.offloaded
    all_params = weights.params + (offloaded.params if offloaded else 0)
    params = f"{short(all_params)} PARAMS" if weights.params else "FROM FILE SIZES"
    parts = [params] + (["KV CACHE"] if kv else []) + (["OVERHEAD"] if estimate.overhead else [])
    summary = f"{gib(estimate.total)} GiB ({' + '.join(parts)})"
    requirements: list[Item] = []
    if (recommended := estimate.recommended_gpu_memory) is not None:
        ranks = (estimate.fit.gpus or 1) if estimate.fit else 1
        topology = f" ACROSS {ranks} GPUS" if ranks > 1 else " ON ONE GPU"
        capacity = f"{gib(recommended)} GiB ({recommended / 10**9:.2f} GB){topology}"
        requirements.append(("GPU CAPACITY", capacity, "RECOMMENDED", 1.0))
    requirements.append(("TOTAL MEMORY", summary, "ALLOCATIONS", 1.0))
    if estimate.startup_reserve_per_gpu:
        requirements.append(
            (
                "STARTUP RESERVE",
                f"{gib(estimate.startup_reserve_per_gpu)} GiB PER GPU (LANGUAGE-ONLY INITIALIZATION)",
                "",
                0.0,
            )
        )
    sections: list[Section] = [(None, requirements)]
    if offloaded:
        cpu = f"{gib(offloaded.bytes)} GiB ({short(offloaded.params)} PARAMS OFFLOADED, NOT ON GPU)"
        sections[0][1].append(
            ("CPU MEMORY", cpu, "HOST MEMORY", offloaded.bytes / (estimate.total + offloaded.bytes))
        )

    def heading(name: str, w: Weights) -> str:
        return (
            f"{name.upper()} ({short(w.params)} PARAMS, {gib(w.bytes)} GiB)"
            if w.params
            else f"{name.upper()} ({gib(w.bytes)} GiB)"
        )

    if estimate.moe is not None:
        moe = estimate.moe
        sections.append((heading("MODEL", moe.base), _dtype_items(moe.base, total)))
        experts = (
            f"EXPERTS ({moe.experts} x {short(moe.expert.params)} PARAMS, {gib(moe.experts_total.bytes)} GiB)"
        )
        sections.append((experts, _dtype_items(moe.experts_total, total)))
    else:
        many = len(estimate.components) > 1
        for name, component in (estimate.components or {"MODEL": weights}).items():
            title = heading(name if many else "MODEL", component)
            if component.dtypes:
                sections.append((title, _dtype_items(component, total)))
            elif many or kv:
                sections.append(
                    (
                        title,
                        [
                            (
                                "FILE SIZE",
                                f"{gib(component.bytes)} / {gib(total)} GiB",
                                "",
                                component.bytes / total,
                            )
                        ],
                    )
                )
    if kv is not None:
        tokens = kv.max_model_len * kv.batch_size
        title = f"KV CACHE ({tokens} TOKENS, {gib(kv.bytes)} GiB)"
        sections.append(
            (title, [(kv.dtype, f"{gib(kv.bytes)} / {gib(total)} GiB", f"{tokens} TOKENS", kv.bytes / total)])
        )
    if overhead := estimate.overhead:
        items = [
            (name, f"{gib(value)} / {gib(total)} GiB", "", value / total)
            for name, value in (
                ("ACTIVATIONS", overhead.activations),
                ("ENCODER", overhead.encoder),
                ("GRAPHS", overhead.graphs),
                ("RUNTIME", overhead.runtime),
                ("REPACK/REPLICA", overhead.weight_adjustment),
            )
            if value
        ]
        sections.append((f"OVERHEAD ({gib(overhead.bytes)} GiB)", items))
    if fit := estimate.fit:
        gpus = fit.gpus or 1
        capacity = gpus * fit.usable
        requirement = max(fit.required, fit.startup_required or 0)
        text = f"{gib(requirement)} / {gib(capacity)} GiB [{requirement / capacity:.0%}] ON {gpus} x {gib(fit.gpu_memory)} GiB"
        if kv and fit.max_model_len:
            text += f", UP TO {fit.max_model_len} TOKENS"
        if fit.fits and fit.needed and fit.needed < gpus:
            verdict = f"FITS ON {fit.needed} GPU" + ("S" if fit.needed > 1 else "")
        else:
            verdict = (
                "FITS"
                if fit.fits
                else f"⚠ NEEDS {fit.needed} GPUS"
                if fit.needed and fit.needed > gpus
                else "⚠ DOES NOT FIT"
            )
        sections[0][1].append(("GPU MEMORY", text, verdict, requirement / capacity))
    titles = _titles(estimate.url, estimate.revision, estimate.file, kv)
    if estimate.overhead and estimate.overhead.reference_engine:
        titles.append(
            f"{estimate.overhead.reference_engine.upper()} REFERENCE, {estimate.gpu_memory_utilization:.0%} ALLOCATION BUDGET"
        )
    return _color(_draw(titles, sections, NAME_WIDTH), color)


def _listing(listing: Listing, color: bool = False) -> str:
    kv = next((f.kv_cache for f in listing.files if f.kv_cache), None)
    largest = max(f.recommended_gpu_memory or f.total for f in listing.files) or 1
    name_width = min(max(len(f.file or "") for f in listing.files), MAX_FILE_WIDTH)
    items: list[Item] = []
    for f in listing.files:
        required = f.recommended_gpu_memory or f.total
        text = f"{gib(required)} GiB" + (" GPU CAPACITY" if f.recommended_gpu_memory else " WEIGHTS")
        if f.kv_cache:
            parts = [f"{gib(f.weights.bytes)} GiB WEIGHTS", f"{gib(f.kv_cache.bytes)} GiB KV"]
            parts += [f"{gib(f.overhead.bytes)} GiB OVERHEAD"] if f.overhead else []
            text += f" ({' + '.join(parts)})"
        items.append(("", text, f.file or "", required / largest))
    titles = _titles(listing.url, listing.revision, None, kv)
    return _color(_draw(titles, [("FILES", items)], name_width), color)


def _color(lines: list[str], enabled: bool = False) -> str:
    if enabled and sys.stdout.isatty() and not os.getenv("NO_COLOR"):
        return "\n".join(f"\x1b[38;2;244;183;63m{line}\x1b[0m" for line in lines)
    return "\n".join(lines)


def markdown(estimate: Estimate | Listing) -> str:
    where = f"[`{estimate.model}`]({estimate.url})" + (f" @ `{estimate.revision}`" if estimate.revision else "")
    if isinstance(estimate, Listing):
        rows = ["| File | GPU capacity | Allocations | Weights | KV cache |", "|---|--:|--:|--:|--:|"]
        for f in estimate.files:
            kv = f"{gib(f.kv_cache.bytes)} GiB" if f.kv_cache else "-"
            capacity = f"{gib(f.recommended_gpu_memory)} GiB" if f.recommended_gpu_memory else "-"
            rows.append(f"| `{f.file}` | {capacity} | {gib(f.total)} GiB | {gib(f.weights.bytes)} GiB | {kv} |")
        return "\n".join([f"Memory estimate for {where}", "", *rows])

    rows = ["| | Memory | Params |", "|---|--:|--:|"]
    if (recommended := estimate.recommended_gpu_memory) is not None:
        rows.append(
            f"| **Recommended GPU capacity** | **{gib(recommended)} GiB ({recommended / 10**9:.2f} GB)** | |"
        )
    if estimate.startup_reserve_per_gpu:
        rows.append(f"| Startup allowance per GPU | {gib(estimate.startup_reserve_per_gpu)} GiB | |")
    rows.append(
        f"| **Total** | **{gib(estimate.total)} GiB** | {short(estimate.weights.params) if estimate.weights.params else '-'} |"
    )
    for name, component in (
        estimate.components if len(estimate.components) > 1 else {"Weights": estimate.weights}
    ).items():
        dtypes = ", ".join(d for d, _ in component.sorted_dtypes())
        label = f"{name} ({dtypes})" if dtypes else name
        rows.append(
            f"| {label} | {gib(component.bytes)} GiB | {short(component.params) if component.params else '-'} |"
        )
    if offloaded := estimate.offloaded:
        rows.append(f"| Offloaded to CPU memory | {gib(offloaded.bytes)} GiB | {short(offloaded.params)} |")
    if cache := estimate.kv_cache:
        rows.append(
            f"| KV cache ({cache.dtype}, {cache.max_model_len} tokens x {cache.batch_size}) | {gib(cache.bytes)} GiB | |"
        )
    if overhead := estimate.overhead:
        parts = ", ".join(
            n
            for n in ("activations", "encoder", "graphs", "runtime", "weight_adjustment")
            if getattr(overhead, n)
        )
        rows.append(f"| Overhead ({parts}) | {gib(overhead.bytes)} GiB | |")
    title = f"Memory estimate for {where}" + (f", file `{estimate.file}`" if estimate.file else "")
    lines = [title, "", *rows]
    if fit := estimate.fit:
        gpus = f"{fit.gpus or 1} x {gib(fit.gpu_memory)} GiB"
        verdict = f"Fits on {gpus}" if fit.fits else f"Does not fit on {gpus}"
        if fit.needed and fit.needed != (fit.gpus or 1):
            action = "fits on" if not fit.fits and fit.needed < (fit.gpus or 1) else "needs"
            verdict += f", {action} {fit.needed} GPU{'s' if fit.needed > 1 else ''}"
        if estimate.kv_cache and fit.max_model_len:
            verdict += f", up to {fit.max_model_len} tokens"
        lines += ["", verdict + "."]
    return "\n".join(lines)
