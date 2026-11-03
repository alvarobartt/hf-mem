import os
import sys

from hf_mem._version import __version__
from hf_mem._weights import Weights
from hf_mem.estimate import Estimate, Listing

NAME_WIDTH = 14
MAX_FILE_WIDTH = 64

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
    filled = round(min(max(fraction, 0.0), 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _fit(name: str, width: int) -> str:
    return (name[: width - 3] + "...") if len(name) > width else f"{name:<{width}}"


def _draw(titles: list[str], sections: list[Section], name_width: int) -> list[str]:
    texts = [t for _, items in sections for _, t, _, _ in items]
    inner = max(
        max(map(len, titles + [s for s, _ in sections if s])) + 8, name_width + max(map(len, texts)) + 7
    )
    data = inner - name_width - 5
    badge = f" hf-mem v{__version__} "
    lines = ["┌" + "┬" * (inner - len(badge)) + badge + "┐", "├" + "┴" * inner + "┤"]
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


def _titles(url: str, revision: str | None, file: str | None, kv: object) -> list[str]:
    titles = ["INFERENCE MEMORY ESTIMATE FOR", f"{url} @ {revision}" if revision else url]
    if file:
        titles.append(f"FOR `{file}`")
    if kv is not None:
        titles.append(f"w/ max-model-len={kv.max_model_len}, batch-size={kv.batch_size}")  # type: ignore[attr-defined]
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


def table(estimate: Estimate | Listing) -> str:
    if isinstance(estimate, Listing):
        return _listing(estimate)
    total, kv, weights = estimate.total or 1, estimate.kv_cache, estimate.weights
    params = f"{short(weights.params)} PARAMS" if weights.params else "FROM FILE SIZES"
    parts = [params] + (["KV CACHE"] if kv else []) + (["OVERHEAD"] if estimate.overhead else [])
    summary = f"{gib(estimate.total)} GiB ({' + '.join(parts)})"
    sections: list[Section] = [(None, [("TOTAL MEMORY", summary, "REQUIREMENTS", 1.0)])]

    def heading(name: str, w: Weights) -> str:
        return (
            f"{name.upper()} ({short(w.params)} PARAMS, {gib(w.bytes)} GiB)"
            if w.params
            else f"{name.upper()} ({gib(w.bytes)} GiB)"
        )

    if estimate.moe is not None:
        moe = estimate.moe
        sections.append((heading("MODEL", moe.base), _dtype_items(moe.base, total)))
        expert = f"{moe.experts} x EXPERTS ({short(moe.expert.params)} PARAMS EACH, {gib(moe.expert.bytes)} GiB EACH)"
        sections.append((expert, _dtype_items(moe.expert, total)))
    else:
        many = len(estimate.components) > 1
        for name, component in (estimate.components or {"MODEL": weights}).items():
            title = heading(name if many else "MODEL", component) if many or kv else None
            if component.dtypes:
                sections.append((title, _dtype_items(component, total)))
            elif many:
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
                ("CUDA GRAPHS", overhead.cuda_graphs),
                ("NON-TORCH", overhead.non_torch),
                ("CUDA CONTEXT", overhead.cuda_context),
            )
            if value
        ]
        sections.append((f"OVERHEAD ({gib(overhead.bytes)} GiB)", items))
    if fit := estimate.fit:
        gpus = fit.gpus or fit.needed or 1
        capacity = gpus * fit.gpu_memory
        if fit.fits:
            text = f"FITS ON {fit.needed} x {gib(fit.gpu_memory)} GiB"
        elif fit.needed:
            text = f"NEEDS {fit.needed} x {gib(fit.gpu_memory)} GiB"
        else:
            text = f"DOES NOT FIT ON 64 x {gib(fit.gpu_memory)} GiB"
        if kv:
            text += f", UP TO {fit.max_model_len} TOKENS"
        used = f"{estimate.total / capacity:.0%} OF {gpus} GPU{'S' if gpus > 1 else ''}"
        sections.append((None, [("GPU FIT", text, used, estimate.total / capacity)]))
    return _color(_draw(_titles(estimate.url, estimate.revision, estimate.file, kv), sections, NAME_WIDTH))


def _listing(listing: Listing) -> str:
    kv = next((f.kv_cache for f in listing.files if f.kv_cache), None)
    largest = max(f.total for f in listing.files) or 1
    name_width = min(max(len(f.file or "") for f in listing.files), MAX_FILE_WIDTH)
    items: list[Item] = []
    for f in listing.files:
        text = f"{gib(f.total)} GiB"
        if f.kv_cache:
            text += f" ({gib(f.weights.bytes)} GiB WEIGHTS + {gib(f.kv_cache.bytes)} GiB KV)"
        items.append(("", text, f.file or "", f.total / largest))
    titles = _titles(listing.url, listing.revision, None, kv)
    return _color(_draw(titles, [("FILES", items)], name_width))


def _color(lines: list[str]) -> str:
    if sys.stdout.isatty() and not os.getenv("NO_COLOR"):
        return "\n".join(f"\x1b[38;2;244;183;63m{line}\x1b[0m" for line in lines)
    return "\n".join(lines)


def markdown(estimate: Estimate | Listing) -> str:
    where = f"[`{estimate.model}`]({estimate.url})" + (f" @ `{estimate.revision}`" if estimate.revision else "")
    if isinstance(estimate, Listing):
        rows = ["| File | Total | Weights | KV cache |", "|---|--:|--:|--:|"]
        for f in estimate.files:
            kv = f"{gib(f.kv_cache.bytes)} GiB" if f.kv_cache else "-"
            rows.append(f"| `{f.file}` | {gib(f.total)} GiB | {gib(f.weights.bytes)} GiB | {kv} |")
        return "\n".join([f"Memory estimate for {where}", "", *rows])

    rows = ["| | Memory | Params |", "|---|--:|--:|"]
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
    if cache := estimate.kv_cache:
        rows.append(
            f"| KV cache ({cache.dtype}, {cache.max_model_len} tokens x {cache.batch_size}) | {gib(cache.bytes)} GiB | |"
        )
    title = f"Memory estimate for {where}" + (f", file `{estimate.file}`" if estimate.file else "")
    return "\n".join([title, "", *rows])
