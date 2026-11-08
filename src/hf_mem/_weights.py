from dataclasses import dataclass, field
from typing import Any

FLOATS = ("F64", "F32", "F16", "BF16")


@dataclass
class Weights:
    params: int = 0
    bytes: int = 0
    dtypes: dict[str, "Weights"] = field(default_factory=dict)

    def add(self, dtype: str, params: int, nbytes: int) -> None:
        self.params += params
        self.bytes += nbytes
        entry = self.dtypes.setdefault(dtype, Weights())
        entry.params += params
        entry.bytes += nbytes

    def merge(self, other: "Weights") -> "Weights":
        for dtype, entry in other.dtypes.items():
            self.add(dtype, entry.params, entry.bytes)
        if not other.dtypes:
            self.params += other.params
            self.bytes += other.bytes
        return self

    def cast(self, dtype: str, size: int) -> "Weights":
        # The floating point weights as loaded by engines that cast them (e.g. Text Embeddings Inference), with
        # `size` bytes per parameter, keeping the integer and quantized tensors as they are
        cast = Weights()
        for name, entry in self.dtypes.items():
            if name in FLOATS:
                cast.add(dtype, entry.params, entry.params * size)
            else:
                cast.add(name, entry.params, entry.bytes)
        if not self.dtypes:
            cast.params, cast.bytes = self.params, self.bytes
        return cast

    def sorted_dtypes(self) -> list[tuple[str, "Weights"]]:
        return sorted(self.dtypes.items(), key=lambda item: item[1].bytes, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"bytes": self.bytes, "params": self.params or None}
        if self.dtypes:
            out["dtypes"] = {d: {"bytes": w.bytes, "params": w.params} for d, w in self.sorted_dtypes()}
        return out


def choices(names: list[str], limit: int = 8) -> str:
    shown = ", ".join(f"`{n}`" for n in sorted(names)[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")
