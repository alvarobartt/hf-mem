from dataclasses import dataclass, field
from typing import Any

DTYPES = {"float16": ("F16", 2), "bfloat16": ("BF16", 2), "float32": ("F32", 4)}
_FLOAT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


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

    def cast(self, dtype: str) -> "Weights":
        target, size = DTYPES[dtype]
        result = Weights()
        for name, entry in self.dtypes.items():
            if name in _FLOAT_BYTES:
                if entry.params:
                    result.add(target, entry.params, entry.params * size)
                if buffers := entry.bytes - entry.params * _FLOAT_BYTES[name]:
                    result.add(name, 0, buffers)
            else:
                result.add(name, entry.params, entry.bytes)
        if not self.dtypes:
            result.params, result.bytes = self.params, self.bytes
        return result

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
