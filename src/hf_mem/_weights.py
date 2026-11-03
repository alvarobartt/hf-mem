from dataclasses import dataclass, field
from typing import Any


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

    def sorted_dtypes(self) -> list[tuple[str, "Weights"]]:
        return sorted(self.dtypes.items(), key=lambda item: item[1].bytes, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"bytes": self.bytes, "params": self.params}
        if self.dtypes:
            out["dtypes"] = {d: {"bytes": w.bytes, "params": w.params} for d, w in self.sorted_dtypes()}
        return out
