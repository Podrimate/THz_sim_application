from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ReferenceStandard:
    kind: str
    stack: Any | None = None

    def __post_init__(self):
        self.kind = str(self.kind).strip().lower()
        if self.kind not in {"identity", "stack"}:
            raise ValueError("reference_standard.kind must be 'identity' or 'stack'")
        if self.kind == "identity":
            self.stack = None
            return
        if self.stack is None:
            raise ValueError("reference_standard.kind='stack' requires a stack or SampleResult")


@dataclass(slots=True)
class Measurement:
    mode: str = "transmission"
    angle_deg: float = 0.0
    polarization: str = "s"
    reference_standard: ReferenceStandard | dict[str, Any] | None = None

    def __post_init__(self):
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"transmission", "reflection"}:
            raise ValueError("mode must be 'transmission' or 'reflection'")

        self.angle_deg = float(self.angle_deg)
        if not (-89.999 <= self.angle_deg < 89.999):
            raise ValueError("angle_deg must be between -89.999 and 89.999 degrees")

        self.polarization = str(self.polarization).strip().lower()
        if self.polarization not in {"s", "p"}:
            raise ValueError("polarization must be 's' or 'p'")

        if isinstance(self.reference_standard, dict):
            self.reference_standard = ReferenceStandard(**self.reference_standard)
        elif self.reference_standard is not None and not isinstance(self.reference_standard, ReferenceStandard):
            raise TypeError("reference_standard must be a ReferenceStandard, dictionary, or None")
