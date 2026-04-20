from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fit import Fit


def _resolve_angle_value(value, *, field_name: str):
    if isinstance(value, Fit):
        _validate_fit_bounds(value, field_name=field_name)
        if not (-89.999 <= float(value.initial) < 89.999):
            raise ValueError(f"{field_name}.initial must be between -89.999 and 89.999 degrees")
        if value.resolved_min is not None and float(value.resolved_min) < -89.999:
            raise ValueError(f"{field_name} lower bound must be >= -89.999 degrees")
        if value.resolved_max is not None and float(value.resolved_max) >= 89.999:
            raise ValueError(f"{field_name} upper bound must be < 89.999 degrees")
        return value

    angle = float(value)
    if not (-89.999 <= angle < 89.999):
        raise ValueError(f"{field_name} must be between -89.999 and 89.999 degrees")
    return angle


def _resolve_mix_value(value, *, field_name: str):
    if isinstance(value, Fit):
        _validate_fit_bounds(value, field_name=field_name)
        if not (0.0 <= float(value.initial) <= 1.0):
            raise ValueError(f"{field_name}.initial must be between 0 and 1")
        if value.resolved_min is not None and float(value.resolved_min) < 0.0:
            raise ValueError(f"{field_name} lower bound must be >= 0")
        if value.resolved_max is not None and float(value.resolved_max) > 1.0:
            raise ValueError(f"{field_name} upper bound must be <= 1")
        return value

    mix = float(value)
    if not (0.0 <= mix <= 1.0):
        raise ValueError(f"{field_name} must be between 0 and 1")
    return mix


def _validate_fit_bounds(value: Fit, *, field_name: str):
    if value.resolved_min is None or value.resolved_max is None:
        raise ValueError(f"{field_name} Fit(...) must define both lower and upper bounds")


@dataclass(slots=True)
class ResolvedMeasurementFitParameter:
    key: str
    label: str
    path: str
    unit: str
    initial_value: float
    bound_min: float
    bound_max: float


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
    angle_deg: float | Fit = 0.0
    polarization: str = "s"
    polarization_mix: float | Fit | None = None
    reference_standard: ReferenceStandard | dict[str, Any] | None = None

    def __post_init__(self):
        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"transmission", "reflection"}:
            raise ValueError("mode must be 'transmission' or 'reflection'")

        self.polarization = str(self.polarization).strip().lower()
        if self.polarization not in {"s", "p", "mixed"}:
            raise ValueError("polarization must be 's', 'p', or 'mixed'")

        self.angle_deg = _resolve_angle_value(self.angle_deg, field_name="angle_deg")

        if self.polarization == "mixed":
            if self.polarization_mix is None:
                self.polarization_mix = 0.5
            self.polarization_mix = _resolve_mix_value(
                self.polarization_mix,
                field_name="polarization_mix",
            )
        elif self.polarization_mix is not None:
            self.polarization_mix = _resolve_mix_value(
                self.polarization_mix,
                field_name="polarization_mix",
            )

        if isinstance(self.reference_standard, dict):
            self.reference_standard = ReferenceStandard(**self.reference_standard)
        elif self.reference_standard is not None and not isinstance(self.reference_standard, ReferenceStandard):
            raise TypeError("reference_standard must be a ReferenceStandard, dictionary, or None")
