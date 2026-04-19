from dataclasses import dataclass, field


@dataclass(slots=True)
class Fit:
    """Explicit fit-parameter declaration with resolved absolute bounds.

    ``rel_min`` and ``rel_max`` are direct multipliers of ``initial``.
    For example, ``Fit(100.0, rel_min=0.5, rel_max=1.5)`` resolves to
    bounds ``(50.0, 150.0)``.
    """

    initial: float
    rel_min: float | None = None
    rel_max: float | None = None
    abs_min: float | None = None
    abs_max: float | None = None
    label: str | None = None
    resolved_min: float | None = field(init=False, default=None)
    resolved_max: float | None = field(init=False, default=None)

    def __post_init__(self):
        self.initial = float(self.initial)
        if self.label is not None and not str(self.label).strip():
            raise ValueError("label must be a non-empty string when provided")

        self.resolved_min = self._resolve_side(
            rel_value=self.rel_min,
            abs_value=self.abs_min,
            side_name="min",
        )
        self.resolved_max = self._resolve_side(
            rel_value=self.rel_max,
            abs_value=self.abs_max,
            side_name="max",
        )

        if self.resolved_min is not None and self.resolved_max is not None:
            if self.resolved_min > self.resolved_max:
                raise ValueError("resolved_min cannot be greater than resolved_max")

    def _resolve_side(self, rel_value, abs_value, side_name):
        if rel_value is not None and abs_value is not None:
            raise ValueError(f"Specify either rel_{side_name} or abs_{side_name}, not both")
        if abs_value is not None:
            return float(abs_value)
        if rel_value is None:
            return None
        if self.initial == 0.0:
            raise ValueError(
                f"rel_{side_name} cannot be resolved when initial is zero; "
                f"provide abs_{side_name} instead"
            )
        return self.initial * float(rel_value)

    @property
    def bounds(self):
        return (self.resolved_min, self.resolved_max)
