from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .fit import Fit

ParameterValue: TypeAlias = float | int | Fit


@dataclass(slots=True)
class NKFile:
    """Imported refractive-index data using the standard ``freq_thz,n,k`` schema."""

    path: str | Path

    def __post_init__(self):
        path = Path(self.path)
        if not str(path).strip():
            raise ValueError("NKFile path must be a non-empty path")
        self.path = path


@dataclass(slots=True)
class ConstantNK:
    n: ParameterValue
    k: ParameterValue = 0.0


@dataclass(slots=True)
class Drude:
    eps_inf: ParameterValue
    plasma_freq_thz: ParameterValue
    gamma_thz: ParameterValue


@dataclass(slots=True)
class TwoDrude:
    eps_inf: ParameterValue
    plasma_freq1_thz: ParameterValue
    gamma1_thz: ParameterValue
    plasma_freq2_thz: ParameterValue
    gamma2_thz: ParameterValue


@dataclass(slots=True)
class Lorentz:
    eps_inf: ParameterValue
    delta_eps: ParameterValue
    resonance_thz: ParameterValue
    gamma_thz: ParameterValue


@dataclass(slots=True)
class LorentzOscillator:
    delta_eps: ParameterValue
    resonance_thz: ParameterValue
    gamma_thz: ParameterValue


@dataclass(slots=True)
class DrudeLorentz:
    eps_inf: ParameterValue
    plasma_freq_thz: ParameterValue = 0.0
    gamma_thz: ParameterValue = 0.0
    oscillators: tuple[LorentzOscillator, ...] = ()

    def __post_init__(self):
        self.oscillators = tuple(self.oscillators)


MaterialSpec: TypeAlias = NKFile | ConstantNK | Drude | TwoDrude | Lorentz | DrudeLorentz


@dataclass(slots=True)
class Layer:
    name: str
    thickness_um: ParameterValue
    material: MaterialSpec

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("Layer name must be a non-empty string")
        self.name = str(self.name)


@dataclass(slots=True)
class ResolvedFitParameter:
    key: str
    label: str
    path: str
    unit: str
    initial_value: float
    bound_min: float | None
    bound_max: float | None
    layer_name: str


@dataclass(slots=True)
class SampleLayerResult:
    index: int
    name: str
    thickness_um: float
    material_kind: str
    parameters: dict[str, Any]
    freq_thz: NDArray[np.float64]
    n: NDArray[np.float64]
    k: NDArray[np.float64]
    fit_parameters: list[ResolvedFitParameter] = field(default_factory=list)
    imported_freq_thz: NDArray[np.float64] | None = None
    imported_n: NDArray[np.float64] | None = None
    imported_k: NDArray[np.float64] | None = None


@dataclass(slots=True)
class SampleResult:
    out_dir: Path
    freq_grid_thz: NDArray[np.float64]
    n_in: float
    n_out: float
    layers: list[SampleLayerResult]
    fit_parameters: list[ResolvedFitParameter]
    resolved_stack: dict[str, Any]
    manifest: dict[str, Any]
    artifact_paths: dict[str, Path]
