from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _validate_uniform_axis(time_ps: NDArray[np.float64]):
    diffs = np.diff(time_ps)
    if np.any(diffs <= 0.0):
        raise ValueError("time_ps must be strictly increasing")
    dt = float(diffs[0])
    tolerance = max(1e-12, abs(dt) * 1e-9)
    if not np.allclose(diffs, dt, rtol=0.0, atol=tolerance):
        raise ValueError("time_ps must be uniformly spaced")


@dataclass(slots=True)
class TraceData:
    time_ps: NDArray[np.float64]
    trace: NDArray[np.float64] | NDArray[np.complex128]
    source_kind: str = "unknown"
    source_path: str | None = None
    pad_factor: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        time_ps = np.asarray(self.time_ps, dtype=np.float64)
        trace = np.asarray(self.trace)

        if time_ps.ndim != 1:
            raise ValueError("time_ps must be 1D")
        if trace.ndim != 1:
            raise ValueError("trace must be 1D")
        if time_ps.size < 2:
            raise ValueError("trace data must contain at least two samples")
        if time_ps.shape != trace.shape:
            raise ValueError("time_ps and trace must have the same shape")
        if not np.isfinite(time_ps).all():
            raise ValueError("time_ps must contain only finite values")

        if np.iscomplexobj(trace):
            trace = np.asarray(trace, dtype=np.complex128)
            if not (np.isfinite(trace.real) & np.isfinite(trace.imag)).all():
                raise ValueError("trace must contain only finite values")
        else:
            trace = np.asarray(trace, dtype=np.float64)
            if not np.isfinite(trace).all():
                raise ValueError("trace must contain only finite values")

        _validate_uniform_axis(time_ps)

        self.time_ps = time_ps
        self.trace = trace
        self.pad_factor = int(self.pad_factor)
        if self.pad_factor < 1:
            raise ValueError("pad_factor must be >= 1")
        self.source_kind = str(self.source_kind)
        self.source_path = None if self.source_path is None else str(self.source_path)
        self.metadata = dict(self.metadata)

    @property
    def sample_count(self) -> int:
        return int(self.time_ps.size)

    @property
    def dt_ps(self) -> float:
        return float(self.time_ps[1] - self.time_ps[0])

    @property
    def time_min_ps(self) -> float:
        return float(self.time_ps[0])

    @property
    def time_max_ps(self) -> float:
        return float(self.time_ps[-1])

    @property
    def time_center_ps(self) -> float:
        return float(np.mean(self.time_ps))

    def with_trace(self, trace, metadata_updates: dict[str, Any] | None = None):
        metadata = dict(self.metadata)
        if metadata_updates:
            metadata.update(metadata_updates)
        return TraceData(
            time_ps=self.time_ps.copy(),
            trace=np.asarray(trace).copy(),
            source_kind=self.source_kind,
            source_path=self.source_path,
            pad_factor=self.pad_factor,
            metadata=metadata,
        )


@dataclass(slots=True)
class SpectrumData:
    freq_thz: NDArray[np.float64]
    real: NDArray[np.float64]
    imag: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    phase_rad: NDArray[np.float64]

    def __post_init__(self):
        freq_thz = np.asarray(self.freq_thz, dtype=np.float64)
        real = np.asarray(self.real, dtype=np.float64)
        imag = np.asarray(self.imag, dtype=np.float64)
        magnitude = np.asarray(self.magnitude, dtype=np.float64)
        phase_rad = np.asarray(self.phase_rad, dtype=np.float64)

        shape = freq_thz.shape
        if freq_thz.ndim != 1:
            raise ValueError("freq_thz must be 1D")
        if shape[0] == 0:
            raise ValueError("spectrum data must be non-empty")
        for name, arr in (
            ("real", real),
            ("imag", imag),
            ("magnitude", magnitude),
            ("phase_rad", phase_rad),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} must match freq_thz shape")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")

        if np.any(freq_thz < 0.0):
            raise ValueError("freq_thz must be nonnegative")
        if shape[0] > 1 and np.any(np.diff(freq_thz) < 0.0):
            raise ValueError("freq_thz must be monotonic increasing")

        self.freq_thz = freq_thz
        self.real = real
        self.imag = imag
        self.magnitude = magnitude
        self.phase_rad = phase_rad


@dataclass(slots=True)
class ReferenceSummary:
    dt_ps: float
    sample_count: int
    time_min_ps: float
    time_max_ps: float
    amplitude_scale: float
    pulse_center_ps: float
    freq_min_thz: float
    freq_max_thz: float
    peak_freq_thz: float
    spectral_centroid_thz: float

    def as_rows(self):
        return [
            ("dt_ps", float(self.dt_ps), "ps"),
            ("sample_count", int(self.sample_count), "count"),
            ("time_min_ps", float(self.time_min_ps), "ps"),
            ("time_max_ps", float(self.time_max_ps), "ps"),
            ("amplitude_scale", float(self.amplitude_scale), "a.u."),
            ("pulse_center_ps", float(self.pulse_center_ps), "ps"),
            ("freq_min_thz", float(self.freq_min_thz), "THz"),
            ("freq_max_thz", float(self.freq_max_thz), "THz"),
            ("peak_freq_thz", float(self.peak_freq_thz), "THz"),
            ("spectral_centroid_thz", float(self.spectral_centroid_thz), "THz"),
        ]


@dataclass(slots=True)
class ReferenceResult:
    run_id: str
    created_at: str
    run_dir: Path
    reference_dir: Path
    trace: TraceData
    spectrum: SpectrumData
    summary: ReferenceSummary
    manifest: dict[str, Any]
    artifact_paths: dict[str, Path]
    run_manifest_path: Path
