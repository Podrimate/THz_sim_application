from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .measurement import ResolvedMeasurementFitParameter
from .reference import ReferenceResult, TraceData
from .sample import SampleResult


@dataclass(slots=True)
class PreparedTracePair:
    raw_reference: TraceData
    raw_sample: TraceData
    aligned_reference: TraceData
    aligned_sample: TraceData
    processed_reference: TraceData
    processed_sample: TraceData
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MeasuredFitResult:
    out_dir: Path
    prepared_traces: PreparedTracePair
    reference_result: ReferenceResult
    sample_result: SampleResult
    fit_result: dict[str, Any]
    measurement_fit_parameters: list[ResolvedMeasurementFitParameter]
    artifact_paths: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)
