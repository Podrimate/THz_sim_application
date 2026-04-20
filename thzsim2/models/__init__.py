"""Typed data models for the notebook-facing workflows."""

from .fit import Fit
from .fit_workflow import MeasuredFitResult, PreparedTracePair
from .measurement import Measurement, ReferenceStandard, ResolvedMeasurementFitParameter
from .reference import ReferenceResult, ReferenceSummary, SpectrumData, TraceData
from .sample import (
    ConstantNK,
    Drude,
    DrudeLorentz,
    Layer,
    Lorentz,
    LorentzOscillator,
    NKFile,
    ResolvedFitParameter,
    SampleLayerResult,
    SampleResult,
)
from .study import StudyCaseResult, StudyResult
from .validation import ValidationCaseResult, ValidationSuiteResult

__all__ = [
    "Fit",
    "PreparedTracePair",
    "MeasuredFitResult",
    "Measurement",
    "ReferenceStandard",
    "ResolvedMeasurementFitParameter",
    "TraceData",
    "SpectrumData",
    "ReferenceSummary",
    "ReferenceResult",
    "NKFile",
    "ConstantNK",
    "Drude",
    "Lorentz",
    "LorentzOscillator",
    "DrudeLorentz",
    "Layer",
    "ResolvedFitParameter",
    "SampleLayerResult",
    "SampleResult",
    "StudyCaseResult",
    "StudyResult",
    "ValidationCaseResult",
    "ValidationSuiteResult",
]
