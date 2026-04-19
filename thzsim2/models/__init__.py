"""Typed data models for the notebook-facing workflows."""

from .fit import Fit
from .measurement import Measurement, ReferenceStandard
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
    "Measurement",
    "ReferenceStandard",
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
