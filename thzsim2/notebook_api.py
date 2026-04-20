"""Minimal public API used by end-user notebooks."""

from .core.fitting import (
    drude_gamma_thz_from_tau_ps,
    drude_plasma_freq_thz_from_sigma_tau,
    sigma_s_per_m_from_drude_plasma_gamma,
    tau_ps_from_drude_gamma_thz,
)
from .models import (
    ConstantNK,
    Drude,
    DrudeLorentz,
    Fit,
    Layer,
    Lorentz,
    LorentzOscillator,
    Measurement,
    NKFile,
    MeasuredFitResult,
    PreparedTracePair,
    ReferenceStandard,
    ResolvedMeasurementFitParameter,
)
from .workflows.fit_workflow import (
    plot_trace_pair_preview,
    prepare_trace_pair_for_fit,
    resolve_measurement_fit_parameters,
    run_measured_fit,
)
from .workflows.reference import generate_reference_pulse, load_reference_csv, prepare_reference
from .workflows.sample_workflow import build_sample
from .workflows.study_workflow import (
    load_study_summary,
    plot_best_and_worst_case,
    plot_study_summary,
    run_single_layer_drude_compat_study,
    run_study,
    show_study_heatmaps,
)
from .workflows.study_setup import build_study_setup, load_study_setup_csv, run_study_from_setup_csv, write_study_setup_csv
from .workflows.validation_workflow import (
    load_validation_summary,
    plot_validation_plot_grid,
    plot_validation_summary,
    run_validation_suite,
)

__all__ = [
    "Fit",
    "Layer",
    "NKFile",
    "Measurement",
    "ReferenceStandard",
    "ResolvedMeasurementFitParameter",
    "PreparedTracePair",
    "MeasuredFitResult",
    "ConstantNK",
    "Drude",
    "Lorentz",
    "LorentzOscillator",
    "DrudeLorentz",
    "load_reference_csv",
    "generate_reference_pulse",
    "prepare_reference",
    "build_sample",
    "prepare_trace_pair_for_fit",
    "plot_trace_pair_preview",
    "resolve_measurement_fit_parameters",
    "run_measured_fit",
    "drude_gamma_thz_from_tau_ps",
    "drude_plasma_freq_thz_from_sigma_tau",
    "tau_ps_from_drude_gamma_thz",
    "sigma_s_per_m_from_drude_plasma_gamma",
    "run_study",
    "run_single_layer_drude_compat_study",
    "load_study_summary",
    "build_study_setup",
    "write_study_setup_csv",
    "load_study_setup_csv",
    "run_study_from_setup_csv",
    "plot_study_summary",
    "show_study_heatmaps",
    "plot_best_and_worst_case",
    "run_validation_suite",
    "load_validation_summary",
    "plot_validation_summary",
    "plot_validation_plot_grid",
]
