"""High-level notebook-facing workflows."""

from .fit_workflow import plot_trace_pair_preview, prepare_trace_pair_for_fit, resolve_measurement_fit_parameters, run_measured_fit
from .reference import generate_reference_pulse, load_reference_csv, prepare_reference
from .sample_workflow import build_sample
from .study_workflow import (
    export_trace_bundle,
    load_study_summary,
    plot_best_and_worst_case,
    plot_study_summary,
    run_single_layer_drude_compat_study,
    run_study,
    show_study_heatmaps,
)
from .study_setup import build_study_setup, load_study_setup_csv, run_study_from_setup_csv, write_study_setup_csv
from .validation_workflow import (
    load_validation_summary,
    plot_validation_plot_grid,
    plot_validation_summary,
    run_validation_suite,
)

__all__ = [
    "prepare_trace_pair_for_fit",
    "plot_trace_pair_preview",
    "resolve_measurement_fit_parameters",
    "run_measured_fit",
    "load_reference_csv",
    "generate_reference_pulse",
    "prepare_reference",
    "build_sample",
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
    "export_trace_bundle",
    "run_validation_suite",
    "load_validation_summary",
    "plot_validation_summary",
    "plot_validation_plot_grid",
]
