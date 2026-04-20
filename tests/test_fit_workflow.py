from pathlib import Path

import numpy as np
import pytest

from thzsim2.core.forward import simulate_sample_from_reference
from thzsim2.core.fitting import fit_sample_trace
from thzsim2.models import ConstantNK, Drude, Fit, Layer, Measurement
from thzsim2.workflows.fit_setup import build_fit_setup, run_measured_fit_from_setup_json, write_fit_setup_json
from thzsim2.workflows.fit_workflow import (
    plot_trace_pair_preview,
    prepare_trace_pair_for_fit,
    resolve_measurement_fit_parameters,
    run_measured_fit,
    summarize_prepared_trace_pair,
    summarize_trace_input,
)
from thzsim2.workflows.reference import generate_reference_pulse, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


def _reference_result(tmp_path):
    reference_input = generate_reference_pulse(
        model="sech_carrier",
        sample_count=384,
        dt_ps=0.03,
        time_center_ps=8.0,
        pulse_center_ps=5.0,
        tau_ps=0.22,
        f0_thz=0.9,
        amp=1.0,
        phi_rad=0.0,
    )
    return prepare_reference(reference_input, output_root=tmp_path, run_label="fit-workflow")


def _data_root():
    return Path(__file__).resolve().parents[1] / "Test_data_for_fitter"


def test_resolve_measurement_fit_parameters_extracts_angle_and_mix():
    measurement, fit_parameters = resolve_measurement_fit_parameters(
        Measurement(
            mode="transmission",
            angle_deg=Fit(18.0, abs_min=5.0, abs_max=35.0, label="incident_angle_deg"),
            polarization="mixed",
            polarization_mix=Fit(0.35, abs_min=0.0, abs_max=1.0, label="effective_mix"),
        )
    )
    assert measurement.polarization == "mixed"
    assert measurement.angle_deg == pytest.approx(18.0)
    assert measurement.polarization_mix == pytest.approx(0.35)
    assert [parameter.key for parameter in fit_parameters] == ["incident_angle_deg", "effective_mix"]


def test_fit_sample_trace_recovers_measurement_angle_and_mix(tmp_path):
    reference_result = _reference_result(tmp_path)
    sample_result = build_sample(
        layers=[
            Layer(
                name="film",
                thickness_um=95.0,
                material=ConstantNK(n=2.2, k=0.03),
            )
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )

    true_measurement = Measurement(
        mode="transmission",
        angle_deg=31.0,
        polarization="mixed",
        polarization_mix=0.72,
    )
    observed_trace = simulate_sample_from_reference(
        reference_result,
        sample_result.resolved_stack,
        measurement=true_measurement,
    )["sample_trace"]

    initial_measurement, measurement_fit_parameters = resolve_measurement_fit_parameters(
        Measurement(
            mode="transmission",
            angle_deg=Fit(18.0, abs_min=10.0, abs_max=40.0, label="fit_angle_deg"),
            polarization="mixed",
            polarization_mix=Fit(0.25, abs_min=0.0, abs_max=1.0, label="fit_polarization_mix"),
        )
    )
    fit_result = fit_sample_trace(
        reference=reference_result,
        observed_trace=observed_trace,
        initial_stack=sample_result.resolved_stack,
        fit_parameters=[],
        measurement_fit_parameters=measurement_fit_parameters,
        metric="mse",
        optimizer={
            "method": "L-BFGS-B",
            "options": {"maxiter": 30},
            "global_options": {"maxiter": 4, "popsize": 6, "seed": 123},
            "fd_rel_step": 1e-4,
        },
        measurement=initial_measurement,
    )

    assert fit_result["success"]
    assert fit_result["recovered_parameters"]["fit_angle_deg"] == pytest.approx(31.0, abs=1.5)
    assert fit_result["recovered_parameters"]["fit_polarization_mix"] == pytest.approx(0.72, abs=0.08)


@pytest.mark.parametrize(
    ("folder_name", "sample_name"),
    [
        ("A11008858_transmission", "SAMPLE.csv"),
        ("A11013460_transmission", "SAMPLE1.csv"),
        ("A11013460_transmission", "SAMPLE2.csv"),
    ],
)
def test_prepare_trace_pair_for_fit_default_processing_retains_main_pulses(folder_name, sample_name):
    data_root = _data_root()
    reference_csv = data_root / folder_name / "REFERENCE.csv"
    sample_csv = data_root / folder_name / sample_name
    if not reference_csv.exists() or not sample_csv.exists():
        pytest.skip("Test_data_for_fitter transmission pair is not available")

    prepared = prepare_trace_pair_for_fit(
        reference_csv,
        sample_csv,
        baseline_mode="auto_pre_pulse",
        baseline_window_samples=40,
        crop_mode="auto",
    )

    assert prepared.raw_reference.sample_count > prepared.processed_reference.sample_count
    assert prepared.processed_reference.sample_count == prepared.processed_sample.sample_count
    assert prepared.metadata["baseline_subtract"] is True
    assert prepared.metadata["processed_reference_peak_retained"] is True
    assert prepared.metadata["processed_sample_peak_retained"] is True
    assert prepared.metadata["aligned_reference_peak"]["time_ps"] == pytest.approx(2611.3, abs=1.5)
    assert prepared.metadata["crop_mode"] == "auto"
    assert not prepared.metadata["warnings"]

    fig, _ = plot_trace_pair_preview(prepared, display=False)
    assert fig is not None
    summary = summarize_prepared_trace_pair(prepared)
    assert summary["processed_reference_peak_retained"] is True


def test_prepare_trace_pair_for_fit_manual_crop_warns_if_main_pulse_is_excluded():
    data_root = _data_root()
    reference_csv = data_root / "A11008858_transmission" / "REFERENCE.csv"
    sample_csv = data_root / "A11008858_transmission" / "SAMPLE.csv"
    if not reference_csv.exists() or not sample_csv.exists():
        pytest.skip("Test_data_for_fitter transmission pair is not available")

    prepared = prepare_trace_pair_for_fit(
        reference_csv,
        sample_csv,
        baseline_mode="first_samples",
        baseline_window_samples=40,
        crop_mode="manual",
        crop_time_window_ps=(2620.0, 2740.0),
    )

    assert prepared.metadata["processed_reference_peak_retained"] is False
    assert prepared.metadata["processed_sample_peak_retained"] is False
    assert prepared.metadata["warnings"]
    assert "dominant reference pulse" in prepared.metadata["warnings"][0]


def test_run_measured_fit_smoke_on_transmission_example(tmp_path):
    data_root = _data_root()
    reference_csv = data_root / "A11008858_transmission" / "REFERENCE.csv"
    sample_csv = data_root / "A11008858_transmission" / "SAMPLE.csv"
    if not reference_csv.exists() or not sample_csv.exists():
        pytest.skip("Test_data_for_fitter transmission pair is not available")

    prepared = prepare_trace_pair_for_fit(
        reference_csv,
        sample_csv,
        baseline_mode="auto_pre_pulse",
        baseline_window_samples=40,
        crop_mode="auto",
    )
    result = run_measured_fit(
        prepared,
        layers=[
            Layer(
                name="drude_layer",
                thickness_um=Fit(550.0, abs_min=300.0, abs_max=800.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(1.1, abs_min=0.1, abs_max=3.0, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(0.08, abs_min=0.005, abs_max=0.5, label="film_gamma_thz"),
                ),
            )
        ],
        out_dir=tmp_path / "measured_fit",
        measurement=Measurement(
            mode="transmission",
            angle_deg=Fit(8.0, abs_min=0.0, abs_max=25.0, label="measurement_angle_deg"),
            polarization="mixed",
            polarization_mix=Fit(0.5, abs_min=0.0, abs_max=1.0, label="measurement_polarization_mix"),
        ),
        optimizer={
            "method": "L-BFGS-B",
            "options": {"maxiter": 8},
            "global_options": {"maxiter": 1, "popsize": 5, "seed": 11},
            "fd_rel_step": 1e-4,
        },
        max_internal_reflections=2,
    )

    assert result.out_dir.exists()
    assert result.artifact_paths["measured_fit_summary_json"].exists()
    assert result.artifact_paths["measured_fit_overlay_png"].exists()
    assert "film_thickness_um" in result.fit_result["recovered_parameters"]
    assert "measurement_angle_deg" in result.fit_result["recovered_parameters"]


def test_trace_import_summary_reports_detected_columns_and_peak():
    data_root = _data_root()
    reference_csv = data_root / "A11008858_transmission" / "REFERENCE.csv"
    if not reference_csv.exists():
        pytest.skip("Test_data_for_fitter transmission reference is not available")

    summary = summarize_trace_input(reference_csv)
    assert summary["time_column"] == "Time_abs/ps"
    assert summary["signal_column"] == "Signal/nA"
    assert summary["peak_time_ps"] == pytest.approx(2611.3, abs=1.0)


def test_fit_setup_json_round_trip_runs_smoke_fit(tmp_path):
    data_root = _data_root()
    reference_csv = data_root / "A11008858_transmission" / "REFERENCE.csv"
    sample_csv = data_root / "A11008858_transmission" / "SAMPLE.csv"
    if not reference_csv.exists() or not sample_csv.exists():
        pytest.skip("Test_data_for_fitter transmission pair is not available")

    setup = build_fit_setup(
        reference_trace={"path": reference_csv},
        sample_trace={"path": sample_csv},
        preprocessing={
            "baseline_mode": "auto_pre_pulse",
            "baseline_window_samples": 40,
            "crop_mode": "auto",
        },
        layers=[
            Layer(
                name="drude_layer",
                thickness_um=Fit(550.0, abs_min=300.0, abs_max=800.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(1.1, abs_min=0.1, abs_max=3.0, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(0.08, abs_min=0.005, abs_max=0.5, label="film_gamma_thz"),
                ),
            )
        ],
        measurement=Measurement(
            mode="transmission",
            angle_deg=Fit(8.0, abs_min=0.0, abs_max=25.0, label="measurement_angle_deg"),
            polarization="mixed",
            polarization_mix=Fit(0.5, abs_min=0.0, abs_max=1.0, label="measurement_polarization_mix"),
        ),
        optimizer={
            "method": "L-BFGS-B",
            "options": {"maxiter": 8},
            "global_options": {"maxiter": 1, "popsize": 5, "seed": 11},
            "fd_rel_step": 1e-4,
        },
        max_internal_reflections=2,
        out_dir=tmp_path / "measured_fit_from_json",
    )
    setup_path = write_fit_setup_json(tmp_path / "fit_setup.json", setup)
    result = run_measured_fit_from_setup_json(setup_path)

    assert result.out_dir.exists()
    assert result.artifact_paths["measured_fit_summary_json"].exists()


def test_run_measured_fit_reflection_smoke_on_example_pair(tmp_path):
    data_root = _data_root() / "A11008858_reflection"
    reference_csv = data_root / "reflection_setup_ref_after_with_AuMirror_A1100858_avg600_onDryAir10min_int55.csv"
    sample_csv = data_root / "reflection_setup_sample_A1100858_avg600_onDryAir10min_int27.csv"
    if not reference_csv.exists() or not sample_csv.exists():
        pytest.skip("Test_data_for_fitter reflection pair is not available")

    prepared = prepare_trace_pair_for_fit(
        reference_csv,
        sample_csv,
        baseline_mode="auto_pre_pulse",
        baseline_window_samples=40,
        crop_mode="auto",
    )
    result = run_measured_fit(
        prepared,
        layers=[
            Layer(
                name="drude_layer",
                thickness_um=Fit(550.0, abs_min=300.0, abs_max=800.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(1.1, abs_min=0.1, abs_max=3.0, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(0.08, abs_min=0.005, abs_max=0.5, label="film_gamma_thz"),
                ),
            )
        ],
        out_dir=tmp_path / "reflection_fit",
        measurement=Measurement(
            mode="reflection",
            angle_deg=Fit(10.0, abs_min=0.0, abs_max=35.0, label="measurement_angle_deg"),
            polarization="mixed",
            polarization_mix=Fit(0.5, abs_min=0.0, abs_max=1.0, label="measurement_polarization_mix"),
            reference_standard={"kind": "identity"},
        ),
        optimizer={
            "method": "L-BFGS-B",
            "options": {"maxiter": 6},
            "global_options": {"maxiter": 1, "popsize": 4, "seed": 17},
            "fd_rel_step": 1e-4,
        },
        max_internal_reflections=1,
    )

    assert result.artifact_paths["measured_fit_summary_json"].exists()
