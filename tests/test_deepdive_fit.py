from pathlib import Path

import numpy as np
import pytest

import thzsim2.workflows.deepdive_fit as deepdive_fit_module
from thzsim2.core.fitting import (
    drude_gamma_thz_from_tau_ps,
    drude_plasma_freq_thz_from_sigma_tau,
    fit_sample_trace,
    objective_metric_value,
    shift_trace_in_time,
)
from thzsim2.core.forward import simulate_sample_from_reference
from thzsim2.models import ConstantNK, Drude, Fit, Layer, Measurement, ReferenceStandard
from thzsim2.workflows.deepdive_fit import run_staged_measured_fit
from thzsim2.workflows.fit_workflow import prepare_trace_pair_for_fit, resolve_measurement_fit_parameters
from thzsim2.workflows.reference import generate_reference_pulse, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


def _reference_result(tmp_path):
    reference_input = generate_reference_pulse(
        model="sech_carrier",
        sample_count=1024,
        dt_ps=0.03,
        time_center_ps=18.0,
        pulse_center_ps=10.0,
        tau_ps=0.22,
        f0_thz=0.9,
        amp=1.0,
        phi_rad=0.0,
    )
    return prepare_reference(reference_input, output_root=tmp_path, run_label="deepdive-fit")


def _data_root():
    return Path(__file__).resolve().parents[1] / "Test_data_for_fitter"


def _fast_stage_sequence():
    return [
        {
            "name": "global_hybrid",
            "metric": "hybrid_transfer",
            "metric_options": {
                "freq_min_thz": 0.25,
                "freq_max_thz": 2.5,
                "time_weight": 1.0,
                "amplitude_weight": 0.35,
                "phase_weight": 0.15,
            },
            "optimizer": {
                "global_method": "differential_evolution",
                "global_restarts": 1,
                "global_options": {
                    "maxiter": 6,
                    "popsize": 8,
                    "seed": 123,
                    "polish": False,
                    "tol": 1e-7,
                    "updating": "deferred",
                },
                "method": None,
                "fd_rel_step": 1e-5,
            },
            "next_search_window_ps": 2.0,
        },
        {
            "name": "local_hybrid",
            "metric": "hybrid_transfer",
            "metric_options": {
                "freq_min_thz": 0.25,
                "freq_max_thz": 2.5,
                "time_weight": 1.0,
                "amplitude_weight": 0.35,
                "phase_weight": 0.15,
            },
            "optimizer": {
                "global_method": "none",
                "method": "L-BFGS-B",
                "options": {"maxiter": 120},
                "fd_rel_step": 1e-5,
            },
            "next_search_window_ps": 0.75,
        },
    ]


def _synthetic_fit_result(
    *,
    residual_trace,
    sample_trace=None,
    data_fit_value=0.1,
    weighted_data_fit_value=None,
    residual_rms_value=0.3,
    delay_ps=0.0,
    metric="data_fit",
):
    residual_trace = np.asarray(residual_trace, dtype=np.float64)
    if sample_trace is None:
        sample_trace = np.zeros_like(residual_trace)
    sample_trace = np.asarray(sample_trace, dtype=np.float64)
    if weighted_data_fit_value is None:
        weighted_data_fit_value = data_fit_value
    return {
        "metric": metric,
        "metric_options": {},
        "residual_trace": residual_trace,
        "fitted_simulation": {
            "time_ps": np.linspace(0.0, float(residual_trace.size - 1), residual_trace.size, dtype=np.float64),
            "sample_trace": sample_trace,
        },
        "residual_metrics": {
            "data_fit": float(data_fit_value),
            "weighted_data_fit": float(weighted_data_fit_value),
            "residual_rms": float(residual_rms_value),
        },
        "objective_weights": None,
        "delay_recovery": {"fitted_delay_ps": float(delay_ps)},
        "recovered_parameters": {},
        "parameter_names": [],
        "max_abs_parameter_correlation": 0.0,
        "mean_abs_parameter_correlation": 0.0,
    }


def test_ambient_replacement_matches_explicit_air_reference_stack(tmp_path):
    reference = _reference_result(tmp_path)
    stack = {
        "n_in": 1.0,
        "n_out": 1.0,
        "layers": [
            {
                "name": "film",
                "thickness_um": 125.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 2.4, "k": 0.04}},
            }
        ],
    }
    explicit_air_stack = {
        "n_in": 1.0,
        "n_out": 1.0,
        "layers": [
            {
                "name": "air_gap",
                "thickness_um": 125.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 1.0, "k": 0.0}},
            }
        ],
    }
    ambient_replacement = simulate_sample_from_reference(
        reference,
        stack,
        max_internal_reflections=4,
        measurement=Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            reference_standard=ReferenceStandard(kind="ambient_replacement"),
        ),
    )
    explicit = simulate_sample_from_reference(
        reference,
        stack,
        max_internal_reflections=4,
        measurement=Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            reference_standard=ReferenceStandard(kind="stack", stack=explicit_air_stack),
        ),
    )
    assert np.max(np.abs(ambient_replacement["sample_trace"] - explicit["sample_trace"])) < 1e-10
    assert np.max(np.abs(ambient_replacement["transfer_function"] - explicit["transfer_function"])) < 1e-10


def test_fit_sample_trace_recovers_trace_scale_offset_and_delay_without_biasing_thickness(tmp_path):
    reference = prepare_reference(
        generate_reference_pulse(
            model="sech_carrier",
            sample_count=1536,
            dt_ps=0.02,
            time_center_ps=18.0,
            pulse_center_ps=10.0,
            tau_ps=0.22,
            f0_thz=0.9,
            amp=1.0,
            phi_rad=0.0,
        ),
        output_root=tmp_path,
        run_label="nuisance-recovery",
    )
    true_sample = build_sample(
        layers=[Layer(name="film", thickness_um=120.0, material=ConstantNK(n=2.25, k=0.02))],
        reference=reference,
        out_dir=reference.run_dir / "true_sample",
    )
    true_scale = 1.08
    true_offset = -0.015
    true_delay_ps = 0.42
    observed = simulate_sample_from_reference(
        reference,
        true_sample.resolved_stack,
        max_internal_reflections=0,
        measurement=Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            trace_scale=true_scale,
            trace_offset=true_offset,
        ),
    )["sample_trace"]
    observed = shift_trace_in_time(observed, reference.trace.time_ps, true_delay_ps)

    fit_sample = build_sample(
        layers=[Layer(name="film", thickness_um=Fit(105.0, abs_min=90.0, abs_max=135.0, label="film_thickness_um"), material=ConstantNK(n=2.25, k=0.02))],
        reference=reference,
        out_dir=reference.run_dir / "fit_sample",
    )
    measurement, measurement_fit_parameters = resolve_measurement_fit_parameters(
        Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            trace_scale=Fit(1.0, abs_min=0.9, abs_max=1.2, label="measurement_trace_scale"),
            trace_offset=Fit(0.0, abs_min=-0.05, abs_max=0.05, label="measurement_trace_offset"),
        )
    )
    fit_result = fit_sample_trace(
        reference=reference,
        observed_trace=observed,
        initial_stack=fit_sample.resolved_stack,
        fit_parameters=fit_sample.fit_parameters,
        metric="hybrid_transfer",
        metric_options={
            "freq_min_thz": 0.25,
            "freq_max_thz": 2.5,
            "time_weight": 1.0,
            "amplitude_weight": 0.35,
            "phase_weight": 0.15,
        },
        max_internal_reflections=0,
        optimizer={
            "global_method": "differential_evolution",
            "global_options": {"maxiter": 6, "popsize": 8, "seed": 123, "polish": False, "tol": 1e-7, "updating": "deferred"},
            "method": "L-BFGS-B",
            "options": {"maxiter": 120},
            "fd_rel_step": 1e-5,
        },
        measurement=measurement,
        measurement_fit_parameters=measurement_fit_parameters,
        delay_options={"enabled": True, "search_window_ps": 2.0},
    )

    assert fit_result["recovered_parameters"]["film_thickness_um"] == pytest.approx(120.0, abs=1.2)
    assert fit_result["recovered_parameters"]["measurement_trace_scale"] == pytest.approx(true_scale, abs=0.03)
    assert fit_result["recovered_parameters"]["measurement_trace_offset"] == pytest.approx(true_offset, abs=0.01)
    assert fit_result["delay_recovery"]["fitted_delay_ps"] == pytest.approx(true_delay_ps, abs=0.08)


def test_relative_lp_penalizes_spikes_more_than_plain_data_fit():
    t = np.linspace(-1.0, 1.0, 201)
    truth = np.exp(-8.0 * t * t)
    spike_candidate = truth.copy()
    spike_candidate[100] += 0.8
    spread_candidate = truth + 0.09 * np.sin(8.0 * np.pi * t)

    assert objective_metric_value(spike_candidate, truth, "data_fit") < objective_metric_value(
        spread_candidate,
        truth,
        "data_fit",
    )
    assert objective_metric_value(
        spread_candidate,
        truth,
        "relative_lp",
        metric_options={"lp_order": 8.0},
    ) < objective_metric_value(
        spike_candidate,
        truth,
        "relative_lp",
        metric_options={"lp_order": 8.0},
    )


def test_staged_fit_selection_prefers_better_global_residual_over_tiny_peak_advantage(monkeypatch):
    staged_candidates = iter(
        [
            _synthetic_fit_result(
                residual_trace=[0.98, 0.0, 0.0],
                data_fit_value=0.24,
                residual_rms_value=0.82,
                metric="hybrid_transfer",
            ),
            _synthetic_fit_result(
                residual_trace=[0.99, 0.0, 0.0],
                data_fit_value=0.015,
                residual_rms_value=0.19,
                metric="relative_lp",
            ),
        ]
    )

    def fake_fit_sample_trace(**kwargs):
        return next(staged_candidates)

    monkeypatch.setattr(deepdive_fit_module, "fit_sample_trace", fake_fit_sample_trace)

    staged = deepdive_fit_module.run_staged_fit_sample_trace(
        reference=object(),
        observed_trace=np.zeros(3, dtype=np.float64),
        initial_stack={"layers": []},
        fit_parameters=[],
        stage_sequence=[
            {"name": "slightly_smaller_peak", "metric": "hybrid_transfer", "metric_options": {}, "optimizer": {}},
            {"name": "better_global_fit", "metric": "relative_lp", "metric_options": {}, "optimizer": {}},
        ],
    )

    assert staged["final_stage_name"] == "better_global_fit"
    assert staged["final_fit_result"]["selection_score"] < staged["stage_results"][0]["selection_score"]
    assert "balanced score" in staged["selection_reason"]


def test_reflection_selection_uses_balanced_score_and_returns_ranked_candidates(monkeypatch, tmp_path):
    def fake_prepare_reference(*args, **kwargs):
        return type("ReferenceResultStub", (), {"trace": type("TraceStub", (), {"trace": np.zeros(4, dtype=np.float64)})(), "run_dir": tmp_path / "ref"})()

    def fake_build_sample(*args, **kwargs):
        return type("SampleResultStub", (), {"resolved_stack": {"layers": []}, "fit_parameters": []})()

    def fake_resolve_measurement_fit_parameters(measurement):
        return measurement, []

    def fake_run_staged_fit_sample_trace(**kwargs):
        reflection_count = int(kwargs["max_internal_reflections"])
        if reflection_count == 0:
            fit_result = _synthetic_fit_result(
                residual_trace=[1.05, 0.0, 0.0, 0.0],
                data_fit_value=0.015,
                residual_rms_value=0.18,
                metric="hybrid_transfer",
            )
            fit_result["selection_score"] = 0.0
            fit_result["selection_reason"] = "reflection 0"
            fit_result["residual_peak_time_ps"] = 0.0
        else:
            fit_result = _synthetic_fit_result(
                residual_trace=[1.01, 0.0, 0.0, 0.0],
                data_fit_value=0.09,
                residual_rms_value=0.45,
                metric="hybrid_transfer",
            )
            fit_result["selection_score"] = 0.0
            fit_result["selection_reason"] = "reflection 8"
            fit_result["residual_peak_time_ps"] = 0.0
        return {
            "stage_results": [],
            "ranked_stage_results": [],
            "final_stage_name": "final",
            "final_fit_result": fit_result,
            "selection_score": 0.0,
            "selection_reason": fit_result["selection_reason"],
        }

    monkeypatch.setattr(deepdive_fit_module, "prepare_reference", fake_prepare_reference)
    monkeypatch.setattr(deepdive_fit_module, "build_sample", fake_build_sample)
    monkeypatch.setattr(deepdive_fit_module, "resolve_measurement_fit_parameters", fake_resolve_measurement_fit_parameters)
    monkeypatch.setattr(deepdive_fit_module, "run_staged_fit_sample_trace", fake_run_staged_fit_sample_trace)

    prepared = type(
        "PreparedTraceStub",
        (),
        {
            "processed_reference": type("TraceStub", (), {"trace": np.zeros(4, dtype=np.float64)})(),
            "processed_sample": type("TraceStub", (), {"trace": np.zeros(4, dtype=np.float64)})(),
        },
    )()

    result = deepdive_fit_module.run_staged_measured_fit(
        prepared,
        [],
        out_dir=tmp_path / "balanced-reflection-test",
        measurement=Measurement(mode="transmission", angle_deg=0.0, polarization="s"),
        reflection_counts=(0, 8),
    )

    assert result["best_reflection_result"]["max_internal_reflections"] == 0
    assert result["ranked_reflection_results"][0]["selection_rank"] == 1
    assert result["ranked_reflection_results"][0]["selection_score"] < result["ranked_reflection_results"][1]["selection_score"]
    assert "selection_reason" in result["best_fit_result"]
    assert "residual_peak_time_ps" in result["best_fit_result"]


def test_hybrid_transfer_reduces_phase_error_vs_time_only_fit(tmp_path):
    reference = _reference_result(tmp_path)
    true_sample = build_sample(
        layers=[Layer(name="film", thickness_um=140.0, material=Drude(eps_inf=11.8, plasma_freq_thz=0.65, gamma_thz=0.42))],
        reference=reference,
        out_dir=reference.run_dir / "true_sample",
    )
    observed = simulate_sample_from_reference(
        reference,
        true_sample.resolved_stack,
        max_internal_reflections=6,
        measurement=Measurement(mode="transmission", angle_deg=0.0, polarization="s"),
    )["sample_trace"]

    fit_sample = build_sample(
        layers=[Layer(name="film", thickness_um=Fit(115.0, abs_min=90.0, abs_max=170.0, label="film_thickness_um"), material=Drude(eps_inf=Fit(10.5, abs_min=8.0, abs_max=14.0, label="film_eps_inf"), plasma_freq_thz=Fit(0.45, abs_min=0.1, abs_max=1.2, label="film_plasma_freq_thz"), gamma_thz=Fit(0.7, abs_min=0.1, abs_max=1.4, label="film_gamma_thz")))],  # noqa: E501
        reference=reference,
        out_dir=reference.run_dir / "fit_sample",
    )
    optimizer = {
        "global_method": "differential_evolution",
        "global_options": {"maxiter": 6, "popsize": 8, "seed": 123, "polish": False, "tol": 1e-7, "updating": "deferred"},
        "method": "L-BFGS-B",
        "options": {"maxiter": 120},
        "fd_rel_step": 1e-5,
    }
    time_fit = fit_sample_trace(
        reference=reference,
        observed_trace=observed,
        initial_stack=fit_sample.resolved_stack,
        fit_parameters=fit_sample.fit_parameters,
        metric="data_fit",
        max_internal_reflections=0,
        optimizer=optimizer,
        measurement=Measurement(mode="transmission", angle_deg=0.0, polarization="s"),
        delay_options={"enabled": True, "search_window_ps": 2.0},
    )
    hybrid_fit = fit_sample_trace(
        reference=reference,
        observed_trace=observed,
        initial_stack=fit_sample.resolved_stack,
        fit_parameters=fit_sample.fit_parameters,
        metric="hybrid_transfer",
        metric_options={
            "freq_min_thz": 0.25,
            "freq_max_thz": 2.5,
            "time_weight": 1.0,
            "amplitude_weight": 0.35,
            "phase_weight": 0.15,
        },
        max_internal_reflections=0,
        optimizer=optimizer,
        measurement=Measurement(mode="transmission", angle_deg=0.0, polarization="s"),
        delay_options={"enabled": True, "search_window_ps": 2.0},
    )

    assert hybrid_fit["residual_metrics"]["transfer_phase_mse"] < time_fit["residual_metrics"]["transfer_phase_mse"]


def test_a11008858_constrained_track_improves_over_old_baseline(tmp_path):
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
    known_tau_ps = 0.25
    known_gamma = drude_gamma_thz_from_tau_ps(known_tau_ps)
    known_plasma = drude_plasma_freq_thz_from_sigma_tau(100.0 / 8.56, known_tau_ps)

    constrained = run_staged_measured_fit(
        prepared,
        [
            Layer(
                name="film",
                thickness_um=Fit(625.0, abs_min=600.0, abs_max=650.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=Fit(9.0, abs_min=7.5, abs_max=12.5, label="film_eps_inf"),
                    plasma_freq_thz=Fit(known_plasma, abs_min=0.05, abs_max=1.5, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(known_gamma, abs_min=0.05, abs_max=2.0, label="film_gamma_thz"),
                ),
            )
        ],
        out_dir=tmp_path / "constrained",
        measurement=Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            reference_standard=ReferenceStandard(kind="ambient_replacement"),
        ),
        delay_options={"enabled": True, "search_window_ps": 15.0},
        weighting={"mode": "trace_amplitude", "floor": 0.02, "power": 2.5, "smooth_window_samples": 51},
        reflection_counts=(0,),
        stage_sequence=None,
    )["best_fit_result"]

    assert np.isfinite(constrained["selection_score"])
    assert np.isfinite(constrained["residual_metrics"]["residual_rms"])
    assert np.isfinite(constrained["residual_metrics"]["data_fit"])
    assert constrained["fitted_measurement"]["polarization"] == "s"
    assert "selection_score" in constrained
    assert "selection_reason" in constrained
    assert "residual_peak_time_ps" in constrained


def test_a11008858_sample_only_balanced_track_reports_selection_diagnostics(tmp_path):
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
    known_tau_ps = 0.25
    known_gamma = drude_gamma_thz_from_tau_ps(known_tau_ps)
    known_plasma = drude_plasma_freq_thz_from_sigma_tau(100.0 / 8.56, known_tau_ps)
    balanced = run_staged_measured_fit(
        prepared,
        [
            Layer(
                name="film",
                thickness_um=Fit(625.0, abs_min=500.0, abs_max=700.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=Fit(9.0, abs_min=6.0, abs_max=14.0, label="film_eps_inf"),
                    plasma_freq_thz=Fit(known_plasma, abs_min=0.05, abs_max=1.5, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(known_gamma, abs_min=0.05, abs_max=2.0, label="film_gamma_thz"),
                ),
            )
        ],
        out_dir=tmp_path / "sample_only_balanced",
        measurement=Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            reference_standard=ReferenceStandard(kind="ambient_replacement"),
        ),
        delay_options={"enabled": True, "search_window_ps": 15.0},
        weighting={"mode": "trace_amplitude", "floor": 0.02, "power": 2.5, "smooth_window_samples": 51},
        reflection_counts=(0, 2, 4),
        stage_sequence=None,
    )

    best = balanced["best_fit_result"]
    assert "selection_score" in best
    assert "selection_reason" in best
    assert "residual_peak_time_ps" in best
    assert len(balanced["ranked_reflection_results"]) == 3
    assert balanced["ranked_reflection_results"][0]["selection_rank"] == 1
    assert "measurement_angle_deg" not in best["recovered_parameters"]
    assert "measurement_polarization_mix" not in best["recovered_parameters"]
