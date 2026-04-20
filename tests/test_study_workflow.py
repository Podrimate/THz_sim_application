import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from thzsim2.core.fitting import (
    drude_gamma_thz_from_tau_ps,
    drude_plasma_freq_thz_from_sigma_tau,
    sigma_s_per_m_from_drude_plasma_gamma,
    tau_ps_from_drude_gamma_thz,
)
from thzsim2.models import ConstantNK, Drude, Fit, Layer, Measurement, ReferenceStandard
from thzsim2.workflows.reference import generate_reference_pulse, load_reference_csv, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample
from thzsim2.workflows.study_setup import (
    build_study_setup,
    load_study_setup_csv,
    load_study_setup_json,
    run_study_from_setup_csv,
    run_study_from_setup_json,
    write_study_setup_csv,
    write_study_setup_json,
)
from thzsim2.workflows.study_workflow import (
    load_study_summary,
    plot_best_and_worst_case,
    plot_study_summary,
    run_study,
    show_study_heatmaps,
)


def _reference_result(tmp_path):
    reference_input = generate_reference_pulse(
        model="sech_carrier",
        sample_count=512,
        dt_ps=0.03,
        time_center_ps=8.0,
        pulse_center_ps=5.0,
        tau_ps=0.22,
        f0_thz=0.9,
        amp=1.0,
        phi_rad=0.0,
    )
    return prepare_reference(reference_input, output_root=tmp_path, run_label="study")


def _sample_result(tmp_path):
    reference_result = _reference_result(tmp_path)
    sample_result = build_sample(
        layers=[
            Layer(
                name="drude_film",
                thickness_um=Fit(150.0, rel_min=0.4, rel_max=1.8, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(0.42, rel_min=0.3, rel_max=2.0, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(0.06, rel_min=0.3, rel_max=2.5, label="film_gamma_thz"),
                ),
            )
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )
    return reference_result, sample_result


def _arbitrary_sample_result(tmp_path):
    reference_result = _reference_result(tmp_path)
    sample_result = build_sample(
        layers=[
            Layer(
                name="coating",
                thickness_um=Fit(120.0, abs_min=70.0, abs_max=190.0, label="coating_thickness_um"),
                material=ConstantNK(
                    n=Fit(2.1, abs_min=1.7, abs_max=2.6, label="coating_n"),
                    k=0.03,
                ),
            ),
            Layer(
                name="substrate",
                thickness_um=450.0,
                material=ConstantNK(n=1.55, k=0.0),
            ),
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )
    return reference_result, sample_result


def test_drude_conversion_helpers_round_trip():
    tau_ps = 5.0
    sigma_s_per_m = 0.02
    gamma_thz = drude_gamma_thz_from_tau_ps(tau_ps)
    plasma_freq_thz = drude_plasma_freq_thz_from_sigma_tau(sigma_s_per_m, tau_ps)

    assert tau_ps_from_drude_gamma_thz(gamma_thz) == pytest.approx(tau_ps)
    assert sigma_s_per_m_from_drude_plasma_gamma(plasma_freq_thz, gamma_thz) == pytest.approx(sigma_s_per_m)


def test_run_study_writes_summary_correlations_traces_and_plots(tmp_path):
    reference_result, sample_result = _sample_result(tmp_path)

    study_result = run_study(
        reference_result,
        sample_result,
        {
            "kind": "single_layer_drude",
            "replicates": 1,
            "seed": 123,
            "metric": "mse",
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 20},
                "global_options": {"maxiter": 2, "popsize": 4, "seed": 123},
                "fd_rel_step": 1e-4,
            },
            "sweep": {
                "true_thickness_um": [120.0, 180.0],
                "true_tau_ps": [3.0],
                "true_sigma_s_per_m": [0.015],
                "noise_dynamic_range_db": [80.0],
            },
        },
    )

    assert study_result.summary_csv_path.exists()
    assert study_result.correlation_csv_path.exists()
    assert study_result.manifest_path.exists()
    assert study_result.config_path.exists()
    assert (study_result.out_dir / "best_and_worst_traces.png").exists()
    assert (study_result.out_dir / "thickness_error_heatmap.png").exists()
    assert len(study_result.summary_rows) == 2
    assert len(study_result.correlation_rows) == 12

    rows = load_study_summary(study_result.summary_csv_path)
    assert rows[0]["true_thickness_um"] in (120.0, 180.0)
    assert "true__film_thickness_um" in rows[0]
    assert "fit__film_plasma_freq_thz" in rows[0]
    assert "sigma__film_gamma_thz" in rows[0]
    assert "tau_error_ps" in rows[0]
    assert "sigma_error_s_per_m" in rows[0]

    case_dir = study_result.cases_dir / "case_0000_rep_0000"
    for name in (
        "reference_trace.csv",
        "sample_true_trace.csv",
        "sample_observed_trace.csv",
        "sample_fit_trace.csv",
        "residual_trace.csv",
    ):
        assert (case_dir / name).exists()

    manifest = json.loads(study_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"] == "study"
    assert manifest["files"]["study_summary_csv"] == "study_summary.csv"

    run_manifest = json.loads((reference_result.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["simulation_manifest"] == "simulation/study_manifest.json"


def test_plot_helpers_work_from_generated_outputs(tmp_path):
    reference_result, sample_result = _sample_result(tmp_path)
    study_result = run_study(
        reference_result,
        sample_result,
        {
            "kind": "single_layer_drude",
            "replicates": 1,
            "seed": 77,
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 15},
                "global_options": {"maxiter": 1, "popsize": 4, "seed": 77},
            },
            "sweep": {
                "true_thickness_um": [100.0, 180.0],
                "true_tau_ps": [2.0, 6.0],
                "true_sigma_s_per_m": [0.01],
                "noise_dynamic_range_db": [90.0],
            },
        },
        out_dir=tmp_path / "study_outputs",
    )

    fig1, _ = plot_study_summary(
        study_result.summary_csv_path,
        x_key="true_tau_ps",
        y_key="true_thickness_um",
        value_key="mse",
    )
    fig2, _ = plot_best_and_worst_case(study_result)
    fig3, _ = show_study_heatmaps(study_result, contains="normalized-mse__", max_images=2, display=False)
    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def test_run_study_supports_arbitrary_sample_truth_paths_and_measurement(tmp_path):
    reference_result, sample_result = _arbitrary_sample_result(tmp_path)

    study_result = run_study(
        reference_result,
        sample_result,
        {
            "truth": {
                "layers[0].thickness_um": [95.0, 140.0],
                "layers[0].material.n": [1.95],
                "layers[0].material.k": 0.03,
            },
            "noise_dynamic_range_db": 85.0,
            "replicates": 1,
            "seed": 19,
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 10},
                "global_options": {"maxiter": 1, "popsize": 4, "seed": 19},
                "fd_rel_step": 1e-4,
            },
            "measurement": {
                "mode": "transmission",
                "angle_deg": 35.0,
                "polarization": "p",
                "reference_standard": {"kind": "identity"},
            },
        },
    )

    assert len(study_result.summary_rows) == 2
    row = study_result.summary_rows[0]
    assert row["measurement_mode"] == "transmission"
    assert row["measurement_angle_deg"] == pytest.approx(35.0)
    assert row["measurement_polarization"] == "p"
    assert "layers[0].thickness_um" in row
    assert "true__coating_thickness_um" in row
    assert "fit__coating_n" in row
    assert "normalized_mse" in row
    assert "signed_err__coating_n" in row
    assert any(key.startswith("normalized-mse__") for key in study_result.artifact_paths)
    assert any(key.startswith("signed-err-coating-n__") for key in study_result.artifact_paths)

    config = json.loads(study_result.config_path.read_text(encoding="utf-8"))
    assert config["kind"] == "arbitrary_sample"
    assert config["measurement"]["mode"] == "transmission"
    assert config["measurement"]["reference_standard_kind"] == "identity"


def test_run_study_recovers_easy_low_noise_drude_case_and_exports_new_metrics(tmp_path):
    reference_result = prepare_reference(
        generate_reference_pulse(
            model="sech_carrier",
            sample_count=1024,
            dt_ps=0.03,
            time_center_ps=20.0,
            pulse_center_ps=10.0,
            tau_ps=0.22,
            f0_thz=0.9,
            amp=1.0,
            phi_rad=0.0,
        ),
        output_root=tmp_path,
        run_label="study-low-noise",
    )
    sample_result = build_sample(
        layers=[
            Layer(
                name="drude_film",
                thickness_um=Fit(550.0, abs_min=450.0, abs_max=650.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(1.1, abs_min=0.7, abs_max=1.5, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(0.9, abs_min=0.4, abs_max=1.2, label="film_gamma_thz"),
                ),
            )
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )

    study_result = run_study(
        reference_result,
        sample_result,
        {
            "truth": {
                "layers[0].thickness_um": 550.0,
                "layers[0].material.plasma_freq_thz": 1.2,
                "layers[0].material.gamma_thz": 0.8,
            },
            "noise_dynamic_range_db": 120.0,
            "replicates": 1,
            "seed": 123,
            "metric": "data_fit",
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 120},
                "global_options": {"maxiter": 8, "popsize": 8, "seed": 123},
                "fd_rel_step": 1e-5,
            },
        },
        out_dir=tmp_path / "study_outputs",
    )

    assert len(study_result.summary_rows) == 1
    row = study_result.summary_rows[0]
    assert row["data_fit"] < 1e-6
    assert row["fit_sigma"] < 1e-3
    assert row["fit__film_thickness_um"] == pytest.approx(550.0, abs=0.1)
    assert row["fit__film_plasma_freq_thz"] == pytest.approx(1.2, abs=1e-3)
    assert row["fit__film_gamma_thz"] == pytest.approx(0.8, abs=1e-3)
    assert row["signed_err__film_thickness_um"] == pytest.approx(0.0, abs=0.1)
    assert row["signed_err__film_plasma_freq_thz"] == pytest.approx(0.0, abs=1e-3)
    assert row["signed_err__film_gamma_thz"] == pytest.approx(0.0, abs=1e-3)
    assert study_result.case_results[0].artifact_paths["noise_trace"].exists()


def test_run_study_supports_reflection_with_explicit_reference_standard(tmp_path):
    reference_result = _reference_result(tmp_path)
    reference_standard = build_sample(
        layers=[
            Layer(
                name="substrate",
                thickness_um=450.0,
                material=ConstantNK(n=1.55, k=0.0),
            )
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "reference_standard",
    )
    sample_result = build_sample(
        layers=[
            Layer(
                name="coating",
                thickness_um=Fit(22.0, abs_min=8.0, abs_max=40.0, label="coating_thickness_um"),
                material=ConstantNK(
                    n=Fit(2.2, abs_min=1.8, abs_max=2.6, label="coating_n"),
                    k=0.0,
                ),
            ),
            Layer(
                name="substrate",
                thickness_um=450.0,
                material=ConstantNK(n=1.55, k=0.0),
            ),
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )

    measurement = Measurement(
        mode="reflection",
        angle_deg=45.0,
        polarization="s",
        reference_standard=ReferenceStandard(kind="stack", stack=reference_standard),
    )
    study_result = run_study(
        reference_result,
        sample_result,
        {
            "truth": {
                "layers[0].thickness_um": 18.0,
                "layers[0].material.n": 2.05,
            },
            "noise_dynamic_range_db": 90.0,
            "replicates": 1,
            "seed": 7,
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 8},
                "global_options": {"maxiter": 1, "popsize": 4, "seed": 7},
                "fd_rel_step": 1e-4,
            },
        },
        measurement=measurement,
    )

    assert len(study_result.summary_rows) == 1
    row = study_result.summary_rows[0]
    assert row["measurement_mode"] == "reflection"
    assert row["measurement_polarization"] == "s"
    config = json.loads(study_result.config_path.read_text(encoding="utf-8"))
    assert config["measurement"]["reference_standard_kind"] == "stack"


def test_study_setup_csv_round_trip_and_run(tmp_path):
    setup = build_study_setup(
        reference={
            "kind": "generated_pulse",
            "generate": {
                "model": "sech_carrier",
                "sample_count": 512,
                "dt_ps": 0.03,
                "time_center_ps": 8.0,
                "pulse_center_ps": 5.0,
                "tau_ps": 0.22,
                "f0_thz": 0.9,
                "amp": 1.0,
                "phi_rad": 0.0,
            },
            "prepare": {
                "output_root": tmp_path / "runs",
                "run_label": "setup-csv",
            },
        },
        layers=[
            Layer(
                name="coating",
                thickness_um=Fit(22.0, abs_min=10.0, abs_max=35.0, label="coating_thickness_um"),
                material=ConstantNK(
                    n=Fit(2.1, abs_min=1.8, abs_max=2.4, label="coating_n"),
                    k=0.0,
                ),
            ),
            Layer(
                name="substrate",
                thickness_um=450.0,
                material=ConstantNK(n=1.55, k=0.0),
            ),
        ],
        measurement={
            "mode": "reflection",
            "angle_deg": 45.0,
            "polarization": "p",
            "reference_standard": {
                "kind": "stack",
                "stack": {
                    "layers": [
                        {
                            "name": "substrate",
                            "thickness_um": 450.0,
                            "material": {"kind": "ConstantNK", "n": 1.55, "k": 0.0},
                        }
                    ]
                },
            },
        },
        study={
            "truth": {
                "layers[0].thickness_um": [18.0, 24.0],
                "layers[0].material.n": 2.05,
            },
            "noise_dynamic_range_db": 90.0,
            "replicates": 1,
            "seed": 7,
            "metric": "mse",
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 8},
                "global_options": {"maxiter": 1, "popsize": 4, "seed": 7},
                "fd_rel_step": 1e-4,
            },
            "out_dir": tmp_path / "study-from-csv",
        },
        sample_out_dir=tmp_path / "sample-from-csv",
        notes="round-trip smoke test",
    )

    csv_path = write_study_setup_csv(tmp_path / "study_setup.csv", setup)
    loaded = load_study_setup_csv(csv_path)

    assert loaded["sample"]["out_dir"] == (tmp_path / "sample-from-csv").resolve().as_posix()
    assert loaded["reference"]["prepare"]["output_root"] == (tmp_path / "runs").resolve().as_posix()
    assert loaded["measurement"]["reference_standard"]["kind"] == "stack"

    study_result = run_study_from_setup_csv(csv_path)
    assert len(study_result.summary_rows) == 2
    row = study_result.summary_rows[0]
    assert row["measurement_mode"] == "reflection"
    assert row["measurement_polarization"] == "p"
    assert "normalized_mse" in row
    assert "fit__coating_n" in row


def test_study_setup_json_round_trip_and_run(tmp_path):
    setup = build_study_setup(
        reference={
            "kind": "generated_pulse",
            "generate": {
                "model": "sech_carrier",
                "sample_count": 384,
                "dt_ps": 0.03,
                "time_center_ps": 8.0,
                "pulse_center_ps": 5.0,
                "tau_ps": 0.22,
                "f0_thz": 0.9,
                "amp": 1.0,
                "phi_rad": 0.0,
            },
            "prepare": {
                "output_root": tmp_path / "runs-json",
                "run_label": "setup-json",
            },
        },
        layers=[
            Layer(
                name="coating",
                thickness_um=Fit(22.0, abs_min=10.0, abs_max=35.0, label="coating_thickness_um"),
                material=ConstantNK(
                    n=Fit(2.1, abs_min=1.8, abs_max=2.4, label="coating_n"),
                    k=0.0,
                ),
            ),
            Layer(
                name="substrate",
                thickness_um=450.0,
                material=ConstantNK(n=1.55, k=0.0),
            ),
        ],
        measurement={
            "mode": "reflection",
            "angle_deg": 45.0,
            "polarization": "p",
            "reference_standard": {
                "kind": "stack",
                "stack": {
                    "layers": [
                        {
                            "name": "substrate",
                            "thickness_um": 450.0,
                            "material": {"kind": "ConstantNK", "n": 1.55, "k": 0.0},
                        }
                    ]
                },
            },
        },
        study={
            "truth": {
                "layers[0].thickness_um": [18.0, 24.0],
                "layers[0].material.n": 2.05,
            },
            "noise_dynamic_range_db": 90.0,
            "replicates": 1,
            "seed": 7,
            "metric": "mse",
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 8},
                "global_options": {"maxiter": 1, "popsize": 4, "seed": 7},
                "fd_rel_step": 1e-4,
            },
            "out_dir": tmp_path / "study-from-json",
        },
    )

    json_path = write_study_setup_json(tmp_path / "study_setup.json", setup)
    loaded = load_study_setup_json(json_path)

    assert loaded["reference"]["prepare"]["output_root"] == (tmp_path / "runs-json").resolve().as_posix()
    assert loaded["measurement"]["reference_standard"]["kind"] == "stack"

    study_result = run_study_from_setup_json(json_path)
    assert len(study_result.summary_rows) == 2
    assert "normalized_mse" in study_result.summary_rows[0]


def test_measured_reference_study_path_keeps_main_pulse_available(tmp_path):
    data_root = Path(__file__).resolve().parents[1] / "Test_data_for_fitter"
    reference_csv = data_root / "A11008858_transmission" / "REFERENCE.csv"
    sample_csv = data_root / "A11008858_transmission" / "SAMPLE.csv"
    if not reference_csv.exists() or not sample_csv.exists():
        pytest.skip("Test_data_for_fitter transmission pair is not available")

    setup = build_study_setup(
        reference={
            "kind": "measured_csv",
            "path": reference_csv,
            "prepare": {
                "output_root": tmp_path / "measured-runs",
                "run_label": "measured-reference-study",
            },
        },
        layers=[
            Layer(
                name="film",
                thickness_um=Fit(550.0, abs_min=300.0, abs_max=800.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(1.1, abs_min=0.1, abs_max=3.0, label="film_plasma_freq_thz"),
                    gamma_thz=Fit(0.08, abs_min=0.005, abs_max=0.5, label="film_gamma_thz"),
                ),
            )
        ],
        measurement={
            "mode": "transmission",
            "angle_deg": 0.0,
            "polarization": "mixed",
            "polarization_mix": 0.5,
            "reference_standard": {"kind": "identity"},
        },
        study={
            "truth": {
                "layers[0].thickness_um": 550.0,
                "layers[0].material.plasma_freq_thz": 1.1,
                "layers[0].material.gamma_thz": 0.08,
            },
            "noise_dynamic_range_db": 80.0,
            "replicates": 1,
            "seed": 5,
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 6},
                "global_options": {"maxiter": 1, "popsize": 4, "seed": 5},
                "fd_rel_step": 1e-4,
            },
            "out_dir": tmp_path / "measured-reference-study",
        },
    )
    json_path = write_study_setup_json(tmp_path / "measured_reference_study.json", setup)
    study_result = run_study_from_setup_json(json_path)
    loaded_reference = load_reference_csv(reference_csv)
    reference_peak_time = float(loaded_reference.time_ps[np.argmax(np.abs(loaded_reference.trace))])
    assert reference_peak_time == pytest.approx(2611.3, abs=1.0)
    assert len(study_result.summary_rows) == 1


def test_build_study_setup_reports_readable_layer_config_errors():
    with pytest.raises(ValueError, match=r"layers\[0\]\.material\.n"):
        build_study_setup(
            reference={
                "kind": "generated_pulse",
                "generate": {
                    "model": "sech_carrier",
                    "sample_count": 128,
                    "dt_ps": 0.03,
                    "time_center_ps": 4.0,
                    "pulse_center_ps": 2.0,
                    "tau_ps": 0.2,
                    "f0_thz": 0.9,
                    "amp": 1.0,
                    "phi_rad": 0.0,
                },
            },
            layers=[
                {
                    "name": "bad_layer",
                    "thickness_um": 10.0,
                    "material": {"kind": "ConstantNK", "k": 0.0},
                }
            ],
            study={"truth": {"layers[0].thickness_um": 10.0}},
        )
