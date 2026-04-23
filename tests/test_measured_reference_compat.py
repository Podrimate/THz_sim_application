import json

import pytest

from thzsim2.workflows.study_workflow import run_single_layer_drude_compat_study


def _measured_csv(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text(
        "\n".join(
            [
                "Time_abs/ps, Signal/nA",
                "530.000, -0.05",
                "530.050, -0.02",
                "530.101, 0.01",
                "530.150, 0.08",
                "530.200, 0.12",
                "530.250, 0.09",
                "530.300, 0.03",
                "530.350, -0.01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_run_single_layer_drude_compat_study_writes_old_style_outputs(tmp_path):
    result = run_single_layer_drude_compat_study(
        _measured_csv(tmp_path),
        output_root=tmp_path,
        run_label="compat-test",
        show_progress=False,
        config_overrides={
            "sweep": {
                "true_thickness_um": [80.0],
                "true_tau_ps": [0.5],
                "true_sigma_s_per_m": [0.005],
                "noise_dynamic_range_db": [60.0],
            },
            "optimizer": {
                "options": {"maxiter": 5},
                "global_options": {"maxiter": 1, "popsize": 4, "seed": 123},
            },
            "eta_pilot_case_count": 1,
            "checkpoint_every_cases": 1,
        },
    )

    assert result.summary_csv_path.exists()
    assert result.correlation_csv_path.exists()
    assert result.manifest_path.exists()
    assert result.config_path.exists()
    assert result.artifact_paths["progress_json"].exists()
    assert result.artifact_paths["traces_dir"].exists()
    assert (result.artifact_paths["traces_dir"] / "case_000000_rep_00.csv").exists()
    assert (result.out_dir / "best_mse_trace.png").exists()
    assert (result.out_dir / "worst_mse_trace.png").exists()

    row = result.summary_rows[0]
    assert "windowed_mse" in row
    assert "peak_normalized_rmse" in row
    assert "noise_sigma" in row
    assert row["trace_file"] == "case_000000_rep_00.csv"

    progress = json.loads(result.artifact_paths["progress_json"].read_text(encoding="utf-8"))
    assert progress["completed_cases"] == 1
    assert progress["total_cases"] == 1
    assert progress["eta_s"] == pytest.approx(0.0)
