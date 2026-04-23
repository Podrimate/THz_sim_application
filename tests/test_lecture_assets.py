from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.generate_lecture_assets import run_lecture_map_from_spec


def _read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_run_lecture_map_from_spec_saves_one_layer_recovery_bundle(tmp_path):
    spec = {
        "slug": "one_layer_test_tx_tau_sigma_low_s_0deg",
        "title": "One-layer test map",
        "mode": "transmission",
        "angle_deg": 0.0,
        "polarization": "s",
        "tau_values": [0.1, 0.55, 1.0],
        "sigma_values": [0.01, 0.3, 1.0],
        "thickness_um": 525.0,
    }

    result = run_lecture_map_from_spec(spec, output_root=tmp_path / "lecture_one_layer", profile="smoke")

    for key in (
        "summary_csv",
        "summary_json",
        "correlation_csv",
        "averaged_correlation_csv",
        "averaged_correlation_json",
        "spec_json",
        "figure_linear_png",
        "figure_log_png",
        "figure_corr_png",
        "figure_triptych_png",
    ):
        assert Path(result[key]).exists(), key

    rows = _read_csv_rows(result["summary_csv"])
    assert len(rows) == 9
    assert all(np.isfinite(float(row["recovery_error"])) for row in rows)

    case_dir = Path(result["study_dir"]) / "cases" / "case_0000"
    for name in (
        "reference_trace.csv",
        "sample_true_trace.csv",
        "sample_observed_trace.csv",
        "sample_fit_trace.csv",
        "sample_residual_trace.csv",
        "sample_noise_trace.csv",
    ):
        assert (case_dir / name).exists(), name

    avg_rows = json.loads(Path(result["averaged_correlation_json"]).read_text(encoding="utf-8"))["rows"]
    diagonal = [row for row in avg_rows if row["param_i"] == row["param_j"]]
    assert diagonal
    assert all(abs(float(row["correlation"]) - 1.0) < 1e-12 for row in diagonal)


def test_run_lecture_map_from_spec_saves_advanced_reflection_bundle(tmp_path):
    spec = {
        "slug": "advanced_test_refl_tau1_tau2_45deg_s_525um",
        "title": "Advanced reflection test map",
        "mode": "reflection",
        "angle_deg": 45.0,
        "polarization": "s",
        "map_kind": "tau",
        "substrate_thickness_um": 525.0,
        "epi_thickness_um": 10.0,
        "sigma_fixed": 0.1,
        "x_values": [0.1, 0.55, 1.0],
        "y_values": [0.1, 0.55, 1.0],
    }

    result = run_lecture_map_from_spec(spec, output_root=tmp_path / "lecture_advanced", profile="smoke")

    rows = _read_csv_rows(result["summary_csv"])
    assert len(rows) == 9
    assert all(np.isfinite(float(row["recovery_error"])) for row in rows)
    assert all(abs(float(row["true_epi_thickness_um"]) - 10.0) < 1e-9 for row in rows)
    assert all(abs(float(row["true_substrate_thickness_um"]) - 525.0) < 1e-9 for row in rows)

    avg_rows = json.loads(Path(result["averaged_correlation_json"]).read_text(encoding="utf-8"))["rows"]
    lookup = {(row["param_i"], row["param_j"]): float(row["correlation"]) for row in avg_rows}
    for (param_i, param_j), value in lookup.items():
        if (param_j, param_i) in lookup:
            assert abs(value - lookup[(param_j, param_i)]) < 1e-9

    assert Path(result["figure_triptych_pdf"]).exists()
