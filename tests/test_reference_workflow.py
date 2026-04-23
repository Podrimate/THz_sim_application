import json

import numpy as np
import pytest

from thzsim2.io.trace_csv import write_trace_csv
from thzsim2.models import TraceData
from thzsim2.workflows.reference import generate_reference_pulse, load_reference_csv, prepare_reference


def _generated_reference(model="gaussian_carrier"):
    return generate_reference_pulse(
        model=model,
        sample_count=2048,
        dt_ps=0.01,
        time_center_ps=0.0,
        pulse_center_ps=0.6,
        tau_ps=0.18,
        f0_thz=1.1,
        amp=1.0,
        phi_rad=0.0,
        pad_factor=1,
    )


@pytest.mark.parametrize("model", ["gaussian_carrier", "sech_carrier"])
def test_prepare_reference_generated_end_to_end(tmp_path, model):
    reference = _generated_reference(model=model)

    result = prepare_reference(reference, output_root=tmp_path, run_label=model)

    assert result.run_dir.exists()
    assert result.reference_dir.exists()
    assert result.run_id.endswith(model.replace("_", "-"))
    assert result.summary.sample_count == 2048
    assert result.summary.dt_ps == pytest.approx(0.01)
    assert result.summary.amplitude_scale > 0.0
    assert result.summary.freq_max_thz > result.summary.freq_min_thz
    assert result.summary.peak_freq_thz >= 0.0

    for path in result.artifact_paths.values():
        assert path.exists()

    manifest_path = result.artifact_paths["reference_manifest_json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"] == "reference"
    assert manifest["source"]["kind"] == "generated"
    assert manifest["files"]["reference_trace_csv"] == "reference/reference_trace.csv"
    for rel_path in manifest["files"].values():
        assert (result.run_dir / rel_path).exists()


def test_prepare_reference_loaded_csv_end_to_end(tmp_path):
    trace = TraceData(
        time_ps=np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float64),
        trace=np.array([0.1, 0.3, -0.2, 0.05], dtype=np.float64),
        source_kind="generated",
    )
    input_csv = tmp_path / "input_trace.csv"
    write_trace_csv(input_csv, trace)

    loaded = load_reference_csv(input_csv)
    result = prepare_reference(loaded, output_root=tmp_path, run_label="loaded")

    assert np.allclose(result.trace.time_ps, trace.time_ps)
    assert result.manifest["source"]["kind"] == "csv"
    assert result.manifest["source"]["path"] == str(input_csv.resolve())
    assert result.summary.sample_count == trace.sample_count


def test_prepare_reference_with_noise_changes_trace_and_records_manifest(tmp_path):
    clean = _generated_reference(model="gaussian_carrier")

    result = prepare_reference(
        clean,
        noise={"model": "white_gaussian", "sigma": 0.02, "seed": 123},
        output_root=tmp_path,
        run_label="noisy",
    )

    assert not np.allclose(result.trace.trace, clean.trace)
    assert result.manifest["source"]["applied_noise"]["sigma"] == pytest.approx(0.02)
    assert result.manifest["source"]["applied_noise"]["seed"] == 123
    assert result.summary.sample_count == clean.sample_count


def test_load_reference_csv_resamples_common_measured_headers(tmp_path):
    input_csv = tmp_path / "measured.csv"
    input_csv.write_text(
        "\n".join(
            [
                "Time_abs/ps, Signal/nA",
                "530.000, -0.1",
                "530.050, -0.2",
                "530.101, 0.0",
                "530.150, 0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_reference_csv(input_csv)

    assert loaded.source_kind == "csv"
    assert loaded.metadata["import_columns"]["time"] == "Time_abs/ps"
    assert loaded.metadata["import_columns"]["signal"] == "Signal/nA"
    assert loaded.metadata["time_axis"]["resampled_to_uniform_grid"] is True
    assert loaded.dt_ps == pytest.approx(0.05, abs=1e-6)


def test_generate_reference_pulse_rejects_pulse_center_outside_window():
    with pytest.raises(ValueError, match="pulse_center_ps must lie inside the generated time window"):
        generate_reference_pulse(
            model="gaussian_carrier",
            sample_count=256,
            dt_ps=0.02,
            time_center_ps=0.0,
            pulse_center_ps=10.0,
            tau_ps=0.2,
            f0_thz=1.0,
        )
