from pathlib import Path

import numpy as np
import pytest

from thzsim2.core.forward import simulate_sample_from_reference
from thzsim2.models import ConstantNK, Layer, Measurement, TraceData
from thzsim2.workflows.notebook_helpers import (
    estimate_single_layer_transmission_nk,
    inspect_trace_input,
    plot_trace_preview,
    preview_sample_response,
    trace_spectrum,
)
from thzsim2.workflows.reference import generate_reference_pulse, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


def _reference_result(tmp_path: Path):
    reference_input = generate_reference_pulse(
        model="sech_carrier",
        sample_count=2048,
        dt_ps=0.03,
        time_center_ps=20.0,
        pulse_center_ps=10.0,
        tau_ps=0.22,
        f0_thz=0.9,
        amp=1.0,
        phi_rad=0.0,
    )
    return prepare_reference(reference_input, output_root=tmp_path, run_label="notebook-helper-check")


def test_trace_spectrum_reports_relative_db_with_floor():
    time_ps = np.linspace(0.0, 9.0, 512)
    trace = np.exp(-((time_ps - 4.5) ** 2) / 0.5) * np.cos(2.0 * np.pi * 0.8 * (time_ps - 4.5))
    trace_data = TraceData(time_ps=time_ps, trace=trace, source_kind="synthetic")

    freq_thz, amplitude_db, phase_rad = trace_spectrum(trace_data, floor_db=-80.0)

    assert freq_thz.shape == amplitude_db.shape == phase_rad.shape
    assert float(np.max(amplitude_db)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.min(amplitude_db)) >= -80.0 - 1e-9


def test_plot_trace_preview_uses_three_stacked_axes_for_fft_preview():
    time_ps = np.linspace(0.0, 12.0, 512)
    trace = np.exp(-((time_ps - 6.0) ** 2) / 0.7) * np.cos(2.0 * np.pi * 0.6 * (time_ps - 6.0))
    trace_info = inspect_trace_input(TraceData(time_ps=time_ps, trace=trace, source_kind="synthetic"))

    fig, axes = plot_trace_preview(trace_info, title_prefix="Synthetic", show_fft=True, display=False)

    assert fig is not None
    assert len(axes) == 3
    assert axes[0].get_title() == "Synthetic: Raw And Prepared Trace"


def test_estimate_single_layer_transmission_nk_recovers_constant_nk(tmp_path):
    reference_result = _reference_result(tmp_path)
    layers = [Layer(name="film", thickness_um=550.0, material=ConstantNK(n=3.4, k=0.1))]
    sample_result = build_sample(
        layers=layers,
        reference=reference_result,
        out_dir=tmp_path / "sample_truth",
    )
    observed_trace = reference_result.trace.with_trace(
        simulate_sample_from_reference(
            reference_result,
            sample_result.resolved_stack,
            measurement=Measurement(mode="transmission", angle_deg=0.0, polarization="s"),
        )["sample_trace"]
    )

    estimate = estimate_single_layer_transmission_nk(
        reference_result.trace,
        observed_trace,
        550.0,
    )

    mask = (estimate["freq_thz"] > 0.3) & (estimate["freq_thz"] < 1.5) & estimate["valid_mask"]
    assert float(np.nanmedian(estimate["n"][mask])) == pytest.approx(3.4, abs=0.03)
    assert float(np.nanmedian(estimate["k"][mask])) == pytest.approx(0.1, abs=0.02)


def test_preview_sample_response_returns_direct_nk_estimate_when_requested(tmp_path):
    reference_result = _reference_result(tmp_path)
    layers = [Layer(name="film", thickness_um=550.0, material=ConstantNK(n=3.4, k=0.1))]
    sample_result = build_sample(
        layers=layers,
        reference=reference_result,
        out_dir=tmp_path / "sample_truth",
    )
    observed_trace = reference_result.trace.with_trace(
        simulate_sample_from_reference(
            reference_result,
            sample_result.resolved_stack,
            measurement=Measurement(mode="transmission", angle_deg=0.0, polarization="s"),
        )["sample_trace"]
    )

    sample_result, simulation, fig, axes = preview_sample_response(
        reference_result=reference_result,
        layers=layers,
        out_dir=tmp_path / "sample_preview",
        observed_trace=observed_trace,
        measurement=Measurement(mode="transmission", angle_deg=0.0, polarization="s"),
        show_fft=True,
        thickness_guess_um=550.0,
        display=False,
    )

    assert sample_result.layers[0].name == "film"
    assert simulation["nk_estimate"] is not None
    assert fig is not None
    assert len(axes) == 5
