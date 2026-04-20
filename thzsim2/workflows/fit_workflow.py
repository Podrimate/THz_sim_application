from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from thzsim2.core.fitting import fit_sample_trace
from thzsim2.io.manifests import write_json
from thzsim2.io.run_folders import slugify
from thzsim2.io.trace_csv import read_trace_csv, write_trace_csv
from thzsim2.models import (
    Fit,
    MeasuredFitResult,
    Measurement,
    PreparedTracePair,
    ReferenceStandard,
    ResolvedMeasurementFitParameter,
    TraceData,
)
from thzsim2.workflows.reference import prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


def _load_trace_input(source, *, time_column=None, signal_column=None):
    if isinstance(source, TraceData):
        return source
    return read_trace_csv(
        source,
        time_column=time_column,
        signal_column=signal_column,
        resample="auto",
    )


def _baseline_correct(trace, *, enabled: bool, window_samples: int):
    values = np.asarray(trace, dtype=np.float64).copy()
    if not enabled:
        return values, 0.0
    window = max(1, min(int(window_samples), values.size))
    baseline = float(np.mean(values[:window]))
    return values - baseline, baseline


def _crop_trace(time_ps, trace, crop_time_window_ps):
    if crop_time_window_ps is None:
        return np.asarray(time_ps, dtype=np.float64), np.asarray(trace, dtype=np.float64)
    t_min, t_max = crop_time_window_ps
    t_min = float(t_min)
    t_max = float(t_max)
    if t_max <= t_min:
        raise ValueError("crop_time_window_ps must have t_max > t_min")
    mask = (np.asarray(time_ps, dtype=np.float64) >= t_min) & (np.asarray(time_ps, dtype=np.float64) <= t_max)
    if np.count_nonzero(mask) < 2:
        raise ValueError("crop_time_window_ps leaves fewer than two samples")
    return np.asarray(time_ps, dtype=np.float64)[mask], np.asarray(trace, dtype=np.float64)[mask]


def _display_figure(fig):
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        pass


def prepare_trace_pair_for_fit(
    reference_input,
    sample_input,
    *,
    reference_time_column=None,
    reference_signal_column=None,
    sample_time_column=None,
    sample_signal_column=None,
    baseline_subtract=False,
    baseline_window_samples=50,
    crop_time_window_ps=None,
):
    raw_reference = _load_trace_input(
        reference_input,
        time_column=reference_time_column,
        signal_column=reference_signal_column,
    )
    raw_sample = _load_trace_input(
        sample_input,
        time_column=sample_time_column,
        signal_column=sample_signal_column,
    )

    common_min = max(raw_reference.time_min_ps, raw_sample.time_min_ps)
    common_max = min(raw_reference.time_max_ps, raw_sample.time_max_ps)
    if common_max <= common_min:
        raise ValueError("reference and sample traces do not overlap in time")

    common_mask = (raw_reference.time_ps >= common_min) & (raw_reference.time_ps <= common_max)
    if np.count_nonzero(common_mask) < 2:
        raise ValueError("reference/sample overlap leaves fewer than two samples")

    aligned_time_ps = raw_reference.time_ps[common_mask]
    aligned_reference_trace = np.asarray(raw_reference.trace, dtype=np.float64)[common_mask]
    aligned_sample_trace = np.interp(
        aligned_time_ps,
        np.asarray(raw_sample.time_ps, dtype=np.float64),
        np.asarray(raw_sample.trace, dtype=np.float64),
    )

    aligned_reference = TraceData(
        time_ps=aligned_time_ps.copy(),
        trace=aligned_reference_trace.copy(),
        source_kind=raw_reference.source_kind,
        source_path=raw_reference.source_path,
        pad_factor=raw_reference.pad_factor,
        metadata=dict(raw_reference.metadata),
    )
    aligned_sample = TraceData(
        time_ps=aligned_time_ps.copy(),
        trace=aligned_sample_trace.copy(),
        source_kind=raw_sample.source_kind,
        source_path=raw_sample.source_path,
        pad_factor=1,
        metadata=dict(raw_sample.metadata),
    )

    processed_reference_trace, ref_baseline = _baseline_correct(
        aligned_reference.trace,
        enabled=bool(baseline_subtract),
        window_samples=int(baseline_window_samples),
    )
    processed_sample_trace, sample_baseline = _baseline_correct(
        aligned_sample.trace,
        enabled=bool(baseline_subtract),
        window_samples=int(baseline_window_samples),
    )

    processed_time_ps, processed_reference_trace = _crop_trace(
        aligned_reference.time_ps,
        processed_reference_trace,
        crop_time_window_ps,
    )
    _, processed_sample_trace = _crop_trace(
        aligned_sample.time_ps,
        processed_sample_trace,
        crop_time_window_ps,
    )

    processed_reference = TraceData(
        time_ps=processed_time_ps.copy(),
        trace=processed_reference_trace.copy(),
        source_kind="processed_reference",
        source_path=raw_reference.source_path,
        metadata={
            "baseline_subtract": bool(baseline_subtract),
            "baseline_window_samples": int(baseline_window_samples),
            "baseline_value": float(ref_baseline),
            "crop_time_window_ps": None
            if crop_time_window_ps is None
            else [float(crop_time_window_ps[0]), float(crop_time_window_ps[1])],
        },
    )
    processed_sample = TraceData(
        time_ps=processed_time_ps.copy(),
        trace=processed_sample_trace.copy(),
        source_kind="processed_sample",
        source_path=raw_sample.source_path,
        metadata={
            "baseline_subtract": bool(baseline_subtract),
            "baseline_window_samples": int(baseline_window_samples),
            "baseline_value": float(sample_baseline),
            "crop_time_window_ps": None
            if crop_time_window_ps is None
            else [float(crop_time_window_ps[0]), float(crop_time_window_ps[1])],
        },
    )

    return PreparedTracePair(
        raw_reference=raw_reference,
        raw_sample=raw_sample,
        aligned_reference=aligned_reference,
        aligned_sample=aligned_sample,
        processed_reference=processed_reference,
        processed_sample=processed_sample,
        metadata={
            "common_time_window_ps": [float(common_min), float(common_max)],
            "baseline_subtract": bool(baseline_subtract),
            "baseline_window_samples": int(baseline_window_samples),
            "crop_time_window_ps": None
            if crop_time_window_ps is None
            else [float(crop_time_window_ps[0]), float(crop_time_window_ps[1])],
        },
    )


def plot_trace_pair_preview(prepared_traces: PreparedTracePair, *, display=True):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    axes[0].plot(prepared_traces.raw_reference.time_ps, prepared_traces.raw_reference.trace, label="raw reference")
    axes[0].plot(prepared_traces.raw_sample.time_ps, prepared_traces.raw_sample.trace, label="raw sample")
    axes[0].set_title("Raw Uploaded Traces")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        prepared_traces.processed_reference.time_ps,
        prepared_traces.processed_reference.trace,
        label="processed reference",
    )
    axes[1].plot(
        prepared_traces.processed_sample.time_ps,
        prepared_traces.processed_sample.trace,
        label="processed sample",
    )
    axes[1].set_title("Processed Traces Used For Fitting")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Signal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    if display:
        _display_figure(fig)
    return fig, axes


def _measurement_fit_key(path: str, label: str | None):
    if label:
        return str(label)
    return f"measurement_{slugify(path)}"


def resolve_measurement_fit_parameters(measurement=None):
    measurement_obj = Measurement(
        mode="transmission",
        angle_deg=0.0,
        polarization="s",
        polarization_mix=None,
        reference_standard=ReferenceStandard(kind="identity"),
    )
    if measurement is not None:
        measurement_obj = Measurement(**measurement) if isinstance(measurement, dict) else measurement
        if not isinstance(measurement_obj, Measurement):
            raise TypeError("measurement must be a Measurement, dictionary, or None")

    fit_parameters = []
    payload = {
        "mode": measurement_obj.mode,
        "angle_deg": measurement_obj.angle_deg,
        "polarization": measurement_obj.polarization,
        "polarization_mix": measurement_obj.polarization_mix,
        "reference_standard": measurement_obj.reference_standard,
    }

    angle_value = payload["angle_deg"]
    if isinstance(angle_value, Fit):
        fit_parameters.append(
            ResolvedMeasurementFitParameter(
                key=_measurement_fit_key("angle_deg", angle_value.label),
                label=angle_value.label or "measurement_angle_deg",
                path="angle_deg",
                unit="deg",
                initial_value=float(angle_value.initial),
                bound_min=float(angle_value.resolved_min),
                bound_max=float(angle_value.resolved_max),
            )
        )
        payload["angle_deg"] = float(angle_value.initial)
    else:
        payload["angle_deg"] = float(angle_value)

    if payload["polarization"] == "mixed":
        mix_value = 0.5 if payload["polarization_mix"] is None else payload["polarization_mix"]
        if isinstance(mix_value, Fit):
            fit_parameters.append(
                ResolvedMeasurementFitParameter(
                    key=_measurement_fit_key("polarization_mix", mix_value.label),
                    label=mix_value.label or "measurement_polarization_mix",
                    path="polarization_mix",
                    unit="",
                    initial_value=float(mix_value.initial),
                    bound_min=float(mix_value.resolved_min),
                    bound_max=float(mix_value.resolved_max),
                )
            )
            payload["polarization_mix"] = float(mix_value.initial)
        else:
            payload["polarization_mix"] = float(mix_value)
    elif payload["polarization_mix"] is not None:
        payload["polarization_mix"] = float(payload["polarization_mix"])

    return Measurement(**payload), fit_parameters


def _plot_fit_overlay(path, prepared_traces: PreparedTracePair, fit_result):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    fitted_trace = np.asarray(fit_result["fitted_simulation"]["sample_trace"], dtype=np.float64)
    residual = np.asarray(fit_result["residual_trace"], dtype=np.float64)

    axes[0].plot(
        prepared_traces.processed_sample.time_ps,
        prepared_traces.processed_sample.trace,
        label="processed sample",
        linewidth=1.4,
    )
    axes[0].plot(
        prepared_traces.processed_reference.time_ps,
        prepared_traces.processed_reference.trace,
        label="processed reference",
        linewidth=1.0,
        alpha=0.8,
    )
    axes[0].plot(
        prepared_traces.processed_sample.time_ps,
        fitted_trace,
        label="fit",
        linewidth=1.4,
    )
    axes[0].set_title("Measured Fit Overlay")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    magnitude = np.abs(np.asarray(fit_result["fitted_simulation"]["transfer_function"], dtype=np.complex128))
    axes[1].plot(
        prepared_traces.processed_sample.time_ps,
        residual,
        label="residual",
        linewidth=1.2,
    )
    axes[1].set_title(
        "Residual Trace"
        + (
            ""
            if fit_result["fitted_measurement"]["polarization_mix"] is None
            else f", mix={fit_result['fitted_measurement']['polarization_mix']:.3f}"
        )
    )
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_measured_fit(
    prepared_traces: PreparedTracePair,
    layers,
    *,
    out_dir,
    measurement=None,
    optimizer=None,
    metric="mse",
    max_internal_reflections=0,
    n_in=1.0,
    n_out=1.0,
    overlay_imported=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_result = prepare_reference(
        prepared_traces.processed_reference,
        output_root=out_dir / "runs",
        run_label="measured-fit-reference",
    )
    sample_result = build_sample(
        layers=layers,
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
        n_in=n_in,
        n_out=n_out,
        overlay_imported=overlay_imported,
    )
    resolved_measurement, measurement_fit_parameters = resolve_measurement_fit_parameters(measurement)
    fit_result = fit_sample_trace(
        reference=reference_result,
        observed_trace=np.asarray(prepared_traces.processed_sample.trace, dtype=np.float64),
        initial_stack=sample_result.resolved_stack,
        fit_parameters=sample_result.fit_parameters,
        measurement_fit_parameters=measurement_fit_parameters,
        metric=metric,
        max_internal_reflections=max_internal_reflections,
        optimizer=optimizer,
        measurement=resolved_measurement,
    )

    fit_dir = reference_result.run_dir / "measured_fit"
    fit_dir.mkdir(parents=True, exist_ok=True)

    processed_reference_csv = fit_dir / "processed_reference_trace.csv"
    processed_sample_csv = fit_dir / "processed_sample_trace.csv"
    fitted_trace_csv = fit_dir / "fitted_sample_trace.csv"
    residual_trace_csv = fit_dir / "residual_trace.csv"
    summary_json = fit_dir / "measured_fit_summary.json"
    overlay_png = fit_dir / "measured_fit_overlay.png"

    write_trace_csv(processed_reference_csv, prepared_traces.processed_reference)
    write_trace_csv(processed_sample_csv, prepared_traces.processed_sample)
    write_trace_csv(
        fitted_trace_csv,
        prepared_traces.processed_sample.with_trace(fit_result["fitted_simulation"]["sample_trace"]),
    )
    write_trace_csv(
        residual_trace_csv,
        prepared_traces.processed_sample.with_trace(fit_result["residual_trace"]),
    )
    _plot_fit_overlay(overlay_png, prepared_traces, fit_result)
    write_json(
        summary_json,
        {
            "measurement": fit_result["fitted_measurement"],
            "recovered_parameters": fit_result["recovered_parameters"],
            "parameter_sigmas": fit_result["parameter_sigmas"],
            "objective_value": float(fit_result["objective_value"]),
            "metric": str(fit_result["metric"]),
            "residual_metrics": deepcopy(fit_result["residual_metrics"]),
        },
    )

    return MeasuredFitResult(
        out_dir=fit_dir,
        prepared_traces=prepared_traces,
        reference_result=reference_result,
        sample_result=sample_result,
        fit_result=fit_result,
        measurement_fit_parameters=measurement_fit_parameters,
        artifact_paths={
            "processed_reference_trace_csv": processed_reference_csv,
            "processed_sample_trace_csv": processed_sample_csv,
            "fitted_sample_trace_csv": fitted_trace_csv,
            "residual_trace_csv": residual_trace_csv,
            "measured_fit_summary_json": summary_json,
            "measured_fit_overlay_png": overlay_png,
        },
        metadata={
            "n_in": float(n_in),
            "n_out": float(n_out),
            "metric": str(metric),
            "max_internal_reflections": int(max_internal_reflections),
        },
    )
