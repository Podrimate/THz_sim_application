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


_CROP_MODES = {"auto", "manual", "none"}
_BASELINE_MODES = {"auto_pre_pulse", "first_samples", "none"}
_AUTO_CROP_PRE_PULSE_PS = 8.0
_AUTO_CROP_POST_PULSE_PS = 30.0
_AUTO_BASELINE_GAP_PS = 1.0
_AUTO_BASELINE_WINDOW_PS = 5.0


def _detect_peak_info(time_ps, trace):
    times = np.asarray(time_ps, dtype=np.float64)
    values = np.asarray(trace, dtype=np.float64)
    index = int(np.argmax(np.abs(values)))
    return {
        "index": index,
        "time_ps": float(times[index]),
        "value": float(values[index]),
        "abs_value": float(abs(values[index])),
    }


def _normalize_crop_mode(crop_mode, *, crop_time_window_ps):
    if crop_mode is None:
        return "manual" if crop_time_window_ps is not None else "none"
    mode = str(crop_mode).strip().lower()
    if mode not in _CROP_MODES:
        raise ValueError(f"crop_mode must be one of {sorted(_CROP_MODES)}")
    return mode


def _normalize_baseline_mode(baseline_mode, *, baseline_subtract):
    if baseline_mode is None:
        return "first_samples" if baseline_subtract else "none"
    mode = str(baseline_mode).strip().lower()
    if mode not in _BASELINE_MODES:
        raise ValueError(f"baseline_mode must be one of {sorted(_BASELINE_MODES)}")
    return mode


def _crop_mask(time_ps, crop_time_window_ps):
    times = np.asarray(time_ps, dtype=np.float64)
    if crop_time_window_ps is None:
        return np.ones(times.shape, dtype=bool), [float(times[0]), float(times[-1])]
    t_min, t_max = crop_time_window_ps
    t_min = float(t_min)
    t_max = float(t_max)
    if t_max <= t_min:
        raise ValueError("crop_time_window_ps must have t_max > t_min")
    mask = (times >= t_min) & (times <= t_max)
    if np.count_nonzero(mask) < 2:
        raise ValueError("crop_time_window_ps leaves fewer than two samples")
    cropped_times = times[mask]
    return mask, [float(cropped_times[0]), float(cropped_times[-1])]


def _resolve_crop_window(time_ps, *, crop_mode, crop_time_window_ps, aligned_reference_peak, aligned_sample_peak):
    times = np.asarray(time_ps, dtype=np.float64)
    full_window = [float(times[0]), float(times[-1])]
    if crop_mode == "none":
        return np.ones(times.shape, dtype=bool), full_window, None
    if crop_mode == "manual":
        if crop_time_window_ps is None:
            raise ValueError("crop_mode='manual' requires crop_time_window_ps")
        mask, actual_window = _crop_mask(times, crop_time_window_ps)
        return mask, actual_window, [float(crop_time_window_ps[0]), float(crop_time_window_ps[1])]

    pulse_min = min(float(aligned_reference_peak["time_ps"]), float(aligned_sample_peak["time_ps"]))
    pulse_max = max(float(aligned_reference_peak["time_ps"]), float(aligned_sample_peak["time_ps"]))
    t_min = max(full_window[0], pulse_min - _AUTO_CROP_PRE_PULSE_PS)
    t_max = min(full_window[1], pulse_max + _AUTO_CROP_POST_PULSE_PS)
    mask, actual_window = _crop_mask(times, (t_min, t_max))
    return mask, actual_window, [float(t_min), float(t_max)]


def _build_first_sample_baseline_mask(time_ps, *, window_samples: int):
    times = np.asarray(time_ps, dtype=np.float64)
    window = max(1, min(int(window_samples), times.size))
    mask = np.zeros(times.shape, dtype=bool)
    mask[:window] = True
    return mask, {
        "mode": "first_samples",
        "interval_ps": [float(times[0]), float(times[window - 1])],
        "sample_count": int(window),
        "fallback_used": False,
    }


def _resolve_baseline_mask(
    time_ps,
    *,
    baseline_mode,
    baseline_window_samples,
    aligned_reference_peak,
    aligned_sample_peak,
):
    times = np.asarray(time_ps, dtype=np.float64)
    if baseline_mode == "none":
        return np.ones(times.shape, dtype=bool), {
            "mode": "none",
            "interval_ps": [float(times[0]), float(times[0])],
            "sample_count": 0,
            "fallback_used": False,
        }
    if baseline_mode == "first_samples":
        return _build_first_sample_baseline_mask(times, window_samples=baseline_window_samples)

    anchor_time = min(float(aligned_reference_peak["time_ps"]), float(aligned_sample_peak["time_ps"]))
    interval_end = anchor_time - _AUTO_BASELINE_GAP_PS
    interval_start = max(float(times[0]), interval_end - _AUTO_BASELINE_WINDOW_PS)
    mask = (times >= interval_start) & (times <= interval_end)
    if np.count_nonzero(mask) >= 2:
        baseline_times = times[mask]
        return mask, {
            "mode": "auto_pre_pulse",
            "interval_ps": [float(baseline_times[0]), float(baseline_times[-1])],
            "sample_count": int(np.count_nonzero(mask)),
            "fallback_used": False,
        }

    mask, info = _build_first_sample_baseline_mask(times, window_samples=baseline_window_samples)
    info.update(
        {
            "mode": "auto_pre_pulse",
            "fallback_used": True,
            "fallback_reason": "not enough pre-pulse samples before the detected pulse",
        }
    )
    return mask, info


def _apply_baseline_mask(trace, baseline_mask, *, baseline_mode):
    values = np.asarray(trace, dtype=np.float64).copy()
    if baseline_mode == "none":
        return values, 0.0
    baseline = float(np.mean(values[np.asarray(baseline_mask, dtype=bool)]))
    return values - baseline, baseline


def _window_contains_peak(window_ps, peak_info):
    return float(window_ps[0]) <= float(peak_info["time_ps"]) <= float(window_ps[1])


def _display_figure(fig):
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        pass


def summarize_trace_input(source, *, time_column=None, signal_column=None):
    trace = _load_trace_input(source, time_column=time_column, signal_column=signal_column)
    peak = _detect_peak_info(trace.time_ps, trace.trace)
    import_columns = dict(trace.metadata.get("import_columns", {}))
    time_axis = dict(trace.metadata.get("time_axis", {}))
    import_units = dict(trace.metadata.get("import_units", {}))
    return {
        "source_path": trace.source_path,
        "sample_count": int(trace.sample_count),
        "dt_ps": float(trace.dt_ps),
        "time_min_ps": float(trace.time_min_ps),
        "time_max_ps": float(trace.time_max_ps),
        "time_column": import_columns.get("time"),
        "signal_column": import_columns.get("signal"),
        "time_unit": import_units.get("time"),
        "signal_unit": import_units.get("signal"),
        "resampled_to_uniform_grid": bool(time_axis.get("resampled_to_uniform_grid", False)),
        "peak_time_ps": float(peak["time_ps"]),
        "peak_value": float(peak["value"]),
    }


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
    baseline_mode=None,
    crop_mode=None,
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

    raw_reference_peak = _detect_peak_info(raw_reference.time_ps, raw_reference.trace)
    raw_sample_peak = _detect_peak_info(raw_sample.time_ps, raw_sample.trace)
    aligned_reference_peak = _detect_peak_info(aligned_reference.time_ps, aligned_reference.trace)
    aligned_sample_peak = _detect_peak_info(aligned_sample.time_ps, aligned_sample.trace)

    resolved_crop_mode = _normalize_crop_mode(crop_mode, crop_time_window_ps=crop_time_window_ps)
    resolved_baseline_mode = _normalize_baseline_mode(
        baseline_mode,
        baseline_subtract=baseline_subtract,
    )
    baseline_mask, baseline_info = _resolve_baseline_mask(
        aligned_reference.time_ps,
        baseline_mode=resolved_baseline_mode,
        baseline_window_samples=int(baseline_window_samples),
        aligned_reference_peak=aligned_reference_peak,
        aligned_sample_peak=aligned_sample_peak,
    )
    processed_reference_trace, ref_baseline = _apply_baseline_mask(
        aligned_reference.trace,
        baseline_mask,
        baseline_mode=resolved_baseline_mode,
    )
    processed_sample_trace, sample_baseline = _apply_baseline_mask(
        aligned_sample.trace,
        baseline_mask,
        baseline_mode=resolved_baseline_mode,
    )

    crop_mask, actual_crop_window_ps, requested_crop_window_ps = _resolve_crop_window(
        aligned_reference.time_ps,
        crop_mode=resolved_crop_mode,
        crop_time_window_ps=crop_time_window_ps,
        aligned_reference_peak=aligned_reference_peak,
        aligned_sample_peak=aligned_sample_peak,
    )
    processed_time_ps = np.asarray(aligned_reference.time_ps, dtype=np.float64)[crop_mask]
    processed_reference_trace = np.asarray(processed_reference_trace, dtype=np.float64)[crop_mask]
    processed_sample_trace = np.asarray(processed_sample_trace, dtype=np.float64)[crop_mask]

    reference_peak_retained = _window_contains_peak(actual_crop_window_ps, aligned_reference_peak)
    sample_peak_retained = _window_contains_peak(actual_crop_window_ps, aligned_sample_peak)
    warnings = []
    if resolved_crop_mode == "manual" and not reference_peak_retained:
        warnings.append("Manual crop window excludes the dominant reference pulse.")
    if resolved_crop_mode == "manual" and not sample_peak_retained:
        warnings.append("Manual crop window excludes the dominant sample pulse.")

    processed_reference = TraceData(
        time_ps=processed_time_ps.copy(),
        trace=processed_reference_trace.copy(),
        source_kind="processed_reference",
        source_path=raw_reference.source_path,
        metadata={
            "baseline_subtract": resolved_baseline_mode != "none",
            "baseline_window_samples": int(baseline_window_samples),
            "baseline_mode": resolved_baseline_mode,
            "baseline_value": float(ref_baseline),
            "crop_mode": resolved_crop_mode,
            "crop_time_window_ps": list(actual_crop_window_ps),
        },
    )
    processed_sample = TraceData(
        time_ps=processed_time_ps.copy(),
        trace=processed_sample_trace.copy(),
        source_kind="processed_sample",
        source_path=raw_sample.source_path,
        metadata={
            "baseline_subtract": resolved_baseline_mode != "none",
            "baseline_window_samples": int(baseline_window_samples),
            "baseline_mode": resolved_baseline_mode,
            "baseline_value": float(sample_baseline),
            "crop_mode": resolved_crop_mode,
            "crop_time_window_ps": list(actual_crop_window_ps),
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
            "baseline_subtract": resolved_baseline_mode != "none",
            "baseline_window_samples": int(baseline_window_samples),
            "baseline_mode": resolved_baseline_mode,
            "baseline_interval_ps": list(baseline_info["interval_ps"]),
            "baseline_sample_count": int(baseline_info["sample_count"]),
            "baseline_fallback_used": bool(baseline_info.get("fallback_used", False)),
            "crop_mode": resolved_crop_mode,
            "crop_time_window_ps": list(actual_crop_window_ps),
            "requested_crop_time_window_ps": requested_crop_window_ps,
            "raw_reference_peak": raw_reference_peak,
            "raw_sample_peak": raw_sample_peak,
            "aligned_reference_peak": aligned_reference_peak,
            "aligned_sample_peak": aligned_sample_peak,
            "processed_reference_peak_retained": bool(reference_peak_retained),
            "processed_sample_peak_retained": bool(sample_peak_retained),
            "warnings": warnings,
        },
    )


def plot_trace_pair_preview(prepared_traces: PreparedTracePair, *, display=True):
    metadata = dict(prepared_traces.metadata)
    crop_window = metadata.get(
        "crop_time_window_ps",
        [float(prepared_traces.processed_reference.time_ps[0]), float(prepared_traces.processed_reference.time_ps[-1])],
    )
    raw_reference_peak = metadata.get(
        "raw_reference_peak",
        _detect_peak_info(prepared_traces.raw_reference.time_ps, prepared_traces.raw_reference.trace),
    )
    raw_sample_peak = metadata.get(
        "raw_sample_peak",
        _detect_peak_info(prepared_traces.raw_sample.time_ps, prepared_traces.raw_sample.trace),
    )
    aligned_reference_peak = metadata.get(
        "aligned_reference_peak",
        _detect_peak_info(prepared_traces.aligned_reference.time_ps, prepared_traces.aligned_reference.trace),
    )
    aligned_sample_peak = metadata.get(
        "aligned_sample_peak",
        _detect_peak_info(prepared_traces.aligned_sample.time_ps, prepared_traces.aligned_sample.trace),
    )
    warnings = list(metadata.get("warnings", []))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

    axes[0].plot(prepared_traces.raw_reference.time_ps, prepared_traces.raw_reference.trace, label="raw reference")
    axes[0].plot(prepared_traces.raw_sample.time_ps, prepared_traces.raw_sample.trace, label="raw sample")
    axes[0].axvline(float(raw_reference_peak["time_ps"]), color="tab:blue", linestyle="--", alpha=0.6)
    axes[0].axvline(float(raw_sample_peak["time_ps"]), color="tab:orange", linestyle="--", alpha=0.6)
    axes[0].axvspan(float(crop_window[0]), float(crop_window[1]), color="tab:green", alpha=0.08, label="processed window")
    axes[0].set_title("Raw Uploaded Traces")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(prepared_traces.aligned_reference.time_ps, prepared_traces.aligned_reference.trace, label="aligned reference")
    axes[1].plot(prepared_traces.aligned_sample.time_ps, prepared_traces.aligned_sample.trace, label="aligned sample")
    axes[1].axvline(float(aligned_reference_peak["time_ps"]), color="tab:blue", linestyle="--", alpha=0.6)
    axes[1].axvline(float(aligned_sample_peak["time_ps"]), color="tab:orange", linestyle="--", alpha=0.6)
    axes[1].axvspan(float(crop_window[0]), float(crop_window[1]), color="tab:green", alpha=0.08, label="processed window")
    axes[1].set_title("Aligned Traces Before Baseline And Crop")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Signal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        prepared_traces.processed_reference.time_ps,
        prepared_traces.processed_reference.trace,
        label="processed reference",
    )
    axes[2].plot(
        prepared_traces.processed_sample.time_ps,
        prepared_traces.processed_sample.trace,
        label="processed sample",
    )
    axes[2].set_title("Processed Traces Used For Fitting")
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel("Signal")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    note_lines = [
        "What you should see: the green window should still contain the dominant reference and sample pulses.",
        f"Baseline mode: {metadata.get('baseline_mode', 'unknown')}, crop mode: {metadata.get('crop_mode', 'unknown')}.",
    ]
    if warnings:
        note_lines.extend([f"Warning: {warning}" for warning in warnings])
    fig.text(0.01, 0.01, "\n".join(note_lines), fontsize=9, va="bottom")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    if display:
        _display_figure(fig)
    return fig, axes


def summarize_prepared_trace_pair(prepared_traces: PreparedTracePair):
    metadata = dict(prepared_traces.metadata)
    return {
        "common_time_window_ps": metadata.get("common_time_window_ps"),
        "baseline_mode": metadata.get("baseline_mode"),
        "baseline_interval_ps": metadata.get("baseline_interval_ps"),
        "crop_mode": metadata.get("crop_mode"),
        "crop_time_window_ps": metadata.get("crop_time_window_ps"),
        "raw_reference_peak_time_ps": metadata.get("raw_reference_peak", {}).get("time_ps"),
        "raw_sample_peak_time_ps": metadata.get("raw_sample_peak", {}).get("time_ps"),
        "aligned_reference_peak_time_ps": metadata.get("aligned_reference_peak", {}).get("time_ps"),
        "aligned_sample_peak_time_ps": metadata.get("aligned_sample_peak", {}).get("time_ps"),
        "processed_reference_peak_retained": metadata.get("processed_reference_peak_retained"),
        "processed_sample_peak_retained": metadata.get("processed_sample_peak_retained"),
        "warnings": list(metadata.get("warnings", [])),
    }


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
