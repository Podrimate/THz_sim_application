from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from thzsim2.core.fitting import build_objective_weights, fit_sample_trace
from thzsim2.core.fft import fft_t_to_w
from thzsim2.models import ConstantNK, Layer
from thzsim2.workflows.fit_workflow import resolve_measurement_fit_parameters
from thzsim2.workflows.reference import prepare_reference
from thzsim2.workflows.sample_workflow import build_sample

_RESIDUAL_TARGET = 1.0
_PEAK_PENALTY_SCALE = 2.0
_PEAK_PENALTY_MULTIPLIER = 5.0
_SUBTARGET_PEAK_WEIGHT = 0.05


def _default_stage_sequence():
    hybrid_options = {
        "freq_min_thz": 0.25,
        "freq_max_thz": 2.5,
        "time_weight": 1.0,
        "amplitude_weight": 0.35,
        "phase_weight": 0.15,
    }
    return [
        {
            "name": "global_hybrid",
            "metric": "hybrid_transfer",
            "metric_options": hybrid_options,
            "optimizer": {
                "global_method": "differential_evolution",
                "global_restarts": 2,
                "global_options": {
                    "maxiter": 12,
                    "popsize": 10,
                    "seed": 123,
                    "polish": False,
                    "tol": 1e-7,
                    "updating": "deferred",
                },
                "method": None,
                "fd_rel_step": 1e-5,
            },
            "next_search_window_ps": 3.0,
        },
        {
            "name": "local_hybrid",
            "metric": "hybrid_transfer",
            "metric_options": hybrid_options,
            "optimizer": {
                "global_method": "none",
                "method": "L-BFGS-B",
                "options": {"maxiter": 220},
                "fd_rel_step": 1e-5,
            },
            "next_search_window_ps": 1.0,
        },
        {
            "name": "local_relative_lp",
            "metric": "relative_lp",
            "metric_options": {"lp_order": 8.0},
            "optimizer": {
                "global_method": "none",
                "method": "L-BFGS-B",
                "options": {"maxiter": 220},
                "fd_rel_step": 1e-5,
            },
            "next_search_window_ps": 0.5,
        },
    ]


def _clone_fit_specs_with_initials(fit_specs, recovered_parameters):
    cloned = deepcopy(list(fit_specs))
    for spec in cloned:
        if spec.key in recovered_parameters:
            value = float(recovered_parameters[spec.key])
            value = min(float(spec.bound_max), max(float(spec.bound_min), value))
            spec.initial_value = value
    return cloned


def _next_delay_options(delay_options, fitted_delay_ps, stage):
    if delay_options is None and fitted_delay_ps is None:
        return None
    updated = {} if delay_options is None else deepcopy(delay_options)
    updated["enabled"] = True
    if fitted_delay_ps is not None:
        updated["initial_ps"] = float(fitted_delay_ps)
    if "next_search_window_ps" in stage and stage["next_search_window_ps"] is not None:
        updated["search_window_ps"] = float(stage["next_search_window_ps"])
    return updated


def top_residual_samples(fit_result, *, top_n=12):
    residual = np.asarray(fit_result["residual_trace"], dtype=np.float64)
    time_ps = np.asarray(fit_result["fitted_simulation"]["time_ps"], dtype=np.float64)
    fitted_trace = np.asarray(fit_result["fitted_simulation"]["sample_trace"], dtype=np.float64)
    observed_trace = fitted_trace + residual
    order = np.argsort(np.abs(residual))[-max(1, int(top_n)) :][::-1]
    return [
        {
            "rank": int(rank + 1),
            "time_ps": float(time_ps[index]),
            "residual": float(residual[index]),
            "abs_residual": float(abs(residual[index])),
            "observed": float(observed_trace[index]),
            "fit": float(fitted_trace[index]),
        }
        for rank, index in enumerate(order)
    ]


def parameter_correlation_rows(fit_result):
    correlation = fit_result.get("parameter_correlation")
    parameter_names = list(fit_result.get("parameter_names", []))
    if correlation is None or not parameter_names:
        return []
    rows = []
    correlation = np.asarray(correlation, dtype=np.float64)
    for i, name_i in enumerate(parameter_names):
        for j in range(i + 1, len(parameter_names)):
            value = float(correlation[i, j])
            if np.isfinite(value):
                rows.append(
                    {
                        "parameter_a": name_i,
                        "parameter_b": parameter_names[j],
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )
    rows.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return rows


def append_air_gap_layer(layers, *, thickness_um, position="front", name="air_gap"):
    sequence = list(layers)
    gap_layer = Layer(name=name, thickness_um=thickness_um, material=ConstantNK(n=1.0, k=0.0))
    if str(position).strip().lower() == "back":
        return sequence + [gap_layer]
    return [gap_layer] + sequence


def _positive_spectrum(trace, time_ps):
    omega, spectrum = fft_t_to_w(
        np.asarray(trace, dtype=np.float64),
        dt=float(np.median(np.diff(np.asarray(time_ps, dtype=np.float64)))) * 1e-12,
        t0=float(np.asarray(time_ps, dtype=np.float64)[0]) * 1e-12,
    )
    freq_thz = omega / (2.0 * np.pi * 1e12)
    mask = freq_thz >= 0.0
    return np.asarray(freq_thz[mask], dtype=np.float64), np.asarray(spectrum[mask], dtype=np.complex128)


def _relative_db(values, *, floor_db=-100.0, reference=None):
    values = np.asarray(values, dtype=np.float64)
    reference = max(float(np.max(values)) if reference is None else float(reference), 1e-30)
    floor_linear = 10.0 ** (float(floor_db) / 20.0)
    return 20.0 * np.log10(np.maximum(values / reference, floor_linear))


def _finite_or_inf(value):
    value = float(value)
    return value if np.isfinite(value) else float("inf")


def _residual_peak_info(fit_result):
    residual = np.asarray(fit_result["residual_trace"], dtype=np.float64)
    time_ps = np.asarray(fit_result["fitted_simulation"]["time_ps"], dtype=np.float64)
    if residual.size == 0 or time_ps.size != residual.size:
        return {
            "index": None,
            "time_ps": float("nan"),
            "value": float("nan"),
            "abs_value": float("inf"),
        }
    index = int(np.argmax(np.abs(residual)))
    value = float(residual[index])
    return {
        "index": index,
        "time_ps": float(time_ps[index]),
        "value": value,
        "abs_value": float(abs(value)),
    }


def _selection_metric_info(fit_result):
    metric_name = "weighted_data_fit" if fit_result.get("objective_weights") is not None else "data_fit"
    metric_value = float(fit_result["residual_metrics"][metric_name])
    return metric_name, metric_value


def _selection_score_components(fit_result):
    peak = _residual_peak_info(fit_result)
    residual_rms_value = float(fit_result["residual_metrics"]["residual_rms"])
    metric_name, global_metric_value = _selection_metric_info(fit_result)
    peak_excess = max(0.0, float(peak["abs_value"]) - _RESIDUAL_TARGET)
    peak_penalty = _PEAK_PENALTY_SCALE * np.log1p(_PEAK_PENALTY_MULTIPLIER * peak_excess)
    under_target_peak_term = _SUBTARGET_PEAK_WEIGHT * min(float(peak["abs_value"]), _RESIDUAL_TARGET)
    selection_score = (
        residual_rms_value
        + np.sqrt(max(global_metric_value, 0.0))
        + float(peak_penalty)
        + under_target_peak_term
    )
    return {
        "selection_score": float(selection_score),
        "selection_metric_name": metric_name,
        "selection_metric_value": float(global_metric_value),
        "peak_penalty": float(peak_penalty),
        "peak_excess": float(peak_excess),
        "under_target_peak_term": float(under_target_peak_term),
        "peak": peak,
        "residual_rms": residual_rms_value,
        "residual_target_passed": bool(float(peak["abs_value"]) <= _RESIDUAL_TARGET),
    }


def _selection_sort_key(selection_summary, *, fit_result):
    return (
        float(selection_summary["selection_score"]),
        float(selection_summary["residual_rms"]),
        float(selection_summary["selection_metric_value"]),
        _finite_or_inf(fit_result.get("max_abs_parameter_correlation", float("inf"))),
        _finite_or_inf(fit_result.get("mean_abs_parameter_correlation", float("inf"))),
        float(selection_summary["peak"]["abs_value"]),
        abs(float(fit_result["delay_recovery"].get("fitted_delay_ps") or 0.0)),
    )


def _selection_reason(selection_summary, *, fit_result, label):
    metric_name = str(selection_summary["selection_metric_name"])
    max_corr = fit_result.get("max_abs_parameter_correlation", float("nan"))
    reason = (
        f"{label} selected by balanced score={selection_summary['selection_score']:.6f} "
        f"(residual_rms={selection_summary['residual_rms']:.6f}, "
        f"{metric_name}={selection_summary['selection_metric_value']:.6f}, "
        f"max|residual|={selection_summary['peak']['abs_value']:.6f}"
    )
    if selection_summary["peak_excess"] > 0.0:
        reason += f", peak penalty={selection_summary['peak_penalty']:.6f}"
    else:
        reason += ", peak already below target penalty threshold"
    reason += (
        f", residual peak at {selection_summary['peak']['time_ps']:.4f} ps, "
        f"max|corr|={float(max_corr):.4f})"
    )
    return reason


def _annotate_fit_result(fit_result, *, label):
    summary = _selection_score_components(fit_result)
    annotated = dict(fit_result)
    annotated["max_abs_residual"] = float(summary["peak"]["abs_value"])
    annotated["residual_peak_time_ps"] = float(summary["peak"]["time_ps"])
    annotated["residual_peak_value"] = float(summary["peak"]["value"])
    annotated["residual_target"] = float(_RESIDUAL_TARGET)
    annotated["residual_target_passed"] = bool(summary["residual_target_passed"])
    annotated["selection_metric_name"] = str(summary["selection_metric_name"])
    annotated["selection_metric_value"] = float(summary["selection_metric_value"])
    annotated["selection_peak_penalty"] = float(summary["peak_penalty"])
    annotated["selection_score"] = float(summary["selection_score"])
    annotated["selection_reason"] = _selection_reason(summary, fit_result=annotated, label=label)
    annotated["selection_components"] = {
        "residual_rms": float(summary["residual_rms"]),
        "selection_metric_name": str(summary["selection_metric_name"]),
        "selection_metric_value": float(summary["selection_metric_value"]),
        "peak_penalty": float(summary["peak_penalty"]),
        "peak_excess": float(summary["peak_excess"]),
        "under_target_peak_term": float(summary["under_target_peak_term"]),
    }
    return annotated, summary


def _annotated_stage_row(stage):
    fit_result, selection_summary = _annotate_fit_result(stage["fit_result"], label=str(stage["name"]))
    return {
        "name": str(stage["name"]),
        "metric": str(stage["metric"]),
        "metric_options": deepcopy(stage["metric_options"]),
        "fit_result": fit_result,
        "selection_score": float(selection_summary["selection_score"]),
        "selection_reason": str(fit_result["selection_reason"]),
        "selection_metric_name": str(selection_summary["selection_metric_name"]),
        "selection_metric_value": float(selection_summary["selection_metric_value"]),
        "residual_rms": float(selection_summary["residual_rms"]),
        "max_abs_residual": float(selection_summary["peak"]["abs_value"]),
        "residual_peak_time_ps": float(selection_summary["peak"]["time_ps"]),
        "residual_target_passed": bool(selection_summary["residual_target_passed"]),
        "selection_sort_key": _selection_sort_key(selection_summary, fit_result=fit_result),
    }


def plot_fit_diagnostics(
    prepared_traces,
    fit_result,
    *,
    title_prefix="Fit",
    freq_limits_thz=None,
    fft_floor_db=-90.0,
    display=True,
):
    import matplotlib.pyplot as plt

    observed_trace = np.asarray(prepared_traces.processed_sample.trace, dtype=np.float64)
    reference_trace = np.asarray(prepared_traces.processed_reference.trace, dtype=np.float64)
    fitted_trace = np.asarray(fit_result["fitted_simulation"]["sample_trace"], dtype=np.float64)
    residual = np.asarray(fit_result["residual_trace"], dtype=np.float64)
    time_ps = np.asarray(prepared_traces.processed_sample.time_ps, dtype=np.float64)

    ref_freq, ref_spec = _positive_spectrum(reference_trace, time_ps)
    obs_freq, obs_spec = _positive_spectrum(observed_trace, time_ps)
    fit_freq, fit_spec = _positive_spectrum(fitted_trace, time_ps)
    amp_reference = max(
        float(np.max(np.abs(ref_spec))),
        float(np.max(np.abs(obs_spec))),
        float(np.max(np.abs(fit_spec))),
        1e-30,
    )

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=False)
    axes[0].plot(time_ps, observed_trace, label="processed sample", linewidth=1.5)
    axes[0].plot(time_ps, fitted_trace, label="fit", linewidth=1.4)
    axes[0].plot(time_ps, reference_trace, label="processed reference", linewidth=1.0, alpha=0.8)
    axes[0].set_title(f"{title_prefix}: Time Trace")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_ps, residual, label="residual", linewidth=1.3)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    peak = _residual_peak_info(fit_result)
    if peak["index"] is not None:
        axes[1].axvline(float(peak["time_ps"]), color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8)
        axes[1].scatter(
            [float(peak["time_ps"])],
            [float(peak["value"])],
            color="tab:red",
            s=30,
            zorder=5,
            label=f"worst sample @ {float(peak['time_ps']):.3f} ps",
        )
    axes[1].set_title(
        f"{title_prefix}: Residual Trace (max |residual| = {float(np.max(np.abs(residual))):.3f}, "
        f"peak @ {float(peak['time_ps']):.3f} ps)"
    )
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(obs_freq, _relative_db(np.abs(obs_spec), floor_db=fft_floor_db, reference=amp_reference), label="processed sample")
    axes[2].plot(fit_freq, _relative_db(np.abs(fit_spec), floor_db=fft_floor_db, reference=amp_reference), label="fit")
    axes[2].plot(ref_freq, _relative_db(np.abs(ref_spec), floor_db=fft_floor_db, reference=amp_reference), label="processed reference", alpha=0.8)
    axes[2].set_title(f"{title_prefix}: FFT Amplitude (Relative)")
    axes[2].set_xlabel("Frequency (THz)")
    axes[2].set_ylabel("Amplitude (dB)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(obs_freq, np.unwrap(np.angle(obs_spec)), label="processed sample")
    axes[3].plot(fit_freq, np.unwrap(np.angle(fit_spec)), label="fit")
    axes[3].plot(ref_freq, np.unwrap(np.angle(ref_spec)), label="processed reference", alpha=0.8)
    axes[3].set_title(f"{title_prefix}: FFT Phase")
    axes[3].set_xlabel("Frequency (THz)")
    axes[3].set_ylabel("Phase (rad)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    if freq_limits_thz is not None:
        for axis in axes[2:]:
            axis.set_xlim(float(freq_limits_thz[0]), float(freq_limits_thz[1]))
    fig.tight_layout()
    if display:
        try:
            from IPython.display import display as ipy_display

            ipy_display(fig)
        except Exception:
            pass
    return fig, axes


def run_staged_fit_sample_trace(
    *,
    reference,
    observed_trace,
    initial_stack,
    fit_parameters,
    measurement=None,
    measurement_fit_parameters=None,
    delay_options=None,
    objective_weights=None,
    max_internal_reflections=0,
    stage_sequence=None,
):
    stage_sequence = _default_stage_sequence() if stage_sequence is None else list(stage_sequence)
    current_fit_parameters = deepcopy(list(fit_parameters))
    current_measurement_fit_parameters = deepcopy([] if measurement_fit_parameters is None else list(measurement_fit_parameters))
    current_delay_options = None if delay_options is None else deepcopy(delay_options)
    stage_results = []

    for stage in stage_sequence:
        if not bool(stage.get("enabled", True)):
            continue
        fit_result = fit_sample_trace(
            reference=reference,
            observed_trace=observed_trace,
            initial_stack=initial_stack,
            fit_parameters=current_fit_parameters,
            measurement=measurement,
            measurement_fit_parameters=current_measurement_fit_parameters,
            metric=stage.get("metric", "data_fit"),
            metric_options=stage.get("metric_options"),
            max_internal_reflections=max_internal_reflections,
            optimizer=stage.get("optimizer"),
            delay_options=current_delay_options,
            objective_weights=objective_weights,
        )
        stage_results.append(
            {
                "name": str(stage.get("name", fit_result["metric"])),
                "metric": str(fit_result["metric"]),
                "metric_options": deepcopy(fit_result["metric_options"]),
                "fit_result": fit_result,
            }
        )
        current_fit_parameters = _clone_fit_specs_with_initials(current_fit_parameters, fit_result["recovered_parameters"])
        current_measurement_fit_parameters = _clone_fit_specs_with_initials(
            current_measurement_fit_parameters,
            fit_result["recovered_parameters"],
        )
        current_delay_options = _next_delay_options(
            current_delay_options,
            fit_result["delay_recovery"]["fitted_delay_ps"],
            stage,
        )

    if not stage_results:
        raise ValueError("stage_sequence did not produce any enabled stages")
    annotated_stage_results = [_annotated_stage_row(stage) for stage in stage_results]
    ranked_stage_results = sorted(annotated_stage_results, key=lambda stage: stage["selection_sort_key"])
    for rank, stage in enumerate(ranked_stage_results, start=1):
        stage["selection_rank"] = int(rank)
    final_stage = ranked_stage_results[0]
    return {
        "stage_results": annotated_stage_results,
        "ranked_stage_results": ranked_stage_results,
        "final_stage_name": str(final_stage["name"]),
        "final_fit_result": final_stage["fit_result"],
        "selection_score": float(final_stage["selection_score"]),
        "selection_reason": str(final_stage["selection_reason"]),
    }


def run_staged_measured_fit(
    prepared_traces,
    layers,
    *,
    out_dir,
    measurement=None,
    weighting=None,
    delay_options=None,
    reflection_counts=(0, 2, 4, 8),
    stage_sequence=None,
    n_in=1.0,
    n_out=1.0,
    overlay_imported=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_result = prepare_reference(
        prepared_traces.processed_reference,
        output_root=out_dir / "runs",
        run_label="deepdive-reference",
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
    weighting = {} if weighting is None else dict(weighting)
    objective_weights = None
    if str(weighting.get("mode", "none")).strip().lower() != "none":
        objective_weights = build_objective_weights(
            prepared_traces.processed_sample.trace,
            mode=weighting.get("mode", "trace_amplitude"),
            floor=weighting.get("floor", 0.05),
            power=weighting.get("power", 2.0),
            smooth_window_samples=weighting.get("smooth_window_samples", 41),
        )

    sweep_results = []
    for reflection_count in tuple(int(value) for value in reflection_counts):
        staged = run_staged_fit_sample_trace(
            reference=reference_result,
            observed_trace=np.asarray(prepared_traces.processed_sample.trace, dtype=np.float64),
            initial_stack=sample_result.resolved_stack,
            fit_parameters=sample_result.fit_parameters,
            measurement=resolved_measurement,
            measurement_fit_parameters=measurement_fit_parameters,
            delay_options=delay_options,
            objective_weights=objective_weights,
            max_internal_reflections=reflection_count,
            stage_sequence=stage_sequence,
        )
        final_fit = staged["final_fit_result"]
        selection_summary = _selection_score_components(final_fit)
        sweep_results.append(
            {
                "max_internal_reflections": reflection_count,
                "stage_results": staged["stage_results"],
                "ranked_stage_results": staged.get("ranked_stage_results", []),
                "final_stage_name": staged["final_stage_name"],
                "fit_result": final_fit,
                "selection_score": float(selection_summary["selection_score"]),
                "selection_reason": str(final_fit["selection_reason"]),
                "selection_metric_name": str(selection_summary["selection_metric_name"]),
                "selection_metric_value": float(selection_summary["selection_metric_value"]),
                "max_abs_residual": float(selection_summary["peak"]["abs_value"]),
                "residual_peak_time_ps": float(selection_summary["peak"]["time_ps"]),
                "residual_rms": float(selection_summary["residual_rms"]),
                "residual_target_passed": bool(selection_summary["residual_target_passed"]),
                "selection_sort_key": _selection_sort_key(selection_summary, fit_result=final_fit),
            }
        )

    ranked_reflection_results = sorted(sweep_results, key=lambda row: row["selection_sort_key"])
    for rank, row in enumerate(ranked_reflection_results, start=1):
        row["selection_rank"] = int(rank)
    best = ranked_reflection_results[0]
    return {
        "prepared_traces": prepared_traces,
        "reference_result": reference_result,
        "sample_result": sample_result,
        "measurement_fit_parameters": measurement_fit_parameters,
        "objective_weights": objective_weights,
        "weighting": deepcopy(weighting),
        "reflection_results": sweep_results,
        "ranked_reflection_results": ranked_reflection_results,
        "best_reflection_result": best,
        "best_fit_result": best["fit_result"],
        "selection_score": float(best["selection_score"]),
        "selection_reason": str(best["selection_reason"]),
    }
