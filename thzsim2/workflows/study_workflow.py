from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime
import itertools
import math
from pathlib import Path
import re
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from thzsim2.core import add_white_gaussian_noise, noise_sigma_from_dynamic_range, simulate_sample_from_reference
from thzsim2.core.fitting import (
    _set_by_path,
    apply_fit_values,
    build_single_layer_drude_true_stack,
    drude_gamma_thz_from_tau_ps,
    drude_plasma_freq_thz_from_sigma_tau,
    fit_sample_trace,
    sigma_s_per_m_from_drude_plasma_gamma,
    stack_path_from_user_path,
    summarize_single_layer_drude_stack,
    tau_ps_from_drude_gamma_thz,
)
from thzsim2.core.forward import normalize_measurement
from thzsim2.io.manifests import build_study_manifest, update_run_manifest, write_json
from thzsim2.io.run_folders import slugify
from thzsim2.io.trace_csv import read_trace_csv, write_trace_csv
from thzsim2.models import (
    Drude,
    Fit,
    Layer,
    Measurement,
    ReferenceResult,
    SampleResult,
    StudyCaseResult,
    StudyResult,
    TraceData,
)
from thzsim2.workflows.reference import load_reference_csv, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def _normalize_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _serialize_csv_value(value):
    value = _normalize_scalar(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return str(value)


def _parse_csv_value(text):
    if text == "None":
        return None
    if text == "True":
        return True
    if text == "False":
        return False
    if text == "nan":
        return float("nan")
    if text == "inf":
        return float("inf")
    if text == "-inf":
        return float("-inf")
    if _INT_RE.match(text):
        try:
            return int(text)
        except ValueError:
            pass
    if _FLOAT_RE.match(text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


def _relative_path(path: Path, root: Path):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _fieldnames(rows):
    names = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def _write_csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_csv_value(row.get(key)) for key in fieldnames})


def load_study_summary(summary_csv_path):
    rows = []
    with Path(summary_csv_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _parse_csv_value(value) for key, value in row.items()})
    return rows


def export_trace_bundle(out_dir, **traces):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, trace in traces.items():
        if trace is None:
            continue
        path = out_dir / f"{name}.csv"
        write_trace_csv(path, trace)
        exported[name] = path
    return exported


def _as_trace_data(reference: ReferenceResult, trace, *, source_kind: str):
    return TraceData(
        time_ps=reference.trace.time_ps.copy(),
        trace=np.asarray(trace, dtype=np.float64),
        source_kind=source_kind,
        metadata={},
    )


def _windowed_mse(observed_trace, fitted_trace, *, fraction=0.05):
    observed = np.asarray(observed_trace, dtype=np.float64)
    fitted = np.asarray(fitted_trace, dtype=np.float64)
    threshold = float(fraction) * float(np.max(np.abs(observed)))
    if threshold <= 0.0:
        return float(np.mean((observed - fitted) ** 2))
    mask = np.abs(observed) >= threshold
    if not np.any(mask):
        return float(np.mean((observed - fitted) ** 2))
    return float(np.mean((observed[mask] - fitted[mask]) ** 2))


def _peak_normalized_rmse(observed_trace, fitted_trace):
    observed = np.asarray(observed_trace, dtype=np.float64)
    fitted = np.asarray(fitted_trace, dtype=np.float64)
    peak = max(float(np.max(np.abs(observed))), 1e-30)
    rmse = float(np.sqrt(np.mean((observed - fitted) ** 2)))
    return rmse / peak


def _validate_single_layer_drude_sample(sample: SampleResult):
    if not isinstance(sample, SampleResult):
        raise TypeError("sample must be a SampleResult")
    if len(sample.resolved_stack["layers"]) != 1:
        raise ValueError("run_study currently supports exactly one layer for kind='single_layer_drude'")
    layer = sample.resolved_stack["layers"][0]
    if layer["material_kind"] != "Drude":
        raise ValueError("run_study kind='single_layer_drude' requires a Drude material")
    if not sample.fit_parameters:
        raise ValueError("sample must contain at least one Fit(...) parameter for run_study")


def _validate_generic_sample(sample: SampleResult):
    if not isinstance(sample, SampleResult):
        raise TypeError("sample must be a SampleResult")
    if not sample.fit_parameters:
        raise ValueError("sample must contain at least one Fit(...) parameter for run_study")


def _is_single_layer_drude_stack(resolved_stack) -> bool:
    layers = list(resolved_stack.get("layers", []))
    return len(layers) == 1 and layers[0].get("material_kind") == "Drude"


def _normalize_float_axis(value, *, field_name: str):
    values = [value] if np.isscalar(value) else list(value)
    if not values:
        raise ValueError(f"{field_name} must be a non-empty scalar or sequence")
    return [float(item) for item in values]


def _normalize_plot_settings(plot_settings, sweep_axes):
    if plot_settings is None:
        return {}
    return dict(plot_settings)


def _plot_slug(text):
    return slugify(str(text).replace("__", "-").replace("[", "-").replace("]", "").replace(".", "-"))


def _normalize_study_config(study):
    if not isinstance(study, dict):
        raise TypeError("study must be a dictionary")
    kind = str(study.get("kind", "")).strip().lower()
    sweep = dict(study.get("sweep", {}))
    is_legacy_drude = kind == "single_layer_drude" or (
        "truth" not in study
        and {"true_thickness_um", "true_tau_ps", "true_sigma_s_per_m"}.issubset(sweep.keys())
    )

    base = {
        "replicates": int(study.get("replicates", 1)),
        "seed": None if study.get("seed") is None else int(study.get("seed")),
        "seed_stride": int(study.get("seed_stride", 1000)),
        "metric": str(study.get("metric", "mse")),
        "max_internal_reflections": int(study.get("max_internal_reflections", 0)),
        "optimizer": dict(
            study.get(
                "optimizer",
                {
                    "method": "L-BFGS-B",
                    "options": {"maxiter": 35},
                    "global_options": {"maxiter": 4, "popsize": 5, "seed": 123},
                    "fd_rel_step": 1e-5,
                },
            )
        ),
    }

    if is_legacy_drude:
        required_axes = (
            "true_thickness_um",
            "true_tau_ps",
            "true_sigma_s_per_m",
            "noise_dynamic_range_db",
        )
        normalized_sweep = {}
        for key in required_axes:
            values = list(sweep.get(key, []))
            if not values:
                raise ValueError(f"study.sweep must contain a non-empty '{key}' list")
            normalized_sweep[key] = [float(value) for value in values]
        plot_settings = dict(
            study.get(
                "plot_settings",
                {
                    "thickness_error": {
                        "x_key": "true_thickness_um",
                        "y_key": "true_sigma_s_per_m",
                        "value_key": "thickness_error_um",
                        "filename": "thickness_error_heatmap.png",
                        "title": "Thickness Error",
                    },
                    "tau_error": {
                        "x_key": "true_tau_ps",
                        "y_key": "true_sigma_s_per_m",
                        "value_key": "tau_error_ps",
                        "filename": "tau_error_heatmap.png",
                        "title": "Tau Error",
                    },
                    "sigma_error": {
                        "x_key": "true_tau_ps",
                        "y_key": "true_sigma_s_per_m",
                        "value_key": "sigma_error_s_per_m",
                        "filename": "sigma_error_heatmap.png",
                        "title": "Sigma Error",
                    },
                    "normalized_mse": {
                        "x_key": "true_tau_ps",
                        "y_key": "true_sigma_s_per_m",
                        "value_key": "normalized_mse",
                        "filename": "normalized_mse_heatmap.png",
                        "title": "Normalized MSE",
                    },
                    "relative_l2": {
                        "x_key": "true_tau_ps",
                        "y_key": "true_sigma_s_per_m",
                        "value_key": "relative_l2",
                        "filename": "relative_l2_heatmap.png",
                        "title": "Relative L2",
                    },
                },
            )
        )
        return {
            **base,
            "kind": "legacy_single_layer_drude",
            "sweep_axes": normalized_sweep,
            "truth_fixed": {},
            "fixed_assignments": {},
            "plot_settings": plot_settings,
        }

    truth = dict(study.get("truth", {}))
    if not truth:
        raise ValueError("study.truth must be a non-empty dictionary for arbitrary-sample studies")

    truth_fixed = {}
    truth_sweep = {}
    for path, value in truth.items():
        key = str(path)
        if isinstance(value, (list, tuple, np.ndarray)):
            values = [float(item) for item in value]
            if not values:
                raise ValueError(f"study.truth['{key}'] must not be empty")
            truth_sweep[key] = values
        else:
            truth_fixed[key] = float(value)

    noise_values = _normalize_float_axis(
        study.get("noise_dynamic_range_db", 80.0),
        field_name="study.noise_dynamic_range_db",
    )
    sweep_axes = dict(truth_sweep)
    fixed_assignments = {}
    if len(noise_values) == 1:
        fixed_assignments["noise_dynamic_range_db"] = float(noise_values[0])
    else:
        sweep_axes["noise_dynamic_range_db"] = noise_values

    return {
        **base,
        "kind": "arbitrary_sample",
        "truth_fixed": truth_fixed,
        "sweep_axes": sweep_axes,
        "fixed_assignments": fixed_assignments,
        "plot_settings": _normalize_plot_settings(study.get("plot_settings"), sweep_axes),
    }


def _effective_seed(base_seed, case_id, replicate_id, seed_stride):
    if base_seed is None:
        return None
    return int(base_seed) + int(case_id) * int(seed_stride) + int(replicate_id)


def _axis_assignments(sweep):
    keys = tuple(sweep.keys())
    values = [sweep[key] for key in keys]
    for combo in itertools.product(*values):
        yield {key: value for key, value in zip(keys, combo, strict=True)}


def _get_true_value_for_fit_parameter(true_stack, fit_parameter):
    path = stack_path_from_user_path(fit_parameter.path)
    current = true_stack
    tokens = re.findall(r"[A-Za-z_]\w*|\[\d+\]", path)
    for token in tokens:
        if token.startswith("["):
            current = current[int(token[1:-1])]
        else:
            current = current[token]
    return float(current)


def _build_true_stack(sample: SampleResult, config, assignments):
    if config["kind"] == "legacy_single_layer_drude":
        return build_single_layer_drude_true_stack(
            sample,
            thickness_um=float(assignments["true_thickness_um"]),
            tau_ps=float(assignments["true_tau_ps"]),
            sigma_s_per_m=float(assignments["true_sigma_s_per_m"]),
        )

    stack = deepcopy(sample.resolved_stack)
    truth_values = dict(config["truth_fixed"])
    for key, value in assignments.items():
        if key == "noise_dynamic_range_db":
            continue
        truth_values[key] = float(value)
    for path, value in truth_values.items():
        _set_by_path(stack, stack_path_from_user_path(path), float(value))
    return stack


def _summary_row(case_id, replicate_id, seed, assignments, true_stack, sample, fit, *, measurement=None):
    observed_trace = fit["fitted_simulation"]["sample_trace"] + fit["residual_trace"]
    fitted_trace = fit["fitted_simulation"]["sample_trace"]
    normalized_mse = _peak_normalized_rmse(observed_trace, fitted_trace) ** 2
    row = {
        "case_id": int(case_id),
        "replicate_id": int(replicate_id),
        "seed": seed,
        "success": bool(fit["success"]),
        "objective_value": float(fit["objective_value"]),
        "mse": float(fit["residual_metrics"]["mse"]),
        "normalized_mse": float(normalized_mse),
        "windowed_mse": _windowed_mse(
            fitted_trace,
            observed_trace,
        ),
        "relative_l2": float(fit["residual_metrics"]["relative_l2"]),
        "peak_normalized_rmse": _peak_normalized_rmse(
            observed_trace,
            fitted_trace,
        ),
        "snr_db": float(fit["residual_metrics"]["snr_db"]),
        "max_abs_parameter_correlation": float(fit["max_abs_parameter_correlation"]),
        "mean_abs_parameter_correlation": float(fit["mean_abs_parameter_correlation"]),
    }
    if measurement is not None:
        row["measurement_mode"] = measurement.mode
        row["measurement_angle_deg"] = float(measurement.angle_deg)
        row["measurement_polarization"] = measurement.polarization
    row.update(assignments)

    for fit_parameter in sample.fit_parameters:
        name = fit_parameter.key
        true_value = _get_true_value_for_fit_parameter(true_stack, fit_parameter)
        init_value = float(fit_parameter.initial_value)
        fit_value = float(fit["recovered_parameters"][name])
        abs_err = abs(fit_value - true_value)
        rel_err = abs_err / max(abs(true_value), 1e-30)
        signed_err = true_value - fit_value

        row[f"true__{name}"] = true_value
        row[f"init__{name}"] = init_value
        row[f"fit__{name}"] = fit_value
        row[f"signed_err__{name}"] = signed_err
        row[f"abs_err__{name}"] = abs_err
        row[f"rel_err__{name}"] = rel_err
        sigma_map = fit["parameter_sigmas"] or {}
        row[f"sigma__{name}"] = float(sigma_map.get(name, float("nan")))

    if _is_single_layer_drude_stack(true_stack) and _is_single_layer_drude_stack(fit["fitted_stack"]):
        true_summary = summarize_single_layer_drude_stack(true_stack)
        fit_summary = summarize_single_layer_drude_stack(fit["fitted_stack"])
        true_thickness_um = float(assignments.get("true_thickness_um", true_summary["thickness_um"]))
        true_tau_ps = float(assignments.get("true_tau_ps", true_summary["tau_ps"]))
        true_sigma_s_per_m = float(assignments.get("true_sigma_s_per_m", true_summary["sigma_s_per_m"]))

        row["fit_thickness_um"] = float(fit_summary["thickness_um"])
        row["true_thickness_m"] = true_thickness_um * 1e-6
        row["fit_thickness_m"] = float(fit_summary["thickness_um"]) * 1e-6
        row["thickness_error_m"] = row["true_thickness_m"] - row["fit_thickness_m"]
        row["fit_tau_ps"] = float(fit_summary["tau_ps"])
        row["fit_sigma_s_per_m"] = float(fit_summary["sigma_s_per_m"])
        row["fit_plasma_freq_thz"] = float(fit_summary["plasma_freq_thz"])
        row["fit_gamma_thz"] = float(fit_summary["gamma_thz"])
        row["true_plasma_freq_thz"] = float(true_summary["plasma_freq_thz"])
        row["true_gamma_thz"] = float(true_summary["gamma_thz"])
        row["thickness_error_um"] = abs(true_thickness_um - float(fit_summary["thickness_um"]))
        row["tau_error_ps"] = abs(true_tau_ps - float(fit_summary["tau_ps"]))
        row["sigma_error_s_per_m"] = abs(true_sigma_s_per_m - float(fit_summary["sigma_s_per_m"]))
    return row


def _parameter_summary_rows(case_id, replicate_id, fit):
    names = list(fit["parameter_names"])
    covariance = fit["parameter_covariance"]
    correlation = fit["parameter_correlation"]
    rows = []
    for i, name_i in enumerate(names):
        for j in range(i, len(names)):
            name_j = names[j]
            rows.append(
                {
                    "case_id": int(case_id),
                    "replicate_id": int(replicate_id),
                    "param_i": name_i,
                    "param_j": name_j,
                    "covariance": float("nan") if covariance is None else float(covariance[i, j]),
                    "correlation": float("nan") if correlation is None else float(correlation[i, j]),
                }
            )
    return rows


def _aggregate_grid(rows, *, x_key, y_key, value_key):
    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = sorted({float(row[y_key]) for row in rows})
    z = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for iy, y_value in enumerate(y_values):
        for ix, x_value in enumerate(x_values):
            values = [
                float(row[value_key])
                for row in rows
                if float(row[x_key]) == x_value and float(row[y_key]) == y_value and np.isfinite(float(row[value_key]))
            ]
            if values:
                z[iy, ix] = float(np.mean(values))
    return x_values, y_values, z


def _auto_plot_settings(config, sample: SampleResult):
    sweep_keys = list(config["sweep_axes"].keys())
    if len(sweep_keys) < 2:
        return {}

    settings = {}
    for x_key, y_key in itertools.combinations(sweep_keys, 2):
        axis_slug = f"{_plot_slug(x_key)}__vs__{_plot_slug(y_key)}"
        for value_key, title in (
            ("normalized_mse", "Normalized MSE"),
            ("relative_l2", "Relative L2"),
        ):
            plot_name = f"{_plot_slug(value_key)}__{axis_slug}"
            settings[plot_name] = {
                "x_key": x_key,
                "y_key": y_key,
                "value_key": value_key,
                "filename": f"{plot_name}.png",
                "title": f"{title}: {x_key} vs {y_key}",
            }

        for fit_parameter in sample.fit_parameters:
            signed_key = f"signed_err__{fit_parameter.key}"
            plot_name = f"{_plot_slug(signed_key)}__{axis_slug}"
            settings[plot_name] = {
                "x_key": x_key,
                "y_key": y_key,
                "value_key": signed_key,
                "filename": f"{plot_name}.png",
                "title": f"True - Fitted {fit_parameter.key}: {x_key} vs {y_key}",
            }
    return settings


def plot_study_summary(summary_source, *, x_key, y_key, value_key, output_path=None, title=None):
    rows = load_study_summary(summary_source) if isinstance(summary_source, (str, Path)) else list(summary_source)
    x_values, y_values, z_values = _aggregate_grid(rows, x_key=x_key, y_key=y_key, value_key=value_key)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(z_values, origin="lower", aspect="auto")
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{value:.4g}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{value:.4g}" for value in y_values])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(value_key if title is None else title)
    fig.colorbar(image, ax=ax, label=value_key)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig, ax


def plot_best_and_worst_case(study_result: StudyResult, *, metric_key="mse", output_path=None):
    rows = list(study_result.summary_rows)
    best_row = min(rows, key=lambda row: float(row[metric_key]))
    worst_row = max(rows, key=lambda row: float(row[metric_key]))

    def _case_dir(row):
        return study_result.cases_dir / f"case_{int(row['case_id']):04d}_rep_{int(row['replicate_id']):04d}"

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for axis, row, title in zip(axes, (best_row, worst_row), ("Best Case", "Worst Case"), strict=True):
        case_dir = _case_dir(row)
        observed = read_trace_csv(case_dir / "sample_observed_trace.csv")
        fitted = read_trace_csv(case_dir / "sample_fit_trace.csv")
        truth = read_trace_csv(case_dir / "sample_true_trace.csv")
        axis.plot(observed.time_ps, observed.trace, label="observed noisy synthetic trace", linewidth=1.4)
        axis.plot(fitted.time_ps, fitted.trace, label="fit", linewidth=1.4)
        axis.plot(truth.time_ps, truth.trace, label="true", linewidth=1.1, linestyle="--")
        axis.set_ylabel("Trace (a.u.)")
        axis.set_title(f"{title}: case {int(row['case_id'])}, mse={float(row['mse']):.3e}")
        axis.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (ps)")
    axes[0].legend()
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig, axes


def run_study(reference, sample, study, *, measurement=None, out_dir=None):
    if not isinstance(reference, ReferenceResult):
        raise TypeError("reference must be a ReferenceResult")
    config = _normalize_study_config(study)
    if config["kind"] == "legacy_single_layer_drude":
        _validate_single_layer_drude_sample(sample)
    else:
        _validate_generic_sample(sample)
    measurement = normalize_measurement(study.get("measurement") if measurement is None else measurement)

    out_dir = reference.run_dir / "simulation" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    correlation_rows = []
    case_results: list[StudyCaseResult] = []

    assignments_list = [
        {**config["fixed_assignments"], **assignment}
        for assignment in _axis_assignments(config["sweep_axes"])
    ]
    for case_id, assignments in enumerate(assignments_list):
        true_stack = _build_true_stack(sample, config, assignments)
        true_simulation = simulate_sample_from_reference(
            reference,
            true_stack,
            max_internal_reflections=config["max_internal_reflections"],
            measurement=measurement,
        )
        noise_sigma = noise_sigma_from_dynamic_range(
            true_simulation["sample_trace"],
            assignments["noise_dynamic_range_db"],
        )

        for replicate_id in range(config["replicates"]):
            seed = _effective_seed(config["seed"], case_id, replicate_id, config["seed_stride"])
            observed_trace = add_white_gaussian_noise(
                true_simulation["sample_trace"],
                sigma=noise_sigma,
                seed=seed,
            )
            fit = fit_sample_trace(
                reference=reference,
                observed_trace=observed_trace,
                initial_stack=sample.resolved_stack,
                fit_parameters=sample.fit_parameters,
                metric=config["metric"],
                max_internal_reflections=config["max_internal_reflections"],
                optimizer=config["optimizer"],
                measurement=measurement,
            )

            case_dir = cases_dir / f"case_{case_id:04d}_rep_{replicate_id:04d}"
            exported = export_trace_bundle(
                case_dir,
                reference_trace=reference.trace,
                sample_true_trace=_as_trace_data(reference, true_simulation["sample_trace"], source_kind="simulation_true"),
                sample_observed_trace=_as_trace_data(reference, observed_trace, source_kind="simulation_observed"),
                sample_fit_trace=_as_trace_data(reference, fit["fitted_simulation"]["sample_trace"], source_kind="simulation_fit"),
                residual_trace=_as_trace_data(reference, fit["residual_trace"], source_kind="simulation_residual"),
            )

            summary_row = _summary_row(
                case_id,
                replicate_id,
                seed,
                assignments,
                true_stack,
                sample,
                fit,
                measurement=measurement,
            )
            summary_rows.append(summary_row)
            correlation_rows.extend(_parameter_summary_rows(case_id, replicate_id, fit))
            case_results.append(
                StudyCaseResult(
                    case_id=case_id,
                    replicate_id=replicate_id,
                    seed=seed,
                    case_dir=case_dir,
                    assignments=deepcopy(assignments),
                    success=bool(fit["success"]),
                    objective_value=float(fit["objective_value"]),
                    metric_value=float(fit["residual_metrics"][config["metric"]]),
                )
            )

    summary_csv_path = out_dir / "study_summary.csv"
    correlation_csv_path = out_dir / "study_correlations.csv"
    config_path = out_dir / "study_config.json"
    manifest_path = out_dir / "study_manifest.json"

    _write_csv_rows(summary_csv_path, summary_rows)
    _write_csv_rows(correlation_csv_path, correlation_rows)
    config_payload = deepcopy(config)
    config_payload["measurement"] = {
        "mode": measurement.mode,
        "angle_deg": float(measurement.angle_deg),
        "polarization": measurement.polarization,
        "reference_standard_kind": measurement.reference_standard.kind if measurement.reference_standard else None,
    }
    write_json(config_path, config_payload)

    artifact_paths = {
        "study_summary_csv": summary_csv_path,
        "study_correlations_csv": correlation_csv_path,
        "study_config_json": config_path,
    }

    final_plot_settings = {
        **_auto_plot_settings(config, sample),
        **config["plot_settings"],
    }
    config_payload["plot_settings"] = final_plot_settings
    write_json(config_path, config_payload)

    plot_files = {}
    for plot_name, settings in final_plot_settings.items():
        output_path = out_dir / settings["filename"]
        fig, _ = plot_study_summary(
            summary_rows,
            x_key=settings["x_key"],
            y_key=settings["y_key"],
            value_key=settings["value_key"],
            output_path=output_path,
            title=settings.get("title"),
        )
        plt.close(fig)
        plot_files[plot_name] = output_path
        artifact_paths[f"{plot_name}_plot"] = output_path

    best_worst_path = out_dir / "best_and_worst_traces.png"
    provisional = StudyResult(
        out_dir=out_dir,
        cases_dir=cases_dir,
        summary_csv_path=summary_csv_path,
        correlation_csv_path=correlation_csv_path,
        manifest_path=manifest_path,
        config_path=config_path,
        summary_rows=summary_rows,
        correlation_rows=correlation_rows,
        case_results=case_results,
        artifact_paths=dict(artifact_paths),
        manifest={},
    )
    fig, _ = plot_best_and_worst_case(provisional, output_path=best_worst_path)
    plt.close(fig)
    artifact_paths["best_and_worst_traces_png"] = best_worst_path

    created_at = datetime.now().astimezone().isoformat()
    manifest = build_study_manifest(
        created_at=created_at,
        config=config_payload,
        case_count=len(assignments_list),
        run_count=len(summary_rows),
        files={
            "study_summary_csv": summary_csv_path.name,
            "study_correlations_csv": correlation_csv_path.name,
            "study_config_json": config_path.name,
            "cases_dir": cases_dir.name,
            "plots": {name: path.name for name, path in plot_files.items()},
            "best_and_worst_traces_png": best_worst_path.name,
        },
    )
    write_json(manifest_path, manifest)
    artifact_paths["study_manifest_json"] = manifest_path

    run_manifest_path = reference.run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        update_run_manifest(run_manifest_path, simulation_manifest=_relative_path(manifest_path, reference.run_dir))

    return StudyResult(
        out_dir=out_dir,
        cases_dir=cases_dir,
        summary_csv_path=summary_csv_path,
        correlation_csv_path=correlation_csv_path,
        manifest_path=manifest_path,
        config_path=config_path,
        summary_rows=summary_rows,
        correlation_rows=correlation_rows,
        case_results=case_results,
        artifact_paths=artifact_paths,
        manifest=manifest,
    )


def _format_seconds(seconds):
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_bar(done, total, elapsed_s, avg_s):
    width = 30
    fraction = 0.0 if total <= 0 else done / total
    filled = int(round(width * fraction))
    eta_s = max(0.0, (total - done) * float(avg_s))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {done}/{total} "
        f"| elapsed {_format_seconds(elapsed_s)} "
        f"| eta {_format_seconds(eta_s)} "
        f"| avg {float(avg_s):6.2f}s/case"
    )


def _print_progress(done, total, elapsed_s, avg_s):
    print("\r" + _progress_bar(done, total, elapsed_s, avg_s), end="", flush=True)
    if done >= total:
        print()


def _write_progress_json(path, *, done, total, elapsed_s, avg_s):
    payload = {
        "completed_cases": int(done),
        "total_cases": int(total),
        "elapsed_s": float(elapsed_s),
        "avg_case_s": float(avg_s),
        "eta_s": max(0.0, (int(total) - int(done)) * float(avg_s)),
        "progress_fraction": 0.0 if total <= 0 else float(done) / float(total),
        "status_line": _progress_bar(done, total, elapsed_s, avg_s),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    write_json(path, payload)


def _pivot_heatmap(rows, x_key, y_key, value_key):
    xs, ys, z = _aggregate_grid(rows, x_key=x_key, y_key=y_key, value_key=value_key)
    return xs, ys, z


def _save_heatmap(rows, x_key, y_key, value_key, title, out_path, cmap="viridis"):
    xs, ys, z = _pivot_heatmap(rows, x_key=x_key, y_key=y_key, value_key=value_key)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f"{x:.4g}" for x in xs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels([f"{y:.4g}" for y in ys])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value_key)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_legacy_trace_plot(summary_rows, traces_dir, pick, out_path):
    target_row = min(summary_rows, key=lambda row: float(row["mse"])) if pick == "best" else max(summary_rows, key=lambda row: float(row["mse"]))
    trace_path = Path(traces_dir) / str(target_row["trace_file"])
    rows = load_study_summary(trace_path)
    rows = sorted(rows, key=lambda row: int(row["sample_index"]))
    t_ps = np.asarray([row["time_ps"] for row in rows], dtype=np.float64)
    observed = np.asarray([row["observed_trace"] for row in rows], dtype=np.float64)
    fitted = np.asarray([row["fitted_trace"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_ps, observed, label="Observed noisy synthetic trace")
    ax.plot(t_ps, fitted, "--", label="Fitted trace")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"{pick.capitalize()} MSE case: case_id={int(target_row['case_id'])}, "
        f"mse={float(target_row['mse']):.3e}, windowed_mse={float(target_row['windowed_mse']):.3e}"
    )
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _legacy_trace_rows(case_id, replicate_id, observed_trace, fitted_trace, reference):
    for sample_index, (time_ps, observed_value, fitted_value) in enumerate(
        zip(reference.trace.time_ps, observed_trace, fitted_trace, strict=True)
    ):
        yield {
            "case_id": int(case_id),
            "replicate_id": int(replicate_id),
            "sample_index": int(sample_index),
            "time_ps": float(time_ps),
            "observed_trace": float(observed_value),
            "fitted_trace": float(fitted_value),
            "residual_trace": float(observed_value - fitted_value),
        }


def _write_legacy_trace_csv(path, case_id, replicate_id, observed_trace, fitted_trace, reference):
    _write_csv_rows(
        path,
        list(_legacy_trace_rows(case_id, replicate_id, observed_trace, fitted_trace, reference)),
    )


def _pilot_case_indices(total_cases, pilot_count):
    pilot_count = max(1, min(int(pilot_count), int(total_cases)))
    if pilot_count == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, total_cases - 1, pilot_count)})


def _run_single_case_compat(reference, sample, assignments, *, noise_sigma, seed, optimizer, max_internal_reflections):
    true_stack = build_single_layer_drude_true_stack(
        sample,
        thickness_um=float(assignments["true_thickness_um"]),
        tau_ps=float(assignments["true_tau_ps"]),
        sigma_s_per_m=float(assignments["true_sigma_s_per_m"]),
    )
    true_simulation = simulate_sample_from_reference(
        reference,
        true_stack,
        max_internal_reflections=max_internal_reflections,
    )
    observed_trace = add_white_gaussian_noise(
        true_simulation["sample_trace"],
        sigma=noise_sigma,
        seed=seed,
    )
    fit = fit_sample_trace(
        reference=reference,
        observed_trace=observed_trace,
        initial_stack=sample.resolved_stack,
        fit_parameters=sample.fit_parameters,
        metric="mse",
        max_internal_reflections=max_internal_reflections,
        optimizer=optimizer,
    )
    return true_stack, true_simulation, observed_trace, fit


def run_single_layer_drude_compat_study(
    reference_csv_path,
    *,
    output_root="notebooks/runs",
    run_label="measured-single-layer-drude-compat",
    show_progress=True,
    config_overrides=None,
):
    reference_input = load_reference_csv(reference_csv_path)
    reference_result = prepare_reference(reference_input, output_root=output_root, run_label=run_label)

    initial_tau_ps = 3.0
    initial_sigma_s_per_m = 0.01
    sample_result = build_sample(
        layers=[
            Layer(
                name="drude_film",
                thickness_um=Fit(
                    150.0,
                    abs_min=40.0,
                    abs_max=320.0,
                    label="film_thickness_um",
                ),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(
                        drude_plasma_freq_thz_from_sigma_tau(initial_sigma_s_per_m, initial_tau_ps),
                        abs_min=drude_plasma_freq_thz_from_sigma_tau(5.0e-4, 30.0),
                        abs_max=drude_plasma_freq_thz_from_sigma_tau(0.2, 0.05),
                        label="film_plasma_freq_thz",
                    ),
                    gamma_thz=Fit(
                        drude_gamma_thz_from_tau_ps(initial_tau_ps),
                        abs_min=drude_gamma_thz_from_tau_ps(30.0),
                        abs_max=drude_gamma_thz_from_tau_ps(0.05),
                        label="film_gamma_thz",
                    ),
                ),
            )
        ],
        reference=reference_result,
        out_dir=reference_result.run_dir / "sample",
    )

    simulation_dir = reference_result.run_dir / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = simulation_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    progress_path = simulation_dir / "progress.json"

    config = {
        "kind": "single_layer_drude_compat",
        "reference_csv_path": str(Path(reference_csv_path).resolve()),
        "layer_nominal": {
            "eps_inf": 12.0,
            "thickness_um": 180.0,
            "tau_ps": 5.0,
            "sigma_s_per_m": 0.02,
        },
        "fit_initial": {
            "thickness_um": 150.0,
            "tau_ps": 3.0,
            "sigma_s_per_m": 0.01,
        },
        "fit_bounds": {
            "thickness_um": [40.0, 320.0],
            "tau_ps": [0.05, 30.0],
            "sigma_s_per_m": [5.0e-4, 0.2],
        },
        "sweep": {
            "true_thickness_um": np.linspace(80.0, 260.0, 4).tolist(),
            "true_tau_ps": np.linspace(0.1, 20.0, 20).tolist(),
            "true_sigma_s_per_m": np.linspace(0.001, 0.1, 20).tolist(),
            "noise_dynamic_range_db": [60.0, 80.0, 100.0],
        },
        "replicates": 1,
        "seed": 123,
        "seed_stride": 1000,
        "max_internal_reflections": 8,
        "optimizer": {
            "method": "L-BFGS-B",
            "options": {"maxiter": 70},
            "global_options": {"maxiter": 8, "popsize": 8, "seed": 123},
            "fd_rel_step": 1e-5,
        },
        "checkpoint_every_cases": 10,
        "eta_pilot_case_count": 5,
    }
    if config_overrides is not None:
        overrides = deepcopy(dict(config_overrides))
        if "sweep" in overrides:
            config["sweep"].update(dict(overrides.pop("sweep")))
        if "optimizer" in overrides:
            optimizer_overrides = dict(overrides.pop("optimizer"))
            if "options" in optimizer_overrides:
                config["optimizer"]["options"].update(dict(optimizer_overrides.pop("options")))
            if "global_options" in optimizer_overrides:
                config["optimizer"]["global_options"].update(dict(optimizer_overrides.pop("global_options")))
            config["optimizer"].update(optimizer_overrides)
        config.update(overrides)

    assignments_list = list(_axis_assignments(config["sweep"]))
    total_cases = len(assignments_list) * int(config["replicates"])
    if total_cases == 0:
        raise ValueError("compatibility study produced zero cases")

    print("Study type: single-layer Drude compatibility study with measured reference.")
    print(f"Reference CSV: {Path(reference_csv_path).resolve()}")
    print(f"Output directory: {simulation_dir}")
    print(f"Total cases: {total_cases}")

    pilot_indices = _pilot_case_indices(len(assignments_list), config["eta_pilot_case_count"])
    pilot_times = []
    for pilot_index in pilot_indices:
        pilot_assignments = assignments_list[pilot_index]
        pilot_true_stack = build_single_layer_drude_true_stack(
            sample_result,
            thickness_um=float(pilot_assignments["true_thickness_um"]),
            tau_ps=float(pilot_assignments["true_tau_ps"]),
            sigma_s_per_m=float(pilot_assignments["true_sigma_s_per_m"]),
        )
        pilot_clean = simulate_sample_from_reference(
            reference_result,
            pilot_true_stack,
            max_internal_reflections=config["max_internal_reflections"],
        )
        pilot_started = time.perf_counter()
        fit_sample_trace(
            reference=reference_result,
            observed_trace=pilot_clean["sample_trace"],
            initial_stack=sample_result.resolved_stack,
            fit_parameters=sample_result.fit_parameters,
            metric="mse",
            max_internal_reflections=config["max_internal_reflections"],
            optimizer=config["optimizer"],
        )
        pilot_times.append(float(time.perf_counter() - pilot_started))
    pilot_runtime = float(np.mean(pilot_times))
    print(
        "Initial runtime estimate from pilot cases spread across the sweep: "
        f"{_format_seconds(pilot_runtime * total_cases)} total "
        f"(about {pilot_runtime:.2f}s per case, from {len(pilot_times)} pilot runs)."
    )

    summary_rows = []
    correlation_rows = []
    case_results = []
    started_at = time.perf_counter()
    elapsed_case_times = []
    completed = 0

    for case_id, assignments in enumerate(assignments_list):
        true_stack = build_single_layer_drude_true_stack(
            sample_result,
            thickness_um=float(assignments["true_thickness_um"]),
            tau_ps=float(assignments["true_tau_ps"]),
            sigma_s_per_m=float(assignments["true_sigma_s_per_m"]),
        )
        true_simulation = simulate_sample_from_reference(
            reference_result,
            true_stack,
            max_internal_reflections=config["max_internal_reflections"],
        )
        noise_sigma = noise_sigma_from_dynamic_range(
            true_simulation["sample_trace"],
            assignments["noise_dynamic_range_db"],
        )

        for replicate_id in range(config["replicates"]):
            seed = _effective_seed(config["seed"], case_id, replicate_id, config["seed_stride"])
            case_started = time.perf_counter()
            _, _, observed_trace, fit = _run_single_case_compat(
                reference_result,
                sample_result,
                assignments,
                noise_sigma=noise_sigma,
                seed=seed,
                optimizer=config["optimizer"],
                max_internal_reflections=config["max_internal_reflections"],
            )
            runtime_s = float(time.perf_counter() - case_started)

            summary_row = _summary_row(case_id, replicate_id, seed, assignments, true_stack, sample_result, fit)
            summary_row["noise_sigma"] = float(noise_sigma)
            trace_file = f"case_{int(case_id):06d}_rep_{int(replicate_id):02d}.csv"
            summary_row["trace_file"] = trace_file
            summary_row["runtime_s"] = runtime_s
            summary_rows.append(summary_row)
            correlation_rows.extend(_parameter_summary_rows(case_id, replicate_id, fit))
            _write_legacy_trace_csv(
                traces_dir / trace_file,
                case_id,
                replicate_id,
                observed_trace,
                fit["fitted_simulation"]["sample_trace"],
                reference_result,
            )

            completed += 1
            elapsed_case_times.append(runtime_s)
            recent = elapsed_case_times[-min(20, len(elapsed_case_times)) :]
            avg_runtime = 0.4 * float(np.mean(elapsed_case_times)) + 0.6 * float(np.mean(recent))
            elapsed_total = float(time.perf_counter() - started_at)
            if show_progress:
                _print_progress(completed, total_cases, elapsed_total, avg_runtime)
            _write_progress_json(progress_path, done=completed, total=total_cases, elapsed_s=elapsed_total, avg_s=avg_runtime)

            case_results.append(
                StudyCaseResult(
                    case_id=case_id,
                    replicate_id=replicate_id,
                    seed=seed,
                    case_dir=traces_dir,
                    assignments=deepcopy(assignments),
                    success=bool(fit["success"]),
                    objective_value=float(fit["objective_value"]),
                    metric_value=float(fit["residual_metrics"]["mse"]),
                )
            )

            if completed % int(config["checkpoint_every_cases"]) == 0:
                _write_csv_rows(simulation_dir / "study_summary.csv", summary_rows)

    _write_csv_rows(simulation_dir / "study_summary.csv", summary_rows)
    _write_csv_rows(simulation_dir / "study_correlations.csv", correlation_rows)

    _save_heatmap(
        summary_rows,
        "true_thickness_um",
        "true_sigma_s_per_m",
        "thickness_error_um",
        "Thickness Error = true - fitted (um)",
        simulation_dir / "thickness_error_heatmap.png",
        cmap="coolwarm",
    )
    _save_heatmap(
        summary_rows,
        "true_tau_ps",
        "true_sigma_s_per_m",
        "tau_error_ps",
        "Tau Error = true - fitted (ps)",
        simulation_dir / "tau_error_heatmap.png",
        cmap="coolwarm",
    )
    _save_heatmap(
        summary_rows,
        "true_tau_ps",
        "true_sigma_s_per_m",
        "sigma_error_s_per_m",
        "Conductivity Error = true - fitted (S/m)",
        simulation_dir / "sigma_error_heatmap.png",
        cmap="coolwarm",
    )
    _save_heatmap(
        summary_rows,
        "true_tau_ps",
        "true_sigma_s_per_m",
        "mse",
        "MSE Heatmap",
        simulation_dir / "mse_heatmap.png",
        cmap="viridis",
    )
    _save_heatmap(
        summary_rows,
        "true_tau_ps",
        "true_sigma_s_per_m",
        "windowed_mse",
        "Windowed MSE Heatmap",
        simulation_dir / "windowed_mse_heatmap.png",
        cmap="viridis",
    )
    _save_legacy_trace_plot(summary_rows, traces_dir, "best", simulation_dir / "best_mse_trace.png")
    _save_legacy_trace_plot(summary_rows, traces_dir, "worst", simulation_dir / "worst_mse_trace.png")

    write_json(simulation_dir / "study_config.json", config)
    manifest = build_study_manifest(
        created_at=datetime.now().astimezone().isoformat(),
        config=config,
        case_count=len(assignments_list),
        run_count=len(summary_rows),
        files={
            "study_summary_csv": "study_summary.csv",
            "study_correlations_csv": "study_correlations.csv",
            "study_config_json": "study_config.json",
            "progress_json": "progress.json",
            "traces_dir": "traces",
            "plots": {
                "thickness_error_heatmap": "thickness_error_heatmap.png",
                "tau_error_heatmap": "tau_error_heatmap.png",
                "sigma_error_heatmap": "sigma_error_heatmap.png",
                "mse_heatmap": "mse_heatmap.png",
                "windowed_mse_heatmap": "windowed_mse_heatmap.png",
                "best_mse_trace": "best_mse_trace.png",
                "worst_mse_trace": "worst_mse_trace.png",
            },
        },
    )
    write_json(simulation_dir / "study_manifest.json", manifest)
    update_run_manifest(reference_result.run_manifest_path, simulation_manifest="simulation/study_manifest.json")

    artifact_paths = {
        "study_summary_csv": simulation_dir / "study_summary.csv",
        "study_correlations_csv": simulation_dir / "study_correlations.csv",
        "study_config_json": simulation_dir / "study_config.json",
        "study_manifest_json": simulation_dir / "study_manifest.json",
        "progress_json": progress_path,
        "traces_dir": traces_dir,
        "thickness_error_heatmap_png": simulation_dir / "thickness_error_heatmap.png",
        "tau_error_heatmap_png": simulation_dir / "tau_error_heatmap.png",
        "sigma_error_heatmap_png": simulation_dir / "sigma_error_heatmap.png",
        "mse_heatmap_png": simulation_dir / "mse_heatmap.png",
        "windowed_mse_heatmap_png": simulation_dir / "windowed_mse_heatmap.png",
        "best_mse_trace_png": simulation_dir / "best_mse_trace.png",
        "worst_mse_trace_png": simulation_dir / "worst_mse_trace.png",
    }

    if show_progress:
        print(f"Finished compatibility study in {_format_seconds(time.perf_counter() - started_at)}.")

    return StudyResult(
        out_dir=simulation_dir,
        cases_dir=traces_dir,
        summary_csv_path=simulation_dir / "study_summary.csv",
        correlation_csv_path=simulation_dir / "study_correlations.csv",
        manifest_path=simulation_dir / "study_manifest.json",
        config_path=simulation_dir / "study_config.json",
        summary_rows=summary_rows,
        correlation_rows=correlation_rows,
        case_results=case_results,
        artifact_paths=artifact_paths,
        manifest=manifest,
    )
