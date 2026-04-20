from __future__ import annotations

from copy import deepcopy
import re

import numpy as np
from scipy.optimize import differential_evolution, minimize

from thzsim2.core.forward import simulate_sample_from_reference
from thzsim2.core.metrics import mse, relative_l2, snr_db
from thzsim2.models import Measurement, ResolvedMeasurementFitParameter

EPS0 = 8.8541878128e-12


def _parse_path(path: str):
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    tokens = re.findall(r"[A-Za-z_]\w*|\[\d+\]", path)
    if not tokens:
        raise ValueError(f"could not parse path: {path}")
    parts = []
    for token in tokens:
        if token.startswith("["):
            parts.append(int(token[1:-1]))
        else:
            parts.append(token)
    return parts


def _get_by_path(obj, path: str):
    current = obj
    for token in _parse_path(path):
        current = current[token]
    return current


def _set_by_path(obj, path: str, value):
    tokens = _parse_path(path)
    current = obj
    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = value


def stack_path_from_fit_path(path: str) -> str:
    return stack_path_from_user_path(path)


def stack_path_from_user_path(path: str) -> str:
    if ".material.parameters." in path:
        return path
    return path.replace(".material.", ".material.parameters.", 1)


def apply_fit_values(resolved_stack, fit_values, fit_parameters):
    stack = deepcopy(resolved_stack)
    values = np.asarray(fit_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("fit_values must be 1D")
    if values.size != len(fit_parameters):
        raise ValueError("fit_values length does not match fit_parameters")
    for value, fit_parameter in zip(values, fit_parameters, strict=True):
        _set_by_path(stack, stack_path_from_user_path(fit_parameter.path), float(value))
    return stack


def apply_measurement_fit_values(measurement, fit_values, measurement_fit_parameters):
    if measurement is None:
        payload = {
            "mode": "transmission",
            "angle_deg": 0.0,
            "polarization": "s",
            "polarization_mix": None,
            "reference_standard": None,
        }
    elif isinstance(measurement, Measurement):
        payload = {
            "mode": measurement.mode,
            "angle_deg": measurement.angle_deg,
            "polarization": measurement.polarization,
            "polarization_mix": measurement.polarization_mix,
            "reference_standard": measurement.reference_standard,
        }
    elif isinstance(measurement, dict):
        payload = deepcopy(dict(measurement))
    else:
        raise TypeError("measurement must be a Measurement, dictionary, or None")

    values = np.asarray(fit_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("measurement fit_values must be 1D")
    if values.size != len(measurement_fit_parameters):
        raise ValueError("measurement fit_values length does not match measurement_fit_parameters")

    for value, fit_parameter in zip(values, measurement_fit_parameters, strict=True):
        payload[str(fit_parameter.path)] = float(value)
    return Measurement(**payload)


def _all_fit_parameter_specs(sample_fit_parameters, measurement_fit_parameters):
    return list(sample_fit_parameters) + list(measurement_fit_parameters)


def objective_metric_value(y_model, y_true, metric: str) -> float:
    if metric == "mse":
        return mse(y_model, y_true)
    if metric == "relative_l2":
        return relative_l2(y_model, y_true)
    raise ValueError("metric must be 'mse' or 'relative_l2'")


def drude_gamma_thz_from_tau_ps(tau_ps: float) -> float:
    tau_ps = float(tau_ps)
    if tau_ps <= 0.0:
        raise ValueError("tau_ps must be positive")
    return 1.0 / (2.0 * np.pi * tau_ps)


def drude_plasma_freq_thz_from_sigma_tau(sigma_s_per_m: float, tau_ps: float) -> float:
    sigma = float(sigma_s_per_m)
    tau_s = float(tau_ps) * 1e-12
    if sigma < 0.0:
        raise ValueError("sigma_s_per_m must be nonnegative")
    if tau_s <= 0.0:
        raise ValueError("tau_ps must be positive")
    plasma_rad_s = np.sqrt(sigma / (EPS0 * tau_s))
    return float(plasma_rad_s / (2.0 * np.pi * 1e12))


def tau_ps_from_drude_gamma_thz(gamma_thz: float) -> float:
    gamma_thz = float(gamma_thz)
    if gamma_thz <= 0.0:
        return float("nan")
    return float(1.0 / (2.0 * np.pi * gamma_thz))


def sigma_s_per_m_from_drude_plasma_gamma(plasma_freq_thz: float, gamma_thz: float) -> float:
    plasma_rad_s = 2.0 * np.pi * float(plasma_freq_thz) * 1e12
    gamma_rad_s = 2.0 * np.pi * float(gamma_thz) * 1e12
    if gamma_rad_s <= 0.0:
        return float("nan")
    return float(EPS0 * plasma_rad_s * plasma_rad_s / gamma_rad_s)


def summarize_single_layer_drude_stack(resolved_stack) -> dict[str, float]:
    layer = resolved_stack["layers"][0]
    params = layer["material"]["parameters"]
    thickness_um = float(layer["thickness_um"])
    plasma_freq_thz = float(params["plasma_freq_thz"])
    gamma_thz = float(params["gamma_thz"])
    return {
        "thickness_um": thickness_um,
        "plasma_freq_thz": plasma_freq_thz,
        "gamma_thz": gamma_thz,
        "tau_ps": tau_ps_from_drude_gamma_thz(gamma_thz),
        "sigma_s_per_m": sigma_s_per_m_from_drude_plasma_gamma(plasma_freq_thz, gamma_thz),
    }


def build_single_layer_drude_true_stack(sample, *, thickness_um: float, tau_ps: float, sigma_s_per_m: float):
    stack = deepcopy(sample.resolved_stack)
    layer = stack["layers"][0]
    layer["thickness_um"] = float(thickness_um)
    layer["material"]["parameters"]["plasma_freq_thz"] = drude_plasma_freq_thz_from_sigma_tau(
        sigma_s_per_m,
        tau_ps,
    )
    layer["material"]["parameters"]["gamma_thz"] = drude_gamma_thz_from_tau_ps(tau_ps)
    return stack


def _residual_vector(trace_model, trace_true):
    residual = np.asarray(trace_true) - np.asarray(trace_model)
    if np.iscomplexobj(residual):
        return np.concatenate([np.real(residual), np.imag(residual)])
    return np.asarray(residual, dtype=np.float64)


def _step_from_bounds(xi, lo, hi, rel_step):
    span = hi - lo
    step = rel_step * max(abs(xi), span, 1.0)
    step = min(step, 0.25 * span)
    if step <= 0.0:
        step = rel_step
    return step


def _correlation_from_covariance(covariance, sigmas):
    if covariance is None or sigmas is None:
        return None
    cov = np.asarray(covariance, dtype=np.float64)
    sigma = np.asarray(sigmas, dtype=np.float64)
    corr = np.full(cov.shape, np.nan, dtype=np.float64)
    for i in range(sigma.size):
        si = float(sigma[i])
        if np.isfinite(si) and si > 0.0:
            corr[i, i] = 1.0
        for j in range(i + 1, sigma.size):
            sj = float(sigma[j])
            if np.isfinite(si) and si > 0.0 and np.isfinite(sj) and sj > 0.0:
                corr[i, j] = cov[i, j] / (si * sj)
                corr[j, i] = corr[i, j]
    return corr


def _correlation_summary(correlation):
    if correlation is None:
        return np.nan, np.nan
    if correlation.shape[0] <= 1:
        return 0.0, 0.0
    mask = ~np.eye(correlation.shape[0], dtype=bool)
    values = np.abs(correlation[mask])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(np.max(values)), float(np.mean(values))


def _estimate_parameter_sigmas(
    x_opt,
    *,
    reference,
    observed_trace,
    initial_stack,
    fit_parameters,
    measurement_fit_parameters,
    max_internal_reflections,
    measurement,
    rel_step,
):
    sample_fit_parameters = list(fit_parameters)
    measurement_fit_parameters = list(measurement_fit_parameters)
    all_fit_parameters = _all_fit_parameter_specs(sample_fit_parameters, measurement_fit_parameters)
    p = len(all_fit_parameters)
    split_index = len(sample_fit_parameters)
    fitted_stack = apply_fit_values(initial_stack, x_opt[:split_index], sample_fit_parameters)
    fitted_measurement = apply_measurement_fit_values(
        measurement,
        x_opt[split_index:],
        measurement_fit_parameters,
    )
    fitted_trace = simulate_sample_from_reference(
        reference,
        fitted_stack,
        max_internal_reflections=max_internal_reflections,
        measurement=fitted_measurement,
    )["sample_trace"]
    residual0 = _residual_vector(fitted_trace, observed_trace)
    m = residual0.size
    if m <= p:
        return None, None

    J = np.zeros((m, p), dtype=np.float64)
    for index, fit_parameter in enumerate(all_fit_parameters):
        lo = float(fit_parameter.bound_min)
        hi = float(fit_parameter.bound_max)
        xj = float(x_opt[index])
        h = _step_from_bounds(xj, lo, hi, rel_step)

        x_plus = np.asarray(x_opt, dtype=np.float64).copy()
        x_minus = np.asarray(x_opt, dtype=np.float64).copy()
        x_plus[index] = min(hi, xj + h)
        x_minus[index] = max(lo, xj - h)
        if x_plus[index] == x_minus[index]:
            continue

        sample_plus = apply_fit_values(initial_stack, x_plus[:split_index], sample_fit_parameters)
        sample_minus = apply_fit_values(initial_stack, x_minus[:split_index], sample_fit_parameters)
        measurement_plus = apply_measurement_fit_values(
            measurement,
            x_plus[split_index:],
            measurement_fit_parameters,
        )
        measurement_minus = apply_measurement_fit_values(
            measurement,
            x_minus[split_index:],
            measurement_fit_parameters,
        )
        trace_plus = simulate_sample_from_reference(
            reference,
            sample_plus,
            max_internal_reflections=max_internal_reflections,
            measurement=measurement_plus,
        )["sample_trace"]
        trace_minus = simulate_sample_from_reference(
            reference,
            sample_minus,
            max_internal_reflections=max_internal_reflections,
            measurement=measurement_minus,
        )["sample_trace"]
        r_plus = _residual_vector(trace_plus, observed_trace)
        r_minus = _residual_vector(trace_minus, observed_trace)
        J[:, index] = (r_plus - r_minus) / (x_plus[index] - x_minus[index])

    rss = float(np.dot(residual0, residual0))
    dof = max(m - p, 1)
    sigma2 = rss / dof
    covariance = sigma2 * np.linalg.pinv(J.T @ J)
    sigmas = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return covariance, sigmas


def fit_sample_trace(
    *,
    reference,
    observed_trace,
    initial_stack,
    fit_parameters,
    metric="mse",
    max_internal_reflections=0,
    optimizer=None,
    measurement=None,
    measurement_fit_parameters=None,
):
    sample_fit_parameters = list(fit_parameters)
    measurement_fit_parameters = [] if measurement_fit_parameters is None else list(measurement_fit_parameters)
    if not sample_fit_parameters and not measurement_fit_parameters:
        raise ValueError("fit_parameters and measurement_fit_parameters cannot both be empty")
    optimizer = {} if optimizer is None else dict(optimizer)

    all_fit_parameters = _all_fit_parameter_specs(sample_fit_parameters, measurement_fit_parameters)
    x0 = np.array([float(parameter.initial_value) for parameter in all_fit_parameters], dtype=np.float64)
    bounds = [(float(parameter.bound_min), float(parameter.bound_max)) for parameter in all_fit_parameters]
    split_index = len(sample_fit_parameters)

    def objective(x):
        sample_values = np.asarray(x[:split_index], dtype=np.float64)
        measurement_values = np.asarray(x[split_index:], dtype=np.float64)
        stack_trial = apply_fit_values(initial_stack, sample_values, sample_fit_parameters)
        measurement_trial = apply_measurement_fit_values(measurement, measurement_values, measurement_fit_parameters)
        simulated = simulate_sample_from_reference(
            reference,
            stack_trial,
            max_internal_reflections=max_internal_reflections,
            measurement=measurement_trial,
        )
        return objective_metric_value(simulated["sample_trace"], observed_trace, metric)

    global_options = {
        "seed": 123,
        "polish": False,
        "maxiter": 4,
        "popsize": 5,
        "tol": 1e-6,
        "updating": "deferred",
    }
    global_options.update(dict(optimizer.get("global_options", {})))
    de_result = differential_evolution(objective, bounds=bounds, **global_options)

    local_result = minimize(
        objective,
        np.asarray(de_result.x, dtype=np.float64),
        method=str(optimizer.get("method", "L-BFGS-B")),
        bounds=bounds,
        options=dict(optimizer.get("options", {"maxiter": 35})),
    )

    if np.isfinite(local_result.fun) and local_result.fun <= de_result.fun:
        chosen = local_result
        x_opt = np.asarray(local_result.x, dtype=np.float64)
        status = int(local_result.status)
        message = str(local_result.message)
        nfev = int(local_result.nfev)
        nit = int(getattr(local_result, "nit", -1))
        objective_value = float(local_result.fun)
    else:
        chosen = de_result
        x_opt = np.asarray(de_result.x, dtype=np.float64)
        status = int(getattr(de_result, "status", 0))
        message = str(de_result.message)
        nfev = int(de_result.nfev)
        nit = int(getattr(de_result, "nit", -1))
        objective_value = float(de_result.fun)

    fitted_stack = apply_fit_values(initial_stack, x_opt[:split_index], sample_fit_parameters)
    fitted_measurement = apply_measurement_fit_values(
        measurement,
        x_opt[split_index:],
        measurement_fit_parameters,
    )
    fitted_simulation = simulate_sample_from_reference(
        reference,
        fitted_stack,
        max_internal_reflections=max_internal_reflections,
        measurement=fitted_measurement,
    )
    fitted_trace = np.asarray(fitted_simulation["sample_trace"], dtype=np.float64)
    observed_trace = np.asarray(observed_trace, dtype=np.float64)
    residual = observed_trace - fitted_trace

    covariance, sigmas = _estimate_parameter_sigmas(
        x_opt,
        reference=reference,
        observed_trace=observed_trace,
        initial_stack=initial_stack,
        fit_parameters=sample_fit_parameters,
        measurement_fit_parameters=measurement_fit_parameters,
        max_internal_reflections=max_internal_reflections,
        measurement=measurement,
        rel_step=float(optimizer.get("fd_rel_step", 1e-5)),
    )
    correlation = _correlation_from_covariance(covariance, sigmas)
    max_abs_corr, mean_abs_corr = _correlation_summary(correlation)

    parameter_names = [parameter.key for parameter in all_fit_parameters]
    sigma_map = None if sigmas is None else {
        parameter.key: float(sigma) for parameter, sigma in zip(all_fit_parameters, sigmas, strict=True)
    }

    return {
        "success": bool(np.isfinite(objective_value) and all(lo <= xi <= hi for xi, (lo, hi) in zip(x_opt, bounds, strict=True))),
        "status": status,
        "message": message,
        "nfev": nfev,
        "nit": nit,
        "optimizer_result": chosen,
        "objective_value": objective_value,
        "x0": x0,
        "x_opt": x_opt,
        "bounds": bounds,
        "metric": metric,
        "parameter_names": parameter_names,
        "recovered_parameters": {
            parameter.key: float(value) for parameter, value in zip(all_fit_parameters, x_opt, strict=True)
        },
        "parameter_sigmas": sigma_map,
        "parameter_covariance": covariance,
        "parameter_correlation": correlation,
        "max_abs_parameter_correlation": max_abs_corr,
        "mean_abs_parameter_correlation": mean_abs_corr,
        "fitted_stack": fitted_stack,
        "fitted_measurement": {
            "mode": fitted_measurement.mode,
            "angle_deg": float(fitted_measurement.angle_deg),
            "polarization": fitted_measurement.polarization,
            "polarization_mix": None
            if fitted_measurement.polarization_mix is None
            else float(fitted_measurement.polarization_mix),
            "reference_standard_kind": None
            if fitted_measurement.reference_standard is None
            else fitted_measurement.reference_standard.kind,
        },
        "fitted_simulation": fitted_simulation,
        "residual_trace": residual,
        "residual_metrics": {
            "mse": mse(fitted_trace, observed_trace),
            "relative_l2": relative_l2(fitted_trace, observed_trace),
            "snr_db": snr_db(observed_trace, residual),
        },
    }
