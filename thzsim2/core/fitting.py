from __future__ import annotations

from copy import deepcopy
import re

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
from scipy.signal import correlate, correlation_lags

from thzsim2.core.fft import fft_t_to_w
from thzsim2.core.forward import simulate_sample_from_reference
from thzsim2.core.metrics import data_fit, fit_sigma, mse, normalized_mse, relative_l2, residual_rms, snr_db
from thzsim2.models import Measurement, ResolvedMeasurementFitParameter

EPS0 = 8.8541878128e-12


class _ScalarFitSpec:
    __slots__ = ("key", "label", "path", "unit", "initial_value", "bound_min", "bound_max")

    def __init__(self, *, key, label, path, unit, initial_value, bound_min, bound_max):
        self.key = str(key)
        self.label = str(label)
        self.path = str(path)
        self.unit = str(unit)
        self.initial_value = float(initial_value)
        self.bound_min = float(bound_min)
        self.bound_max = float(bound_max)


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
            "trace_scale": 1.0,
            "trace_offset": 0.0,
            "reference_standard": None,
        }
    elif isinstance(measurement, Measurement):
        payload = {
            "mode": measurement.mode,
            "angle_deg": measurement.angle_deg,
            "polarization": measurement.polarization,
            "polarization_mix": measurement.polarization_mix,
            "trace_scale": measurement.trace_scale,
            "trace_offset": measurement.trace_offset,
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


def _normalize_objective_weights(weights, size: int):
    if weights is None:
        return None
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size != int(size):
        raise ValueError("objective weights must be a 1D array matching the trace length")
    if not np.isfinite(values).all():
        raise ValueError("objective weights must be finite")
    if np.any(values < 0.0):
        raise ValueError("objective weights must be nonnegative")
    mean_value = float(np.mean(values))
    if mean_value <= 0.0:
        raise ValueError("objective weights must have positive mean")
    return values / mean_value


def build_objective_weights(
    trace,
    *,
    mode="trace_amplitude",
    floor=0.05,
    power=2.0,
    smooth_window_samples=41,
):
    mode = "none" if mode is None else str(mode).strip().lower()
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("trace must be 1D when building objective weights")
    if mode == "none":
        return np.ones(values.shape, dtype=np.float64)
    if mode != "trace_amplitude":
        raise ValueError("weighting mode must be 'none' or 'trace_amplitude'")

    floor = float(floor)
    power = float(power)
    smooth_window_samples = max(1, int(smooth_window_samples))
    if smooth_window_samples % 2 == 0:
        smooth_window_samples += 1
    if not (0.0 <= floor < 1.0):
        raise ValueError("weight floor must satisfy 0 <= floor < 1")
    if power <= 0.0:
        raise ValueError("weight power must be positive")

    envelope = np.abs(values)
    if smooth_window_samples > 1:
        kernel = np.ones(smooth_window_samples, dtype=np.float64) / float(smooth_window_samples)
        envelope = np.convolve(envelope, kernel, mode="same")
    peak = max(float(np.max(envelope)), 1e-30)
    normalized = np.power(np.clip(envelope / peak, 0.0, None), power)
    weights = floor + (1.0 - floor) * normalized
    return _normalize_objective_weights(weights, values.size)


def _weighted_data_fit(y_model, y_true, weights) -> float:
    y_model = np.asarray(y_model, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    weights = _normalize_objective_weights(weights, y_true.size)
    residual = y_model - y_true
    numerator = float(np.sum(weights * residual * residual))
    denominator = max(float(np.sum(weights * y_true * y_true)), 1e-30)
    return numerator / denominator


def _normalize_metric_options(metric_options):
    options = {} if metric_options is None else dict(metric_options)
    normalized = {
        "lp_order": float(options.get("lp_order", 8.0)),
        "freq_min_thz": options.get("freq_min_thz"),
        "freq_max_thz": options.get("freq_max_thz"),
        "time_weight": float(options.get("time_weight", 1.0)),
        "amplitude_weight": float(options.get("amplitude_weight", 1.0)),
        "phase_weight": float(options.get("phase_weight", 1.0)),
        "transfer_floor_ratio": float(options.get("transfer_floor_ratio", 1e-6)),
    }
    if normalized["lp_order"] <= 0.0:
        raise ValueError("metric_options.lp_order must be positive")
    if normalized["time_weight"] < 0.0 or normalized["amplitude_weight"] < 0.0 or normalized["phase_weight"] < 0.0:
        raise ValueError("hybrid transfer metric weights must be nonnegative")
    if normalized["transfer_floor_ratio"] <= 0.0:
        raise ValueError("metric_options.transfer_floor_ratio must be positive")
    if normalized["freq_min_thz"] is not None:
        normalized["freq_min_thz"] = float(normalized["freq_min_thz"])
    if normalized["freq_max_thz"] is not None:
        normalized["freq_max_thz"] = float(normalized["freq_max_thz"])
    if (
        normalized["freq_min_thz"] is not None
        and normalized["freq_max_thz"] is not None
        and normalized["freq_max_thz"] <= normalized["freq_min_thz"]
    ):
        raise ValueError("metric_options.freq_max_thz must be greater than freq_min_thz")
    return normalized


def _relative_lp(y_model, y_true, weights=None, *, p=8.0) -> float:
    y_model = np.asarray(y_model, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    p = float(p)
    if p <= 0.0:
        raise ValueError("relative_lp order must be positive")
    residual_power = np.power(np.abs(y_model - y_true), p)
    reference_power = np.power(np.abs(y_true), p)
    if weights is None:
        numerator = float(np.mean(residual_power))
        denominator = max(float(np.mean(reference_power)), 1e-30)
        return numerator / denominator
    normalized_weights = _normalize_objective_weights(weights, y_true.size)
    numerator = float(np.sum(normalized_weights * residual_power))
    denominator = max(float(np.sum(normalized_weights * reference_power)), 1e-30)
    return numerator / denominator


def _positive_spectrum(trace, time_ps):
    trace = np.asarray(trace, dtype=np.float64)
    time_ps = np.asarray(time_ps, dtype=np.float64)
    omega, spectrum = fft_t_to_w(trace, dt=float(np.median(np.diff(time_ps))) * 1e-12, t0=float(time_ps[0]) * 1e-12)
    freq_thz = omega / (2.0 * np.pi * 1e12)
    positive = freq_thz > 0.0
    return np.asarray(freq_thz[positive], dtype=np.float64), np.asarray(spectrum[positive], dtype=np.complex128)


def _prepare_transfer_objective_cache(reference, observed_trace, metric_options):
    time_ps = np.asarray(reference.trace.time_ps, dtype=np.float64)
    reference_trace = np.asarray(reference.trace.trace, dtype=np.float64)
    if observed_trace.size != reference_trace.size:
        raise ValueError("observed_trace must match the reference trace length for hybrid_transfer")
    freq_thz, reference_spectrum = _positive_spectrum(reference_trace, time_ps)
    observed_freq_thz, observed_spectrum = _positive_spectrum(observed_trace, time_ps)
    if freq_thz.shape != observed_freq_thz.shape or not np.allclose(freq_thz, observed_freq_thz, rtol=0.0, atol=1e-12):
        raise ValueError("reference and observed traces must share the same frequency grid")
    observed_transfer = np.divide(
        observed_spectrum,
        reference_spectrum,
        out=np.zeros_like(observed_spectrum, dtype=np.complex128),
        where=np.abs(reference_spectrum) > 1e-30,
    )
    floor_ratio = float(metric_options["transfer_floor_ratio"])
    valid = np.isfinite(observed_transfer.real) & np.isfinite(observed_transfer.imag)
    valid &= np.abs(reference_spectrum) >= floor_ratio * max(float(np.max(np.abs(reference_spectrum))), 1e-30)
    valid &= np.abs(observed_spectrum) >= floor_ratio * max(float(np.max(np.abs(observed_spectrum))), 1e-30)
    if metric_options["freq_min_thz"] is not None:
        valid &= freq_thz >= float(metric_options["freq_min_thz"])
    if metric_options["freq_max_thz"] is not None:
        valid &= freq_thz <= float(metric_options["freq_max_thz"])
    return {
        "time_ps": time_ps,
        "freq_thz": freq_thz,
        "reference_trace": reference_trace,
        "reference_spectrum": reference_spectrum,
        "observed_transfer": observed_transfer,
        "valid_mask": valid,
    }


def _transfer_mismatch_terms(model_trace, transfer_cache):
    _, model_spectrum = _positive_spectrum(model_trace, transfer_cache["time_ps"])
    model_transfer = np.divide(
        model_spectrum,
        transfer_cache["reference_spectrum"],
        out=np.zeros_like(model_spectrum, dtype=np.complex128),
        where=np.abs(transfer_cache["reference_spectrum"]) > 1e-30,
    )
    valid = np.asarray(transfer_cache["valid_mask"], dtype=bool)
    valid &= np.isfinite(model_transfer.real) & np.isfinite(model_transfer.imag)
    if not np.any(valid):
        return {
            "amplitude_mse": float("inf"),
            "phase_mse": float("inf"),
            "model_transfer": model_transfer,
            "valid_mask": valid,
        }
    observed_transfer = transfer_cache["observed_transfer"][valid]
    model_transfer_valid = model_transfer[valid]
    amplitude_mse = float(
        np.mean(
            np.square(
                np.log(np.maximum(np.abs(model_transfer_valid), 1e-30))
                - np.log(np.maximum(np.abs(observed_transfer), 1e-30))
            )
        )
    )
    phase_mse = float(
        np.mean(
            np.square(
                (np.unwrap(np.angle(model_transfer_valid)) - np.unwrap(np.angle(observed_transfer))) / np.pi
            )
        )
    )
    return {
        "amplitude_mse": amplitude_mse,
        "phase_mse": phase_mse,
        "model_transfer": model_transfer,
        "valid_mask": valid,
    }


def _hybrid_transfer_metric(model_trace, observed_trace, objective_weights, metric_options, transfer_cache):
    time_term = (
        _weighted_data_fit(model_trace, observed_trace, objective_weights)
        if objective_weights is not None
        else data_fit(model_trace, observed_trace)
    )
    transfer_terms = _transfer_mismatch_terms(model_trace, transfer_cache)
    return (
        float(metric_options["time_weight"]) * float(time_term)
        + float(metric_options["amplitude_weight"]) * float(transfer_terms["amplitude_mse"])
        + float(metric_options["phase_weight"]) * float(transfer_terms["phase_mse"])
    )


def _run_global_optimizer(objective, bounds, optimizer):
    global_method = str(optimizer.get("global_method", "differential_evolution")).strip().lower()
    if global_method in {"", "none"}:
        return None

    if global_method == "differential_evolution":
        base_options = {
            "seed": 123,
            "polish": False,
            "maxiter": 12,
            "popsize": 10,
            "tol": 1e-7,
            "updating": "deferred",
        }
        base_options.update(dict(optimizer.get("global_options", {})))
        restarts = max(1, int(optimizer.get("global_restarts", 1)))
        seed = base_options.get("seed")
        results = []
        for restart_index in range(restarts):
            options = dict(base_options)
            if seed is not None:
                options["seed"] = int(seed) + restart_index
            results.append(differential_evolution(objective, bounds=bounds, **options))
        finite_results = [result for result in results if np.isfinite(float(result.fun))]
        return None if not finite_results else min(finite_results, key=lambda result: float(result.fun))

    if global_method == "dual_annealing":
        base_options = {
            "seed": 123,
            "maxiter": 800,
            "no_local_search": True,
        }
        base_options.update(dict(optimizer.get("global_options", {})))
        return dual_annealing(objective, bounds=bounds, **base_options)

    raise ValueError("global_method must be 'differential_evolution', 'dual_annealing', or 'none'")


def _run_local_optimizer(objective, x_start, bounds, optimizer):
    method = optimizer.get("method", "L-BFGS-B")
    if method is None or str(method).strip().lower() == "none":
        return None
    return minimize(
        objective,
        np.asarray(x_start, dtype=np.float64),
        method=str(method),
        bounds=bounds,
        options=dict(optimizer.get("options", {"maxiter": 120})),
    )


def objective_metric_value(
    y_model,
    y_true,
    metric: str,
    *,
    objective_weights=None,
    metric_options=None,
    reference=None,
    transfer_cache=None,
) -> float:
    metric_options = _normalize_metric_options(metric_options)
    if metric == "data_fit":
        return data_fit(y_model, y_true)
    if metric == "weighted_data_fit":
        return _weighted_data_fit(y_model, y_true, objective_weights)
    if metric == "mse":
        return mse(y_model, y_true)
    if metric == "normalized_mse":
        return normalized_mse(y_model, y_true)
    if metric == "relative_l2":
        return relative_l2(y_model, y_true)
    if metric == "relative_lp":
        return _relative_lp(y_model, y_true, objective_weights, p=metric_options["lp_order"])
    if metric == "hybrid_transfer":
        if reference is None:
            raise ValueError("hybrid_transfer requires a reference result")
        transfer_cache = (
            transfer_cache if transfer_cache is not None else _prepare_transfer_objective_cache(reference, y_true, metric_options)
        )
        return _hybrid_transfer_metric(y_model, y_true, objective_weights, metric_options, transfer_cache)
    raise ValueError(
        "metric must be 'data_fit', 'weighted_data_fit', 'mse', 'normalized_mse', 'relative_l2', 'relative_lp', or 'hybrid_transfer'"
    )


def estimate_trace_delay_ps(model_trace, observed_trace, time_ps) -> float:
    time_ps = np.asarray(time_ps, dtype=np.float64)
    if time_ps.ndim != 1 or time_ps.size < 2:
        raise ValueError("time_ps must contain at least two samples")
    dt_ps = float(np.median(np.diff(time_ps)))
    model = np.asarray(model_trace, dtype=np.float64)
    observed = np.asarray(observed_trace, dtype=np.float64)
    model = model - float(np.mean(model))
    observed = observed - float(np.mean(observed))
    corr = correlate(observed, model, mode="full", method="fft")
    lags = correlation_lags(observed.size, model.size, mode="full")
    best_lag = int(lags[int(np.argmax(corr))])
    return float(best_lag) * dt_ps


def shift_trace_in_time(trace, time_ps, delta_t_ps):
    time_ps = np.asarray(time_ps, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    return np.interp(
        time_ps - float(delta_t_ps),
        time_ps,
        trace,
        left=0.0,
        right=0.0,
    )


def _resolve_delay_fit_spec(
    *,
    delay_options,
    reference,
    observed_trace,
    initial_stack,
    measurement,
    max_internal_reflections,
):
    if not delay_options or not bool(delay_options.get("enabled", False)):
        return None, 0.0

    initial_simulation = simulate_sample_from_reference(
        reference,
        initial_stack,
        max_internal_reflections=max_internal_reflections,
        measurement=measurement,
    )
    coarse_delay_ps = delay_options.get("initial_ps")
    if coarse_delay_ps is None:
        coarse_delay_ps = estimate_trace_delay_ps(
            initial_simulation["sample_trace"],
            observed_trace,
            reference.trace.time_ps,
        )
    coarse_delay_ps = float(coarse_delay_ps)
    search_window_ps = float(delay_options.get("search_window_ps", max(25.0, abs(coarse_delay_ps) * 0.5 + 10.0)))
    bound_min = delay_options.get("abs_min")
    bound_max = delay_options.get("abs_max")
    if bound_min is None:
        bound_min = coarse_delay_ps - search_window_ps
    if bound_max is None:
        bound_max = coarse_delay_ps + search_window_ps
    if float(bound_max) <= float(bound_min):
        raise ValueError("delay_options bounds must satisfy abs_max > abs_min")

    return (
        _ScalarFitSpec(
            key=str(delay_options.get("label", "delta_t_ps")),
            label=str(delay_options.get("label", "delta_t_ps")),
            path="delta_t_ps",
            unit="ps",
            initial_value=coarse_delay_ps,
            bound_min=float(bound_min),
            bound_max=float(bound_max),
        ),
        coarse_delay_ps,
    )


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


def summarize_two_drude_layer(layer) -> dict[str, float]:
    params = layer["material"]["parameters"]
    plasma1 = float(params["plasma_freq1_thz"])
    gamma1 = float(params["gamma1_thz"])
    plasma2 = float(params["plasma_freq2_thz"])
    gamma2 = float(params["gamma2_thz"])
    return {
        "thickness_um": float(layer["thickness_um"]),
        "eps_inf": float(params["eps_inf"]),
        "plasma_freq1_thz": plasma1,
        "gamma1_thz": gamma1,
        "tau1_ps": tau_ps_from_drude_gamma_thz(gamma1),
        "sigma1_s_per_m": sigma_s_per_m_from_drude_plasma_gamma(plasma1, gamma1),
        "plasma_freq2_thz": plasma2,
        "gamma2_thz": gamma2,
        "tau2_ps": tau_ps_from_drude_gamma_thz(gamma2),
        "sigma2_s_per_m": sigma_s_per_m_from_drude_plasma_gamma(plasma2, gamma2),
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


def _split_fit_vector(x, sample_count, measurement_count, delay_parameter):
    sample_values = np.asarray(x[:sample_count], dtype=np.float64)
    measurement_values = np.asarray(x[sample_count : sample_count + measurement_count], dtype=np.float64)
    delay_value = 0.0
    if delay_parameter is not None:
        delay_value = float(x[sample_count + measurement_count])
    return sample_values, measurement_values, delay_value


def _simulate_trial(
    *,
    x,
    reference,
    initial_stack,
    sample_fit_parameters,
    measurement,
    measurement_fit_parameters,
    delay_parameter,
    max_internal_reflections,
):
    sample_count = len(sample_fit_parameters)
    measurement_count = len(measurement_fit_parameters)
    sample_values, measurement_values, delay_value = _split_fit_vector(
        x,
        sample_count,
        measurement_count,
        delay_parameter,
    )
    stack_trial = apply_fit_values(initial_stack, sample_values, sample_fit_parameters)
    measurement_trial = apply_measurement_fit_values(measurement, measurement_values, measurement_fit_parameters)
    simulated = simulate_sample_from_reference(
        reference,
        stack_trial,
        max_internal_reflections=max_internal_reflections,
        measurement=measurement_trial,
    )
    trace = np.asarray(simulated["sample_trace"], dtype=np.float64)
    if delay_parameter is not None:
        trace = shift_trace_in_time(trace, reference.trace.time_ps, delay_value)
        simulated = dict(simulated)
        simulated["sample_trace_unshifted"] = np.asarray(simulated["sample_trace"], dtype=np.float64)
        simulated["sample_trace"] = trace
    return simulated, stack_trial, measurement_trial, float(delay_value)


def _estimate_parameter_sigmas(
    x_opt,
    *,
    reference,
    observed_trace,
    initial_stack,
    fit_parameters,
    measurement_fit_parameters,
    delay_parameter,
    max_internal_reflections,
    measurement,
    rel_step,
):
    sample_fit_parameters = list(fit_parameters)
    measurement_fit_parameters = list(measurement_fit_parameters)
    all_fit_parameters = _all_fit_parameter_specs(sample_fit_parameters, measurement_fit_parameters)
    if delay_parameter is not None:
        all_fit_parameters = all_fit_parameters + [delay_parameter]
    p = len(all_fit_parameters)
    fitted_simulation, _, _, _ = _simulate_trial(
        x=x_opt,
        reference=reference,
        initial_stack=initial_stack,
        sample_fit_parameters=sample_fit_parameters,
        measurement=measurement,
        measurement_fit_parameters=measurement_fit_parameters,
        delay_parameter=delay_parameter,
        max_internal_reflections=max_internal_reflections,
    )
    fitted_trace = fitted_simulation["sample_trace"]
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

        trace_plus = _simulate_trial(
            x=x_plus,
            reference=reference,
            initial_stack=initial_stack,
            sample_fit_parameters=sample_fit_parameters,
            measurement=measurement,
            measurement_fit_parameters=measurement_fit_parameters,
            delay_parameter=delay_parameter,
            max_internal_reflections=max_internal_reflections,
        )[0]["sample_trace"]
        trace_minus = _simulate_trial(
            x=x_minus,
            reference=reference,
            initial_stack=initial_stack,
            sample_fit_parameters=sample_fit_parameters,
            measurement=measurement,
            measurement_fit_parameters=measurement_fit_parameters,
            delay_parameter=delay_parameter,
            max_internal_reflections=max_internal_reflections,
        )[0]["sample_trace"]
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
    metric="data_fit",
    max_internal_reflections=0,
    optimizer=None,
    measurement=None,
    measurement_fit_parameters=None,
    delay_options=None,
    objective_weights=None,
    metric_options=None,
):
    sample_fit_parameters = list(fit_parameters)
    measurement_fit_parameters = [] if measurement_fit_parameters is None else list(measurement_fit_parameters)
    optimizer = {} if optimizer is None else dict(optimizer)
    observed_trace = np.asarray(observed_trace, dtype=np.float64)
    objective_weights = _normalize_objective_weights(objective_weights, observed_trace.size) if objective_weights is not None else None
    metric_options = _normalize_metric_options(metric_options)
    transfer_cache = (
        _prepare_transfer_objective_cache(reference, observed_trace, metric_options)
        if metric == "hybrid_transfer"
        else None
    )

    delay_parameter, coarse_delay_ps = _resolve_delay_fit_spec(
        delay_options=delay_options,
        reference=reference,
        observed_trace=observed_trace,
        initial_stack=initial_stack,
        measurement=measurement,
        max_internal_reflections=max_internal_reflections,
    )
    all_fit_parameters = _all_fit_parameter_specs(sample_fit_parameters, measurement_fit_parameters)
    if delay_parameter is not None:
        all_fit_parameters = all_fit_parameters + [delay_parameter]
    x0 = np.array([float(parameter.initial_value) for parameter in all_fit_parameters], dtype=np.float64)
    bounds = [(float(parameter.bound_min), float(parameter.bound_max)) for parameter in all_fit_parameters]

    initial_simulation, _, _, initial_delay_ps = _simulate_trial(
        x=x0,
        reference=reference,
        initial_stack=initial_stack,
        sample_fit_parameters=sample_fit_parameters,
        measurement=measurement,
        measurement_fit_parameters=measurement_fit_parameters,
        delay_parameter=delay_parameter,
        max_internal_reflections=max_internal_reflections,
    )
    initial_trace = np.asarray(initial_simulation["sample_trace"], dtype=np.float64)
    initial_objective_value = objective_metric_value(
        initial_trace,
        observed_trace,
        metric,
        objective_weights=objective_weights,
        metric_options=metric_options,
        reference=reference,
        transfer_cache=transfer_cache,
    )

    def objective(x):
        simulated, _, _, _ = _simulate_trial(
            x=x,
            reference=reference,
            initial_stack=initial_stack,
            sample_fit_parameters=sample_fit_parameters,
            measurement=measurement,
            measurement_fit_parameters=measurement_fit_parameters,
            delay_parameter=delay_parameter,
            max_internal_reflections=max_internal_reflections,
        )
        return objective_metric_value(
            simulated["sample_trace"],
            observed_trace,
            metric,
            objective_weights=objective_weights,
            metric_options=metric_options,
            reference=reference,
            transfer_cache=transfer_cache,
        )

    global_result = None if x0.size == 0 else _run_global_optimizer(objective, bounds, optimizer)
    local_start = x0 if global_result is None else np.asarray(global_result.x, dtype=np.float64)
    local_result = None if x0.size == 0 else _run_local_optimizer(objective, local_start, bounds, optimizer)

    candidates = [
        {
            "kind": "initial",
            "x": np.asarray(x0, dtype=np.float64),
            "objective_value": float(initial_objective_value),
            "status": 0,
            "message": "initial guess",
            "nfev": 1,
            "nit": 0,
            "optimizer_result": None,
        }
    ]
    if global_result is not None and np.isfinite(global_result.fun):
        candidates.append(
            {
                "kind": "global",
                "x": np.asarray(global_result.x, dtype=np.float64),
                "objective_value": float(global_result.fun),
                "status": int(getattr(global_result, "status", 0)),
                "message": str(global_result.message),
                "nfev": int(getattr(global_result, "nfev", 0)),
                "nit": int(getattr(global_result, "nit", -1)),
                "optimizer_result": global_result,
            }
        )
    if local_result is not None and np.isfinite(local_result.fun):
        candidates.append(
            {
                "kind": "local",
                "x": np.asarray(local_result.x, dtype=np.float64),
                "objective_value": float(local_result.fun),
                "status": int(local_result.status),
                "message": str(local_result.message),
                "nfev": int(local_result.nfev),
                "nit": int(getattr(local_result, "nit", -1)),
                "optimizer_result": local_result,
            }
        )

    best = min(candidates, key=lambda candidate: candidate["objective_value"])
    chosen = best["optimizer_result"]
    x_opt = np.asarray(best["x"], dtype=np.float64)
    status = int(best["status"])
    message = str(best["message"])
    nfev = int(best["nfev"])
    nit = int(best["nit"])
    objective_value = float(best["objective_value"])

    fitted_simulation, fitted_stack, fitted_measurement, fitted_delay_ps = _simulate_trial(
        x=x_opt,
        reference=reference,
        initial_stack=initial_stack,
        sample_fit_parameters=sample_fit_parameters,
        measurement=measurement,
        measurement_fit_parameters=measurement_fit_parameters,
        delay_parameter=delay_parameter,
        max_internal_reflections=max_internal_reflections,
    )
    fitted_trace = np.asarray(fitted_simulation["sample_trace"], dtype=np.float64)
    residual = observed_trace - fitted_trace

    covariance, sigmas = _estimate_parameter_sigmas(
        x_opt,
        reference=reference,
        observed_trace=observed_trace,
        initial_stack=initial_stack,
        fit_parameters=sample_fit_parameters,
        measurement_fit_parameters=measurement_fit_parameters,
        delay_parameter=delay_parameter,
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
    transfer_metric_cache = transfer_cache
    if transfer_metric_cache is None:
        transfer_metric_cache = _prepare_transfer_objective_cache(reference, observed_trace, metric_options)
    fitted_transfer_terms = _transfer_mismatch_terms(fitted_trace, transfer_metric_cache)
    residual_metrics = {
        "data_fit": data_fit(fitted_trace, observed_trace),
        "weighted_data_fit": _weighted_data_fit(fitted_trace, observed_trace, objective_weights)
        if objective_weights is not None
        else data_fit(fitted_trace, observed_trace),
        "mse": mse(fitted_trace, observed_trace),
        "normalized_mse": normalized_mse(fitted_trace, observed_trace),
        "relative_l2": relative_l2(fitted_trace, observed_trace),
        "relative_lp": _relative_lp(fitted_trace, observed_trace, objective_weights, p=metric_options["lp_order"]),
        "hybrid_transfer": _hybrid_transfer_metric(
            fitted_trace,
            observed_trace,
            objective_weights,
            metric_options,
            transfer_metric_cache,
        ),
        "transfer_amplitude_mse": float(fitted_transfer_terms["amplitude_mse"]),
        "transfer_phase_mse": float(fitted_transfer_terms["phase_mse"]),
        "residual_rms": residual_rms(fitted_trace, observed_trace),
        "fit_sigma": fit_sigma(fitted_trace, observed_trace),
        "snr_db": snr_db(observed_trace, residual),
    }
    initial_residual = observed_trace - initial_trace
    initial_transfer_terms = _transfer_mismatch_terms(initial_trace, transfer_metric_cache)
    initial_residual_metrics = {
        "data_fit": data_fit(initial_trace, observed_trace),
        "weighted_data_fit": _weighted_data_fit(initial_trace, observed_trace, objective_weights)
        if objective_weights is not None
        else data_fit(initial_trace, observed_trace),
        "mse": mse(initial_trace, observed_trace),
        "normalized_mse": normalized_mse(initial_trace, observed_trace),
        "relative_l2": relative_l2(initial_trace, observed_trace),
        "relative_lp": _relative_lp(initial_trace, observed_trace, objective_weights, p=metric_options["lp_order"]),
        "hybrid_transfer": _hybrid_transfer_metric(
            initial_trace,
            observed_trace,
            objective_weights,
            metric_options,
            transfer_metric_cache,
        ),
        "transfer_amplitude_mse": float(initial_transfer_terms["amplitude_mse"]),
        "transfer_phase_mse": float(initial_transfer_terms["phase_mse"]),
        "residual_rms": residual_rms(initial_trace, observed_trace),
        "fit_sigma": fit_sigma(initial_trace, observed_trace),
        "snr_db": snr_db(observed_trace, initial_residual),
    }
    recovered_parameters = {
        parameter.key: float(value) for parameter, value in zip(all_fit_parameters, x_opt, strict=True)
    }

    return {
        "success": bool(np.isfinite(objective_value) and all(lo <= xi <= hi for xi, (lo, hi) in zip(x_opt, bounds, strict=True))),
        "status": status,
        "message": message,
        "nfev": nfev,
        "nit": nit,
        "optimizer_result": chosen,
        "optimizer_stage": best["kind"],
        "objective_value": objective_value,
        "x0": x0,
        "x_opt": x_opt,
        "bounds": bounds,
        "metric": metric,
        "metric_options": deepcopy(metric_options),
        "objective_weights": None if objective_weights is None else np.asarray(objective_weights, dtype=np.float64),
        "parameter_names": parameter_names,
        "recovered_parameters": recovered_parameters,
        "parameter_sigmas": sigma_map,
        "parameter_covariance": covariance,
        "parameter_correlation": correlation,
        "max_abs_parameter_correlation": max_abs_corr,
        "mean_abs_parameter_correlation": mean_abs_corr,
        "initial_objective_value": float(initial_objective_value),
        "initial_residual_metrics": initial_residual_metrics,
        "fitted_stack": fitted_stack,
        "fitted_measurement": {
            "mode": fitted_measurement.mode,
            "angle_deg": float(fitted_measurement.angle_deg),
            "polarization": fitted_measurement.polarization,
            "polarization_mix": None
            if fitted_measurement.polarization_mix is None
            else float(fitted_measurement.polarization_mix),
            "trace_scale": float(fitted_measurement.trace_scale),
            "trace_offset": float(fitted_measurement.trace_offset),
            "reference_standard_kind": None
            if fitted_measurement.reference_standard is None
            else fitted_measurement.reference_standard.kind,
        },
        "delay_recovery": {
            "enabled": delay_parameter is not None,
            "coarse_delay_ps": None if delay_parameter is None else float(coarse_delay_ps),
            "initial_delay_ps": None if delay_parameter is None else float(initial_delay_ps),
            "fitted_delay_ps": None if delay_parameter is None else float(fitted_delay_ps),
        },
        "initial_simulation": initial_simulation,
        "fitted_simulation": fitted_simulation,
        "residual_trace": residual,
        "residual_metrics": residual_metrics,
    }
