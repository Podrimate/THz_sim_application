from __future__ import annotations

import numpy as np


def _real_residual(y_model, y_true):
    residual = np.asarray(y_model) - np.asarray(y_true)
    if np.iscomplexobj(residual):
        residual = np.abs(residual)
    return np.asarray(residual, dtype=np.float64)


def mse(y_model, y_true) -> float:
    residual = _real_residual(y_model, y_true)
    return float(np.mean(np.square(residual)))


def normalized_mse(y_model, y_true) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    peak = max(float(np.max(np.abs(true))), 1e-30)
    return mse(y_model, y_true) / (peak * peak)


def data_fit(y_model, y_true) -> float:
    residual = _real_residual(y_model, y_true)
    true = np.asarray(y_true, dtype=np.float64)
    numerator = float(np.sum(np.square(residual)))
    denominator = float(np.sum(np.square(true)))
    return numerator / max(denominator, 1e-30)


def relative_l2(y_model, y_true) -> float:
    model = np.asarray(y_model)
    true = np.asarray(y_true)
    numerator = float(np.linalg.norm(model - true))
    denominator = float(np.linalg.norm(true))
    return numerator / max(denominator, 1e-30)


def residual_rms(y_model, y_true) -> float:
    residual = _real_residual(y_model, y_true)
    return float(np.sqrt(np.mean(np.square(residual))))


def fit_sigma(y_model, y_true) -> float:
    residual = _real_residual(y_model, y_true)
    true = np.asarray(y_true, dtype=np.float64)
    residual_std = float(np.std(residual))
    signal_std = float(np.std(true))
    if signal_std <= 0.0:
        return float("inf") if residual_std > 0.0 else 0.0
    return residual_std / signal_std


def snr_db(signal, noise) -> float:
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    signal_rms = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2)))
    if noise_rms <= 0.0:
        return float("inf")
    return 20.0 * np.log10(signal_rms / noise_rms)
