from __future__ import annotations

import numpy as np


def mse(y_model, y_true) -> float:
    residual = np.asarray(y_model) - np.asarray(y_true)
    if np.iscomplexobj(residual):
        residual = np.abs(residual)
    return float(np.mean(np.square(np.asarray(residual, dtype=np.float64))))


def relative_l2(y_model, y_true) -> float:
    model = np.asarray(y_model)
    true = np.asarray(y_true)
    numerator = float(np.linalg.norm(model - true))
    denominator = float(np.linalg.norm(true))
    return numerator / max(denominator, 1e-30)


def snr_db(signal, noise) -> float:
    signal = np.asarray(signal)
    noise = np.asarray(noise)
    signal_rms = float(np.sqrt(np.mean(np.abs(signal) ** 2)))
    noise_rms = float(np.sqrt(np.mean(np.abs(noise) ** 2)))
    if noise_rms <= 0.0:
        return float("inf")
    return 20.0 * np.log10(signal_rms / noise_rms)
