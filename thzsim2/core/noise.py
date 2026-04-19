from __future__ import annotations

import numpy as np


def noise_sigma_from_dynamic_range(signal, dynamic_range_db: float) -> float:
    amplitude = float(np.max(np.abs(np.asarray(signal))))
    if amplitude <= 0.0:
        raise ValueError("cannot derive noise sigma from a zero-amplitude trace")
    return amplitude / (10.0 ** (float(dynamic_range_db) / 20.0))


def add_white_gaussian_noise(trace, *, sigma: float, seed: int | None = None):
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("noise sigma must be nonnegative")
    values = np.asarray(trace, dtype=np.float64)
    if sigma == 0.0:
        return values.copy()
    rng = np.random.default_rng(seed)
    return values + rng.normal(loc=0.0, scale=sigma, size=values.shape)
