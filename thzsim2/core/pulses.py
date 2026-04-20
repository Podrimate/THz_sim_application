import numpy as np


def gaussian_carrier_pulse(t, amp=1.0, t0=0.0, tau=0.2e-12, f0=1.0e12, phi=0.0):
    """Return a Gaussian-envelope pulse with a cosine carrier."""
    t = np.asarray(t, dtype=np.float64)
    if tau <= 0:
        raise ValueError("tau must be positive")
    env = np.exp(-0.5 * ((t - t0) / tau) ** 2)
    car = np.cos(2.0 * np.pi * f0 * (t - t0) + phi)
    return amp * env * car


def sech_carrier_pulse(t, amp=1.0, t0=0.0, tau=0.2e-12, f0=1.0e12, phi=0.0):
    """Return a sech-envelope pulse with a cosine carrier."""
    t = np.asarray(t, dtype=np.float64)
    if tau <= 0:
        raise ValueError("tau must be positive")
    x = (t - t0) / tau
    env = 1.0 / np.cosh(np.clip(x, -700.0, 700.0))
    car = np.cos(2.0 * np.pi * f0 * (t - t0) + phi)
    return amp * env * car


def make_pulse(t, pulse_spec):
    """Create a pulse from a dictionary specification."""
    if not isinstance(pulse_spec, dict):
        raise TypeError("pulse_spec must be a dictionary")
    if "model" not in pulse_spec:
        raise KeyError("pulse_spec must contain 'model'")

    model = pulse_spec["model"]
    params = dict(pulse_spec.get("params", {}))

    if model == "gaussian_carrier":
        return gaussian_carrier_pulse(t, **params)

    if model == "sech_carrier":
        return sech_carrier_pulse(t, **params)

    if model == "custom":
        func = pulse_spec.get("callable", params.pop("callable", None))
        if func is None:
            raise KeyError("custom pulse_spec must provide a callable")
        if not callable(func):
            raise TypeError("custom callable must be callable")
        return np.asarray(func(np.asarray(t), **params))

    raise ValueError(f"unsupported pulse model: {model}")
