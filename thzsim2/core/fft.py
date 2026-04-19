import numpy as np

from .grids import make_omega_grid


def fft_t_to_w(trace, dt, t0=None):
    """Fourier transform a 1D time trace onto the angular-frequency grid."""
    x = np.asarray(trace)
    if x.ndim != 1:
        raise ValueError("trace must be 1D")

    n = x.size
    dt = float(dt)
    if n == 0:
        raise ValueError("trace must be non-empty")
    if dt <= 0:
        raise ValueError("dt must be positive")

    if t0 is None:
        t0 = 0.0
    t0 = float(t0)

    omega = make_omega_grid(n, dt)
    spectrum = dt * n * np.fft.ifft(x) * np.exp(1j * omega * t0)
    return omega, spectrum


def ifft_w_to_t(spectrum, dt, t0=None):
    """Inverse Fourier transform from angular-frequency space back to time."""
    s = np.asarray(spectrum)
    if s.ndim != 1:
        raise ValueError("spectrum must be 1D")

    n = s.size
    dt = float(dt)
    if n == 0:
        raise ValueError("spectrum must be non-empty")
    if dt <= 0:
        raise ValueError("dt must be positive")

    if t0 is None:
        t0 = 0.0
    t0 = float(t0)

    omega = make_omega_grid(n, dt)
    corrected = s * np.exp(-1j * omega * t0)
    trace = np.fft.fft(corrected) / (n * dt)
    return trace


def zero_pad_trace(trace, pad_factor=1):
    """Zero-pad a 1D trace by an integer factor while keeping the original prefix."""
    x = np.asarray(trace)
    if x.ndim != 1:
        raise ValueError("trace must be 1D")

    pad_factor = int(pad_factor)
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")

    if pad_factor == 1:
        return x.copy()

    out = np.zeros(x.size * pad_factor, dtype=x.dtype)
    out[: x.size] = x
    return out
