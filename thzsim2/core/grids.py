import numpy as np


def make_time_grid(n, dt, t_center=0.0):
    """Return a uniformly spaced time axis centered around ``t_center``."""
    n = int(n)
    dt = float(dt)
    if n <= 0:
        raise ValueError("n must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    m = np.arange(n, dtype=np.float64)
    return (m - 0.5 * (n - 1)) * dt + float(t_center)


def make_omega_grid(n, dt):
    """Return the FFT angular-frequency grid for ``n`` samples at spacing ``dt``."""
    n = int(n)
    dt = float(dt)
    if n <= 0:
        raise ValueError("n must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    return 2.0 * np.pi * np.fft.fftfreq(n, d=dt)


def make_grids(n, dt, t_center=0.0):
    """Return the matching time and angular-frequency grids."""
    t = make_time_grid(n=n, dt=dt, t_center=t_center)
    w = make_omega_grid(n=n, dt=dt)
    return t, w
