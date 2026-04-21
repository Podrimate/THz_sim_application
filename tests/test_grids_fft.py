import numpy as np

from thzsim2.core.fft import fft_t_to_w, ifft_w_to_t, zero_pad_trace
from thzsim2.core.grids import make_grids, make_omega_grid, make_time_grid


def test_make_time_grid_spacing_and_center():
    n = 9
    dt = 0.01e-12
    t_center = 2.5e-12

    t = make_time_grid(n=n, dt=dt, t_center=t_center)

    assert t.shape == (n,)
    assert np.allclose(np.diff(t), dt)
    assert np.isclose(t.mean(), t_center)
    assert np.isclose(t[-1] - t[0], (n - 1) * dt)


def test_make_omega_grid_spacing_and_period():
    n = 4096
    dt = 0.01e-12

    w = make_omega_grid(n=n, dt=dt)
    dw = 2.0 * np.pi / (n * dt)

    w_sorted = np.sort(w)
    diffs = np.diff(w_sorted)

    assert np.allclose(diffs, dw)
    assert np.isclose(2.0 * np.pi / dw, n * dt)


def test_make_grids_consistency():
    n = 1024
    dt = 0.02e-12
    t_center = -0.3e-12

    t, w = make_grids(n=n, dt=dt, t_center=t_center)

    assert np.allclose(t, make_time_grid(n=n, dt=dt, t_center=t_center))
    assert np.allclose(w, make_omega_grid(n=n, dt=dt))


def test_fft_ifft_roundtrip_real():
    n = 2048
    dt = 0.01e-12
    t = make_time_grid(n=n, dt=dt, t_center=0.0)

    x = np.exp(-0.5 * ((t - 0.4e-12) / (0.15e-12)) ** 2) * np.cos(
        2.0 * np.pi * 1.1e12 * (t - 0.4e-12)
    )

    w, s = fft_t_to_w(x, dt=dt, t0=t[0])
    xr = ifft_w_to_t(s, dt=dt, t0=t[0])

    assert np.allclose(w, make_omega_grid(n=n, dt=dt))
    assert np.max(np.abs(xr.imag)) < 1e-12
    assert np.allclose(xr.real, x, rtol=1e-10, atol=1e-12)


def test_fft_ifft_roundtrip_complex():
    n = 1024
    dt = 0.015e-12
    t = make_time_grid(n=n, dt=dt, t_center=0.0)

    x = np.exp(-0.5 * ((t + 0.2e-12) / (0.18e-12)) ** 2) * np.exp(
        1j * (2.0 * np.pi * 0.7e12 * t + 0.4)
    )

    _, s = fft_t_to_w(x, dt=dt, t0=t[0])
    xr = ifft_w_to_t(s, dt=dt, t0=t[0])

    assert np.allclose(xr, x, rtol=1e-10, atol=1e-12)


def test_time_shift_gives_expected_linear_phase():
    n = 4096
    dt = 0.005e-12
    t = make_time_grid(n=n, dt=dt, t_center=0.0)

    shift = 0.35e-12
    sigma = 0.12e-12
    f0 = 0.8e12

    x0 = np.exp(-0.5 * (t / sigma) ** 2) * np.exp(1j * 2.0 * np.pi * f0 * t)
    x1 = np.exp(-0.5 * ((t - shift) / sigma) ** 2) * np.exp(
        1j * 2.0 * np.pi * f0 * (t - shift)
    )

    w, s0 = fft_t_to_w(x0, dt=dt, t0=t[0])
    _, s1 = fft_t_to_w(x1, dt=dt, t0=t[0])

    mask = np.abs(s0) > 1e-8 * np.max(np.abs(s0))
    expected = s0[mask] * np.exp(1j * w[mask] * shift)

    assert np.allclose(s1[mask], expected, rtol=1e-6, atol=1e-9)


def test_zero_padding_preserves_original_segment():
    x = np.array([1.0, -2.0, 3.0, 4.0])
    y = zero_pad_trace(x, pad_factor=3)

    assert y.shape == (12,)
    assert np.array_equal(y[: x.size], x)
    assert np.array_equal(y[x.size :], np.zeros(8))


def test_zero_padding_preserves_complex_dtype_and_values():
    x = np.array([1.0 + 2.0j, 3.0 - 1.0j, -2.0 + 0.5j])
    y = zero_pad_trace(x, pad_factor=2)

    assert y.dtype == x.dtype
    assert np.array_equal(y[: x.size], x)
    assert np.array_equal(y[x.size :], np.zeros(x.size, dtype=x.dtype))
