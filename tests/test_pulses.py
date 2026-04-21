import numpy as np

from thzsim2.core.pulses import gaussian_carrier_pulse, make_pulse, sech_carrier_pulse


def test_model_swap_keeps_same_interface():
    t = np.linspace(-3e-12, 3e-12, 2001)

    common_params = {
        "amp": 1.0,
        "t0": 0.4e-12,
        "tau": 0.2e-12,
        "f0": 1.1e12,
        "phi": 0.3,
    }

    pulse_spec_gauss = {
        "model": "gaussian_carrier",
        "params": common_params,
    }
    pulse_spec_sech = {
        "model": "sech_carrier",
        "params": common_params,
    }

    g = make_pulse(t, pulse_spec_gauss)
    s = make_pulse(t, pulse_spec_sech)

    assert g.shape == t.shape
    assert s.shape == t.shape
    assert np.isfinite(g).all()
    assert np.isfinite(s).all()
    assert not np.allclose(g, s)


def test_changing_t0_shifts_pulse():
    t = np.linspace(-4e-12, 4e-12, 4001)

    spec1 = {
        "model": "gaussian_carrier",
        "params": {
            "amp": 1.0,
            "t0": -0.5e-12,
            "tau": 0.25e-12,
            "f0": 0.0,
            "phi": 0.0,
        },
    }
    spec2 = {
        "model": "gaussian_carrier",
        "params": {
            "amp": 1.0,
            "t0": 0.8e-12,
            "tau": 0.25e-12,
            "f0": 0.0,
            "phi": 0.0,
        },
    }

    p1 = make_pulse(t, spec1)
    p2 = make_pulse(t, spec2)

    t_peak_1 = t[np.argmax(p1)]
    t_peak_2 = t[np.argmax(p2)]

    assert np.isclose(t_peak_1, spec1["params"]["t0"], atol=t[1] - t[0])
    assert np.isclose(t_peak_2, spec2["params"]["t0"], atol=t[1] - t[0])
    assert np.isclose(t_peak_2 - t_peak_1, 1.3e-12, atol=2.0 * (t[1] - t[0]))


def test_custom_pulse_callable_works():
    t = np.linspace(-2.0, 2.0, 101)

    def custom_func(t, amp=1.0, t0=0.0, width=1.0):
        y = np.zeros_like(t)
        mask = np.abs(t - t0) <= width
        y[mask] = amp
        return y

    pulse_spec = {
        "model": "custom",
        "callable": custom_func,
        "params": {
            "amp": 2.5,
            "t0": 0.25,
            "width": 0.5,
        },
    }

    y = make_pulse(t, pulse_spec)

    assert y.shape == t.shape
    assert np.max(y) == 2.5
    assert np.min(y) == 0.0
    assert np.count_nonzero(y) > 0


def test_direct_pulse_functions_exist_and_match_make_pulse():
    t = np.linspace(-3e-12, 3e-12, 1001)
    params = {
        "amp": 0.8,
        "t0": 0.1e-12,
        "tau": 0.3e-12,
        "f0": 0.9e12,
        "phi": -0.2,
    }

    g1 = gaussian_carrier_pulse(t, **params)
    g2 = make_pulse(t, {"model": "gaussian_carrier", "params": params})
    s1 = sech_carrier_pulse(t, **params)
    s2 = make_pulse(t, {"model": "sech_carrier", "params": params})

    assert np.allclose(g1, g2)
    assert np.allclose(s1, s2)
