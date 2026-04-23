import numpy as np

from thzsim2.core.materials import evaluate_material_nk
from thzsim2.models import Drude, DrudeLorentz, Lorentz, LorentzOscillator, TwoDrude


def test_drude_generation_returns_finite_aligned_nk():
    freq_grid_thz = np.linspace(0.1, 2.0, 256)
    nk = evaluate_material_nk(
        freq_grid_thz,
        Drude(eps_inf=3.4, plasma_freq_thz=1.2, gamma_thz=0.15),
    )

    assert nk.shape == freq_grid_thz.shape
    assert np.isfinite(nk).all()
    assert np.all(np.imag(nk) >= -1e-12)


def test_lorentz_generation_returns_finite_aligned_nk():
    freq_grid_thz = np.linspace(0.1, 2.0, 256)
    nk = evaluate_material_nk(
        freq_grid_thz,
        Lorentz(eps_inf=2.2, delta_eps=1.4, resonance_thz=1.0, gamma_thz=0.08),
    )

    assert nk.shape == freq_grid_thz.shape
    assert np.isfinite(nk).all()
    assert np.std(np.real(nk)) > 1e-4


def test_two_drude_generation_returns_finite_aligned_nk():
    freq_grid_thz = np.linspace(0.1, 2.5, 300)
    nk = evaluate_material_nk(
        freq_grid_thz,
        TwoDrude(
            eps_inf=11.7,
            plasma_freq1_thz=0.9,
            gamma1_thz=0.4,
            plasma_freq2_thz=2.2,
            gamma2_thz=3.0,
        ),
    )

    assert nk.shape == freq_grid_thz.shape
    assert np.isfinite(nk).all()
    assert np.all(np.imag(nk) >= -1e-12)
    assert np.std(np.real(nk)) > 1e-4


def test_drude_lorentz_generation_returns_finite_aligned_nk():
    freq_grid_thz = np.linspace(0.1, 2.5, 300)
    nk = evaluate_material_nk(
        freq_grid_thz,
        DrudeLorentz(
            eps_inf=1.9,
            plasma_freq_thz=0.9,
            gamma_thz=0.12,
            oscillators=(
                LorentzOscillator(delta_eps=0.8, resonance_thz=1.4, gamma_thz=0.07),
                LorentzOscillator(delta_eps=0.3, resonance_thz=2.1, gamma_thz=0.15),
            ),
        ),
    )

    assert nk.shape == freq_grid_thz.shape
    assert np.isfinite(nk).all()
    assert np.all(np.imag(nk) >= -1e-12)
