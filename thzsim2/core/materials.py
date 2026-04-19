from __future__ import annotations

import numpy as np

from thzsim2.models.sample import ConstantNK, Drude, DrudeLorentz, Lorentz


def _as_freq_grid(freq_grid_thz):
    freq = np.asarray(freq_grid_thz, dtype=np.float64)
    if freq.ndim != 1:
        raise ValueError("freq_grid_thz must be 1D")
    if freq.size == 0:
        raise ValueError("freq_grid_thz must be non-empty")
    if not np.isfinite(freq).all():
        raise ValueError("freq_grid_thz must contain only finite values")
    if np.any(freq < 0.0):
        raise ValueError("freq_grid_thz must be nonnegative")
    if freq.size > 1 and np.any(np.diff(freq) <= 0.0):
        raise ValueError("freq_grid_thz must be strictly increasing")
    return freq


def _omega_safe(omega, floor=1e-30):
    omega = np.asarray(omega, dtype=np.float64)
    return np.where(np.abs(omega) < floor, floor, omega)


def eps_drude(freq_grid_thz, *, eps_inf=1.0, plasma_freq_thz=0.0, gamma_thz=0.0):
    freq = _as_freq_grid(freq_grid_thz)
    omega = 2.0 * np.pi * freq * 1e12
    plasma = 2.0 * np.pi * float(plasma_freq_thz) * 1e12
    gamma = 2.0 * np.pi * float(gamma_thz) * 1e12
    om = _omega_safe(omega)
    return complex(eps_inf) - (plasma**2) / (om * (om + 1j * gamma))


def eps_lorentz(freq_grid_thz, *, eps_inf=1.0, delta_eps=1.0, resonance_thz=1.0, gamma_thz=0.0):
    freq = _as_freq_grid(freq_grid_thz)
    omega = 2.0 * np.pi * freq * 1e12
    omega0 = 2.0 * np.pi * float(resonance_thz) * 1e12
    gamma = 2.0 * np.pi * float(gamma_thz) * 1e12
    denom = (omega0**2) - (omega**2) - 1j * gamma * omega
    return complex(eps_inf) + float(delta_eps) * (omega0**2) / denom


def eps_drude_lorentz(
    freq_grid_thz,
    *,
    eps_inf=1.0,
    plasma_freq_thz=0.0,
    gamma_thz=0.0,
    oscillators=(),
):
    freq = _as_freq_grid(freq_grid_thz)
    omega = 2.0 * np.pi * freq * 1e12
    eps = np.full(freq.shape, complex(eps_inf), dtype=np.complex128)

    plasma = 2.0 * np.pi * float(plasma_freq_thz) * 1e12
    gamma = 2.0 * np.pi * float(gamma_thz) * 1e12
    if plasma != 0.0:
        om_safe = _omega_safe(omega)
        eps = eps - (plasma**2) / (om_safe * (om_safe + 1j * gamma))

    for oscillator in oscillators:
        omega0 = 2.0 * np.pi * float(oscillator["resonance_thz"]) * 1e12
        gamma_i = 2.0 * np.pi * float(oscillator["gamma_thz"]) * 1e12
        delta_eps = float(oscillator["delta_eps"])
        denom = (omega0**2) - (omega**2) - 1j * gamma_i * omega
        eps = eps + delta_eps * (omega0**2) / denom

    return eps


def nk_from_eps(eps):
    eps = np.asarray(eps, dtype=np.complex128)
    nk = np.sqrt(eps)
    mask = np.imag(nk) < 0.0
    nk = np.where(mask, -nk, nk)
    return nk


def evaluate_material_nk(freq_grid_thz, material):
    freq = _as_freq_grid(freq_grid_thz)

    if isinstance(material, ConstantNK):
        nk = np.full(freq.shape, complex(float(material.n), float(material.k)), dtype=np.complex128)
        return nk

    if isinstance(material, Drude):
        return nk_from_eps(
            eps_drude(
                freq,
                eps_inf=float(material.eps_inf),
                plasma_freq_thz=float(material.plasma_freq_thz),
                gamma_thz=float(material.gamma_thz),
            )
        )

    if isinstance(material, Lorentz):
        return nk_from_eps(
            eps_lorentz(
                freq,
                eps_inf=float(material.eps_inf),
                delta_eps=float(material.delta_eps),
                resonance_thz=float(material.resonance_thz),
                gamma_thz=float(material.gamma_thz),
            )
        )

    if isinstance(material, DrudeLorentz):
        oscillators = [
            {
                "delta_eps": float(osc.delta_eps),
                "resonance_thz": float(osc.resonance_thz),
                "gamma_thz": float(osc.gamma_thz),
            }
            for osc in material.oscillators
        ]
        return nk_from_eps(
            eps_drude_lorentz(
                freq,
                eps_inf=float(material.eps_inf),
                plasma_freq_thz=float(material.plasma_freq_thz),
                gamma_thz=float(material.gamma_thz),
                oscillators=oscillators,
            )
        )

    raise TypeError(
        "evaluate_material_nk only supports resolved ConstantNK, Drude, Lorentz, and DrudeLorentz materials"
    )
