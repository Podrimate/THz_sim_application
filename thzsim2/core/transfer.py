from __future__ import annotations

from copy import deepcopy

import numpy as np

C0 = 299792458.0


def _as_1d_array(x, name: str):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_complex_array(x, name: str, shape=None):
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim == 0:
        if shape is None:
            raise ValueError(f"{name} scalar needs shape")
        arr = np.full(shape, arr, dtype=np.complex128)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1D")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} has incompatible shape")
    return arr


def _normalize_branch_polarization(polarization: str):
    pol = str(polarization).strip().lower()
    if pol not in {"s", "p"}:
        raise ValueError("polarization must be 's' or 'p'")
    return pol


def _normalize_polarization_mode(polarization: str):
    pol = str(polarization).strip().lower()
    if pol not in {"s", "p", "mixed"}:
        raise ValueError("polarization must be 's', 'p', or 'mixed'")
    return pol


def _normalize_polarization_mix(polarization_mix):
    mix = float(0.5 if polarization_mix is None else polarization_mix)
    if not (0.0 <= mix <= 1.0):
        raise ValueError("polarization_mix must be between 0 and 1")
    return mix


def _normalize_mode(mode: str):
    value = str(mode).strip().lower()
    if value not in {"transmission", "reflection"}:
        raise ValueError("mode must be 'transmission' or 'reflection'")
    return value


def fresnel_t(nj, nk):
    nj = np.asarray(nj, dtype=np.complex128)
    nk = np.asarray(nk, dtype=np.complex128)
    return 2.0 * nj / (nj + nk)


def fresnel_r(nj, nk):
    nj = np.asarray(nj, dtype=np.complex128)
    nk = np.asarray(nk, dtype=np.complex128)
    return (nj - nk) / (nj + nk)


def fresnel_t_oblique(nj, nk, cos_j, cos_k, *, polarization="s"):
    pol = _normalize_branch_polarization(polarization)
    nj = np.asarray(nj, dtype=np.complex128)
    nk = np.asarray(nk, dtype=np.complex128)
    cos_j = np.asarray(cos_j, dtype=np.complex128)
    cos_k = np.asarray(cos_k, dtype=np.complex128)
    if pol == "s":
        denom = nj * cos_j + nk * cos_k
    else:
        denom = nk * cos_j + nj * cos_k
    return 2.0 * nj * cos_j / denom


def fresnel_r_oblique(nj, nk, cos_j, cos_k, *, polarization="s"):
    pol = _normalize_branch_polarization(polarization)
    nj = np.asarray(nj, dtype=np.complex128)
    nk = np.asarray(nk, dtype=np.complex128)
    cos_j = np.asarray(cos_j, dtype=np.complex128)
    cos_k = np.asarray(cos_k, dtype=np.complex128)
    if pol == "s":
        numer = nj * cos_j - nk * cos_k
        denom = nj * cos_j + nk * cos_k
    else:
        numer = nk * cos_j - nj * cos_k
        denom = nk * cos_j + nj * cos_k
    return numer / denom


def propagation_factor(omega_rad_s, n, thickness_m):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    n = _as_complex_array(n, "n", shape=omega.shape)
    thickness_m = float(thickness_m)
    if thickness_m < 0.0:
        raise ValueError("thickness_m must be nonnegative")
    return np.exp(1j * omega * n * thickness_m / C0)


def _propagation_factor_oblique(omega_rad_s, n, cos_theta, thickness_m):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    n = _as_complex_array(n, "n", shape=omega.shape)
    cos_theta = _as_complex_array(cos_theta, "cos_theta", shape=omega.shape)
    thickness_m = float(thickness_m)
    if thickness_m < 0.0:
        raise ValueError("thickness_m must be nonnegative")
    return np.exp(1j * omega * n * cos_theta * thickness_m / C0)


def _finite_fp_factor(z, max_internal_reflections):
    if max_internal_reflections is None:
        return 1.0 / (1.0 - z)
    mmax = int(max_internal_reflections)
    if mmax < 0:
        raise ValueError("max_internal_reflections must be >= 0 or None")
    out = np.ones_like(z, dtype=np.complex128)
    term = np.ones_like(z, dtype=np.complex128)
    for _ in range(mmax):
        term = term * z
        out = out + term
    return out


def _matrix_multiply(a11, a12, a21, a22, b11, b12, b21, b22):
    return (
        a11 * b11 + a12 * b21,
        a11 * b12 + a12 * b22,
        a21 * b11 + a22 * b21,
        a21 * b12 + a22 * b22,
    )


def _omega_safe(omega_rad_s, floor=1e-30):
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    return np.where(np.abs(omega) < floor, floor, np.abs(omega))


def _nk_from_eps(eps):
    eps = np.asarray(eps, dtype=np.complex128)
    nk = np.sqrt(eps)
    mask = np.imag(nk) < 0.0
    return np.where(mask, -nk, nk)


def _forward_cos_theta(n, transverse_n):
    n = np.asarray(n, dtype=np.complex128)
    transverse_n = np.asarray(transverse_n, dtype=np.complex128)
    ratio = np.divide(
        transverse_n,
        n,
        out=np.zeros_like(n, dtype=np.complex128),
        where=np.abs(n) > 1e-30,
    )
    cos_theta = np.sqrt(1.0 - ratio * ratio + 0.0j)
    kz_like = n * cos_theta
    flip = (np.imag(kz_like) < 0.0) | (
        np.isclose(np.imag(kz_like), 0.0, atol=1e-14) & (np.real(kz_like) < 0.0)
    )
    return np.where(flip, -cos_theta, cos_theta)


def _interface_matrix(nj, nk, cos_j, cos_k, *, polarization):
    r_jk = fresnel_r_oblique(nj, nk, cos_j, cos_k, polarization=polarization)
    t_jk = fresnel_t_oblique(nj, nk, cos_j, cos_k, polarization=polarization)
    return (
        1.0 / t_jk,
        r_jk / t_jk,
        r_jk / t_jk,
        1.0 / t_jk,
    )


def _physical_response(response, *, polarization, mode):
    if _normalize_mode(mode) == "reflection" and _normalize_branch_polarization(polarization) == "p":
        return -np.asarray(response, dtype=np.complex128)
    return np.asarray(response, dtype=np.complex128)


def _multilayer_response_recursive(omega_rad_s, n_arrays, cos_arrays, d_list, *, polarization):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    if len(d_list) == 0:
        return np.zeros_like(omega, dtype=np.complex128), np.ones_like(omega, dtype=np.complex128)

    last = len(d_list) - 1
    reflection_next = fresnel_r_oblique(
        n_arrays[last + 1],
        n_arrays[last + 2],
        cos_arrays[last + 1],
        cos_arrays[last + 2],
        polarization=polarization,
    )
    transmission_next = fresnel_t_oblique(
        n_arrays[last + 1],
        n_arrays[last + 2],
        cos_arrays[last + 1],
        cos_arrays[last + 2],
        polarization=polarization,
    )

    for index in range(last, -1, -1):
        phase = _propagation_factor_oblique(omega, n_arrays[index + 1], cos_arrays[index + 1], d_list[index])
        r_forward = fresnel_r_oblique(
            n_arrays[index],
            n_arrays[index + 1],
            cos_arrays[index],
            cos_arrays[index + 1],
            polarization=polarization,
        )
        r_backward = fresnel_r_oblique(
            n_arrays[index + 1],
            n_arrays[index],
            cos_arrays[index + 1],
            cos_arrays[index],
            polarization=polarization,
        )
        t_forward = fresnel_t_oblique(
            n_arrays[index],
            n_arrays[index + 1],
            cos_arrays[index],
            cos_arrays[index + 1],
            polarization=polarization,
        )
        t_backward = fresnel_t_oblique(
            n_arrays[index + 1],
            n_arrays[index],
            cos_arrays[index + 1],
            cos_arrays[index],
            polarization=polarization,
        )
        roundtrip = r_backward * phase * reflection_next * phase
        denominator = 1.0 - roundtrip
        transmission_next = t_forward * phase * transmission_next / denominator
        reflection_next = r_forward + t_forward * phase * reflection_next * phase * t_backward / denominator

    return reflection_next, transmission_next


def _medium_stack_response_positive(
    omega_rad_s,
    n_list,
    thicknesses_m,
    *,
    angle_deg=0.0,
    polarization="s",
    polarization_mix=None,
    mode="transmission",
    max_internal_reflections=0,
):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    pol_mode = _normalize_polarization_mode(polarization)
    mode = _normalize_mode(mode)

    if pol_mode == "mixed":
        mix = _normalize_polarization_mix(polarization_mix)
        response_s = _medium_stack_response_positive(
            omega,
            n_list,
            thicknesses_m,
            angle_deg=angle_deg,
            polarization="s",
            polarization_mix=None,
            mode=mode,
            max_internal_reflections=max_internal_reflections,
        )
        response_p = _medium_stack_response_positive(
            omega,
            n_list,
            thicknesses_m,
            angle_deg=angle_deg,
            polarization="p",
            polarization_mix=None,
            mode=mode,
            max_internal_reflections=max_internal_reflections,
        )
        return mix * np.asarray(response_p, dtype=np.complex128) + (1.0 - mix) * np.asarray(
            response_s,
            dtype=np.complex128,
        )
    pol = _normalize_branch_polarization(pol_mode)

    if len(n_list) != len(thicknesses_m) + 2:
        raise ValueError("len(n_list) must equal len(thicknesses_m) + 2")

    n_arrays = [_as_complex_array(n, f"n_list[{index}]", shape=omega.shape) for index, n in enumerate(n_list)]
    d_list = [float(value) for value in thicknesses_m]
    for thickness_m in d_list:
        if thickness_m < 0.0:
            raise ValueError("all thicknesses_m must be nonnegative")

    angle_rad = np.deg2rad(float(angle_deg))
    transverse_n = n_arrays[0] * np.sin(angle_rad)
    cos_arrays = [_forward_cos_theta(n, transverse_n) for n in n_arrays]

    if len(d_list) == 0:
        if mode == "transmission":
            response = fresnel_t_oblique(
                n_arrays[0],
                n_arrays[1],
                cos_arrays[0],
                cos_arrays[1],
                polarization=pol,
            )
        else:
            response = fresnel_r_oblique(
                n_arrays[0],
                n_arrays[1],
                cos_arrays[0],
                cos_arrays[1],
                polarization=pol,
            )
        return _physical_response(response, polarization=pol, mode=mode)

    # Exact finite-round-trip sums for a single layer keep the old behavior where
    # internal reflections can be truncated explicitly.
    if len(d_list) == 1 and max_internal_reflections is not None:
        n_in, n_layer, n_out = n_arrays
        cos_in, cos_layer, cos_out = cos_arrays
        thickness_m = d_list[0]
        p1 = _propagation_factor_oblique(omega, n_layer, cos_layer, thickness_m)
        r01 = fresnel_r_oblique(n_in, n_layer, cos_in, cos_layer, polarization=pol)
        r10 = fresnel_r_oblique(n_layer, n_in, cos_layer, cos_in, polarization=pol)
        r12 = fresnel_r_oblique(n_layer, n_out, cos_layer, cos_out, polarization=pol)
        t01 = fresnel_t_oblique(n_in, n_layer, cos_in, cos_layer, polarization=pol)
        t12 = fresnel_t_oblique(n_layer, n_out, cos_layer, cos_out, polarization=pol)
        roundtrip = r10 * p1 * r12 * p1

        if mode == "transmission":
            fp = _finite_fp_factor(roundtrip, max_internal_reflections)
            return _physical_response(t01 * p1 * t12 * fp, polarization=pol, mode=mode)

        mmax = int(max_internal_reflections)
        if mmax == 0:
            return _physical_response(r01, polarization=pol, mode=mode)
        t10 = fresnel_t_oblique(n_layer, n_in, cos_layer, cos_in, polarization=pol)
        fp = _finite_fp_factor(roundtrip, mmax - 1)
        return _physical_response(r01 + t01 * p1 * r12 * p1 * t10 * fp, polarization=pol, mode=mode)

    reflection, transmission = _multilayer_response_recursive(
        omega,
        n_arrays,
        cos_arrays,
        d_list,
        polarization=pol,
    )
    if mode == "transmission":
        return _physical_response(transmission, polarization=pol, mode=mode)
    return _physical_response(reflection, polarization=pol, mode=mode)


def single_layer_transfer(
    omega_rad_s,
    n_in,
    n_layer,
    n_out,
    thickness_m,
    max_internal_reflections=0,
    *,
    angle_deg=0.0,
    polarization="s",
    polarization_mix=None,
    mode="transmission",
):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    thickness_m = float(thickness_m)
    if thickness_m < 0.0:
        raise ValueError("thickness_m must be nonnegative")
    if thickness_m == 0.0:
        return _medium_stack_response_positive(
            omega,
            [n_in, n_out],
            [],
            angle_deg=angle_deg,
            polarization=polarization,
            polarization_mix=polarization_mix,
            mode=mode,
            max_internal_reflections=max_internal_reflections,
        )
    return _medium_stack_response_positive(
        omega,
        [n_in, n_layer, n_out],
        [thickness_m],
        angle_deg=angle_deg,
        polarization=polarization,
        polarization_mix=polarization_mix,
        mode=mode,
        max_internal_reflections=max_internal_reflections,
    )


def multilayer_transfer(
    omega_rad_s,
    n_list,
    thicknesses_m,
    max_internal_reflections=0,
    *,
    angle_deg=0.0,
    polarization="s",
    polarization_mix=None,
    mode="transmission",
):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    return _medium_stack_response_positive(
        omega,
        n_list,
        thicknesses_m,
        angle_deg=angle_deg,
        polarization=polarization,
        polarization_mix=polarization_mix,
        mode=mode,
        max_internal_reflections=max_internal_reflections,
    )


def _nk_from_material_dict(omega_rad_s, material):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    kind = material["kind"]
    params = deepcopy(material.get("parameters", {}))
    abs_freq_thz = np.abs(omega) / (2.0 * np.pi * 1e12)

    if kind == "NKFile":
        freq_thz = np.asarray(material["freq_thz"], dtype=np.float64)
        n_vals = np.asarray(material["n"], dtype=np.float64)
        k_vals = np.asarray(material["k"], dtype=np.float64)
        aligned_n = np.interp(abs_freq_thz, freq_thz, n_vals)
        aligned_k = np.interp(abs_freq_thz, freq_thz, k_vals)
        nk = aligned_n + 1j * aligned_k
    elif kind == "ConstantNK":
        nk = np.full(omega.shape, complex(float(params["n"]), float(params["k"])), dtype=np.complex128)
    elif kind == "Drude":
        eps_inf = float(params["eps_inf"])
        plasma = 2.0 * np.pi * float(params["plasma_freq_thz"]) * 1e12
        gamma = 2.0 * np.pi * float(params["gamma_thz"]) * 1e12
        om = _omega_safe(omega)
        eps = eps_inf - (plasma**2) / (om * (om + 1j * gamma))
        nk = _nk_from_eps(eps)
    elif kind == "Lorentz":
        eps_inf = float(params["eps_inf"])
        delta_eps = float(params["delta_eps"])
        resonance = 2.0 * np.pi * float(params["resonance_thz"]) * 1e12
        gamma = 2.0 * np.pi * float(params["gamma_thz"]) * 1e12
        om = np.abs(omega)
        eps = eps_inf + delta_eps * (resonance**2) / ((resonance**2) - (om**2) - 1j * gamma * om)
        nk = _nk_from_eps(eps)
    elif kind == "DrudeLorentz":
        eps = np.full(omega.shape, complex(float(params["eps_inf"])), dtype=np.complex128)
        plasma = 2.0 * np.pi * float(params.get("plasma_freq_thz", 0.0)) * 1e12
        gamma = 2.0 * np.pi * float(params.get("gamma_thz", 0.0)) * 1e12
        if plasma != 0.0:
            om_safe = _omega_safe(omega)
            eps = eps - (plasma**2) / (om_safe * (om_safe + 1j * gamma))
        for oscillator in params.get("oscillators", []):
            delta_eps = float(
                oscillator["delta_eps"]["value"]
                if isinstance(oscillator["delta_eps"], dict)
                else oscillator["delta_eps"]
            )
            resonance = float(
                oscillator["resonance_thz"]["value"]
                if isinstance(oscillator["resonance_thz"], dict)
                else oscillator["resonance_thz"]
            )
            osc_gamma = float(
                oscillator["gamma_thz"]["value"]
                if isinstance(oscillator["gamma_thz"], dict)
                else oscillator["gamma_thz"]
            )
            omega0 = 2.0 * np.pi * resonance * 1e12
            gamma_i = 2.0 * np.pi * osc_gamma * 1e12
            om = np.abs(omega)
            eps = eps + delta_eps * (omega0**2) / ((omega0**2) - (om**2) - 1j * gamma_i * om)
        nk = _nk_from_eps(eps)
    else:
        raise ValueError(f"unsupported material kind: {kind}")

    out = np.asarray(nk, dtype=np.complex128)
    neg = omega < 0.0
    out[neg] = np.conj(out[neg])
    zero = np.isclose(omega, 0.0)
    out[zero] = np.real(out[zero]) + 0.0j
    return out


def stack_response_function(
    omega_rad_s,
    resolved_stack,
    max_internal_reflections=0,
    *,
    angle_deg=0.0,
    polarization="s",
    polarization_mix=None,
    mode="transmission",
):
    omega = _as_1d_array(omega_rad_s, "omega_rad_s")
    mode = _normalize_mode(mode)
    layers = [layer for layer in resolved_stack["layers"] if float(layer["thickness_um"]) > 0.0]
    n_list = [np.full(omega.shape, complex(float(resolved_stack["n_in"])), dtype=np.complex128)]
    thicknesses_m = []
    omega_abs = np.abs(omega)
    for layer in layers:
        n_list.append(_nk_from_material_dict(omega_abs, layer["material"]))
        thicknesses_m.append(float(layer["thickness_um"]) * 1e-6)
    n_list.append(np.full(omega.shape, complex(float(resolved_stack["n_out"])), dtype=np.complex128))

    response = multilayer_transfer(
        omega_abs,
        n_list,
        thicknesses_m,
        max_internal_reflections=max_internal_reflections,
        angle_deg=angle_deg,
        polarization=polarization,
        polarization_mix=polarization_mix,
        mode=mode,
    )
    out = np.asarray(response, dtype=np.complex128)
    neg = omega < 0.0
    out[neg] = np.conj(out[neg])
    zero = np.isclose(omega, 0.0)
    out[zero] = np.real(out[zero]) + 0.0j
    return out


def stack_transfer_function(
    omega_rad_s,
    resolved_stack,
    max_internal_reflections=0,
    *,
    angle_deg=0.0,
    polarization="s",
    polarization_mix=None,
    mode="transmission",
):
    return stack_response_function(
        omega_rad_s,
        resolved_stack,
        max_internal_reflections=max_internal_reflections,
        angle_deg=angle_deg,
        polarization=polarization,
        polarization_mix=polarization_mix,
        mode=mode,
    )
