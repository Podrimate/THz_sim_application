import numpy as np
import pytest

from thzsim2.core.forward import simulate_sample_from_reference
from thzsim2.core.transfer import fresnel_r_oblique, single_layer_transfer, stack_response_function
from thzsim2.models import Measurement, ReferenceStandard
from thzsim2.workflows.reference import generate_reference_pulse, prepare_reference


def _manual_forward_cos_theta(n, transverse_n):
    n = np.asarray(n, dtype=np.complex128)
    transverse_n = np.asarray(transverse_n, dtype=np.complex128)
    ratio = np.divide(
        transverse_n,
        n,
        out=np.zeros_like(n, dtype=np.complex128),
        where=np.abs(n) > 1e-30,
    )
    cos_theta = np.sqrt(1.0 - ratio * ratio + 0.0j)
    kz = n * cos_theta
    flip = (np.imag(kz) < 0.0) | ((np.abs(np.imag(kz)) <= 1e-14) & (np.real(kz) < 0.0))
    return np.where(flip, -cos_theta, cos_theta)


def _manual_fresnel_r(nj, nk, cos_j, cos_k, polarization):
    if polarization == "s":
        return (nj * cos_j - nk * cos_k) / (nj * cos_j + nk * cos_k)
    return (nk * cos_j - nj * cos_k) / (nk * cos_j + nj * cos_k)


def _manual_fresnel_t(nj, nk, cos_j, cos_k, polarization):
    if polarization == "s":
        return 2.0 * nj * cos_j / (nj * cos_j + nk * cos_k)
    return 2.0 * nj * cos_j / (nk * cos_j + nj * cos_k)


def _manual_single_layer_response(omega, *, n_in, n_layer, n_out, thickness_m, angle_deg, polarization, mode):
    omega = np.asarray(omega, dtype=np.float64)
    transverse_n = complex(n_in) * np.sin(np.deg2rad(angle_deg))
    cos_in = _manual_forward_cos_theta(np.full(omega.shape, complex(n_in)), transverse_n)
    cos_layer = _manual_forward_cos_theta(np.full(omega.shape, complex(n_layer)), transverse_n)
    cos_out = _manual_forward_cos_theta(np.full(omega.shape, complex(n_out)), transverse_n)

    r01 = _manual_fresnel_r(n_in, n_layer, cos_in, cos_layer, polarization)
    r10 = _manual_fresnel_r(n_layer, n_in, cos_layer, cos_in, polarization)
    r12 = _manual_fresnel_r(n_layer, n_out, cos_layer, cos_out, polarization)
    t01 = _manual_fresnel_t(n_in, n_layer, cos_in, cos_layer, polarization)
    t10 = _manual_fresnel_t(n_layer, n_in, cos_layer, cos_in, polarization)
    t12 = _manual_fresnel_t(n_layer, n_out, cos_layer, cos_out, polarization)
    phase = np.exp(1j * omega * complex(n_layer) * cos_layer * float(thickness_m) / 299792458.0)
    denominator = 1.0 - r10 * r12 * phase * phase

    if mode == "transmission":
        return t01 * phase * t12 / denominator
    reflected = r01 + t01 * phase * r12 * phase * t10 / denominator
    if polarization == "p":
        reflected = -reflected
    return reflected


def _reference_result(tmp_path):
    reference_input = generate_reference_pulse(
        model="sech_carrier",
        sample_count=512,
        dt_ps=0.03,
        time_center_ps=8.0,
        pulse_center_ps=5.0,
        tau_ps=0.22,
        f0_thz=0.9,
        amp=1.0,
        phi_rad=0.0,
    )
    return prepare_reference(reference_input, output_root=tmp_path, run_label="oblique-transfer")


def test_normal_incidence_is_polarization_independent_for_isotropic_multilayers():
    omega = 2.0 * np.pi * np.linspace(0.1, 2.5, 96) * 1e12
    stack = {
        "n_in": 1.0,
        "n_out": 1.4,
        "layers": [
            {
                "name": "film_a",
                "thickness_um": 120.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 2.1, "k": 0.04}},
            },
            {
                "name": "film_b",
                "thickness_um": 75.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 1.7, "k": 0.01}},
            },
        ],
    }

    transmission_s = stack_response_function(omega, stack, max_internal_reflections=None, angle_deg=0.0, polarization="s")
    transmission_p = stack_response_function(omega, stack, max_internal_reflections=None, angle_deg=0.0, polarization="p")
    reflection_s = stack_response_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=0.0,
        polarization="s",
        mode="reflection",
    )
    reflection_p = stack_response_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=0.0,
        polarization="p",
        mode="reflection",
    )

    assert np.max(np.abs(transmission_s - transmission_p)) < 1e-12
    assert np.max(np.abs(reflection_s - reflection_p)) < 1e-12


def test_p_polarized_single_interface_reflection_vanishes_at_brewster_angle():
    n_in = 1.0 + 0.0j
    n_out = 1.5 + 0.0j
    brewster_angle_deg = np.rad2deg(np.arctan2(np.real(n_out), np.real(n_in)))
    cos_in = np.cos(np.deg2rad(brewster_angle_deg))
    sin_out = np.sin(np.deg2rad(brewster_angle_deg)) * np.real(n_in) / np.real(n_out)
    cos_out = np.sqrt(1.0 - sin_out**2)
    reflection = fresnel_r_oblique(n_in, n_out, cos_in, cos_out, polarization="p")
    assert abs(reflection) < 1e-12


@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("mode", ["transmission", "reflection"])
def test_oblique_single_layer_matches_closed_form_series(polarization, mode):
    omega = 2.0 * np.pi * np.linspace(0.15, 2.2, 80) * 1e12
    n_in = 1.0 + 0.0j
    n_layer = 2.2 + 0.05j
    n_out = 1.45 + 0.0j
    thickness_m = 135.0e-6
    numeric = single_layer_transfer(
        omega,
        n_in,
        n_layer,
        n_out,
        thickness_m,
        max_internal_reflections=None,
        angle_deg=37.0,
        polarization=polarization,
        mode=mode,
    )
    manual = _manual_single_layer_response(
        omega,
        n_in=n_in,
        n_layer=n_layer,
        n_out=n_out,
        thickness_m=thickness_m,
        angle_deg=37.0,
        polarization=polarization,
        mode=mode,
    )
    assert np.max(np.abs(numeric - manual)) < 1e-12


def test_reference_standard_normalization_is_identity_for_matching_stack(tmp_path):
    reference = _reference_result(tmp_path)
    stack = {
        "n_in": 1.0,
        "n_out": 1.0,
        "layers": [
            {
                "name": "film",
                "thickness_um": 110.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 2.3, "k": 0.03}},
            }
        ],
    }
    simulation = simulate_sample_from_reference(
        reference,
        stack,
        max_internal_reflections=None,
        measurement=Measurement(
            mode="transmission",
            angle_deg=32.0,
            polarization="p",
            reference_standard=ReferenceStandard(kind="stack", stack=stack),
        ),
    )
    assert np.max(np.abs(simulation["transfer_function"] - 1.0)) < 1e-12
    assert np.max(np.abs(simulation["sample_trace"] - reference.trace.trace)) < 1e-10


def test_mixed_polarization_collapses_to_pure_limits_for_isotropic_stack():
    omega = 2.0 * np.pi * np.linspace(0.15, 2.0, 64) * 1e12
    stack = {
        "n_in": 1.0,
        "n_out": 1.4,
        "layers": [
            {
                "name": "film",
                "thickness_um": 90.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 2.1, "k": 0.03}},
            }
        ],
    }
    pure_s = stack_response_function(
        omega,
        stack,
        angle_deg=37.0,
        polarization="s",
        mode="reflection",
    )
    pure_p = stack_response_function(
        omega,
        stack,
        angle_deg=37.0,
        polarization="p",
        mode="reflection",
    )
    mixed_s = stack_response_function(
        omega,
        stack,
        angle_deg=37.0,
        polarization="mixed",
        polarization_mix=0.0,
        mode="reflection",
    )
    mixed_p = stack_response_function(
        omega,
        stack,
        angle_deg=37.0,
        polarization="mixed",
        polarization_mix=1.0,
        mode="reflection",
    )
    assert np.max(np.abs(mixed_s - pure_s)) < 1e-12
    assert np.max(np.abs(mixed_p - pure_p)) < 1e-12


def test_layer_free_stack_acts_as_direct_interface():
    omega = 2.0 * np.pi * np.linspace(0.2, 2.0, 48) * 1e12
    stack = {"n_in": 1.0, "n_out": 1.6, "layers": []}
    transmission = stack_response_function(
        omega,
        stack,
        angle_deg=25.0,
        polarization="s",
        mode="transmission",
    )
    reflection = stack_response_function(
        omega,
        stack,
        angle_deg=25.0,
        polarization="s",
        mode="reflection",
    )
    cos_in = np.cos(np.deg2rad(25.0))
    sin_out = np.sin(np.deg2rad(25.0)) / 1.6
    cos_out = np.sqrt(1.0 - sin_out**2)
    expected_t = _manual_fresnel_t(1.0 + 0.0j, 1.6 + 0.0j, cos_in, cos_out, "s")
    expected_r = _manual_fresnel_r(1.0 + 0.0j, 1.6 + 0.0j, cos_in, cos_out, "s")
    assert np.max(np.abs(transmission - expected_t)) < 1e-12
    assert np.max(np.abs(reflection - expected_r)) < 1e-12


def test_reflection_requires_explicit_reference_standard(tmp_path):
    reference = _reference_result(tmp_path)
    stack = {
        "n_in": 1.0,
        "n_out": 1.0,
        "layers": [
            {
                "name": "film",
                "thickness_um": 80.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 2.0, "k": 0.0}},
            }
        ],
    }
    with pytest.raises(ValueError, match="explicit reference_standard"):
        simulate_sample_from_reference(
            reference,
            stack,
            measurement=Measurement(mode="reflection", angle_deg=30.0, polarization="s"),
        )
