from __future__ import annotations

import csv
from datetime import datetime
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from thzsim2.core.fft import fft_t_to_w
from thzsim2.core.fitting import (
    build_single_layer_drude_true_stack,
    drude_gamma_thz_from_tau_ps,
    drude_plasma_freq_thz_from_sigma_tau,
    fit_sample_trace,
    sigma_s_per_m_from_drude_plasma_gamma,
    tau_ps_from_drude_gamma_thz,
)
from thzsim2.core.forward import simulate_sample_from_reference
from thzsim2.core.materials import evaluate_material_nk, eps_lorentz
from thzsim2.core.metrics import relative_l2
from thzsim2.core.transfer import C0, stack_transfer_function
from thzsim2.io.manifests import write_json
from thzsim2.models import (
    Drude,
    DrudeLorentz,
    Fit,
    Layer,
    Lorentz,
    LorentzOscillator,
    ReferenceResult,
    ReferenceSummary,
    SpectrumData,
    TraceData,
    ValidationCaseResult,
    ValidationSuiteResult,
)
from thzsim2.workflows.reference import generate_reference_pulse
from thzsim2.workflows.sample_workflow import build_sample


_TEST_DESCRIPTIONS = {
    "drude_roundtrip": "Checks sigma/tau <-> plasma/gamma conversion consistency.",
    "empty_stack_identity": "Checks that an empty stack leaves the reference trace unchanged.",
    "zero_thickness_identity": "Checks that a zero-thickness layer behaves like no sample.",
    "constant_nk_single_layer_analytic": "Compares a single constant-n,k layer against the closed-form slab transfer formula.",
    "oblique_polarization_consistency": "Checks that isotropic stacks split into s/p responses at oblique incidence and collapse back together at normal incidence.",
    "noiseless_drude_recovery": "Checks exact noiseless thickness recovery when the Drude material is fixed to the true model.",
    "internal_reflection_convergence": "Checks convergence toward the infinite internal-reflection sum.",
    "grid_convergence": "Checks that the simulated trace converges as the time grid is refined.",
    "kk_consistency_lorentz": "Checks finite-band Kramers-Kronig consistency for an analytic Lorentz dielectric function.",
    "passive_material_models": "Checks that passive model exports keep k >= 0 for Drude, Lorentz, and Drude-Lorentz.",
    "photonic_crystal_bandgap_convergence": "Checks that a periodic quarter-wave stack develops a stronger stop-band with more periods.",
}


def _settings(mode: str):
    mode = str(mode).lower()
    if mode not in {"fast", "standard"}:
        raise ValueError("mode must be 'fast' or 'standard'")
    if mode == "fast":
        return {
            "reference": {
                "sample_count": 768,
                "dt_ps": 0.025,
                "time_center_ps": 18.0,
                "pulse_center_ps": 18.0,
                "tau_ps": 0.24,
                "f0_thz": 0.9,
            },
            "optimizer": {
                "method": "L-BFGS-B",
                "options": {"maxiter": 22},
                "global_options": {"maxiter": 3, "popsize": 5, "seed": 123},
                "fd_rel_step": 1e-5,
            },
            "grid_validation": {"dts_ps": [0.04, 0.02, 0.01], "window_ps": 20.48},
            "kk_points": 220,
            "pc_points": 260,
        }
    return {
        "reference": {
            "sample_count": 1024,
            "dt_ps": 0.02,
            "time_center_ps": 20.0,
            "pulse_center_ps": 20.0,
            "tau_ps": 0.22,
            "f0_thz": 0.9,
        },
        "optimizer": {
            "method": "L-BFGS-B",
            "options": {"maxiter": 32},
            "global_options": {"maxiter": 4, "popsize": 6, "seed": 123},
            "fd_rel_step": 1e-5,
        },
        "grid_validation": {"dts_ps": [0.04, 0.02, 0.01], "window_ps": 20.48},
        "kk_points": 320,
        "pc_points": 360,
    }


def _slugify(text: str):
    chars = []
    for char in str(text).lower():
        chars.append(char if char.isalnum() else "-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "validation"


def _timestamped_out_dir(output_root, run_label):
    output_root = Path(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"{stamp}__validation__{_slugify(run_label)}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _relative_path(path: Path, root: Path):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_csv_rows(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_csv_value(text):
    if text in {"True", "False"}:
        return text == "True"
    try:
        if text.strip() == "":
            return text
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except Exception:
        return text


def load_validation_summary(summary_csv_path):
    rows = []
    with Path(summary_csv_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _parse_csv_value(value) for key, value in row.items()})
    return rows


def _build_reference_result(trace_data: TraceData, run_dir: Path):
    dt_s = trace_data.dt_ps * 1e-12
    t0_s = trace_data.time_ps[0] * 1e-12
    omega, spectrum = fft_t_to_w(trace_data.trace, dt=dt_s, t0=t0_s)
    freq_thz = omega / (2.0 * np.pi * 1e12)
    mask = freq_thz >= 0.0
    freq_pos = freq_thz[mask]
    spec_pos = spectrum[mask]
    order = np.argsort(freq_pos)
    freq_pos = freq_pos[order]
    spec_pos = spec_pos[order]
    spectrum_data = SpectrumData(
        freq_thz=freq_pos,
        real=np.real(spec_pos),
        imag=np.imag(spec_pos),
        magnitude=np.abs(spec_pos),
        phase_rad=np.angle(spec_pos),
    )
    weights = spectrum_data.magnitude
    weight_sum = float(np.sum(weights))
    spectral_centroid = 0.0 if weight_sum <= 0.0 else float(np.sum(spectrum_data.freq_thz * weights) / weight_sum)
    summary = ReferenceSummary(
        dt_ps=trace_data.dt_ps,
        sample_count=trace_data.sample_count,
        time_min_ps=trace_data.time_min_ps,
        time_max_ps=trace_data.time_max_ps,
        amplitude_scale=float(np.max(np.abs(trace_data.trace))),
        pulse_center_ps=float(trace_data.time_ps[np.argmax(np.abs(trace_data.trace))]),
        freq_min_thz=float(spectrum_data.freq_thz[0]),
        freq_max_thz=float(spectrum_data.freq_thz[-1]),
        peak_freq_thz=float(spectrum_data.freq_thz[np.argmax(spectrum_data.magnitude)]),
        spectral_centroid_thz=spectral_centroid,
    )
    return ReferenceResult(
        run_id=_slugify(run_dir.name),
        created_at=datetime.now().astimezone().isoformat(),
        run_dir=run_dir,
        reference_dir=run_dir,
        trace=trace_data,
        spectrum=spectrum_data,
        summary=summary,
        manifest={},
        artifact_paths={},
        run_manifest_path=run_dir / "run_manifest.json",
    )


def _make_reference(run_dir: Path, settings):
    reference_input = generate_reference_pulse(
        model="sech_carrier",
        sample_count=settings["sample_count"],
        dt_ps=settings["dt_ps"],
        time_center_ps=settings["time_center_ps"],
        pulse_center_ps=settings["pulse_center_ps"],
        tau_ps=settings["tau_ps"],
        f0_thz=settings["f0_thz"],
        amp=1.0,
        phi_rad=0.0,
    )
    return _build_reference_result(reference_input, run_dir)


def _save_xy_plot(path: Path, x, y_series, *, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for series in y_series:
        ax.plot(series["x"] if "x" in series else x, series["y"], label=series["label"], linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any(series.get("label") for series in y_series):
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _resolved_constant_stack(*, thickness_um, n, k=0.0, n_in=1.0, n_out=1.0):
    return {
        "n_in": float(n_in),
        "n_out": float(n_out),
        "layers": [
            {
                "name": "layer",
                "thickness_um": float(thickness_um),
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": float(n), "k": float(k)}},
            }
        ],
    }


def _manual_single_layer_transfer(omega, *, n_in, n_layer, n_out, thickness_m, max_internal_reflections=0):
    t01 = 2.0 * n_in / (n_in + n_layer)
    t12 = 2.0 * n_layer / (n_layer + n_out)
    r10 = (n_layer - n_in) / (n_layer + n_in)
    r12 = (n_layer - n_out) / (n_layer + n_out)
    p = np.exp(1j * omega * n_layer * thickness_m / C0)
    z = r12 * p * r10 * p
    fp = np.ones_like(omega, dtype=np.complex128)
    if max_internal_reflections is None:
        fp = 1.0 / (1.0 - z)
    else:
        term = np.ones_like(omega, dtype=np.complex128)
        for _ in range(int(max_internal_reflections)):
            term = term * z
            fp = fp + term
    return t01 * p * t12 * fp


def _make_case_result(*, test_name, passed, score_name, score_value, tolerance, notes, details, plot_path):
    return ValidationCaseResult(
        test_name=test_name,
        description=_TEST_DESCRIPTIONS[test_name],
        passed=bool(passed),
        score_name=score_name,
        score_value=float(score_value),
        tolerance=float(tolerance),
        notes=str(notes),
        details=dict(details),
        plot_path=plot_path,
    )


def _validation_summary_row(case_result: ValidationCaseResult, root: Path):
    return {
        "test_name": case_result.test_name,
        "description": case_result.description,
        "passed": case_result.passed,
        "score_name": case_result.score_name,
        "score_value": case_result.score_value,
        "tolerance": case_result.tolerance,
        "notes": case_result.notes,
        "plot_path": "" if case_result.plot_path is None else _relative_path(case_result.plot_path, root),
    }


def _validate_drude_roundtrip(*, out_dir, settings):
    tau_values = np.array([0.1, 0.25, 0.5, 1.0, 5.0, 10.0, 20.0], dtype=np.float64)
    sigma_values = np.array([5e-4, 1e-3, 0.01, 0.05, 0.2], dtype=np.float64)
    tau_errors = []
    sigma_errors = []
    for tau_ps in tau_values:
        gamma_thz = drude_gamma_thz_from_tau_ps(tau_ps)
        tau_back = tau_ps_from_drude_gamma_thz(gamma_thz)
        tau_errors.append(abs(tau_back - tau_ps) / tau_ps)
    for tau_ps in tau_values:
        for sigma in sigma_values:
            plasma_thz = drude_plasma_freq_thz_from_sigma_tau(sigma, tau_ps)
            gamma_thz = drude_gamma_thz_from_tau_ps(tau_ps)
            sigma_back = sigma_s_per_m_from_drude_plasma_gamma(plasma_thz, gamma_thz)
            sigma_errors.append(abs(sigma_back - sigma) / sigma)
    plot_path = out_dir / "drude_roundtrip.png"
    _save_xy_plot(
        plot_path,
        tau_values,
        [
            {"y": tau_errors, "label": "tau roundtrip rel. error"},
            {"y": np.full(tau_values.shape, max(sigma_errors)), "label": "max sigma roundtrip rel. error"},
        ],
        xlabel="tau (ps)",
        ylabel="Relative error",
        title="Drude Parameter Roundtrip",
    )
    score = max(max(tau_errors), max(sigma_errors))
    return _make_case_result(
        test_name="drude_roundtrip",
        passed=score < 1e-12,
        score_name="max_relative_error",
        score_value=score,
        tolerance=1e-12,
        notes="Drude parameter conversions should be numerically reversible.",
        details={"max_tau_rel_error": max(tau_errors), "max_sigma_rel_error": max(sigma_errors)},
        plot_path=plot_path,
    )


def _validate_empty_stack_identity(*, out_dir, reference):
    resolved_stack = {"n_in": 1.0, "n_out": 1.0, "layers": []}
    simulated = simulate_sample_from_reference(reference, resolved_stack, max_internal_reflections=0)
    diff = simulated["sample_trace"] - reference.trace.trace
    score = float(np.max(np.abs(diff)))
    plot_path = out_dir / "empty_stack_identity.png"
    _save_xy_plot(
        plot_path,
        reference.trace.time_ps,
        [
            {"y": reference.trace.trace, "label": "reference"},
            {"y": simulated["sample_trace"], "label": "sample"},
        ],
        xlabel="Time (ps)",
        ylabel="Amplitude",
        title="Empty Stack Identity",
    )
    return _make_case_result(
        test_name="empty_stack_identity",
        passed=score < 1e-10,
        score_name="max_abs_trace_error",
        score_value=score,
        tolerance=1e-10,
        notes="An empty stack should leave the trace unchanged.",
        details={},
        plot_path=plot_path,
    )


def _validate_zero_thickness_identity(*, out_dir, reference):
    resolved_stack = _resolved_constant_stack(thickness_um=0.0, n=2.5, k=0.02)
    simulated = simulate_sample_from_reference(reference, resolved_stack, max_internal_reflections=8)
    score = float(np.max(np.abs(simulated["sample_trace"] - reference.trace.trace)))
    plot_path = out_dir / "zero_thickness_identity.png"
    _save_xy_plot(
        plot_path,
        reference.trace.time_ps,
        [
            {"y": reference.trace.trace, "label": "reference"},
            {"y": simulated["sample_trace"], "label": "zero-thickness sample"},
        ],
        xlabel="Time (ps)",
        ylabel="Amplitude",
        title="Zero-Thickness Identity",
    )
    return _make_case_result(
        test_name="zero_thickness_identity",
        passed=score < 1e-10,
        score_name="max_abs_trace_error",
        score_value=score,
        tolerance=1e-10,
        notes="A zero-thickness layer should act like no sample.",
        details={},
        plot_path=plot_path,
    )


def _validate_constant_nk_single_layer_analytic(*, out_dir, reference):
    thickness_um = 120.0
    n_layer = 2.2 + 0.03j
    thickness_m = thickness_um * 1e-6
    omega = np.linspace(2.0 * np.pi * 0.1e12, 2.0 * np.pi * 2.5e12, 220)
    resolved_stack = _resolved_constant_stack(thickness_um=thickness_um, n=n_layer.real, k=n_layer.imag)
    transfer_numeric = stack_transfer_function(omega, resolved_stack, max_internal_reflections=None)
    transfer_manual = _manual_single_layer_transfer(
        omega,
        n_in=1.0 + 0.0j,
        n_layer=n_layer,
        n_out=1.0 + 0.0j,
        thickness_m=thickness_m,
        max_internal_reflections=None,
    )
    score = float(np.max(np.abs(transfer_numeric - transfer_manual)))
    plot_path = out_dir / "constant_nk_single_layer_analytic.png"
    _save_xy_plot(
        plot_path,
        omega / (2.0 * np.pi * 1e12),
        [
            {"y": np.abs(transfer_numeric), "label": "|T| numeric"},
            {"y": np.abs(transfer_manual), "label": "|T| analytic"},
        ],
        xlabel="Frequency (THz)",
        ylabel="Transmission magnitude",
        title="Single-Layer Constant n,k Analytic Check",
    )
    return _make_case_result(
        test_name="constant_nk_single_layer_analytic",
        passed=score < 1e-12,
        score_name="max_abs_transfer_error",
        score_value=score,
        tolerance=1e-12,
        notes="Closed-form and implemented single-layer transfer should match.",
        details={},
        plot_path=plot_path,
    )


def _validate_oblique_polarization_consistency(*, out_dir):
    freq_thz = np.linspace(0.2, 2.8, 260)
    omega = 2.0 * np.pi * freq_thz * 1e12
    stack = {
        "n_in": 1.0,
        "n_out": 1.45,
        "layers": [
            {
                "name": "film",
                "thickness_um": 140.0,
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": 2.25, "k": 0.04}},
            }
        ],
    }

    transmission_normal_s = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=0.0,
        polarization="s",
        mode="transmission",
    )
    transmission_normal_p = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=0.0,
        polarization="p",
        mode="transmission",
    )
    reflection_normal_s = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=0.0,
        polarization="s",
        mode="reflection",
    )
    reflection_normal_p = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=0.0,
        polarization="p",
        mode="reflection",
    )

    transmission_oblique_s = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=45.0,
        polarization="s",
        mode="transmission",
    )
    transmission_oblique_p = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=45.0,
        polarization="p",
        mode="transmission",
    )
    reflection_oblique_s = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=45.0,
        polarization="s",
        mode="reflection",
    )
    reflection_oblique_p = stack_transfer_function(
        omega,
        stack,
        max_internal_reflections=None,
        angle_deg=45.0,
        polarization="p",
        mode="reflection",
    )

    normal_collapse = float(
        max(
            np.max(np.abs(transmission_normal_s - transmission_normal_p)),
            np.max(np.abs(reflection_normal_s - reflection_normal_p)),
        )
    )
    oblique_split = float(
        max(
            np.max(np.abs(transmission_oblique_s - transmission_oblique_p)),
            np.max(np.abs(reflection_oblique_s - reflection_oblique_p)),
        )
    )

    plot_path = out_dir / "oblique_polarization_consistency.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(freq_thz, np.abs(transmission_oblique_s), label=r"$|T_s|$ at 45 deg", linewidth=1.5)
    axes[0].plot(freq_thz, np.abs(transmission_oblique_p), label=r"$|T_p|$ at 45 deg", linewidth=1.5)
    axes[0].plot(freq_thz, np.abs(reflection_oblique_s), label=r"$|R_s|$ at 45 deg", linewidth=1.4)
    axes[0].plot(freq_thz, np.abs(reflection_oblique_p), label=r"$|R_p|$ at 45 deg", linewidth=1.4)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Oblique-incidence polarization response of an isotropic film")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(freq_thz, np.abs(transmission_normal_s - transmission_normal_p), label=r"$|T_s-T_p|$ at 0 deg")
    axes[1].plot(freq_thz, np.abs(reflection_normal_s - reflection_normal_p), label=r"$|R_s-R_p|$ at 0 deg")
    axes[1].plot(freq_thz, np.abs(transmission_oblique_s - transmission_oblique_p), label=r"$|T_s-T_p|$ at 45 deg")
    axes[1].plot(freq_thz, np.abs(reflection_oblique_s - reflection_oblique_p), label=r"$|R_s-R_p|$ at 45 deg")
    axes[1].set_xlabel("Frequency (THz)")
    axes[1].set_ylabel("Polarization split")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    return _make_case_result(
        test_name="oblique_polarization_consistency",
        passed=normal_collapse < 1e-12 and oblique_split > 1e-3,
        score_name="normal_incidence_max_sp_difference",
        score_value=normal_collapse,
        tolerance=1e-12,
        notes="Isotropic stacks must be polarization-degenerate at normal incidence, while oblique incidence should split the s and p responses.",
        details={
            "normal_incidence_max_difference": normal_collapse,
            "oblique_incidence_max_difference": oblique_split,
        },
        plot_path=plot_path,
    )


def _validate_noiseless_drude_recovery(*, out_dir, reference, settings):
    true_tau_ps = 5.0
    true_sigma_s_per_m = 0.02
    sample_out_dir = out_dir / "noiseless_drude_recovery_sample"
    sample_result = build_sample(
        layers=[
            Layer(
                name="film",
                thickness_um=Fit(150.0, abs_min=40.0, abs_max=320.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=drude_plasma_freq_thz_from_sigma_tau(true_sigma_s_per_m, true_tau_ps),
                    gamma_thz=drude_gamma_thz_from_tau_ps(true_tau_ps),
                ),
            )
        ],
        reference=reference,
        out_dir=sample_out_dir,
    )
    true_stack = build_single_layer_drude_true_stack(
        sample_result,
        thickness_um=180.0,
        tau_ps=true_tau_ps,
        sigma_s_per_m=true_sigma_s_per_m,
    )
    observed = simulate_sample_from_reference(reference, true_stack, max_internal_reflections=4)
    fit = fit_sample_trace(
        reference=reference,
        observed_trace=observed["sample_trace"],
        initial_stack=sample_result.resolved_stack,
        fit_parameters=sample_result.fit_parameters,
        metric="mse",
        max_internal_reflections=4,
        optimizer=settings["optimizer"],
    )
    recovered = fit["fitted_stack"]["layers"][0]
    rel_errors = {"thickness_um": abs(recovered["thickness_um"] - 180.0) / 180.0}
    score = rel_errors["thickness_um"]
    plot_path = out_dir / "noiseless_drude_recovery.png"
    _save_xy_plot(
        plot_path,
        reference.trace.time_ps,
        [
            {"y": observed["sample_trace"], "label": "true sample"},
            {"y": fit["fitted_simulation"]["sample_trace"], "label": "fitted sample"},
        ],
        xlabel="Time (ps)",
        ylabel="Amplitude",
        title="Noiseless Drude Recovery",
    )
    return _make_case_result(
        test_name="noiseless_drude_recovery",
        passed=score < 2e-4,
        score_name="thickness_relative_error",
        score_value=score,
        tolerance=2e-4,
        notes="This benchmark fixes the Drude material to the true model and checks that a noiseless trace recovers the true thickness.",
        details=rel_errors,
        plot_path=plot_path,
    )


def _validate_internal_reflection_convergence(*, out_dir):
    omega = 2.0 * np.pi * np.linspace(0.05, 2.5, 240) * 1e12
    stack = _resolved_constant_stack(thickness_um=350.0, n=3.2, k=0.02)
    transfer_inf = stack_transfer_function(omega, stack, max_internal_reflections=None)
    counts = [0, 2, 4, 8, 16]
    errors = []
    for count in counts:
        transfer = stack_transfer_function(omega, stack, max_internal_reflections=count)
        errors.append(relative_l2(transfer, transfer_inf))
    monotonic = all(errors[i + 1] <= errors[i] for i in range(len(errors) - 1))
    score = errors[-1]
    plot_path = out_dir / "internal_reflection_convergence.png"
    _save_xy_plot(
        plot_path,
        counts,
        [{"y": errors, "label": "relative L2 to infinite sum"}],
        xlabel="Max internal reflections",
        ylabel="Relative L2 error",
        title="Internal-Reflection Convergence",
    )
    return _make_case_result(
        test_name="internal_reflection_convergence",
        passed=score < 2e-3 and monotonic,
        score_name="relative_l2_to_infinite_sum",
        score_value=score,
        tolerance=2e-3,
        notes="Increasing the reflection count should approach the infinite geometric sum.",
        details={"errors": errors, "monotonic": monotonic},
        plot_path=plot_path,
    )


def _interp_trace(time_target, time_source, values):
    return np.interp(np.asarray(time_target, dtype=np.float64), np.asarray(time_source, dtype=np.float64), np.asarray(values, dtype=np.float64))


def _validate_grid_convergence(*, out_dir, settings):
    dts_ps = list(settings["grid_validation"]["dts_ps"])
    window_ps = float(settings["grid_validation"]["window_ps"])
    traces = []
    errors = []
    sample = _resolved_constant_stack(thickness_um=180.0, n=2.6, k=0.015)
    for dt_ps in dts_ps:
        sample_count = int(round(window_ps / dt_ps))
        ref = _build_reference_result(
            generate_reference_pulse(
                model="sech_carrier",
                sample_count=sample_count,
                dt_ps=dt_ps,
                time_center_ps=20.0,
                pulse_center_ps=20.0,
                tau_ps=0.22,
                f0_thz=0.9,
                amp=1.0,
                phi_rad=0.0,
            ),
            out_dir / f"grid_{_slugify(str(dt_ps))}",
        )
        traces.append((dt_ps, ref, simulate_sample_from_reference(ref, sample, max_internal_reflections=4)["sample_trace"]))
    fine_dt, fine_ref, fine_trace = traces[-1]
    for dt_ps, ref, trace in traces[:-1]:
        aligned = _interp_trace(fine_ref.trace.time_ps, ref.trace.time_ps, trace)
        errors.append(relative_l2(aligned, fine_trace))
    score = errors[-1]
    improving = errors[-1] < errors[0]
    plot_path = out_dir / "grid_convergence.png"
    _save_xy_plot(
        plot_path,
        dts_ps[:-1],
        [{"y": errors, "label": "relative L2 vs finest grid"}],
        xlabel="dt (ps)",
        ylabel="Relative L2 error",
        title="Grid Convergence",
    )
    return _make_case_result(
        test_name="grid_convergence",
        passed=improving and score < 1.5e-2,
        score_name="medium_grid_relative_l2_to_finest",
        score_value=score,
        tolerance=1.5e-2,
        notes="The medium grid should be closer to the finest grid than the coarse grid is.",
        details={"errors": errors, "fine_dt_ps": fine_dt},
        plot_path=plot_path,
    )


def _validate_kk_consistency_lorentz(*, out_dir, settings):
    freq_pos_thz = np.linspace(0.01, 20.0, int(settings["kk_points"]) * 8)
    eps_pos = eps_lorentz(freq_pos_thz, eps_inf=3.0, delta_eps=2.0, resonance_thz=1.0, gamma_thz=0.08)
    freq_thz = np.concatenate([-freq_pos_thz[:0:-1], freq_pos_thz])
    eps = np.concatenate([np.conj(eps_pos[:0:-1]), eps_pos])
    chi = eps - 3.0
    reconstructed_imag = np.imag(hilbert(np.real(chi)))
    central = slice(freq_thz.size // 10, -freq_thz.size // 10)
    actual = np.imag(chi)[central]
    reconstructed = reconstructed_imag[central]
    score = relative_l2(reconstructed, actual)
    plot_path = out_dir / "kk_consistency_lorentz.png"
    _save_xy_plot(
        plot_path,
        freq_thz[central],
        [
            {"y": actual, "label": "Im(chi) exact"},
            {"y": reconstructed, "label": "Hilbert reconstruction"},
        ],
        xlabel="Frequency (THz)",
        ylabel="Im(chi)",
        title="Lorentz Kramers-Kronig Consistency",
    )
    return _make_case_result(
        test_name="kk_consistency_lorentz",
        passed=score < 1e-2,
        score_name="relative_l2_imag_chi_reconstruction",
        score_value=score,
        tolerance=1e-2,
        notes="This finite-band analytic Lorentz benchmark checks the even/odd Hilbert-pair structure expected from a causal response.",
        details={},
        plot_path=plot_path,
    )


def _validate_passive_material_models(*, out_dir):
    freq_thz = np.linspace(0.05, 4.0, 240)
    models = {
        "Drude": evaluate_material_nk(freq_thz, Drude(eps_inf=12.0, plasma_freq_thz=1.2, gamma_thz=0.2)),
        "Lorentz": evaluate_material_nk(freq_thz, Lorentz(eps_inf=3.0, delta_eps=2.0, resonance_thz=1.0, gamma_thz=0.08)),
        "DrudeLorentz": evaluate_material_nk(
            freq_thz,
            DrudeLorentz(
                eps_inf=4.0,
                plasma_freq_thz=0.7,
                gamma_thz=0.15,
                oscillators=(LorentzOscillator(delta_eps=1.0, resonance_thz=1.4, gamma_thz=0.07),),
            ),
        ),
    }
    min_k = min(float(np.min(np.imag(nk))) for nk in models.values())
    plot_path = out_dir / "passive_material_models.png"
    _save_xy_plot(
        plot_path,
        freq_thz,
        [{"y": np.imag(nk), "label": name} for name, nk in models.items()],
        xlabel="Frequency (THz)",
        ylabel="k",
        title="Passive Material k >= 0 Check",
    )
    return _make_case_result(
        test_name="passive_material_models",
        passed=min_k >= -1e-12,
        score_name="min_k_value",
        score_value=abs(min(0.0, min_k)),
        tolerance=1e-12,
        notes="Passive materials should export nonnegative extinction coefficients.",
        details={"min_k": min_k},
        plot_path=plot_path,
    )


def _characteristic_matrix(n, thickness_m, omega):
    delta = omega * n * thickness_m / C0
    return np.array(
        [
            [np.cos(delta), 1j * np.sin(delta) / n],
            [1j * n * np.sin(delta), np.cos(delta)],
        ],
        dtype=np.complex128,
    )


def _shade_boolean_regions(ax, x, mask, *, color="0.85", alpha=0.5, label=None):
    x = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 1 or mask.ndim != 1 or x.size != mask.size:
        raise ValueError("x and mask must be 1D arrays with matching shapes")
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    first = True
    for start, end in zip(starts, ends, strict=True):
        ax.axvspan(
            float(x[start]),
            float(x[end]),
            color=color,
            alpha=alpha,
            label=label if first else None,
        )
        first = False


def _photonic_stack(periods, n_a, n_b, d_a_um, d_b_um):
    layers = []
    for index in range(periods):
        layers.append(
            {
                "name": f"A{index}",
                "thickness_um": float(d_a_um),
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": float(n_a), "k": 0.0}},
            }
        )
        layers.append(
            {
                "name": f"B{index}",
                "thickness_um": float(d_b_um),
                "material_kind": "ConstantNK",
                "material": {"kind": "ConstantNK", "parameters": {"n": float(n_b), "k": 0.0}},
            }
        )
    return {"n_in": 1.0, "n_out": 1.0, "layers": layers}


def _validate_photonic_crystal_bandgap_convergence(*, out_dir, settings):
    f0_thz = 1.0
    n_a = 1.5
    n_b = 3.0
    d_a_um = (C0 / (4.0 * f0_thz * 1e12 * n_a)) * 1e6
    d_b_um = (C0 / (4.0 * f0_thz * 1e12 * n_b)) * 1e6
    freq_thz = np.linspace(0.2, 5.0, int(settings["pc_points"]) * 2)
    omega = 2.0 * np.pi * freq_thz * 1e12
    periods_list = [1, 2, 4, 8, 12]
    center_transmissions = []
    spectra = []
    for periods in periods_list:
        transfer = stack_transfer_function(
            omega,
            _photonic_stack(periods, n_a, n_b, d_a_um, d_b_um),
            max_internal_reflections=None,
        )
        power = np.abs(transfer) ** 2
        spectra.append((periods, power))
        center_transmissions.append(float(power[np.argmin(np.abs(freq_thz - f0_thz))]))
    bloch_half_trace = np.empty_like(freq_thz)
    for index, omega_i in enumerate(omega):
        unit_matrix = _characteristic_matrix(n_a, d_a_um * 1e-6, omega_i) @ _characteristic_matrix(
            n_b, d_b_um * 1e-6, omega_i
        )
        bloch_half_trace[index] = abs((unit_matrix[0, 0] + unit_matrix[1, 1]) / 2.0)
    stop_band_mask = bloch_half_trace > 1.0
    first_gap_center_idx = int(np.argmin(np.abs(freq_thz - f0_thz)))
    monotonic = all(center_transmissions[i + 1] <= center_transmissions[i] for i in range(len(center_transmissions) - 1))
    score = center_transmissions[-1]
    plot_path = out_dir / "photonic_crystal_bandgap_convergence.png"
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for periods, power in spectra:
        ax.plot(freq_thz, power, linewidth=1.6, label=f"N={periods}")
    _shade_boolean_regions(ax, freq_thz, stop_band_mask, color="0.8", alpha=0.45, label="Bloch stop-band")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Transmission")
    ax.set_title("Photonic-crystal / Bloch test")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return _make_case_result(
        test_name="photonic_crystal_bandgap_convergence",
        passed=bool(stop_band_mask[first_gap_center_idx]) and monotonic and score < 0.2,
        score_name="center_frequency_transmission_magnitude",
        score_value=score,
        tolerance=0.2,
        notes="At the quarter-wave design frequency the Bloch stop-band should deepen as the period count increases; the plotted quantity is power transmission.",
        details={
            "center_transmissions": center_transmissions,
            "first_gap_center_frequency_thz": float(freq_thz[first_gap_center_idx]),
            "bloch_half_trace_at_first_gap_center": float(bloch_half_trace[first_gap_center_idx]),
        },
        plot_path=plot_path,
    )


_VALIDATORS = {
    "drude_roundtrip": _validate_drude_roundtrip,
    "empty_stack_identity": _validate_empty_stack_identity,
    "zero_thickness_identity": _validate_zero_thickness_identity,
    "constant_nk_single_layer_analytic": _validate_constant_nk_single_layer_analytic,
    "oblique_polarization_consistency": _validate_oblique_polarization_consistency,
    "noiseless_drude_recovery": _validate_noiseless_drude_recovery,
    "internal_reflection_convergence": _validate_internal_reflection_convergence,
    "grid_convergence": _validate_grid_convergence,
    "kk_consistency_lorentz": _validate_kk_consistency_lorentz,
    "passive_material_models": _validate_passive_material_models,
    "photonic_crystal_bandgap_convergence": _validate_photonic_crystal_bandgap_convergence,
}


def plot_validation_summary(validation, *, out_path=None):
    if isinstance(validation, ValidationSuiteResult):
        rows = list(validation.summary_rows)
    elif isinstance(validation, (str, Path)):
        rows = load_validation_summary(validation)
    else:
        rows = list(validation)
    names = [row["test_name"] for row in rows]
    ratios = []
    colors = []
    for row in rows:
        tolerance = max(float(row["tolerance"]), 1e-30)
        ratios.append(max(float(row["score_value"]) / tolerance, 1e-16))
        colors.append("#2E8B57" if bool(row["passed"]) else "#B22222")
    fig_height = max(4.5, 0.55 * len(rows))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = np.arange(len(rows))
    ax.barh(y, ratios, color=colors, alpha=0.85)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, label="pass threshold")
    ax.set_yticks(y, labels=names)
    ax.set_xscale("log")
    ax.set_xlabel("Score / tolerance")
    ax.set_title("THzSim2 Physical Validation Summary")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig, ax


def plot_validation_plot_grid(validation_result: ValidationSuiteResult, *, columns=2, out_path=None):
    if not isinstance(validation_result, ValidationSuiteResult):
        raise TypeError("plot_validation_plot_grid expects a ValidationSuiteResult")
    plot_cases = [case for case in validation_result.case_results if case.plot_path is not None and Path(case.plot_path).exists()]
    if not plot_cases:
        raise ValueError("validation_result does not contain any saved plot paths")
    columns = max(1, int(columns))
    rows = int(math.ceil(len(plot_cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, case in zip(axes, plot_cases, strict=False):
        image = mpimg.imread(case.plot_path)
        ax.imshow(image)
        ax.set_title(case.test_name.replace("_", " "))
        ax.axis("off")
    for ax in axes[len(plot_cases) :]:
        ax.axis("off")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig, axes


def run_validation_suite(*, output_root="notebooks/validation_runs", run_label="physical-validation", tests=None, mode="standard"):
    settings = _settings(mode)
    out_dir = _timestamped_out_dir(output_root, run_label)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    selected_tests = list(_VALIDATORS.keys()) if tests is None else [str(name) for name in tests]
    unknown = [name for name in selected_tests if name not in _VALIDATORS]
    if unknown:
        raise ValueError(f"unknown validation tests requested: {unknown}")

    reference = _make_reference(out_dir / "reference_fixture", settings["reference"])
    case_results = []
    for test_name in selected_tests:
        validator = _VALIDATORS[test_name]
        kwargs = {"out_dir": plots_dir}
        if "settings" in validator.__code__.co_varnames:
            kwargs["settings"] = settings
        if "reference" in validator.__code__.co_varnames:
            kwargs["reference"] = reference
        case_results.append(validator(**kwargs))

    summary_rows = [_validation_summary_row(case, out_dir) for case in case_results]
    summary_csv_path = out_dir / "validation_summary.csv"
    manifest_path = out_dir / "validation_manifest.json"
    summary_plot_path = out_dir / "validation_summary.png"
    plot_grid_path = out_dir / "validation_plot_grid.png"

    _write_csv_rows(summary_csv_path, summary_rows)
    overview_fig, _ = plot_validation_summary(summary_rows, out_path=summary_plot_path)
    plt.close(overview_fig)
    temp_result = ValidationSuiteResult(
        out_dir=out_dir,
        summary_csv_path=summary_csv_path,
        manifest_path=manifest_path,
        summary_rows=summary_rows,
        case_results=case_results,
    )
    grid_fig, _ = plot_validation_plot_grid(temp_result, out_path=plot_grid_path)
    plt.close(grid_fig)

    manifest = {
        "schema_name": "thzsim2.validation_manifest",
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "tests": selected_tests,
        "summary_csv": summary_csv_path.name,
        "summary_plot": summary_plot_path.name,
        "plot_grid": plot_grid_path.name,
        "cases": [
            {
                "test_name": case.test_name,
                "description": case.description,
                "passed": case.passed,
                "score_name": case.score_name,
                "score_value": case.score_value,
                "tolerance": case.tolerance,
                "notes": case.notes,
                "details": case.details,
                "plot_path": None if case.plot_path is None else _relative_path(case.plot_path, out_dir),
            }
            for case in case_results
        ],
    }
    write_json(manifest_path, manifest)

    artifact_paths = {
        "validation_summary_csv": summary_csv_path,
        "validation_manifest_json": manifest_path,
        "validation_summary_png": summary_plot_path,
        "validation_plot_grid_png": plot_grid_path,
    }
    for case in case_results:
        if case.plot_path is not None:
            artifact_paths[f"{case.test_name}_png"] = case.plot_path

    return ValidationSuiteResult(
        out_dir=out_dir,
        summary_csv_path=summary_csv_path,
        manifest_path=manifest_path,
        summary_rows=summary_rows,
        case_results=case_results,
        artifact_paths=artifact_paths,
        manifest=manifest,
    )
