from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


repo_root = _repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from thzsim2.core import add_white_gaussian_noise, noise_sigma_from_dynamic_range, simulate_sample_from_reference
from thzsim2.core.fitting import (
    build_single_layer_drude_true_stack,
    drude_gamma_thz_from_tau_ps,
    drude_plasma_freq_thz_from_sigma_tau,
    fit_sample_trace,
)
from thzsim2.models import ConstantNK, Drude, Fit, Layer, Measurement, ReferenceStandard
from thzsim2.workflows.reference import generate_reference_pulse, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


def _save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _make_reference(output_root: Path):
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
    return prepare_reference(reference_input, output_root=output_root, run_label="time-domain-fit-doc")


def _plot_reference_trace(reference, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(reference.trace.time_ps, reference.trace.trace, linewidth=1.6)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Measured / generated reference trace")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_dir / "transmission_reference_trace.png")


def _plot_reference_spectrum(reference, output_dir: Path):
    freq_thz = reference.spectrum.freq_thz
    phase = np.unwrap(reference.spectrum.phase_rad)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.5), sharex=True)
    axes[0].plot(freq_thz, reference.spectrum.magnitude, linewidth=1.5)
    axes[0].set_ylabel(r"$|E_{ref}(\omega)|$")
    axes[0].set_title("Reference spectrum after fft_t_to_w")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freq_thz, phase, linewidth=1.4)
    axes[1].set_xlabel("Frequency (THz)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].grid(True, alpha=0.3)
    _save_figure(fig, output_dir / "transmission_reference_spectrum.png")


def _plot_complex_response(freq_thz, response, path: Path, *, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.5), sharex=True)
    axes[0].plot(freq_thz, np.abs(response), linewidth=1.5)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freq_thz, np.unwrap(np.angle(response)), linewidth=1.4)
    axes[1].set_xlabel("Frequency (THz)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].grid(True, alpha=0.3)
    _save_figure(fig, path)


def _plot_trace_fit(reference, observed_trace, fitted_trace, true_trace, path: Path, *, title: str):
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    ax.plot(reference.trace.time_ps, observed_trace, label="Observed", linewidth=1.4)
    ax.plot(reference.trace.time_ps, fitted_trace, label="Fitted", linewidth=1.4)
    ax.plot(reference.trace.time_ps, true_trace, label="True", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save_figure(fig, path)


def _plot_residual(reference, residual_trace, path: Path):
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(reference.trace.time_ps, residual_trace, linewidth=1.4)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Residual")
    ax.set_title("Time-domain residual trace")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, path)


def _plot_reflection_normalization(freq_thz, sample_response, standard_response, transfer, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True)
    axes[0].plot(freq_thz, np.abs(sample_response), label=r"$|H_{sample}|$", linewidth=1.5)
    axes[0].plot(freq_thz, np.abs(standard_response), label=r"$|H_{ref,std}|$", linewidth=1.5)
    axes[0].plot(freq_thz, np.abs(transfer), label=r"$|H_{sample}/H_{ref,std}|$", linewidth=1.4)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Reflection response normalization")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(freq_thz, np.unwrap(np.angle(transfer)), linewidth=1.4)
    axes[1].set_xlabel("Frequency (THz)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].grid(True, alpha=0.3)
    _save_figure(fig, path)


def _plot_reflection_trace(reference, sample_trace, path: Path):
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(reference.trace.time_ps, reference.trace.trace, label="Reference trace", linewidth=1.2)
    ax.plot(reference.trace.time_ps, sample_trace, label="Predicted reflection trace", linewidth=1.5)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Reflection worked example in the time domain")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _save_figure(fig, path)


def generate_figures(output_dir: Path | None = None):
    output_dir = Path(output_dir) if output_dir is not None else repo_root / "docs" / "time_domain_fit_derivation_figures"
    run_root = repo_root / "docs" / "_time_domain_fit_doc_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    reference = _make_reference(run_root)
    _plot_reference_trace(reference, output_dir)
    _plot_reference_spectrum(reference, output_dir)

    initial_tau_ps = 4.0
    initial_sigma_s_per_m = 0.018
    transmission_sample = build_sample(
        layers=[
            Layer(
                name="drude_film",
                thickness_um=Fit(150.0, abs_min=80.0, abs_max=240.0, label="film_thickness_um"),
                material=Drude(
                    eps_inf=12.0,
                    plasma_freq_thz=Fit(
                        drude_plasma_freq_thz_from_sigma_tau(initial_sigma_s_per_m, initial_tau_ps),
                        abs_min=drude_plasma_freq_thz_from_sigma_tau(5.0e-4, 30.0),
                        abs_max=drude_plasma_freq_thz_from_sigma_tau(0.15, 0.08),
                        label="film_plasma_freq_thz",
                    ),
                    gamma_thz=Fit(
                        drude_gamma_thz_from_tau_ps(initial_tau_ps),
                        abs_min=drude_gamma_thz_from_tau_ps(30.0),
                        abs_max=drude_gamma_thz_from_tau_ps(0.08),
                        label="film_gamma_thz",
                    ),
                ),
            )
        ],
        reference=reference,
        out_dir=reference.run_dir / "doc_transmission_sample",
    )
    true_transmission_stack = build_single_layer_drude_true_stack(
        transmission_sample,
        thickness_um=182.0,
        tau_ps=5.2,
        sigma_s_per_m=0.028,
    )
    transmission_measurement = Measurement(
        mode="transmission",
        angle_deg=30.0,
        polarization="p",
        reference_standard=ReferenceStandard(kind="identity"),
    )
    transmission_simulation = simulate_sample_from_reference(
        reference,
        true_transmission_stack,
        max_internal_reflections=8,
        measurement=transmission_measurement,
    )
    noise_sigma = noise_sigma_from_dynamic_range(transmission_simulation["sample_trace"], 85.0)
    observed_trace = add_white_gaussian_noise(transmission_simulation["sample_trace"], sigma=noise_sigma, seed=123)
    transmission_fit = fit_sample_trace(
        reference=reference,
        observed_trace=observed_trace,
        initial_stack=transmission_sample.resolved_stack,
        fit_parameters=transmission_sample.fit_parameters,
        metric="mse",
        max_internal_reflections=8,
        optimizer={
            "method": "L-BFGS-B",
            "options": {"maxiter": 18},
            "global_options": {"maxiter": 2, "popsize": 5, "seed": 123},
            "fd_rel_step": 1e-4,
        },
        measurement=transmission_measurement,
    )

    _plot_complex_response(
        reference.spectrum.freq_thz,
        transmission_simulation["transfer_function"][: reference.spectrum.freq_thz.size],
        output_dir / "transmission_stack_response.png",
        title="Transmission stack response used in the forward model",
    )
    _plot_trace_fit(
        reference,
        observed_trace,
        transmission_fit["fitted_simulation"]["sample_trace"],
        transmission_simulation["sample_trace"],
        output_dir / "transmission_time_fit.png",
        title="Transmission time-domain fit example",
    )
    _plot_residual(reference, transmission_fit["residual_trace"], output_dir / "transmission_residual.png")

    reference_standard_sample = build_sample(
        layers=[],
        reference=reference,
        n_out=1.52,
        out_dir=reference.run_dir / "doc_reference_standard_sample",
    )
    reflection_sample = build_sample(
        layers=[
            Layer(name="coating", thickness_um=22.0, material=ConstantNK(n=2.2, k=0.0)),
        ],
        reference=reference,
        n_out=1.52,
        out_dir=reference.run_dir / "doc_reflection_sample",
    )
    reflection_measurement = Measurement(
        mode="reflection",
        angle_deg=45.0,
        polarization="s",
        reference_standard=ReferenceStandard(kind="stack", stack=reference_standard_sample),
    )
    reflection_simulation = simulate_sample_from_reference(
        reference,
        reflection_sample.resolved_stack,
        max_internal_reflections=None,
        measurement=reflection_measurement,
    )
    positive = reference.spectrum.freq_thz.size
    sample_response_pos = reflection_simulation["sample_response"][:positive]
    standard_response_pos = reflection_simulation["reference_standard_response"][:positive]
    transfer_pos = reflection_simulation["transfer_function"][:positive]

    _plot_reflection_normalization(
        reference.spectrum.freq_thz,
        sample_response_pos,
        standard_response_pos,
        transfer_pos,
        output_dir / "reflection_normalized_response.png",
    )
    _plot_reflection_trace(reference, reflection_simulation["sample_trace"], output_dir / "reflection_time_trace.png")

    return {
        "output_dir": output_dir,
        "reference_run_dir": reference.run_dir,
        "transmission_fit": transmission_fit,
        "reflection_simulation": reflection_simulation,
    }


if __name__ == "__main__":
    result = generate_figures()
    print(result["output_dir"])
