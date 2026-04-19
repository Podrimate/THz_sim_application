from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from thzsim2.core import fft_t_to_w, make_pulse, make_time_grid, zero_pad_trace
from thzsim2.io.manifests import build_reference_manifest, build_run_manifest, write_json
from thzsim2.io.run_folders import create_reference_run_folders
from thzsim2.io.summaries import write_reference_summary_csv, write_reference_summary_txt
from thzsim2.io.trace_csv import read_trace_csv, write_trace_csv
from thzsim2.models import ReferenceResult, ReferenceSummary, SpectrumData, TraceData


def load_reference_csv(path, *, time_column=None, signal_column=None):
    """Load a reference trace from the standard schema or a common measured-data CSV."""
    return read_trace_csv(
        path,
        time_column=time_column,
        signal_column=signal_column,
        resample="auto",
    )


def generate_reference_pulse(
    *,
    model,
    sample_count,
    dt_ps,
    time_center_ps,
    pulse_center_ps,
    tau_ps,
    f0_thz,
    amp=1.0,
    phi_rad=0.0,
    pad_factor=1,
):
    """Generate a reference pulse using notebook-facing public units."""
    sample_count = int(sample_count)
    pad_factor = int(pad_factor)
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")
    if sample_count % pad_factor != 0:
        raise ValueError("sample_count must be divisible by pad_factor")

    raw_count = sample_count // pad_factor
    dt_s = float(dt_ps) * 1e-12
    time_center_s = float(time_center_ps) * 1e-12
    pulse_center_s = float(pulse_center_ps) * 1e-12
    tau_s = float(tau_ps) * 1e-12
    f0_hz = float(f0_thz) * 1e12

    pulse_spec = {
        "model": model,
        "params": {
            "amp": float(amp),
            "t0": pulse_center_s,
            "tau": tau_s,
            "f0": f0_hz,
            "phi": float(phi_rad),
        },
    }

    t_raw = make_time_grid(raw_count, dt_s, time_center_s)
    trace_raw = np.asarray(make_pulse(t_raw, pulse_spec), dtype=np.float64)
    trace = zero_pad_trace(trace_raw, pad_factor=pad_factor)
    time_s = t_raw[0] + np.arange(trace.size, dtype=np.float64) * dt_s

    if not (float(time_s[0]) <= pulse_center_s <= float(time_s[-1])):
        raise ValueError("pulse_center_ps must lie inside the generated time window")
    if float(np.max(np.abs(trace))) <= max(1e-18, abs(float(amp)) * 1e-6):
        raise ValueError(
            "generated pulse amplitude is effectively zero on the chosen time grid; "
            "check time_center_ps, pulse_center_ps, dt_ps, and sample_count"
        )

    return TraceData(
        time_ps=time_s * 1e12,
        trace=trace,
        source_kind="generated",
        pad_factor=pad_factor,
        metadata={
            "pulse": {
                "model": model,
                "amp": float(amp),
                "pulse_center_ps": float(pulse_center_ps),
                "tau_ps": float(tau_ps),
                "f0_thz": float(f0_thz),
                "phi_rad": float(phi_rad),
            },
            "requested_grid": {
                "sample_count": sample_count,
                "raw_sample_count": raw_count,
                "dt_ps": float(dt_ps),
                "time_center_ps": float(time_center_ps),
            },
        },
    )


def _normalize_reference_input(reference_input):
    if isinstance(reference_input, TraceData):
        return reference_input
    if isinstance(reference_input, (str, Path)):
        return load_reference_csv(reference_input)
    raise TypeError("reference_input must be a TraceData instance or a path to a trace CSV")


def _normalize_noise(noise):
    if noise is None:
        return None
    if not isinstance(noise, dict):
        raise TypeError("noise must be a dictionary or None")
    model = noise.get("model", "white_gaussian")
    if model != "white_gaussian":
        raise ValueError("only white_gaussian noise is supported in phase 1")
    if "sigma" not in noise:
        raise KeyError("noise must contain 'sigma'")
    sigma = float(noise["sigma"])
    if sigma < 0.0:
        raise ValueError("noise sigma must be nonnegative")
    seed = noise.get("seed")
    return {"model": model, "sigma": sigma, "seed": None if seed is None else int(seed)}


def _apply_noise(trace_data: TraceData, noise):
    noise_spec = _normalize_noise(noise)
    if noise_spec is None or noise_spec["sigma"] == 0.0:
        return trace_data
    rng = np.random.default_rng(noise_spec["seed"])
    noisy_trace = np.asarray(trace_data.trace, dtype=np.float64) + rng.normal(
        loc=0.0,
        scale=noise_spec["sigma"],
        size=trace_data.trace.shape,
    )
    return trace_data.with_trace(
        noisy_trace,
        metadata_updates={"applied_noise": noise_spec},
    )


def _compute_positive_spectrum(trace_data: TraceData):
    dt_s = trace_data.dt_ps * 1e-12
    t0_s = trace_data.time_ps[0] * 1e-12
    omega, spectrum = fft_t_to_w(trace_data.trace, dt=dt_s, t0=t0_s)
    freq_thz = omega / (2.0 * np.pi * 1e12)
    mask = freq_thz >= 0.0
    freq_pos = freq_thz[mask]
    spectrum_pos = spectrum[mask]
    order = np.argsort(freq_pos)
    freq_pos = freq_pos[order]
    spectrum_pos = spectrum_pos[order]
    return SpectrumData(
        freq_thz=freq_pos,
        real=np.real(spectrum_pos),
        imag=np.imag(spectrum_pos),
        magnitude=np.abs(spectrum_pos),
        phase_rad=np.angle(spectrum_pos),
    )


def _summarize_reference(trace_data: TraceData, spectrum_data: SpectrumData):
    amplitude_scale = float(np.max(np.abs(trace_data.trace)))
    pulse_center_ps = float(trace_data.time_ps[np.argmax(np.abs(trace_data.trace))])
    freq_weights = spectrum_data.magnitude
    weight_sum = float(np.sum(freq_weights))
    if weight_sum > 0.0:
        spectral_centroid = float(np.sum(spectrum_data.freq_thz * freq_weights) / weight_sum)
    else:
        spectral_centroid = 0.0
    peak_freq = float(spectrum_data.freq_thz[np.argmax(freq_weights)])

    return ReferenceSummary(
        dt_ps=trace_data.dt_ps,
        sample_count=trace_data.sample_count,
        time_min_ps=trace_data.time_min_ps,
        time_max_ps=trace_data.time_max_ps,
        amplitude_scale=amplitude_scale,
        pulse_center_ps=pulse_center_ps,
        freq_min_thz=float(spectrum_data.freq_thz[0]),
        freq_max_thz=float(spectrum_data.freq_thz[-1]),
        peak_freq_thz=peak_freq,
        spectral_centroid_thz=spectral_centroid,
    )


def _write_spectrum_csv(path, spectrum_data: SpectrumData):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("freq_thz", "real", "imag", "magnitude", "phase_rad")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for freq_thz, real, imag, magnitude, phase_rad in zip(
            spectrum_data.freq_thz,
            spectrum_data.real,
            spectrum_data.imag,
            spectrum_data.magnitude,
            spectrum_data.phase_rad,
            strict=True,
        ):
            writer.writerow(
                {
                    "freq_thz": format(float(freq_thz), ".16g"),
                    "real": format(float(real), ".16g"),
                    "imag": format(float(imag), ".16g"),
                    "magnitude": format(float(magnitude), ".16g"),
                    "phase_rad": format(float(phase_rad), ".16g"),
                }
            )


def _save_trace_plot(path, trace_data: TraceData):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.real(trace_data.trace) if np.iscomplexobj(trace_data.trace) else trace_data.trace
    ax.plot(trace_data.time_ps, y, linewidth=1.4)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Trace (a.u.)")
    ax.set_title("Reference Trace")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_spectrum_plot(path, spectrum_data: SpectrumData):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(spectrum_data.freq_thz, spectrum_data.magnitude, linewidth=1.4)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Reference Spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _relative_path(path: Path, root: Path):
    return path.relative_to(root).as_posix()


def prepare_reference(reference_input, *, noise=None, output_root="runs", run_label=None):
    """Prepare, summarize, plot, and export a reference trace into a run folder."""
    trace_data = _normalize_reference_input(reference_input)
    trace_data = _apply_noise(trace_data, noise=noise)
    spectrum_data = _compute_positive_spectrum(trace_data)
    summary = _summarize_reference(trace_data, spectrum_data)
    folders = create_reference_run_folders(output_root=output_root, run_label=run_label)

    trace_csv_path = folders.reference_dir / "reference_trace.csv"
    spectrum_csv_path = folders.reference_dir / "reference_spectrum.csv"
    summary_csv_path = folders.reference_dir / "reference_summary.csv"
    summary_txt_path = folders.reference_dir / "reference_summary.txt"
    reference_manifest_path = folders.reference_dir / "manifest.json"
    trace_plot_path = folders.reference_dir / "reference_trace.png"
    spectrum_plot_path = folders.reference_dir / "reference_spectrum.png"
    run_manifest_path = folders.run_dir / "run_manifest.json"

    write_trace_csv(trace_csv_path, trace_data)
    _write_spectrum_csv(spectrum_csv_path, spectrum_data)
    write_reference_summary_csv(summary_csv_path, summary)
    write_reference_summary_txt(summary_txt_path, summary)
    _save_trace_plot(trace_plot_path, trace_data)
    _save_spectrum_plot(spectrum_plot_path, spectrum_data)

    artifact_paths = {
        "reference_trace_csv": trace_csv_path,
        "reference_spectrum_csv": spectrum_csv_path,
        "reference_summary_csv": summary_csv_path,
        "reference_summary_txt": summary_txt_path,
        "reference_manifest_json": reference_manifest_path,
        "reference_trace_png": trace_plot_path,
        "reference_spectrum_png": spectrum_plot_path,
    }
    relative_files = {
        name: _relative_path(path, folders.run_dir) for name, path in artifact_paths.items()
    }

    manifest = build_reference_manifest(
        run_id=folders.run_id,
        created_at=folders.created_at,
        trace_data=trace_data,
        summary=summary,
        files=relative_files,
    )
    run_manifest = build_run_manifest(
        run_id=folders.run_id,
        created_at=folders.created_at,
        reference_manifest_path=_relative_path(reference_manifest_path, folders.run_dir),
    )
    write_json(reference_manifest_path, manifest)
    write_json(run_manifest_path, run_manifest)

    return ReferenceResult(
        run_id=folders.run_id,
        created_at=folders.created_at,
        run_dir=folders.run_dir,
        reference_dir=folders.reference_dir,
        trace=trace_data,
        spectrum=spectrum_data,
        summary=summary,
        manifest=deepcopy(manifest),
        artifact_paths=artifact_paths,
        run_manifest_path=run_manifest_path,
    )
