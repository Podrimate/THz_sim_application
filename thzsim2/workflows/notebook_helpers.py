from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from pprint import pformat
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from thzsim2.core import add_white_gaussian_noise, noise_sigma_from_dynamic_range
from thzsim2.core.fft import fft_t_to_w
from thzsim2.io.trace_csv import (
    _infer_signal_column,
    _infer_time_column,
    _split_header_unit,
    _time_scale_to_ps,
    read_trace_csv,
)
from thzsim2.io.run_folders import slugify
from thzsim2.models import Fit, Measurement, TraceData
from thzsim2.workflows.fit_setup import build_fit_setup, write_fit_setup_json
from thzsim2.workflows.sample_workflow import build_sample
from thzsim2.workflows.study_setup import _layers_from_config, build_study_setup, write_study_setup_json
from thzsim2.workflows.study_workflow import (
    _axis_assignments,
    _build_true_stack,
    _normalize_study_config,
    run_study,
)
from thzsim2.core.fitting import build_objective_weights, fit_sample_trace
from thzsim2.core.forward import normalize_measurement, simulate_sample_from_reference


_BUILTIN_MATERIAL_CSV = {
    "si_thz_nk.csv": """freq_thz,n,k
0.10,3.418,0.002
0.25,3.418,0.002
0.50,3.419,0.002
0.75,3.419,0.002
1.00,3.420,0.003
1.25,3.420,0.003
1.50,3.421,0.003
1.75,3.421,0.004
2.00,3.422,0.004
2.25,3.422,0.004
2.50,3.423,0.005
2.75,3.423,0.005
3.00,3.424,0.006
""",
    "sio2_lossy_thz_nk.csv": """freq_thz,n,k
0.10,1.940,0.180
0.25,1.945,0.190
0.50,1.950,0.210
0.75,1.958,0.235
1.00,1.965,0.260
1.25,1.972,0.285
1.50,1.980,0.315
1.75,1.988,0.345
2.00,1.995,0.380
2.25,2.003,0.420
2.50,2.010,0.460
2.75,2.018,0.505
3.00,2.025,0.550
""",
}


def fit_param(initial, minimum, maximum, *, label=None):
    return Fit(float(initial), abs_min=float(minimum), abs_max=float(maximum), label=label)


def sweep_axis(minimum, maximum, count, *, scale="linear"):
    count = int(count)
    if count < 1:
        raise ValueError("count must be at least 1")
    minimum = float(minimum)
    maximum = float(maximum)
    scale = str(scale).strip().lower()
    if scale == "linear":
        return [float(value) for value in np.linspace(minimum, maximum, count)]
    if scale == "log":
        if minimum <= 0.0 or maximum <= 0.0:
            raise ValueError("log sweep axes require positive minimum and maximum")
        return [float(value) for value in np.geomspace(minimum, maximum, count)]
    raise ValueError("scale must be 'linear' or 'log'")


def layers_from_definition(definition):
    return _layers_from_config(list(definition))


def create_run_output_dir(run_name, *, root="notebooks/runs"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{timestamp}__{slugify(run_name)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def enable_inline_plots():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        shell.run_line_magic("matplotlib", "inline")
        return True
    except Exception:
        return False


def resolve_workflow_root(local_folder_name, *, use_google_drive=False, google_drive_subdir="THz_sim_application_outputs"):
    if use_google_drive:
        try:
            from google.colab import drive
        except Exception as exc:
            raise RuntimeError("Google Drive output is only available inside Google Colab.") from exc
        mount_root = Path("/content/drive")
        drive.mount(str(mount_root), force_remount=False)
        root = mount_root / "MyDrive" / str(google_drive_subdir).strip().strip("/\\") / str(local_folder_name).strip()
    else:
        root = Path.cwd() / str(local_folder_name)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def write_python_snapshot(path, **variables):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Auto-generated notebook snapshot", ""]
    for name, value in variables.items():
        lines.append(f"{name} = {pformat(value, sort_dicts=False)}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def ensure_builtin_material_file(name, *, out_dir):
    if name not in _BUILTIN_MATERIAL_CSV:
        raise ValueError(f"unknown builtin material '{name}'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(_BUILTIN_MATERIAL_CSV[name], encoding="utf-8")
    return path.resolve()


def _read_raw_trace_csv(path, *, time_column=None, signal_column=None):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError("trace CSV is missing a header row")
        fieldnames = list(reader.fieldnames)
        time_column = _infer_time_column(fieldnames, time_column)
        signal_column = _infer_signal_column(fieldnames, signal_column)
        scale_to_ps, time_unit = _time_scale_to_ps(time_column)
        _, signal_unit = _split_header_unit(signal_column)

        time_ps = []
        trace = []
        for row in reader:
            time_ps.append(float(row[time_column]) * scale_to_ps)
            trace.append(float(row[signal_column]))

    return {
        "time_ps": np.asarray(time_ps, dtype=np.float64),
        "trace": np.asarray(trace, dtype=np.float64),
        "time_column": time_column,
        "signal_column": signal_column,
        "time_unit": time_unit,
        "signal_unit": signal_unit or "a.u.",
    }


def inspect_trace_input(trace_input, *, time_column=None, signal_column=None):
    if isinstance(trace_input, TraceData):
        trace = trace_input
        peak_index = int(np.argmax(np.abs(trace.trace)))
        return {
            "kind": "trace_data",
            "raw_time_ps": np.asarray(trace.time_ps, dtype=np.float64),
            "raw_trace": np.asarray(trace.trace, dtype=np.float64),
            "prepared_trace": trace,
            "summary": {
                "sample_count": int(trace.sample_count),
                "dt_ps": float(trace.dt_ps),
                "time_min_ps": float(trace.time_min_ps),
                "time_max_ps": float(trace.time_max_ps),
                "peak_time_ps": float(trace.time_ps[peak_index]),
                "peak_value": float(trace.trace[peak_index]),
                "time_column": "time_ps",
                "signal_column": "trace",
                "time_unit": "ps",
                "signal_unit": "a.u.",
                "resampled_to_uniform_grid": False,
            },
        }

    raw = _read_raw_trace_csv(trace_input, time_column=time_column, signal_column=signal_column)
    prepared = read_trace_csv(
        trace_input,
        time_column=raw["time_column"],
        signal_column=raw["signal_column"],
        resample="auto",
    )
    peak_index = int(np.argmax(np.abs(prepared.trace)))
    time_axis = dict(prepared.metadata.get("time_axis", {}))
    return {
        "kind": "csv",
        "raw_time_ps": raw["time_ps"],
        "raw_trace": raw["trace"],
        "prepared_trace": prepared,
        "summary": {
            "sample_count": int(prepared.sample_count),
            "dt_ps": float(prepared.dt_ps),
            "time_min_ps": float(prepared.time_min_ps),
            "time_max_ps": float(prepared.time_max_ps),
            "peak_time_ps": float(prepared.time_ps[peak_index]),
            "peak_value": float(prepared.trace[peak_index]),
            "time_column": raw["time_column"],
            "signal_column": raw["signal_column"],
            "time_unit": raw["time_unit"],
            "signal_unit": raw["signal_unit"],
            "resampled_to_uniform_grid": bool(time_axis.get("resampled_to_uniform_grid", False)),
        },
    }


def trace_spectrum(trace_data: TraceData):
    dt_s = float(trace_data.dt_ps) * 1e-12
    t0_s = float(trace_data.time_ps[0]) * 1e-12
    omega, spectrum = fft_t_to_w(trace_data.trace, dt=dt_s, t0=t0_s)
    freq_thz = omega / (2.0 * np.pi * 1e12)
    positive = freq_thz >= 0.0
    freq_thz = np.asarray(freq_thz[positive], dtype=np.float64)
    spectrum = np.asarray(spectrum[positive], dtype=np.complex128)
    amplitude_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-30))
    phase_rad = np.unwrap(np.angle(spectrum))
    return freq_thz, amplitude_db, phase_rad


def plot_trace_preview(trace_info, *, title_prefix, show_fft=False, freq_limits_thz=None, display=True):
    raw_time_ps = np.asarray(trace_info["raw_time_ps"], dtype=np.float64)
    raw_trace = np.asarray(trace_info["raw_trace"], dtype=np.float64)
    prepared = trace_info["prepared_trace"]
    fig, axes = plt.subplots(2 if show_fft else 1, 2 if show_fft else 1, figsize=(12, 8 if show_fft else 4.5))
    axes = np.atleast_1d(axes).ravel()

    ax = axes[0]
    ax.plot(raw_time_ps, raw_trace, label="raw")
    ax.plot(prepared.time_ps, prepared.trace, label="prepared", linewidth=1.4)
    ax.set_title(f"{title_prefix}: Raw And Prepared Trace")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Signal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show_fft:
        raw_fft_label = "raw"
        try:
            raw_fft_trace = TraceData(time_ps=raw_time_ps, trace=raw_trace, source_kind="raw_preview")
        except ValueError:
            raw_fft_trace = TraceData(
                time_ps=np.asarray(prepared.time_ps, dtype=np.float64),
                trace=np.interp(np.asarray(prepared.time_ps, dtype=np.float64), raw_time_ps, raw_trace),
                source_kind="raw_preview_resampled",
                metadata={"fft_preview_only": True},
            )
            raw_fft_label = "raw (resampled for FFT preview)"
        freq_thz, raw_amp_db, raw_phase = trace_spectrum(raw_fft_trace)
        prep_freq_thz, prep_amp_db, prep_phase = trace_spectrum(prepared)
        amp_ax = axes[1]
        phase_ax = axes[2]
        amp_ax.plot(freq_thz, raw_amp_db, label=raw_fft_label)
        amp_ax.plot(prep_freq_thz, prep_amp_db, label="prepared")
        amp_ax.set_title(f"{title_prefix}: FFT Amplitude")
        amp_ax.set_xlabel("Frequency (THz)")
        amp_ax.set_ylabel("Amplitude (dB)")
        amp_ax.grid(True, alpha=0.3)
        amp_ax.legend()
        phase_ax.plot(freq_thz, raw_phase, label=raw_fft_label)
        phase_ax.plot(prep_freq_thz, prep_phase, label="prepared")
        phase_ax.set_title(f"{title_prefix}: FFT Phase")
        phase_ax.set_xlabel("Frequency (THz)")
        phase_ax.set_ylabel("Phase (rad)")
        phase_ax.grid(True, alpha=0.3)
        phase_ax.legend()
        if freq_limits_thz is not None:
            for axis in (amp_ax, phase_ax):
                axis.set_xlim(float(freq_limits_thz[0]), float(freq_limits_thz[1]))
    fig.tight_layout()
    if display:
        try:
            from IPython.display import display as ipy_display

            ipy_display(fig)
        except Exception:
            pass
    return fig, axes


def plot_objective_weighting(trace_data, weighting, *, title="Objective Weighting Preview", display=True):
    weighting = {} if weighting is None else dict(weighting)
    mode = str(weighting.get("mode", "none")).strip().lower()
    weights = build_objective_weights(
        trace_data.trace,
        mode=mode,
        floor=weighting.get("floor", 0.05),
        power=weighting.get("power", 2.0),
        smooth_window_samples=weighting.get("smooth_window_samples", 41),
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(trace_data.time_ps, trace_data.trace, label="trace")
    axes[0].set_title(title)
    axes[0].set_ylabel("Signal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(trace_data.time_ps, weights, label=f"weights ({mode})", color="tab:red")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Relative weight")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    if display:
        try:
            from IPython.display import display as ipy_display

            ipy_display(fig)
        except Exception:
            plt.show(block=False)
    return weights, fig, axes


def preview_sample_response(
    *,
    reference_result,
    layers,
    out_dir,
    measurement=None,
    n_in=1.0,
    n_out=1.0,
    overlay_imported=True,
    max_internal_reflections=0,
    show_fft=False,
    freq_limits_thz=None,
    display=True,
):
    sample_result = build_sample(
        layers=layers,
        reference=reference_result,
        out_dir=Path(out_dir),
        n_in=n_in,
        n_out=n_out,
        overlay_imported=overlay_imported,
    )
    measurement = normalize_measurement(measurement)
    simulation = simulate_sample_from_reference(
        reference_result,
        sample_result.resolved_stack,
        max_internal_reflections=max_internal_reflections,
        measurement=measurement,
    )
    sample_trace = reference_result.trace.with_trace(simulation["sample_trace"])
    fig, axes = plt.subplots(2 if show_fft else 1, 2 if show_fft else 1, figsize=(12, 8 if show_fft else 4.5))
    axes = np.atleast_1d(axes).ravel()

    time_ax = axes[0]
    time_ax.plot(reference_result.trace.time_ps, reference_result.trace.trace, label="reference")
    time_ax.plot(sample_trace.time_ps, sample_trace.trace, label="sample preview")
    time_ax.set_title("Reference And Simulated Sample Trace")
    time_ax.set_xlabel("Time (ps)")
    time_ax.set_ylabel("Signal")
    time_ax.grid(True, alpha=0.3)
    time_ax.legend()

    if show_fft:
        ref_freq, ref_amp_db, ref_phase = trace_spectrum(reference_result.trace)
        sam_freq, sam_amp_db, sam_phase = trace_spectrum(sample_trace)
        amp_ax = axes[1]
        phase_ax = axes[2]
        amp_ax.plot(ref_freq, ref_amp_db, label="reference")
        amp_ax.plot(sam_freq, sam_amp_db, label="sample preview")
        amp_ax.set_title("Reference And Sample FFT Amplitude")
        amp_ax.set_xlabel("Frequency (THz)")
        amp_ax.set_ylabel("Amplitude (dB)")
        amp_ax.grid(True, alpha=0.3)
        amp_ax.legend()
        phase_ax.plot(ref_freq, ref_phase, label="reference")
        phase_ax.plot(sam_freq, sam_phase, label="sample preview")
        phase_ax.set_title("Reference And Sample FFT Phase")
        phase_ax.set_xlabel("Frequency (THz)")
        phase_ax.set_ylabel("Phase (rad)")
        phase_ax.grid(True, alpha=0.3)
        phase_ax.legend()
        if freq_limits_thz is not None:
            for axis in (amp_ax, phase_ax):
                axis.set_xlim(float(freq_limits_thz[0]), float(freq_limits_thz[1]))
    fig.tight_layout()
    if display:
        try:
            from IPython.display import display as ipy_display

            ipy_display(fig)
        except Exception:
            pass
    return sample_result, simulation, fig, axes


def preview_study_noise(
    *,
    reference_result,
    sample_result,
    measurement=None,
    noise_dynamic_range_db,
    max_internal_reflections=0,
    seed=123,
    show_fft=False,
    freq_limits_thz=None,
    display=True,
):
    measurement = normalize_measurement(measurement)
    simulation = simulate_sample_from_reference(
        reference_result,
        sample_result.resolved_stack,
        max_internal_reflections=max_internal_reflections,
        measurement=measurement,
    )
    clean_trace = np.asarray(simulation["sample_trace"], dtype=np.float64)
    sigma = noise_sigma_from_dynamic_range(clean_trace, float(noise_dynamic_range_db))
    noisy_trace = add_white_gaussian_noise(clean_trace, sigma=sigma, seed=int(seed))
    noise_trace = noisy_trace - clean_trace

    clean_td = reference_result.trace.with_trace(clean_trace)
    noisy_td = reference_result.trace.with_trace(noisy_trace)
    noise_td = reference_result.trace.with_trace(noise_trace)

    fig, axes = plt.subplots(2 if show_fft else 1, 2 if show_fft else 1, figsize=(12, 8 if show_fft else 4.5))
    axes = np.atleast_1d(axes).ravel()
    time_ax = axes[0]
    time_ax.plot(clean_td.time_ps, clean_td.trace, label="noiseless sample")
    time_ax.plot(noisy_td.time_ps, noisy_td.trace, label="noisy sample", alpha=0.85)
    time_ax.plot(noise_td.time_ps, noise_td.trace, label="noise only", alpha=0.85)
    time_ax.set_title("Study Noise Preview")
    time_ax.set_xlabel("Time (ps)")
    time_ax.set_ylabel("Signal")
    time_ax.grid(True, alpha=0.3)
    time_ax.legend()

    if show_fft:
        clean_freq, clean_amp_db, clean_phase = trace_spectrum(clean_td)
        noisy_freq, noisy_amp_db, noisy_phase = trace_spectrum(noisy_td)
        noise_freq, noise_amp_db, noise_phase = trace_spectrum(noise_td)
        amp_ax = axes[1]
        phase_ax = axes[2]
        amp_ax.plot(clean_freq, clean_amp_db, label="noiseless sample")
        amp_ax.plot(noisy_freq, noisy_amp_db, label="noisy sample")
        amp_ax.plot(noise_freq, noise_amp_db, label="noise only")
        amp_ax.set_title("Study Noise FFT Amplitude")
        amp_ax.set_xlabel("Frequency (THz)")
        amp_ax.set_ylabel("Amplitude (dB)")
        amp_ax.grid(True, alpha=0.3)
        amp_ax.legend()
        phase_ax.plot(clean_freq, clean_phase, label="noiseless sample")
        phase_ax.plot(noisy_freq, noisy_phase, label="noisy sample")
        phase_ax.plot(noise_freq, noise_phase, label="noise only")
        phase_ax.set_title("Study Noise FFT Phase")
        phase_ax.set_xlabel("Frequency (THz)")
        phase_ax.set_ylabel("Phase (rad)")
        phase_ax.grid(True, alpha=0.3)
        phase_ax.legend()
        if freq_limits_thz is not None:
            for axis in (amp_ax, phase_ax):
                axis.set_xlim(float(freq_limits_thz[0]), float(freq_limits_thz[1]))
    fig.tight_layout()
    if display:
        try:
            from IPython.display import display as ipy_display

            ipy_display(fig)
        except Exception:
            pass
    return {
        "simulation": simulation,
        "noise_sigma": float(sigma),
        "clean_trace": clean_td,
        "noisy_trace": noisy_td,
        "noise_trace": noise_td,
        "figure": fig,
        "axes": axes,
    }


def estimate_study_runtime(
    reference_result,
    sample_result,
    study,
    *,
    measurement=None,
    pilot_case_count=10,
):
    config = _normalize_study_config(study)
    measurement = normalize_measurement(study.get("measurement") if measurement is None else measurement)
    assignments_list = [
        {**config["fixed_assignments"], **assignment}
        for assignment in _axis_assignments(config["sweep_axes"])
    ]
    total_runs = len(assignments_list) * int(config["replicates"])
    pilot_case_count = max(1, min(int(pilot_case_count), total_runs))
    sampled = np.linspace(0, total_runs - 1, pilot_case_count, dtype=int)

    started = time.perf_counter()
    for flat_index in sampled:
        case_id = int(flat_index) // int(config["replicates"])
        replicate_id = int(flat_index) % int(config["replicates"])
        assignments = assignments_list[case_id]
        true_stack = _build_true_stack(sample_result, config, assignments)
        true_simulation = simulate_sample_from_reference(
            reference_result,
            true_stack,
            max_internal_reflections=config["max_internal_reflections"],
            measurement=measurement,
        )
        noise_sigma = noise_sigma_from_dynamic_range(
            true_simulation["sample_trace"],
            assignments["noise_dynamic_range_db"],
        )
        seed = int(config["seed"] or 0) + case_id * int(config["seed_stride"]) + replicate_id
        observed_trace = add_white_gaussian_noise(
            true_simulation["sample_trace"],
            sigma=noise_sigma,
            seed=seed,
        )
        fit_sample_trace(
            reference=reference_result,
            observed_trace=observed_trace,
            initial_stack=sample_result.resolved_stack,
            fit_parameters=sample_result.fit_parameters,
            metric=config["metric"],
            max_internal_reflections=config["max_internal_reflections"],
            optimizer=config["optimizer"],
            measurement=measurement,
            objective_weights=build_objective_weights(
                observed_trace,
                mode=config["weighting"].get("mode", "none"),
                floor=config["weighting"].get("floor", 0.05),
                power=config["weighting"].get("power", 2.0),
                smooth_window_samples=config["weighting"].get("smooth_window_samples", 41),
            )
            if str(config["weighting"].get("mode", "none")).strip().lower() != "none"
            else None,
        )
    elapsed = time.perf_counter() - started
    avg_case_s = elapsed / pilot_case_count
    return {
        "pilot_case_count": int(pilot_case_count),
        "total_runs": int(total_runs),
        "avg_case_s": float(avg_case_s),
        "estimated_total_s": float(avg_case_s * total_runs),
        "estimated_total_h": float(avg_case_s * total_runs / 3600.0),
    }


def _aggregate_grid(rows, *, x_key, y_key, value_key):
    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = sorted({float(row[y_key]) for row in rows})
    z = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for iy, y_value in enumerate(y_values):
        for ix, x_value in enumerate(x_values):
            values = [
                float(row[value_key])
                for row in rows
                if float(row[x_key]) == x_value and float(row[y_key]) == y_value and np.isfinite(float(row[value_key]))
            ]
            if values:
                z[iy, ix] = float(np.mean(values))
    return x_values, y_values, z


def find_nearest_study_row(study_result, *, x_key, x_value, y_key, y_value):
    rows = list(study_result.summary_rows)
    target_x = float(x_value)
    target_y = float(y_value)
    return min(
        rows,
        key=lambda row: abs(float(row[x_key]) - target_x) + abs(float(row[y_key]) - target_y),
    )


def plot_study_heatmap_selector(
    study_result,
    *,
    x_key,
    y_key,
    value_key,
    color_scale="linear",
    selected_row=None,
    title=None,
    output_path=None,
):
    rows = list(study_result.summary_rows)
    x_values, y_values, z_values = _aggregate_grid(rows, x_key=x_key, y_key=y_key, value_key=value_key)
    fig, ax = plt.subplots(figsize=(7, 5))
    finite = z_values[np.isfinite(z_values)]
    norm = None
    if str(color_scale).strip().lower() == "log" and finite.size and np.all(finite > 0.0):
        norm = mcolors.LogNorm(vmin=float(np.min(finite)), vmax=float(np.max(finite)))
    image = ax.imshow(z_values, origin="lower", aspect="auto", norm=norm)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{value:.4g}" for value in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{value:.4g}" for value in y_values])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title or f"{value_key}: {x_key} vs {y_key}")
    if selected_row is not None:
        x_match = float(selected_row[x_key])
        y_match = float(selected_row[y_key])
        if x_match in x_values and y_match in y_values:
            ax.plot(x_values.index(x_match), y_values.index(y_match), marker="o", color="white", markersize=10)
            ax.plot(x_values.index(x_match), y_values.index(y_match), marker="x", color="black", markersize=8)
    fig.colorbar(image, ax=ax, label=value_key)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig, ax


def plot_study_case_detail(
    study_result,
    *,
    row,
    show_fft=False,
    freq_limits_thz=None,
):
    case_dir = study_result.cases_dir / f"case_{int(row['case_id']):04d}_rep_{int(row['replicate_id']):04d}"
    observed = read_trace_csv(case_dir / "sample_observed_trace.csv")
    fitted = read_trace_csv(case_dir / "sample_fit_trace.csv")
    truth = read_trace_csv(case_dir / "sample_true_trace.csv")
    fig, axes = plt.subplots(2 if show_fft else 1, 2 if show_fft else 1, figsize=(12, 8 if show_fft else 4.5))
    axes = np.atleast_1d(axes).ravel()
    time_ax = axes[0]
    time_ax.plot(observed.time_ps, observed.trace, label="observed")
    time_ax.plot(fitted.time_ps, fitted.trace, label="fit")
    time_ax.plot(truth.time_ps, truth.trace, label="true", linestyle="--")
    time_ax.set_title(
        f"Study Case {int(row['case_id'])}/{int(row['replicate_id'])}: {row.get('data_fit', row.get('objective_value')):.3e}"
    )
    time_ax.set_xlabel("Time (ps)")
    time_ax.set_ylabel("Signal")
    time_ax.grid(True, alpha=0.3)
    time_ax.legend()

    if show_fft:
        obs_freq, obs_amp_db, obs_phase = trace_spectrum(observed)
        fit_freq, fit_amp_db, fit_phase = trace_spectrum(fitted)
        true_freq, true_amp_db, true_phase = trace_spectrum(truth)
        amp_ax = axes[1]
        phase_ax = axes[2]
        amp_ax.plot(obs_freq, obs_amp_db, label="observed")
        amp_ax.plot(fit_freq, fit_amp_db, label="fit")
        amp_ax.plot(true_freq, true_amp_db, label="true", linestyle="--")
        amp_ax.set_title("FFT Amplitude")
        amp_ax.set_xlabel("Frequency (THz)")
        amp_ax.set_ylabel("Amplitude (dB)")
        amp_ax.grid(True, alpha=0.3)
        amp_ax.legend()
        phase_ax.plot(obs_freq, obs_phase, label="observed")
        phase_ax.plot(fit_freq, fit_phase, label="fit")
        phase_ax.plot(true_freq, true_phase, label="true", linestyle="--")
        phase_ax.set_title("FFT Phase")
        phase_ax.set_xlabel("Frequency (THz)")
        phase_ax.set_ylabel("Phase (rad)")
        phase_ax.grid(True, alpha=0.3)
        phase_ax.legend()
        if freq_limits_thz is not None:
            for axis in (amp_ax, phase_ax):
                axis.set_xlim(float(freq_limits_thz[0]), float(freq_limits_thz[1]))
    fig.tight_layout()
    return fig, axes


def save_fit_setup_snapshot(path, *, reference_trace, sample_trace, layers, measurement, preprocessing, optimizer, metric, max_internal_reflections, out_dir, notes=None, delay_options=None, n_in=1.0, n_out=1.0, overlay_imported=True):
    setup = build_fit_setup(
        reference_trace=reference_trace,
        sample_trace=sample_trace,
        layers=layers,
        preprocessing=preprocessing,
        measurement=measurement,
        optimizer=optimizer,
        metric=metric,
        max_internal_reflections=max_internal_reflections,
        delay_options=delay_options,
        n_in=n_in,
        n_out=n_out,
        overlay_imported=overlay_imported,
        out_dir=out_dir,
        notes=notes,
    )
    return write_fit_setup_json(path, setup)


def save_study_setup_snapshot(path, *, reference, layers, measurement, study, n_in=1.0, n_out=1.0, overlay_imported=True, sample_out_dir=None, notes=None):
    setup = build_study_setup(
        reference=reference,
        layers=layers,
        measurement=measurement,
        study=study,
        n_in=n_in,
        n_out=n_out,
        overlay_imported=overlay_imported,
        sample_out_dir=sample_out_dir,
        notes=notes,
    )
    return write_study_setup_json(path, setup)


def run_study_with_progress(reference, sample, study, *, measurement=None, out_dir=None):
    return run_study(reference, sample, study, measurement=measurement, out_dir=out_dir, show_progress=True)
