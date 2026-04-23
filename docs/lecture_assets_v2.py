from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


HEATMAP_CMAP = "plasma"
HEATMAP_LEVEL_COUNT = 256

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thzsim2.core import add_white_gaussian_noise, noise_sigma_from_dynamic_range, simulate_sample_from_reference
from thzsim2.core.fitting import (
    _set_by_path,
    build_objective_weights,
    drude_gamma_thz_from_tau_ps,
    drude_plasma_freq_thz_from_sigma_tau,
    fit_sample_trace,
    stack_path_from_user_path,
    summarize_single_layer_drude_stack,
    summarize_two_drude_layer,
)
from thzsim2.io.trace_csv import write_trace_csv
from thzsim2.models import ConstantNK, Drude, Fit, Layer, Measurement, ReferenceStandard, TraceData, TwoDrude
from thzsim2.workflows.deepdive_fit import parameter_correlation_rows, run_staged_measured_fit
from thzsim2.workflows.fit_workflow import (
    prepare_reflection_first_peak_pair,
    prepare_trace_pair_for_fit,
    resolve_measurement_fit_parameters,
    run_measured_fit,
)
from thzsim2.workflows.reference import generate_reference_pulse, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample


LECTURE_BUILD_ROOT = REPO_ROOT / "docs" / "lecture_build"
LATEST_BUILD_ROOT = LECTURE_BUILD_ROOT / "latest"
FINAL_LECTURE_ROOT = REPO_ROOT / "docs" / "final_lecture"
GENERATED_LECTURE_ROOT = FINAL_LECTURE_ROOT / "generated"
NOTEBOOK_PATH = GENERATED_LECTURE_ROOT / "THzTDS_Lecture_Fit_Study.ipynb"
NOTES_TEX_PATH = GENERATED_LECTURE_ROOT / "lecture_thz_tds_fit_study_notes.tex"
SLIDES_TEX_PATH = GENERATED_LECTURE_ROOT / "lecture_thz_tds_fit_study_slides.tex"

_RECOVERY_LABEL = r"$\mathcal{E}_{\mathrm{rec}}=\sqrt{\frac{1}{2}\sum_i\log_{10}^2\!\left(\hat{p}_i/p_i^\star\right)}$"
_WEIGHTED_OBJECTIVE_LABEL = (
    r"\mathcal{J}_{w}=\frac{\sum_k w_k\left(E_{\mathrm{fit}}(t_k)-E_{\mathrm{obs}}(t_k)\right)^2}"
    r"{\sum_k w_k E_{\mathrm{obs}}^2(t_k)}"
)


@dataclass(slots=True)
class LectureGridStudySpec:
    slug: str
    title: str
    section: str
    x_name: str
    y_name: str
    x_values: list[float]
    y_values: list[float]
    x_label: str
    y_label: str
    measurement: Measurement
    noise_dynamic_range_db: float
    max_internal_reflections: int
    recovery_keys: tuple[str, str]
    fixed_note: str
    truth_update_builder: callable
    fit_summary_builder: callable
    spec_payload: dict


def _lecture_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfaf7",
            "axes.edgecolor": "#2a2a2a",
            "axes.labelcolor": "#1f1f1f",
            "axes.titleweight": "bold",
            "axes.titlesize": 13.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "grid.color": "#b8b8b8",
            "grid.alpha": 0.20,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "savefig.bbox": "tight",
        }
    )


def _apply_ruler_ticks(ax):
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=7, width=1.0)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _zoom_bounds(x_values, y_primary, y_secondary=None, *, center_index: int | None = None, half_window: int = 18):
    x = np.asarray(x_values, dtype=np.float64)
    y1 = np.asarray(y_primary, dtype=np.float64)
    y2 = None if y_secondary is None else np.asarray(y_secondary, dtype=np.float64)
    if x.size == 0:
        return 0.0, 1.0, -1.0, 1.0
    if center_index is None:
        metric = np.abs(y1 if y2 is None else (y1 - y2))
        center_index = int(np.argmax(metric))
    lo = max(0, int(center_index) - int(half_window))
    hi = min(x.size - 1, int(center_index) + int(half_window))
    if hi <= lo:
        hi = min(x.size - 1, lo + 1)
    x_slice = x[lo : hi + 1]
    y_parts = [y1[lo : hi + 1]]
    if y2 is not None:
        y_parts.append(y2[lo : hi + 1])
    y_slice = np.concatenate(y_parts)
    y_min = float(np.min(y_slice))
    y_max = float(np.max(y_slice))
    pad = max(0.08 * (y_max - y_min), 1e-6)
    return float(x_slice[0]), float(x_slice[-1]), y_min - pad, y_max + pad


def _add_zoom_inset(
    ax,
    x_values,
    y_primary,
    y_secondary=None,
    *,
    colors,
    labels=None,
    loc="lower left",
    bbox_to_anchor=(0.06, 0.08, 0.42, 0.42),
    center_index=None,
    half_window=18,
    y_limits=None,
    x_limits=None,
    show_connectors=False,
):
    x0, x1, y0, y1 = _zoom_bounds(
        x_values,
        y_primary,
        y_secondary,
        center_index=center_index,
        half_window=half_window,
    )
    if x_limits is not None:
        x0, x1 = float(x_limits[0]), float(x_limits[1])
    if y_limits is not None:
        y0, y1 = float(y_limits[0]), float(y_limits[1])
    inset = inset_axes(
        ax,
        width="100%",
        height="100%",
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    inset.plot(x_values, y_primary, color=colors[0], linewidth=1.35)
    if y_secondary is not None:
        inset.plot(x_values, y_secondary, color=colors[1], linewidth=1.2)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.set_facecolor("#fffdf8")
    _apply_ruler_ticks(inset)
    inset.tick_params(axis="both", which="major", labelsize=7.5, pad=1.5, length=5.5)
    inset.tick_params(axis="both", which="minor", length=2.5)
    for spine in inset.spines.values():
        spine.set_edgecolor("#6a6a6a")
        spine.set_linewidth(0.9)
    inset.grid(True, which="major", alpha=0.14, linestyle="-", linewidth=0.4)
    inset.grid(True, which="minor", alpha=0.07, linestyle="-", linewidth=0.25)
    if show_connectors:
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#808080", lw=0.8, alpha=0.9)
    if labels:
        inset.text(
            0.03,
            0.97,
            labels,
            transform=inset.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#fffdf8", "edgecolor": "none", "alpha": 0.88},
        )
    return inset


def _timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _json_dump(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_rows_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames: list[str] = []
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


def _save_figure(fig, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "png": stem.with_suffix(".png").resolve().as_posix(),
        "pdf": stem.with_suffix(".pdf").resolve().as_posix(),
    }


def _copy_tree_latest(source_dir: Path, latest_dir: Path):
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(source_dir, latest_dir)


def _new_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _new_code_cell(source: str, *, metadata: dict | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {} if metadata is None else dict(metadata),
        "outputs": [],
        "source": source,
    }


def _new_notebook(*, cells: list[dict], metadata: dict) -> dict:
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _reference_input_config() -> dict:
    return {
        "kind": "generated_pulse",
        "generate": {
            "model": "sech_carrier",
            "sample_count": 1024,
            "dt_ps": 0.03,
            "time_center_ps": 18.0,
            "pulse_center_ps": 9.0,
            "tau_ps": 0.22,
            "f0_thz": 0.9,
            "amp": 1.0,
            "phi_rad": 0.0,
        },
        "prepare": {},
    }


def _prepare_reference_from_config(config: dict, *, output_root: Path, run_label: str):
    generate = dict(config["generate"])
    pulse = generate_reference_pulse(**generate)
    return prepare_reference(pulse, output_root=output_root, run_label=run_label)


def _as_trace_data(reference_result, trace, *, source_kind: str):
    return TraceData(
        time_ps=reference_result.trace.time_ps.copy(),
        trace=np.asarray(trace, dtype=np.float64),
        source_kind=source_kind,
        metadata={},
    )


def _write_trace_bundle(case_dir: Path, *, reference_result, true_trace, observed_trace, fit_trace, residual_trace):
    case_dir.mkdir(parents=True, exist_ok=True)
    noise_trace = np.asarray(observed_trace, dtype=np.float64) - np.asarray(true_trace, dtype=np.float64)
    write_trace_csv(case_dir / "reference_trace.csv", reference_result.trace)
    write_trace_csv(case_dir / "sample_true_trace.csv", _as_trace_data(reference_result, true_trace, source_kind="simulation_true"))
    write_trace_csv(
        case_dir / "sample_observed_trace.csv",
        _as_trace_data(reference_result, observed_trace, source_kind="simulation_observed"),
    )
    write_trace_csv(case_dir / "sample_fit_trace.csv", _as_trace_data(reference_result, fit_trace, source_kind="simulation_fit"))
    write_trace_csv(case_dir / "sample_residual_trace.csv", _as_trace_data(reference_result, residual_trace, source_kind="simulation_residual"))
    write_trace_csv(case_dir / "sample_noise_trace.csv", _as_trace_data(reference_result, noise_trace, source_kind="simulation_noise"))


def _measurement_record(measurement: Measurement) -> dict:
    return {
        "mode": measurement.mode,
        "angle_deg": float(measurement.angle_deg),
        "polarization": measurement.polarization,
        "polarization_mix": None if measurement.polarization_mix is None else float(measurement.polarization_mix),
        "reference_standard_kind": None if measurement.reference_standard is None else measurement.reference_standard.kind,
    }


def _mean_value(values):
    values = [float(value) for value in values if np.isfinite(float(value))]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _signed_fisher_mean(correlation_values: list[float]) -> float:
    finite = [float(value) for value in correlation_values if np.isfinite(float(value))]
    if not finite:
        return float("nan")
    clipped = np.clip(np.asarray(finite, dtype=np.float64), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def _aggregate_grid(rows: list[dict], *, x_key: str, y_key: str, value_key: str):
    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = sorted({float(row[y_key]) for row in rows})
    z = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for iy, y_value in enumerate(y_values):
        for ix, x_value in enumerate(x_values):
            cell_values = [
                float(row[value_key])
                for row in rows
                if math.isclose(float(row[x_key]), x_value, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(float(row[y_key]), y_value, rel_tol=0.0, abs_tol=1e-12)
                and np.isfinite(float(row[value_key]))
            ]
            if cell_values:
                z[iy, ix] = float(np.mean(cell_values))
    return x_values, y_values, z


def _positive_grid_for_log(z_values):
    z = np.asarray(z_values, dtype=np.float64).copy()
    finite = z[np.isfinite(z)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        z[np.isfinite(z)] = 1.0
        return z
    floor = max(float(np.min(positive)) * 0.5, 1e-18)
    z[np.isfinite(z) & (z <= 0.0)] = floor
    return z


def _contourf_heatmap(
    x_values,
    y_values,
    z_values,
    *,
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
    log_color: bool,
):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    z = np.asarray(z_values, dtype=np.float64)
    if log_color:
        z = _positive_grid_for_log(z)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("heatmap requires at least one finite z value")

    if log_color:
        positive = finite[finite > 0.0]
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
        if math.isclose(vmin, vmax):
            vmax = vmin * 1.01
        levels = np.geomspace(vmin, vmax, HEATMAP_LEVEL_COUNT)
        contour = ax.contourf(
            x,
            y,
            z,
            levels=levels,
            cmap=HEATMAP_CMAP,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            extend="both",
        )
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if math.isclose(vmin, vmax):
            vmax = vmin + max(abs(vmin) * 1e-3, 1e-12)
        levels = np.linspace(vmin, vmax, HEATMAP_LEVEL_COUNT)
        contour = ax.contourf(x, y, z, levels=levels, cmap=HEATMAP_CMAP, extend="both")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(False)
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def _pretty_fit_parameter_label(name: str) -> str:
    mapping = {
        "film_thickness_um": r"$d$ ($\mu$m)",
        "film_plasma_freq_thz": r"$\omega_{p}$ (THz)",
        "film_gamma_thz": r"$\gamma$ (THz)",
        "epi_thickness_um": r"$d_{\mathrm{epi}}$ ($\mu$m)",
        "oxide_thickness_um": r"$d_{\mathrm{ox}}$ ($\mu$m)",
        "epi_plasma1_thz": r"$\omega_{p1}$ (THz)",
        "epi_gamma1_thz": r"$\gamma_1$ (THz)",
        "epi_plasma2_thz": r"$\omega_{p2}$ (THz)",
        "epi_gamma2_thz": r"$\gamma_2$ (THz)",
        "wafer_thickness_um": r"$d$ ($\mu$m)",
        "wafer_plasma_freq_thz": r"$\omega_p$ (THz)",
        "wafer_gamma_thz": r"$\gamma$ (THz)",
        "delta_t_ps": r"$\Delta t$ (ps)",
    }
    return mapping.get(name, name.replace("_", r"\_"))


def _plot_correlation_matrix(matrix_rows: list[dict], *, title: str):
    labels = sorted({str(row["param_i"]) for row in matrix_rows} | {str(row["param_j"]) for row in matrix_rows})
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    for row in matrix_rows:
        i = index_by_label[str(row["param_i"])]
        j = index_by_label[str(row["param_j"])]
        matrix[i, j] = float(row["correlation"])
        matrix[j, i] = float(row["correlation"])
    for idx in range(len(labels)):
        matrix[idx, idx] = 1.0

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([_pretty_fit_parameter_label(label) for label in labels], rotation=45, ha="right")
    ax.set_yticklabels([_pretty_fit_parameter_label(label) for label in labels])
    fig.colorbar(image, ax=ax, label=r"$\rho_{ij}$")
    fig.tight_layout()
    return fig


def _plot_triptych(linear_png, log_png, corr_png, *, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    for axis, image_path, panel_title in zip(
        axes,
        (linear_png, log_png, corr_png),
        ("Linear scale", "Log scale", "Average fit correlation"),
        strict=True,
    ):
        image = plt.imread(str(image_path))
        axis.imshow(image)
        axis.set_title(panel_title)
        axis.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _fit_correlation_rows(case_id: int, fit_result: dict) -> list[dict]:
    names = list(fit_result.get("parameter_names", []))
    matrix = fit_result.get("parameter_correlation")
    if matrix is None or not names:
        return []
    corr = np.asarray(matrix, dtype=np.float64)
    rows = []
    for i, label_i in enumerate(names):
        for j, label_j in enumerate(names):
            rows.append(
                {
                    "case_id": int(case_id),
                    "param_i": str(label_i),
                    "param_j": str(label_j),
                    "correlation": float(corr[i, j]),
                }
            )
    return rows


def _average_correlation_rows(correlation_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = {}
    labels = sorted({str(row["param_i"]) for row in correlation_rows} | {str(row["param_j"]) for row in correlation_rows})
    for label_i in labels:
        for label_j in labels:
            grouped[(label_i, label_j)] = []
    for row in correlation_rows:
        grouped[(str(row["param_i"]), str(row["param_j"]))].append(float(row["correlation"]))

    averaged = []
    for label_i in labels:
        for label_j in labels:
            if label_i == label_j:
                value = 1.0
            else:
                value = _signed_fisher_mean(grouped[(label_i, label_j)])
            averaged.append({"param_i": label_i, "param_j": label_j, "correlation": value})
    return averaged


def _map_recovery_error(truth_summary: dict, fit_summary: dict, recovery_keys: tuple[str, str]) -> float:
    errors = []
    for key in recovery_keys:
        true_value = float(truth_summary[key])
        fit_value = float(fit_summary[key])
        ratio = max(abs(fit_value) / max(abs(true_value), 1e-30), 1e-30)
        errors.append(math.log10(ratio) ** 2)
    return float(math.sqrt(np.mean(errors)))


def _summarize_advanced_stack(resolved_stack) -> dict[str, float]:
    epi_summary = summarize_two_drude_layer(resolved_stack["layers"][0])
    return {
        "epi_thickness_um": float(epi_summary["thickness_um"]),
        "oxide_thickness_um": float(resolved_stack["layers"][1]["thickness_um"]),
        "substrate_thickness_um": float(resolved_stack["layers"][2]["thickness_um"]),
        "eps_inf": float(epi_summary["eps_inf"]),
        "tau1_ps": float(epi_summary["tau1_ps"]),
        "sigma1_s_per_m": float(epi_summary["sigma1_s_per_m"]),
        "tau2_ps": float(epi_summary["tau2_ps"]),
        "sigma2_s_per_m": float(epi_summary["sigma2_s_per_m"]),
        "plasma_freq1_thz": float(epi_summary["plasma_freq1_thz"]),
        "gamma1_thz": float(epi_summary["gamma1_thz"]),
        "plasma_freq2_thz": float(epi_summary["plasma_freq2_thz"]),
        "gamma2_thz": float(epi_summary["gamma2_thz"]),
    }


def _one_layer_fit_sample(reference_result, *, thickness_um: float, sigma_nominal: float, tau_nominal: float, output_dir: Path):
    return build_sample(
        layers=[
            Layer(
                name="film",
                thickness_um=Fit(
                    thickness_um,
                    abs_min=thickness_um - 10.0,
                    abs_max=thickness_um + 10.0,
                    label="film_thickness_um",
                ),
                material=Drude(
                    eps_inf=11.7,
                    plasma_freq_thz=Fit(
                        drude_plasma_freq_thz_from_sigma_tau(sigma_nominal, tau_nominal),
                        abs_min=drude_plasma_freq_thz_from_sigma_tau(0.01, 1.0),
                        abs_max=drude_plasma_freq_thz_from_sigma_tau(1000.0, 0.1),
                        label="film_plasma_freq_thz",
                    ),
                    gamma_thz=Fit(
                        drude_gamma_thz_from_tau_ps(tau_nominal),
                        abs_min=drude_gamma_thz_from_tau_ps(1.0),
                        abs_max=drude_gamma_thz_from_tau_ps(0.1),
                        label="film_gamma_thz",
                    ),
                ),
            )
        ],
        reference=reference_result,
        out_dir=output_dir,
        n_in=1.0,
        n_out=1.0,
    )


def _advanced_fit_sample(
    reference_result,
    *,
    epi_thickness_um: float,
    substrate_thickness_um: float,
    oxide_thickness_um: float,
    tau1_nominal: float,
    sigma1_nominal: float,
    tau2_nominal: float,
    sigma2_nominal: float,
    output_dir: Path,
):
    return build_sample(
        layers=[
            Layer(
                name="epi",
                thickness_um=Fit(
                    epi_thickness_um,
                    abs_min=max(1.0, epi_thickness_um - 5.0),
                    abs_max=epi_thickness_um + 5.0,
                    label="epi_thickness_um",
                ),
                material=TwoDrude(
                    eps_inf=11.7,
                    plasma_freq1_thz=Fit(
                        drude_plasma_freq_thz_from_sigma_tau(sigma1_nominal, tau1_nominal),
                        abs_min=drude_plasma_freq_thz_from_sigma_tau(0.01, 1.0),
                        abs_max=drude_plasma_freq_thz_from_sigma_tau(1000.0, 0.1),
                        label="epi_plasma1_thz",
                    ),
                    gamma1_thz=Fit(
                        drude_gamma_thz_from_tau_ps(tau1_nominal),
                        abs_min=drude_gamma_thz_from_tau_ps(1.0),
                        abs_max=drude_gamma_thz_from_tau_ps(0.1),
                        label="epi_gamma1_thz",
                    ),
                    plasma_freq2_thz=Fit(
                        drude_plasma_freq_thz_from_sigma_tau(sigma2_nominal, tau2_nominal),
                        abs_min=drude_plasma_freq_thz_from_sigma_tau(0.01, 1.0),
                        abs_max=drude_plasma_freq_thz_from_sigma_tau(1000.0, 0.1),
                        label="epi_plasma2_thz",
                    ),
                    gamma2_thz=Fit(
                        drude_gamma_thz_from_tau_ps(tau2_nominal),
                        abs_min=drude_gamma_thz_from_tau_ps(1.0),
                        abs_max=drude_gamma_thz_from_tau_ps(0.1),
                        label="epi_gamma2_thz",
                    ),
                ),
            ),
            Layer(
                name="oxide",
                thickness_um=Fit(
                    oxide_thickness_um,
                    abs_min=max(1.0, oxide_thickness_um - 2.0),
                    abs_max=oxide_thickness_um + 2.0,
                    label="oxide_thickness_um",
                ),
                material=ConstantNK(n=1.95, k=0.005),
            ),
            Layer(
                name="substrate",
                thickness_um=substrate_thickness_um,
                material=ConstantNK(n=3.42, k=0.005),
            ),
        ],
        reference=reference_result,
        out_dir=output_dir,
        n_in=1.0,
        n_out=1.0,
    )


def _advanced_reflection_standard(reference_result, *, substrate_thickness_um: float, oxide_thickness_um: float, output_dir: Path):
    return build_sample(
        layers=[
            Layer(name="oxide", thickness_um=oxide_thickness_um, material=ConstantNK(n=1.95, k=0.005)),
            Layer(name="substrate", thickness_um=substrate_thickness_um, material=ConstantNK(n=3.42, k=0.005)),
        ],
        reference=reference_result,
        out_dir=output_dir,
        n_in=1.0,
        n_out=1.0,
    )


def _run_saved_grid_study(
    *,
    build_root: Path,
    reference_result,
    sample_result,
    spec: LectureGridStudySpec,
    optimizer: dict,
    weighting: dict,
):
    study_dir = build_root / "data" / spec.section / spec.slug
    cases_dir = study_dir / "cases"
    study_dir.mkdir(parents=True, exist_ok=True)
    measurement, measurement_fit_parameters = resolve_measurement_fit_parameters(spec.measurement)

    summary_rows = []
    correlation_rows = []
    case_id = 0

    for y_value in spec.y_values:
        for x_value in spec.x_values:
            true_stack = deepcopy(sample_result.resolved_stack)
            updates, truth_summary = spec.truth_update_builder(float(x_value), float(y_value))
            for user_path, value in updates.items():
                _set_by_path(true_stack, stack_path_from_user_path(user_path), float(value))

            simulation = simulate_sample_from_reference(
                reference_result,
                true_stack,
                measurement=measurement,
                max_internal_reflections=spec.max_internal_reflections,
            )
            noise_sigma = noise_sigma_from_dynamic_range(simulation["sample_trace"], spec.noise_dynamic_range_db)
            observed_trace = add_white_gaussian_noise(
                simulation["sample_trace"],
                sigma=noise_sigma,
                seed=1000 + int(case_id),
            )
            objective_weights = build_objective_weights(
                observed_trace,
                mode=weighting.get("mode", "trace_amplitude"),
                floor=weighting.get("floor", 0.03),
                power=weighting.get("power", 2.0),
                smooth_window_samples=weighting.get("smooth_window_samples", 41),
            )
            fit = fit_sample_trace(
                reference=reference_result,
                observed_trace=observed_trace,
                initial_stack=sample_result.resolved_stack,
                fit_parameters=sample_result.fit_parameters,
                measurement=measurement,
                measurement_fit_parameters=measurement_fit_parameters,
                metric="weighted_data_fit",
                max_internal_reflections=spec.max_internal_reflections,
                optimizer=optimizer,
                objective_weights=objective_weights,
            )
            fit_summary = spec.fit_summary_builder(fit["fitted_stack"])
            recovery_error = _map_recovery_error(truth_summary, fit_summary, spec.recovery_keys)
            row = {
                "case_id": int(case_id),
                spec.x_name: float(x_value),
                spec.y_name: float(y_value),
                "measurement_mode": measurement.mode,
                "measurement_angle_deg": float(measurement.angle_deg),
                "measurement_polarization": measurement.polarization,
                "noise_dynamic_range_db": float(spec.noise_dynamic_range_db),
                "weighted_data_fit": float(fit["residual_metrics"]["weighted_data_fit"]),
                "data_fit": float(fit["residual_metrics"]["data_fit"]),
                "relative_l2": float(fit["residual_metrics"]["relative_l2"]),
                "residual_rms": float(fit["residual_metrics"]["residual_rms"]),
                "fit_sigma": float(fit["residual_metrics"]["fit_sigma"]),
                "max_abs_residual": float(np.max(np.abs(fit["residual_trace"]))),
                "max_abs_parameter_correlation": float(fit["max_abs_parameter_correlation"]),
                "mean_abs_parameter_correlation": float(fit["mean_abs_parameter_correlation"]),
                "recovery_error": float(recovery_error),
            }
            for key, value in truth_summary.items():
                row[f"true_{key}"] = float(value)
            for key, value in fit_summary.items():
                row[f"fit_{key}"] = float(value)
            for key in sorted(set(truth_summary) & set(fit_summary)):
                signed_error = float(truth_summary[key]) - float(fit_summary[key])
                row[f"signed_err_{key}"] = signed_error
                row[f"abs_err_{key}"] = abs(signed_error)
            summary_rows.append(row)
            correlation_rows.extend(_fit_correlation_rows(case_id, fit))
            _write_trace_bundle(
                cases_dir / f"case_{case_id:04d}",
                reference_result=reference_result,
                true_trace=simulation["sample_trace"],
                observed_trace=observed_trace,
                fit_trace=fit["fitted_simulation"]["sample_trace"],
                residual_trace=fit["residual_trace"],
            )
            case_id += 1

    summary_csv_path = study_dir / "study_summary.csv"
    summary_json_path = study_dir / "study_summary.json"
    correlation_csv_path = study_dir / "study_correlations.csv"
    spec_json_path = study_dir / "lecture_study_spec.json"
    averaged_corr_csv_path = study_dir / "averaged_correlation.csv"
    averaged_corr_json_path = study_dir / "averaged_correlation.json"
    _write_rows_csv(summary_csv_path, summary_rows)
    _write_rows_csv(correlation_csv_path, correlation_rows)
    averaged_corr_rows = _average_correlation_rows(correlation_rows)
    _write_rows_csv(averaged_corr_csv_path, averaged_corr_rows)
    _json_dump(
        averaged_corr_json_path,
        {
            "title": spec.title,
            "slug": spec.slug,
            "rows": averaged_corr_rows,
        },
    )
    _json_dump(
        summary_json_path,
        {
            "slug": spec.slug,
            "title": spec.title,
            "row_count": len(summary_rows),
            "measurement": _measurement_record(measurement),
            "max_internal_reflections": int(spec.max_internal_reflections),
            "fixed_note": spec.fixed_note,
            "recovery_keys": list(spec.recovery_keys),
        },
    )
    _json_dump(spec_json_path, spec.spec_payload)

    x_grid, y_grid, recovery_grid = _aggregate_grid(summary_rows, x_key=spec.x_name, y_key=spec.y_name, value_key="recovery_error")
    linear_figure = _contourf_heatmap(
        x_grid,
        y_grid,
        recovery_grid,
        title=spec.title,
        x_label=spec.x_label,
        y_label=spec.y_label,
        cbar_label=_RECOVERY_LABEL,
        log_color=False,
    )
    linear_paths = _save_figure(linear_figure, study_dir / f"{spec.slug}__linear")
    log_figure = _contourf_heatmap(
        x_grid,
        y_grid,
        recovery_grid,
        title=spec.title,
        x_label=spec.x_label,
        y_label=spec.y_label,
        cbar_label=_RECOVERY_LABEL,
        log_color=True,
    )
    log_paths = _save_figure(log_figure, study_dir / f"{spec.slug}__log")
    corr_figure = _plot_correlation_matrix(averaged_corr_rows, title=f"{spec.title}: average fit correlation")
    corr_paths = _save_figure(corr_figure, study_dir / f"{spec.slug}__corr")
    triptych_figure = _plot_triptych(
        Path(linear_paths["png"]),
        Path(log_paths["png"]),
        Path(corr_paths["png"]),
        title=spec.title,
    )
    triptych_paths = _save_figure(triptych_figure, build_root / "figures" / spec.slug)

    return {
        "slug": spec.slug,
        "title": spec.title,
        "study_dir": study_dir.resolve().as_posix(),
        "summary_csv": summary_csv_path.resolve().as_posix(),
        "summary_json": summary_json_path.resolve().as_posix(),
        "correlation_csv": correlation_csv_path.resolve().as_posix(),
        "averaged_correlation_csv": averaged_corr_csv_path.resolve().as_posix(),
        "averaged_correlation_json": averaged_corr_json_path.resolve().as_posix(),
        "spec_json": spec_json_path.resolve().as_posix(),
        "figure_linear_png": linear_paths["png"],
        "figure_linear_pdf": linear_paths["pdf"],
        "figure_log_png": log_paths["png"],
        "figure_log_pdf": log_paths["pdf"],
        "figure_corr_png": corr_paths["png"],
        "figure_corr_pdf": corr_paths["pdf"],
        "figure_triptych_png": triptych_paths["png"],
        "figure_triptych_pdf": triptych_paths["pdf"],
    }


def _measured_fit_layers() -> list[Layer]:
    sigma_s_per_m = 100.0 / 67.0
    tau_ps = 0.25
    return [
        Layer(
            name="wafer",
            thickness_um=Fit(625.0, abs_min=600.0, abs_max=650.0, label="wafer_thickness_um"),
            material=Drude(
                eps_inf=11.7,
                plasma_freq_thz=Fit(
                    drude_plasma_freq_thz_from_sigma_tau(sigma_s_per_m, tau_ps),
                    abs_min=0.05,
                    abs_max=3.5,
                    label="wafer_plasma_freq_thz",
                ),
                gamma_thz=Fit(
                    drude_gamma_thz_from_tau_ps(tau_ps),
                    abs_min=0.1,
                    abs_max=2.5,
                    label="wafer_gamma_thz",
                ),
            ),
        )
    ]


def _measured_reflection_layers() -> list[Layer]:
    sigma_s_per_m = 100.0 / 67.0
    tau_ps = 0.25
    return [
        Layer(
            name="wafer",
            thickness_um=Fit(625.0, abs_min=575.0, abs_max=675.0, label="wafer_thickness_um"),
            material=Drude(
                eps_inf=Fit(11.7, abs_min=4.0, abs_max=20.0, label="wafer_eps_inf"),
                plasma_freq_thz=Fit(
                    drude_plasma_freq_thz_from_sigma_tau(sigma_s_per_m, tau_ps),
                    abs_min=0.01,
                    abs_max=4.0,
                    label="wafer_plasma_freq_thz",
                ),
                gamma_thz=Fit(
                    drude_gamma_thz_from_tau_ps(tau_ps),
                    abs_min=0.01,
                    abs_max=4.0,
                    label="wafer_gamma_thz",
                ),
            ),
        )
    ]


def _positive_spectrum(trace, time_ps):
    dt_s = float(np.median(np.diff(np.asarray(time_ps, dtype=np.float64)))) * 1e-12
    t0_s = float(np.asarray(time_ps, dtype=np.float64)[0]) * 1e-12
    from thzsim2.core.fft import fft_t_to_w

    omega, spectrum = fft_t_to_w(np.asarray(trace, dtype=np.float64), dt=dt_s, t0=t0_s)
    freq_thz = omega / (2.0 * np.pi * 1e12)
    mask = freq_thz > 0.0
    return np.asarray(freq_thz[mask], dtype=np.float64), np.asarray(spectrum[mask], dtype=np.complex128)


def _relative_db(values, *, floor_db=-110.0):
    values = np.asarray(values, dtype=np.float64)
    reference = max(float(np.max(values)), 1e-30)
    floor_linear = 10.0 ** (float(floor_db) / 20.0)
    return 20.0 * np.log10(np.maximum(values / reference, floor_linear))


def _plot_measured_fit_overview(prepared_traces, fit_result, *, title: str):
    observed = np.asarray(prepared_traces.processed_sample.trace, dtype=np.float64)
    fitted = np.asarray(fit_result["fitted_simulation"]["sample_trace"], dtype=np.float64)
    residual = np.asarray(fit_result["residual_trace"], dtype=np.float64)
    time_ps = np.asarray(prepared_traces.processed_sample.time_ps, dtype=np.float64)

    freq_thz, observed_spec = _positive_spectrum(observed, time_ps)
    _, fitted_spec = _positive_spectrum(fitted, time_ps)
    observed_amp = _relative_db(np.abs(observed_spec))
    fitted_amp = _relative_db(np.abs(fitted_spec))
    observed_phase = np.unwrap(np.angle(observed_spec))
    fitted_phase = np.unwrap(np.angle(fitted_spec))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    axes[0, 0].plot(time_ps, observed, linewidth=1.9, color="#0f4c81", label="Observed")
    axes[0, 0].plot(time_ps, fitted, linewidth=1.55, color="#d1495b", label="Fit")
    axes[0, 0].set_title("Time-domain waveform")
    axes[0, 0].set_xlabel(r"$t$ (ps)")
    axes[0, 0].set_ylabel(r"$E(t)$")
    axes[0, 0].legend(loc="upper left")
    peak_index = int(np.argmax(np.abs(observed)))
    _add_zoom_inset(
        axes[0, 0],
        time_ps,
        observed,
        fitted,
        colors=("#0f4c81", "#d1495b"),
        labels="main peak zoom",
        loc="upper right",
        bbox_to_anchor=(0.57, 0.54, 0.36, 0.33),
        center_index=peak_index,
        half_window=14,
    )

    axes[0, 1].plot(time_ps, residual, linewidth=1.4, color="#3b7d3a")
    residual_peak_index = int(np.argmax(np.abs(residual)))
    axes[0, 1].axhline(0.0, color="#555555", linewidth=0.9, alpha=0.7)
    axes[0, 1].scatter([time_ps[residual_peak_index]], [residual[residual_peak_index]], color="#b22222", zorder=3)
    axes[0, 1].annotate(
        f"{time_ps[residual_peak_index]:.2f} ps",
        xy=(time_ps[residual_peak_index], residual[residual_peak_index]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0, 1].set_title("Residual trace")
    axes[0, 1].set_xlabel(r"$t$ (ps)")
    axes[0, 1].set_ylabel(r"$r(t)=E_{\mathrm{fit}}(t)-E_{\mathrm{data}}(t)$")
    axes[0, 1].text(
        0.03,
        0.06,
        r"$r(t)$ is plotted with the fit-minus-data sign convention",
        transform=axes[0, 1].transAxes,
        fontsize=8.8,
        color="#424242",
    )

    axes[1, 0].plot(freq_thz, observed_amp, linewidth=1.9, color="#0f4c81", label="Observed")
    axes[1, 0].plot(freq_thz, fitted_amp, linewidth=1.55, color="#d1495b", label="Fit")
    axes[1, 0].set_title("Spectral amplitude")
    axes[1, 0].set_xlabel(r"$f$ (THz)")
    axes[1, 0].set_ylabel("Amplitude (dB)")
    axes[1, 0].set_xlim(0.05, min(3.0, float(freq_thz[-1])))
    amp_center = int(np.argmax(np.abs(observed_amp - fitted_amp)))
    amp_x0 = max(0.05, float(freq_thz[max(0, amp_center - 12)]))
    amp_x1 = min(3.0, float(freq_thz[min(freq_thz.size - 1, amp_center + 12)]))
    _add_zoom_inset(
        axes[1, 0],
        freq_thz,
        observed_amp,
        fitted_amp,
        colors=("#0f4c81", "#d1495b"),
        labels="largest amplitude mismatch",
        loc="lower left",
        bbox_to_anchor=(0.10, 0.12, 0.38, 0.36),
        center_index=amp_center,
        half_window=12,
        y_limits=(-60.0, 2.0),
        x_limits=(amp_x0, amp_x1),
    )

    axes[1, 1].plot(freq_thz, observed_phase, linewidth=1.9, color="#0f4c81", label="Observed")
    axes[1, 1].plot(freq_thz, fitted_phase, linewidth=1.55, color="#d1495b", label="Fit")
    axes[1, 1].set_title("Unwrapped spectral phase")
    axes[1, 1].set_xlabel(r"$f$ (THz)")
    axes[1, 1].set_ylabel(r"$\phi(f)$ (rad)")
    axes[1, 1].set_xlim(0.05, min(3.0, float(freq_thz[-1])))
    phase_center = int(np.argmax(np.abs(observed_phase - fitted_phase)))
    phase_x0 = max(0.05, float(freq_thz[max(0, phase_center - 12)]))
    phase_x1 = min(3.0, float(freq_thz[min(freq_thz.size - 1, phase_center + 12)]))
    _add_zoom_inset(
        axes[1, 1],
        freq_thz,
        observed_phase,
        fitted_phase,
        colors=("#0f4c81", "#d1495b"),
        labels="largest phase mismatch",
        loc="lower left",
        bbox_to_anchor=(0.10, 0.13, 0.38, 0.36),
        center_index=phase_center,
        half_window=12,
        y_limits=(-200.0, 5.0),
        x_limits=(phase_x0, phase_x1),
    )

    for axis in axes.flat:
        _apply_ruler_ticks(axis)
        axis.grid(True, which="major", alpha=0.18, linestyle="-", linewidth=0.6)
        axis.grid(True, which="minor", alpha=0.08, linestyle="-", linewidth=0.35)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.20, hspace=0.28)
    return fig


def _plot_reflection_self_reference_overview(
    result,
    *,
    title: str,
    reference_label: str = "Extracted first-peak reference",
    target_label: str = "Observed reflection target",
    reference_panel_title: str = "Self-reference construction",
    inset_label: str = "first-peak window",
):
    prepared = result.prepared_traces
    fit_result = result.fit_result
    reference = np.asarray(prepared.processed_reference.trace, dtype=np.float64)
    observed = np.asarray(prepared.processed_sample.trace, dtype=np.float64)
    fitted = np.asarray(fit_result["fitted_simulation"]["sample_trace"], dtype=np.float64)
    residual = np.asarray(fit_result["residual_trace"], dtype=np.float64)
    time_ps = np.asarray(prepared.processed_sample.time_ps, dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    axes[0, 0].plot(time_ps, reference, linewidth=1.75, color="#5b84b1", label=reference_label)
    axes[0, 0].plot(time_ps, observed, linewidth=1.45, color="#d1495b", label=target_label)
    axes[0, 0].set_title(reference_panel_title)
    axes[0, 0].set_xlabel(r"$t$ (ps)")
    axes[0, 0].set_ylabel(r"$E(t)$")
    axes[0, 0].legend(loc="upper left")
    ref_peak_index = int(np.argmax(np.abs(reference)))
    _add_zoom_inset(
        axes[0, 0],
        time_ps,
        reference,
        observed,
        colors=("#5b84b1", "#d1495b"),
        labels=inset_label,
        loc="upper right",
        bbox_to_anchor=(0.57, 0.54, 0.36, 0.33),
        center_index=ref_peak_index,
        half_window=14,
    )

    axes[0, 1].plot(time_ps, observed, linewidth=1.8, color="#0f4c81", label="Observed")
    axes[0, 1].plot(time_ps, fitted, linewidth=1.45, color="#d1495b", label="Fit")
    axes[0, 1].set_title("Reflection fit")
    axes[0, 1].set_xlabel(r"$t$ (ps)")
    axes[0, 1].set_ylabel(r"$E(t)$")
    axes[0, 1].legend(loc="upper left")
    refl_peak_index = int(np.argmax(np.abs(observed)))
    _add_zoom_inset(
        axes[0, 1],
        time_ps,
        observed,
        fitted,
        colors=("#0f4c81", "#d1495b"),
        labels="main peak zoom",
        loc="upper right",
        bbox_to_anchor=(0.57, 0.54, 0.36, 0.33),
        center_index=refl_peak_index,
        half_window=14,
    )

    axes[1, 0].plot(time_ps, residual, linewidth=1.4, color="#3b7d3a")
    axes[1, 0].axhline(0.0, color="#555555", linewidth=0.9, alpha=0.7)
    axes[1, 0].set_title("Residual trace")
    axes[1, 0].set_xlabel(r"$t$ (ps)")
    axes[1, 0].set_ylabel(r"$r(t)=E_{\mathrm{fit}}(t)-E_{\mathrm{data}}(t)$")
    residual_peak_index = int(np.argmax(np.abs(residual)))
    axes[1, 0].scatter([time_ps[residual_peak_index]], [residual[residual_peak_index]], color="#b22222", zorder=3)
    axes[1, 0].annotate(
        f"{time_ps[residual_peak_index]:.2f} ps",
        xy=(time_ps[residual_peak_index], residual[residual_peak_index]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    freq_thz, observed_spec = _positive_spectrum(observed, time_ps)
    _, fitted_spec = _positive_spectrum(fitted, time_ps)
    observed_amp = _relative_db(np.abs(observed_spec))
    fitted_amp = _relative_db(np.abs(fitted_spec))
    axes[1, 1].plot(freq_thz, observed_amp, linewidth=1.8, color="#0f4c81", label="Observed")
    axes[1, 1].plot(freq_thz, fitted_amp, linewidth=1.45, color="#d1495b", label="Fit")
    axes[1, 1].set_title("Spectral amplitude")
    axes[1, 1].set_xlabel(r"$f$ (THz)")
    axes[1, 1].set_ylabel("Amplitude (dB)")
    axes[1, 1].set_xlim(0.05, min(3.0, float(freq_thz[-1])))
    axes[1, 1].legend()
    amp_center = int(np.argmax(np.abs(observed_amp - fitted_amp)))
    amp_x0 = max(0.05, float(freq_thz[max(0, amp_center - 12)]))
    amp_x1 = min(3.0, float(freq_thz[min(freq_thz.size - 1, amp_center + 12)]))
    _add_zoom_inset(
        axes[1, 1],
        freq_thz,
        observed_amp,
        fitted_amp,
        colors=("#0f4c81", "#d1495b"),
        labels="largest amplitude mismatch",
        loc="lower left",
        bbox_to_anchor=(0.10, 0.12, 0.38, 0.36),
        center_index=amp_center,
        half_window=12,
        y_limits=(-60.0, 2.0),
        x_limits=(amp_x0, amp_x1),
    )

    for axis in axes.flat:
        _apply_ruler_ticks(axis)
        axis.grid(True, which="major", alpha=0.18, linestyle="-", linewidth=0.6)
        axis.grid(True, which="minor", alpha=0.08, linestyle="-", linewidth=0.35)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.90, wspace=0.20, hspace=0.28)
    return fig


def run_lecture_measured_transmission_example(
    output_root: Path | None = None,
    *,
    profile: str = "full",
):
    profile_settings = _profile_settings(profile)
    if output_root is None:
        output_root = LECTURE_BUILD_ROOT / "notebook_measured_transmission"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    data_dir = output_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    transmission_root = REPO_ROOT / "Test_data_for_fitter" / "A11013460_transmission"
    prepared_tx = prepare_trace_pair_for_fit(
        transmission_root / "REFERENCE.csv",
        transmission_root / "SAMPLE1.csv",
        baseline_mode="auto_pre_pulse",
        baseline_window_samples=40,
        crop_mode="auto",
    )
    staged = run_staged_measured_fit(
        prepared_tx,
        _measured_fit_layers(),
        out_dir=output_root / "runs" / "measured_a11013460_tx",
        measurement=Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            reference_standard=ReferenceStandard(kind="ambient_replacement"),
        ),
        weighting={"mode": "trace_amplitude", "floor": 0.03, "power": 2.0, "smooth_window_samples": 41},
        delay_options={"enabled": True, "search_window_ps": 20.0},
        reflection_counts=profile_settings["fit_reflection_counts"],
    )
    tx_fit = staged["best_fit_result"]
    tx_overview = _plot_measured_fit_overview(prepared_tx, tx_fit, title="Measured transmission fit: A11013460")
    tx_overview_paths = _save_figure(tx_overview, figures_dir / "measured_transmission_overview")
    tx_corr = _plot_correlation_matrix(
        _fit_correlation_rows(0, tx_fit),
        title="Measured transmission fit: local parameter correlation",
    )
    tx_corr_paths = _save_figure(tx_corr, figures_dir / "measured_transmission_correlation")
    summary_path = data_dir / "measured_transmission_summary.json"
    _json_dump(
        summary_path,
        {
            "dataset": "A11013460_transmission",
            "sample": "SAMPLE1.csv",
            "selection_reason": str(staged["selection_reason"]),
            "selection_score": float(staged["selection_score"]),
            "residual_metrics": {
                key: float(value) for key, value in tx_fit["residual_metrics"].items() if np.isfinite(float(value))
            },
            "recovered_parameters": {key: float(value) for key, value in tx_fit["recovered_parameters"].items()},
        },
    )
    return {
        "figure_overview_png": tx_overview_paths["png"],
        "figure_overview_pdf": tx_overview_paths["pdf"],
        "figure_correlation_png": tx_corr_paths["png"],
        "figure_correlation_pdf": tx_corr_paths["pdf"],
        "summary_json": summary_path.resolve().as_posix(),
        "output_root": output_root.resolve().as_posix(),
    }


def run_lecture_measured_reflection_example(
    output_root: Path | None = None,
    *,
    profile: str = "full",
):
    profile_settings = _profile_settings(profile)
    if output_root is None:
        output_root = LECTURE_BUILD_ROOT / "notebook_measured_reflection"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    data_dir = output_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    reflection_reference_path = (
        REPO_ROOT
        / "Test_data_for_fitter"
        / "A11013460_reflection"
        / "reflection_setup_ref_after_with_AuMirror_A11013460_avg600_onDryAir10min_int56.csv"
    )
    reflection_sample_path = (
        REPO_ROOT
        / "Test_data_for_fitter"
        / "A11013460_reflection"
        / "reflection_setup_sample_A11013460_avg600_onDryAir10min_int30.csv"
    )
    prepared_refl = prepare_trace_pair_for_fit(
        reflection_reference_path,
        reflection_sample_path,
        baseline_mode="auto_pre_pulse",
        baseline_window_samples=40,
        crop_mode="auto",
    )
    reflection_result = run_measured_fit(
        prepared_refl,
        _measured_reflection_layers(),
        out_dir=output_root / "runs" / "measured_a11013460_reflection",
        measurement=Measurement(
            mode="reflection",
            angle_deg=Fit(10.0, abs_min=0.0, abs_max=45.0, label="measurement_angle_deg"),
            polarization="mixed",
            polarization_mix=Fit(0.5, abs_min=0.0, abs_max=1.0, label="measurement_polarization_mix"),
            trace_scale=Fit(-1.0, abs_min=-2.5, abs_max=0.5, label="measurement_trace_scale"),
            trace_offset=Fit(0.0, abs_min=-1.0, abs_max=1.0, label="measurement_trace_offset"),
            reference_standard={"kind": "identity"},
        ),
        optimizer={
            "method": "L-BFGS-B",
            "options": {"maxiter": 180 if profile_settings["profile"] == "full" else 140},
            "global_options": {"maxiter": 10, "popsize": 10, "seed": 123},
            "fd_rel_step": 1e-5,
        },
        max_internal_reflections=2,
        delay_options={"enabled": True, "search_window_ps": 20.0, "initial_ps": 0.0},
        weighting={"mode": "trace_amplitude", "floor": 0.03, "power": 2.0, "smooth_window_samples": 41},
        metric="weighted_data_fit",
    )
    refl_overview = _plot_reflection_self_reference_overview(
        reflection_result,
        title="Measured reflection fit: A11013460 mirror-reference pair",
        reference_label="Measured Au-mirror reference",
        target_label="Observed wafer reflection",
        reference_panel_title="Mirror-reference pair",
        inset_label="main reference pulse",
    )
    refl_overview_paths = _save_figure(refl_overview, figures_dir / "measured_reflection_overview")
    refl_corr = _plot_correlation_matrix(
        _fit_correlation_rows(0, reflection_result.fit_result),
        title="Measured reflection fit: local parameter correlation",
    )
    refl_corr_paths = _save_figure(refl_corr, figures_dir / "measured_reflection_correlation")
    summary_path = data_dir / "measured_reflection_summary.json"
    _json_dump(
        summary_path,
        {
            "dataset": "A11013460_reflection",
            "reference": reflection_reference_path.name,
            "sample": reflection_sample_path.name,
            "construction": "mirror_reference_pair",
            "prepared_metadata": prepared_refl.metadata,
            "residual_metrics": {
                key: float(value)
                for key, value in reflection_result.fit_result["residual_metrics"].items()
                if np.isfinite(float(value))
            },
            "max_abs_residual": float(np.max(np.abs(reflection_result.fit_result["residual_trace"]))),
            "fitted_measurement": {
                key: (float(value) if value is not None else None)
                for key, value in reflection_result.fit_result["fitted_measurement"].items()
                if key in {"angle_deg", "polarization_mix", "trace_scale", "trace_offset"}
            },
            "recovered_parameters": {
                key: float(value) for key, value in reflection_result.fit_result["recovered_parameters"].items()
            },
        },
    )
    return {
        "figure_overview_png": refl_overview_paths["png"],
        "figure_overview_pdf": refl_overview_paths["pdf"],
        "figure_correlation_png": refl_corr_paths["png"],
        "figure_correlation_pdf": refl_corr_paths["pdf"],
        "summary_json": summary_path.resolve().as_posix(),
        "output_root": output_root.resolve().as_posix(),
    }


def _run_measured_examples(build_root: Path, profile_settings: dict):
    figures_dir = build_root / "figures"
    data_dir = build_root / "data" / "measured_examples"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tx_result = run_lecture_measured_transmission_example(
        build_root / "measured_transmission",
        profile=profile_settings["profile"],
    )
    refl_result = run_lecture_measured_reflection_example(
        build_root / "measured_reflection",
        profile=profile_settings["profile"],
    )
    shutil.copy2(tx_result["figure_overview_png"], figures_dir / "fit_a11013460_overview.png")
    shutil.copy2(tx_result["figure_overview_pdf"], figures_dir / "fit_a11013460_overview.pdf")
    shutil.copy2(tx_result["figure_correlation_png"], figures_dir / "fit_a11013460_correlation.png")
    shutil.copy2(tx_result["figure_correlation_pdf"], figures_dir / "fit_a11013460_correlation.pdf")
    shutil.copy2(refl_result["figure_overview_png"], figures_dir / "fit_a11013460_reflection_overview.png")
    shutil.copy2(refl_result["figure_overview_pdf"], figures_dir / "fit_a11013460_reflection_overview.pdf")
    shutil.copy2(refl_result["figure_correlation_png"], figures_dir / "fit_a11013460_reflection_correlation.png")
    shutil.copy2(refl_result["figure_correlation_pdf"], figures_dir / "fit_a11013460_reflection_correlation.pdf")
    shutil.copy2(tx_result["summary_json"], data_dir / "measured_transmission_summary.json")
    shutil.copy2(refl_result["summary_json"], data_dir / "measured_reflection_summary.json")

    return {
        "fit_a11013460_overview": {
            "png": (figures_dir / "fit_a11013460_overview.png").resolve().as_posix(),
            "pdf": (figures_dir / "fit_a11013460_overview.pdf").resolve().as_posix(),
        },
        "fit_a11013460_correlation": {
            "png": (figures_dir / "fit_a11013460_correlation.png").resolve().as_posix(),
            "pdf": (figures_dir / "fit_a11013460_correlation.pdf").resolve().as_posix(),
        },
        "fit_a11013460_reflection_overview": {
            "png": (figures_dir / "fit_a11013460_reflection_overview.png").resolve().as_posix(),
            "pdf": (figures_dir / "fit_a11013460_reflection_overview.pdf").resolve().as_posix(),
        },
        "fit_a11013460_reflection_correlation": {
            "png": (figures_dir / "fit_a11013460_reflection_correlation.png").resolve().as_posix(),
            "pdf": (figures_dir / "fit_a11013460_reflection_correlation.pdf").resolve().as_posix(),
        },
    }


def _one_layer_map_specs(*, profile: str):
    grid_count = 25 if profile == "full" else 6
    full = profile == "full"
    specs = [
        {
            "slug": "one_layer_tx_tau_sigma_low_s_0deg",
            "title": "One-layer Drude transmission: $\\tau$ vs $\\sigma$ (0 deg, s-pol, $d=525\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 0.0,
            "polarization": "s",
            "tau_range": (0.1, 1.0),
            "sigma_range": (0.01, 1.0),
            "thickness_um": 525.0,
        },
        {
            "slug": "one_layer_tx_tau_sigma_low_s_45deg",
            "title": "One-layer Drude transmission: $\\tau$ vs $\\sigma$ (45 deg, s-pol, $d=525\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 45.0,
            "polarization": "s",
            "tau_range": (0.1, 1.0),
            "sigma_range": (0.01, 1.0),
            "thickness_um": 525.0,
        },
        {
            "slug": "one_layer_tx_tau_sigma_high_s_0deg",
            "title": "One-layer Drude transmission: $\\tau$ vs $\\sigma$ (0 deg, s-pol, $d=725\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 0.0,
            "polarization": "s",
            "tau_range": (0.1, 1.0),
            "sigma_range": (100.0, 1000.0),
            "thickness_um": 725.0,
        },
        {
            "slug": "one_layer_tx_tau_sigma_high_p_45deg",
            "title": "One-layer Drude transmission: $\\tau$ vs $\\sigma$ (45 deg, p-pol, $d=725\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 45.0,
            "polarization": "p",
            "tau_range": (0.1, 1.0),
            "sigma_range": (100.0, 1000.0),
            "thickness_um": 725.0,
        },
        {
            "slug": "one_layer_refl_tau_sigma_low_s_45deg",
            "title": "One-layer Drude reflection: $\\tau$ vs $\\sigma$ (45 deg, s-pol, $d=525\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 45.0,
            "polarization": "s",
            "tau_range": (0.1, 1.0),
            "sigma_range": (0.01, 1.0),
            "thickness_um": 525.0,
        },
        {
            "slug": "one_layer_refl_tau_sigma_low_s_60deg",
            "title": "One-layer Drude reflection: $\\tau$ vs $\\sigma$ (60 deg, s-pol, $d=525\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 60.0,
            "polarization": "s",
            "tau_range": (0.1, 1.0),
            "sigma_range": (0.01, 1.0),
            "thickness_um": 525.0,
        },
        {
            "slug": "one_layer_refl_tau_sigma_high_p_45deg",
            "title": "One-layer Drude reflection: $\\tau$ vs $\\sigma$ (45 deg, p-pol, $d=725\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 45.0,
            "polarization": "p",
            "tau_range": (0.1, 1.0),
            "sigma_range": (100.0, 1000.0),
            "thickness_um": 725.0,
        },
        {
            "slug": "one_layer_refl_tau_sigma_high_p_60deg",
            "title": "One-layer Drude reflection: $\\tau$ vs $\\sigma$ (60 deg, p-pol, $d=725\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 60.0,
            "polarization": "p",
            "tau_range": (0.1, 1.0),
            "sigma_range": (100.0, 1000.0),
            "thickness_um": 725.0,
        },
    ]
    if not full:
        specs = specs[:2]
    for spec in specs:
        spec["tau_values"] = np.linspace(spec["tau_range"][0], spec["tau_range"][1], grid_count).tolist()
        spec["sigma_values"] = np.linspace(spec["sigma_range"][0], spec["sigma_range"][1], grid_count).tolist()
    return specs


def _run_one_layer_studies(build_root: Path, profile_settings: dict):
    reference_result = _prepare_reference_from_config(
        _reference_input_config(),
        output_root=build_root / "runs",
        run_label="lecture-one-layer-reference",
    )
    optimizer = dict(profile_settings["study_optimizer"])
    weighting = {"mode": "trace_amplitude", "floor": 0.03, "power": 2.0, "smooth_window_samples": 41}
    results = {}

    for spec_config in _one_layer_map_specs(profile=profile_settings["profile"]):
        sigma_values = list(spec_config["sigma_values"])
        tau_values = list(spec_config["tau_values"])
        nominal_sigma = float(np.sqrt(min(sigma_values) * max(sigma_values)))
        nominal_tau = 0.316
        sample_result = _one_layer_fit_sample(
            reference_result,
            thickness_um=float(spec_config["thickness_um"]),
            sigma_nominal=nominal_sigma,
            tau_nominal=nominal_tau,
            output_dir=build_root / "runs" / spec_config["slug"],
        )
        measurement = Measurement(
            mode=spec_config["mode"],
            angle_deg=float(spec_config["angle_deg"]),
            polarization=str(spec_config["polarization"]),
            reference_standard=ReferenceStandard(kind="identity"),
        )

        def truth_builder(tau_ps, sigma_s_per_m, *, thickness_um=float(spec_config["thickness_um"])):
            return (
                {
                    "layers[0].thickness_um": thickness_um,
                    "layers[0].material.plasma_freq_thz": drude_plasma_freq_thz_from_sigma_tau(sigma_s_per_m, tau_ps),
                    "layers[0].material.gamma_thz": drude_gamma_thz_from_tau_ps(tau_ps),
                },
                {
                    "thickness_um": thickness_um,
                    "eps_inf": 11.7,
                    "tau_ps": tau_ps,
                    "sigma_s_per_m": sigma_s_per_m,
                },
            )

        study_spec = LectureGridStudySpec(
            slug=str(spec_config["slug"]),
            title=str(spec_config["title"]),
            section="one_layer",
            x_name="tau_ps",
            y_name="sigma_s_per_m",
            x_values=tau_values,
            y_values=sigma_values,
            x_label=r"$\tau$ (ps)",
            y_label=r"$\sigma$ (S/m)",
            measurement=measurement,
            noise_dynamic_range_db=float(profile_settings["noise_db"]),
            max_internal_reflections=8,
            recovery_keys=("tau_ps", "sigma_s_per_m"),
            fixed_note=rf"$\varepsilon_\infty = 11.7$, $d = {float(spec_config['thickness_um']):.1f}\,\mu\mathrm{{m}}$",
            truth_update_builder=truth_builder,
            fit_summary_builder=lambda stack: summarize_single_layer_drude_stack(stack),
            spec_payload={
                "slug": spec_config["slug"],
                "title": spec_config["title"],
                "measurement": _measurement_record(measurement),
                "x_values": tau_values,
                "y_values": sigma_values,
                "fixed_parameters": {"thickness_um": float(spec_config["thickness_um"]), "eps_inf": 11.7},
                "noise_dynamic_range_db": float(profile_settings["noise_db"]),
                "optimizer": optimizer,
                "weighting": weighting,
                "reference_input": _reference_input_config(),
            },
        )
        result = _run_saved_grid_study(
            build_root=build_root,
            reference_result=reference_result,
            sample_result=sample_result,
            spec=study_spec,
            optimizer=optimizer,
            weighting=weighting,
        )
        results[study_spec.slug] = result
    return results


def _advanced_map_specs(*, profile: str):
    grid_count = 25 if profile == "full" else 5
    full = profile == "full"
    specs = [
        {
            "slug": "advanced_tx_tau1_tau2_0deg_s_525um",
            "title": "Two-Drude transmission: $\\tau_1$ vs $\\tau_2$ (0 deg, s-pol, $d_{\\mathrm{sub}}=525\\,\\mu$m, $d_{\\mathrm{epi}}=10\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 0.0,
            "polarization": "s",
            "map_kind": "tau",
            "substrate_thickness_um": 525.0,
            "epi_thickness_um": 10.0,
            "tau_range": (0.1, 1.0),
            "sigma_fixed": 0.1,
        },
        {
            "slug": "advanced_tx_sigma1_sigma2_0deg_s_525um",
            "title": "Two-Drude transmission: $\\sigma_1$ vs $\\sigma_2$ (0 deg, s-pol, $d_{\\mathrm{sub}}=525\\,\\mu$m, $d_{\\mathrm{epi}}=10\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 0.0,
            "polarization": "s",
            "map_kind": "sigma",
            "substrate_thickness_um": 525.0,
            "epi_thickness_um": 10.0,
            "sigma_range": (0.01, 1.0),
            "tau_fixed": 0.316,
        },
        {
            "slug": "advanced_tx_tau1_tau2_45deg_s_725um",
            "title": "Two-Drude transmission: $\\tau_1$ vs $\\tau_2$ (45 deg, s-pol, $d_{\\mathrm{sub}}=725\\,\\mu$m, $d_{\\mathrm{epi}}=50\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 45.0,
            "polarization": "s",
            "map_kind": "tau",
            "substrate_thickness_um": 725.0,
            "epi_thickness_um": 50.0,
            "tau_range": (0.1, 1.0),
            "sigma_fixed": 316.2,
        },
        {
            "slug": "advanced_tx_sigma1_sigma2_45deg_s_725um",
            "title": "Two-Drude transmission: $\\sigma_1$ vs $\\sigma_2$ (45 deg, s-pol, $d_{\\mathrm{sub}}=725\\,\\mu$m, $d_{\\mathrm{epi}}=50\\,\\mu$m)",
            "mode": "transmission",
            "angle_deg": 45.0,
            "polarization": "s",
            "map_kind": "sigma",
            "substrate_thickness_um": 725.0,
            "epi_thickness_um": 50.0,
            "sigma_range": (100.0, 1000.0),
            "tau_fixed": 0.316,
        },
        {
            "slug": "advanced_refl_tau1_tau2_45deg_s_525um",
            "title": "Two-Drude reflection: $\\tau_1$ vs $\\tau_2$ (45 deg, s-pol, $d_{\\mathrm{sub}}=525\\,\\mu$m, $d_{\\mathrm{epi}}=10\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 45.0,
            "polarization": "s",
            "map_kind": "tau",
            "substrate_thickness_um": 525.0,
            "epi_thickness_um": 10.0,
            "tau_range": (0.1, 1.0),
            "sigma_fixed": 0.1,
        },
        {
            "slug": "advanced_refl_sigma1_sigma2_45deg_s_525um",
            "title": "Two-Drude reflection: $\\sigma_1$ vs $\\sigma_2$ (45 deg, s-pol, $d_{\\mathrm{sub}}=525\\,\\mu$m, $d_{\\mathrm{epi}}=10\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 45.0,
            "polarization": "s",
            "map_kind": "sigma",
            "substrate_thickness_um": 525.0,
            "epi_thickness_um": 10.0,
            "sigma_range": (0.01, 1.0),
            "tau_fixed": 0.316,
        },
        {
            "slug": "advanced_refl_tau1_tau2_60deg_p_725um",
            "title": "Two-Drude reflection: $\\tau_1$ vs $\\tau_2$ (60 deg, p-pol, $d_{\\mathrm{sub}}=725\\,\\mu$m, $d_{\\mathrm{epi}}=50\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 60.0,
            "polarization": "p",
            "map_kind": "tau",
            "substrate_thickness_um": 725.0,
            "epi_thickness_um": 50.0,
            "tau_range": (0.1, 1.0),
            "sigma_fixed": 316.2,
        },
        {
            "slug": "advanced_refl_sigma1_sigma2_60deg_s_725um",
            "title": "Two-Drude reflection: $\\sigma_1$ vs $\\sigma_2$ (60 deg, s-pol, $d_{\\mathrm{sub}}=725\\,\\mu$m, $d_{\\mathrm{epi}}=50\\,\\mu$m)",
            "mode": "reflection",
            "angle_deg": 60.0,
            "polarization": "s",
            "map_kind": "sigma",
            "substrate_thickness_um": 725.0,
            "epi_thickness_um": 50.0,
            "sigma_range": (100.0, 1000.0),
            "tau_fixed": 0.316,
        },
    ]
    if not full:
        specs = specs[:2]
    for spec in specs:
        if spec["map_kind"] == "tau":
            values = np.linspace(spec["tau_range"][0], spec["tau_range"][1], grid_count).tolist()
            spec["x_values"] = values
            spec["y_values"] = values
        else:
            values = np.linspace(spec["sigma_range"][0], spec["sigma_range"][1], grid_count).tolist()
            spec["x_values"] = values
            spec["y_values"] = values
    return specs


def _run_advanced_studies(build_root: Path, profile_settings: dict):
    reference_result = _prepare_reference_from_config(
        _reference_input_config(),
        output_root=build_root / "runs",
        run_label="lecture-advanced-reference",
    )
    optimizer = dict(profile_settings["study_optimizer"])
    weighting = {"mode": "trace_amplitude", "floor": 0.03, "power": 2.0, "smooth_window_samples": 41}
    results = {}

    for spec_config in _advanced_map_specs(profile=profile_settings["profile"]):
        substrate_thickness_um = float(spec_config["substrate_thickness_um"])
        epi_thickness_um = float(spec_config["epi_thickness_um"])
        oxide_thickness_um = 10.0
        if spec_config["map_kind"] == "tau":
            tau_nominal = 0.316
            sigma_nominal = float(spec_config["sigma_fixed"])
            sample_result = _advanced_fit_sample(
                reference_result,
                epi_thickness_um=epi_thickness_um,
                substrate_thickness_um=substrate_thickness_um,
                oxide_thickness_um=oxide_thickness_um,
                tau1_nominal=tau_nominal,
                sigma1_nominal=sigma_nominal,
                tau2_nominal=tau_nominal,
                sigma2_nominal=sigma_nominal,
                output_dir=build_root / "runs" / spec_config["slug"],
            )

            def truth_builder(value_x, value_y, *, sigma_fixed=sigma_nominal):
                return (
                    {
                        "layers[0].thickness_um": epi_thickness_um,
                        "layers[0].material.plasma_freq1_thz": drude_plasma_freq_thz_from_sigma_tau(sigma_fixed, value_x),
                        "layers[0].material.gamma1_thz": drude_gamma_thz_from_tau_ps(value_x),
                        "layers[0].material.plasma_freq2_thz": drude_plasma_freq_thz_from_sigma_tau(sigma_fixed, value_y),
                        "layers[0].material.gamma2_thz": drude_gamma_thz_from_tau_ps(value_y),
                        "layers[1].thickness_um": oxide_thickness_um,
                        "layers[2].thickness_um": substrate_thickness_um,
                    },
                    {
                        "epi_thickness_um": epi_thickness_um,
                        "oxide_thickness_um": oxide_thickness_um,
                        "substrate_thickness_um": substrate_thickness_um,
                        "eps_inf": 11.7,
                        "tau1_ps": value_x,
                        "sigma1_s_per_m": sigma_fixed,
                        "tau2_ps": value_y,
                        "sigma2_s_per_m": sigma_fixed,
                    },
                )

            x_name = "tau1_ps"
            y_name = "tau2_ps"
            x_label = r"$\tau_1$ (ps)"
            y_label = r"$\tau_2$ (ps)"
            recovery_keys = ("tau1_ps", "tau2_ps")
        else:
            tau_fixed = float(spec_config["tau_fixed"])
            sigma_nominal = math.sqrt(min(spec_config["x_values"]) * max(spec_config["x_values"]))
            sample_result = _advanced_fit_sample(
                reference_result,
                epi_thickness_um=epi_thickness_um,
                substrate_thickness_um=substrate_thickness_um,
                oxide_thickness_um=oxide_thickness_um,
                tau1_nominal=tau_fixed,
                sigma1_nominal=sigma_nominal,
                tau2_nominal=tau_fixed,
                sigma2_nominal=sigma_nominal,
                output_dir=build_root / "runs" / spec_config["slug"],
            )

            def truth_builder(value_x, value_y, *, tau_fixed=tau_fixed):
                return (
                    {
                        "layers[0].thickness_um": epi_thickness_um,
                        "layers[0].material.plasma_freq1_thz": drude_plasma_freq_thz_from_sigma_tau(value_x, tau_fixed),
                        "layers[0].material.gamma1_thz": drude_gamma_thz_from_tau_ps(tau_fixed),
                        "layers[0].material.plasma_freq2_thz": drude_plasma_freq_thz_from_sigma_tau(value_y, tau_fixed),
                        "layers[0].material.gamma2_thz": drude_gamma_thz_from_tau_ps(tau_fixed),
                        "layers[1].thickness_um": oxide_thickness_um,
                        "layers[2].thickness_um": substrate_thickness_um,
                    },
                    {
                        "epi_thickness_um": epi_thickness_um,
                        "oxide_thickness_um": oxide_thickness_um,
                        "substrate_thickness_um": substrate_thickness_um,
                        "eps_inf": 11.7,
                        "tau1_ps": tau_fixed,
                        "sigma1_s_per_m": value_x,
                        "tau2_ps": tau_fixed,
                        "sigma2_s_per_m": value_y,
                    },
                )

            x_name = "sigma1_s_per_m"
            y_name = "sigma2_s_per_m"
            x_label = r"$\sigma_1$ (S/m)"
            y_label = r"$\sigma_2$ (S/m)"
            recovery_keys = ("sigma1_s_per_m", "sigma2_s_per_m")

        if spec_config["mode"] == "reflection":
            standard_sample = _advanced_reflection_standard(
                reference_result,
                substrate_thickness_um=substrate_thickness_um,
                oxide_thickness_um=oxide_thickness_um,
                output_dir=build_root / "runs" / f"{spec_config['slug']}_standard",
            )
            reference_standard = ReferenceStandard(kind="stack", stack=standard_sample)
        else:
            reference_standard = ReferenceStandard(kind="identity")

        measurement = Measurement(
            mode=str(spec_config["mode"]),
            angle_deg=float(spec_config["angle_deg"]),
            polarization=str(spec_config["polarization"]),
            reference_standard=reference_standard,
        )
        study_spec = LectureGridStudySpec(
            slug=str(spec_config["slug"]),
            title=str(spec_config["title"]),
            section="advanced",
            x_name=x_name,
            y_name=y_name,
            x_values=list(spec_config["x_values"]),
            y_values=list(spec_config["y_values"]),
            x_label=x_label,
            y_label=y_label,
            measurement=measurement,
            noise_dynamic_range_db=float(profile_settings["noise_db"]),
            max_internal_reflections=4,
            recovery_keys=recovery_keys,
            fixed_note=(
                rf"$d_{{\mathrm{{epi}}}}={epi_thickness_um:.1f}\,\mu\mathrm{{m}}$, "
                rf"$d_{{\mathrm{{ox}}}}={oxide_thickness_um:.1f}\,\mu\mathrm{{m}}$, "
                rf"$d_{{\mathrm{{sub}}}}={substrate_thickness_um:.1f}\,\mu\mathrm{{m}}$"
            ),
            truth_update_builder=truth_builder,
            fit_summary_builder=_summarize_advanced_stack,
            spec_payload={
                "slug": spec_config["slug"],
                "title": spec_config["title"],
                "measurement": _measurement_record(measurement),
                "x_values": list(spec_config["x_values"]),
                "y_values": list(spec_config["y_values"]),
                "fixed_parameters": {
                    "epi_thickness_um": epi_thickness_um,
                    "oxide_thickness_um": oxide_thickness_um,
                    "substrate_thickness_um": substrate_thickness_um,
                    "eps_inf": 11.7,
                },
                "noise_dynamic_range_db": float(profile_settings["noise_db"]),
                "optimizer": optimizer,
                "weighting": weighting,
                "reference_input": _reference_input_config(),
            },
        )
        result = _run_saved_grid_study(
            build_root=build_root,
            reference_result=reference_result,
            sample_result=sample_result,
            spec=study_spec,
            optimizer=optimizer,
            weighting=weighting,
        )
        results[study_spec.slug] = result
    return results


def _notes_tex_source(manifest: dict, *, graphics_root: str) -> str:
    one_layer_slugs = manifest["study_groups"]["one_layer"]
    advanced_slugs = manifest["study_groups"]["advanced"]
    measured_tx = "fit_a11013460_overview"
    measured_tx_corr = "fit_a11013460_correlation"
    measured_refl = "fit_a11013460_reflection_overview"
    measured_refl_corr = "fit_a11013460_reflection_correlation"

    def include_triptych(slug: str) -> str:
        return rf"""
\begin{{figure}}[htbp]
\centering
\includegraphics[width=\linewidth]{{{slug}.pdf}}
\caption{{{manifest["key_figure_titles"][slug]}. Left: linear recovery-confidence map. Middle: logarithmic recovery-confidence map. Right: Fisher-$z$ averaged local fit-correlation matrix over the whole map.}}
\label{{fig:{slug}}}
\end{{figure}}
"""

    bibliography = r"""
\begin{thebibliography}{9}
\bibitem{ellipsometry_intro}
P. Dean \emph{et al.}, ``An introduction to terahertz time-domain spectroscopic ellipsometry,'' 2022.

\bibitem{gan_ellipsometry}
N. Tayama \emph{et al.}, ``Terahertz time-domain ellipsometry with high precision for the evaluation of GaN crystals with carrier densities up to $10^{20}\,\mathrm{cm}^{-3}$,'' \emph{Scientific Reports}, 2021.

\bibitem{paint_ellipsometer}
F. Gentele \emph{et al.}, ``THz Time-Domain Ellipsometer for Material Characterization and Paint Quality Control with More Than 5 THz Bandwidth,'' \emph{Applied Sciences}, 2022.

\bibitem{photoresist_2025}
H. Zhang \emph{et al.}, ``In-situ non-contact monitoring of photoresist thickness and degree of cure using terahertz time-domain spectroscopy,'' \emph{NDT \& E International}, 2025.

\bibitem{fraunhofer_semicon}
Fraunhofer ITWM, ``Activity Report 2024/2025,'' THz-SEMICON program summary, 2025.
\end{thebibliography}
"""

    return (rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb,mathtools}}
\usepackage{{siunitx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{float}}
\graphicspath{{{{{graphics_root}}}}}
\title{{Lecture Notes: THz-TDS Time-Domain Fit and Confidence-Map Studies}}
\author{{THzSim2 Lecture Package}}
\date{{\today}}
\begin{{document}}
\maketitle

\begin{{abstract}}
Terahertz time-domain spectroscopy (THz-TDS) has moved from a niche laboratory technique toward a serious wafer-metrology platform because it can probe thickness, conductivity, carrier scattering time, multilayer interfaces, and front-side reflection standards without contact or destructive sample preparation. A dated keyword survey on April 22, 2026 shows that the semiconductor-wafer, thin-film, and ellipsometry literature for THz-TDS is already on the order of hundreds of papers and application reports, with representative recent work spanning high-precision THz ellipsometry on semiconductors, in-situ photoresist monitoring, and dedicated industrial semiconductor inspection programs \cite{{ellipsometry_intro,gan_ellipsometry,photoresist_2025,fraunhofer_semicon}}. The end goal behind these notes is a practical THz-TDS ellipsometer: a tool that does not only record a waveform, but converts that waveform into trustworthy thickness, carrier, and multilayer parameters under realistic angle, polarization, and reflection-standard constraints.
\end{{abstract}}

\section{{Why Time-Domain Fitting and Study Matter}}
THz-TDS measures an electric-field transient rather than only a scalar spectrum. The waveform carries amplitude, phase, pulse arrival time, internal echoes, and angle-dependent interface information in one object. A time-domain fit therefore solves a stronger inverse problem than a workflow that first compresses the data into a transmission magnitude, a phase-unwrapped refractive-index curve, or an echo delay alone.

In these notes, \emph{{time-domain fitting}} means: choose a parametric optical stack, propagate the measured reference pulse through that stack, compare the predicted sample waveform with the observed sample waveform, and optimize the unknown parameters until the residual trace is minimized. The residual is not just a scalar score; it shows \emph{{where}} and \emph{{when}} the model still fails.

The \emph{{study}} workflow asks a different question. Instead of fitting one trace, it sweeps truth parameters, simulates a synthetic measurement, adds noise to that simulated trace, refits the noisy trace, and compares the recovered parameters to the truth. This is not a pure detector-sensitivity benchmark. It is a confidence study of the \emph{{model plus measurement technique}}. It answers: given this optical model, this angle/polarization setup, this reflection standard, and this noise level, how confidently can we believe that the fitted parameters are the right physical values?

If the synthetic sample trace were refit without added noise and with a complete forward model, the recovered parameters should come back almost exactly. The study becomes interesting only after noise is injected, because then the maps reveal which parameter combinations remain stable and which become ambiguous. That is why the lecture heatmaps are recovery-confidence maps, not only waveform-misfit maps.

\section{{Why Time-Domain Fit is Often Better}}
\begin{{itemize}}[leftmargin=2em]
\item It preserves waveform timing, so internal reflections and delayed echoes remain visible instead of being hidden inside a smooth spectral ratio.
\item It uses the measured reference pulse as the source term, so the forward model automatically carries the instrument bandwidth, pulse chirp, and finite spectral support.
\item It avoids a fragile intermediate phase-extraction step when the sample is thick, noisy, or strongly dispersive.
\item It handles transmission, reflection, oblique incidence, and explicit reference standards inside one forward model.
\end{{itemize}}

The limitations matter just as much:
\begin{{itemize}}[leftmargin=2em]
\item local parameter correlations can produce a visually good fit but still leave the solution non-unique,
\item preprocessing choices such as cropping and baseline subtraction move the fit,
\item reflection standards and normalization assumptions matter strongly,
\item a low objective value does not prove the physical model is complete.
\end{{itemize}}

\section{{Forward Model Theory}}
Let $E_{{\mathrm{{ref}}}}(t)$ be the processed reference waveform and let its Fourier transform be
\[
\hat{{E}}_{{\mathrm{{ref}}}}(\omega)=\int_{{-\infty}}^{{\infty}}E_{{\mathrm{{ref}}}}(t)e^{{-i\omega t}}\,dt.
\]
The predicted sample waveform is constructed in frequency domain through
\[
\hat{{E}}_{{\mathrm{{sam}}}}(\omega)=H(\omega)\,\hat{{E}}_{{\mathrm{{ref}}}}(\omega),
\]
where $H(\omega)$ is the complex transfer function of the multilayer under the selected measurement mode.

For layer $j$ with thickness $d_j$ and complex permittivity $\varepsilon_j(\omega)$, the code first forms the complex refractive index
\[
\tilde{{n}}_j(\omega)=\sqrt{{\varepsilon_j(\omega)}}.
\]
The physical branch is chosen so that $\operatorname{{Im}}\tilde{{n}}_j(\omega)\ge 0$, which enforces attenuation rather than unphysical gain in passive media.

At oblique incidence, the conserved transverse index is
\[
\tilde{{n}}_0\sin\theta_0,
\]
and the branch-consistent longitudinal cosine used by the code is
\[
\cos\theta_j=\sqrt{{1-\left(\frac{{\tilde{{n}}_0\sin\theta_0}}{{\tilde{{n}}_j}}\right)^2}}.
\]
The sign is selected so that the wave propagates or attenuates in the physically forward direction. The propagation factor through layer $j$ is then
\[
P_j(\omega)=\exp\!\left(i\omega \tilde{{n}}_j(\omega)\cos\theta_j\frac{{d_j}}{{c_0}}\right).
\]

For an interface from medium $j$ to medium $k$, the Fresnel coefficients are
\[
r_{{jk}}^s=\frac{{\tilde{{n}}_j\cos\theta_j-\tilde{{n}}_k\cos\theta_k}}{{\tilde{{n}}_j\cos\theta_j+\tilde{{n}}_k\cos\theta_k}}, \qquad
t_{{jk}}^s=\frac{{2\tilde{{n}}_j\cos\theta_j}}{{\tilde{{n}}_j\cos\theta_j+\tilde{{n}}_k\cos\theta_k}},
\]
for $s$ polarization, and
\[
r_{{jk}}^p=\frac{{\tilde{{n}}_k\cos\theta_j-\tilde{{n}}_j\cos\theta_k}}{{\tilde{{n}}_k\cos\theta_j+\tilde{{n}}_j\cos\theta_k}}, \qquad
t_{{jk}}^p=\frac{{2\tilde{{n}}_j\cos\theta_j}}{{\tilde{{n}}_k\cos\theta_j+\tilde{{n}}_j\cos\theta_k}},
\]
for $p$ polarization.

For a single finite layer between input medium $0$ and output medium $2$, the code keeps an explicit finite Fabry--Pérot sum. If the maximum number of internal round trips is $M$, then with
\[
z(\omega)=r_{{10}}(\omega)P_1(\omega)r_{{12}}(\omega)P_1(\omega),
\]
the transmission channel is
\[
T_M(\omega)=t_{{01}}(\omega)P_1(\omega)t_{{12}}(\omega)\sum_{{m=0}}^M z^m(\omega),
\]
and the reflection channel is
\[
R_M(\omega)=r_{{01}}(\omega)+t_{{01}}(\omega)P_1(\omega)r_{{12}}(\omega)P_1(\omega)t_{{10}}(\omega)\sum_{{m=0}}^{{M-1}} z^m(\omega).
\]
This is the exact finite-round-trip sum implemented in the single-layer fast path.

For an arbitrary isotropic stack, the code uses a recursive construction from the last interface back to the first. If $R_{{j+1}}$ and $T_{{j+1}}$ are the effective reflection and transmission responses of the part of the stack behind layer $j$, then with
\[
D_j(\omega)=1-r_{{j+1,j}}(\omega)P_j(\omega)R_{{j+1}}(\omega)P_j(\omega),
\]
the recursion is
\[
T_j(\omega)=\frac{{t_{{j,j+1}}(\omega)P_j(\omega)T_{{j+1}}(\omega)}}{{D_j(\omega)}},
\]
\[
R_j(\omega)=r_{{j,j+1}}(\omega)+\frac{{t_{{j,j+1}}(\omega)P_j(\omega)R_{{j+1}}(\omega)P_j(\omega)t_{{j+1,j}}(\omega)}}{{D_j(\omega)}}.
\]
Starting from the last interface and applying this recursion to every layer yields the effective stack transmission or reflection response seen by the incident reference pulse.

Transmission and reflection then differ only in the extracted channel and the normalization standard. If a reference-standard stack $H_{{\mathrm{{std}}}}(\omega)$ is provided, the physical measured transfer is
\[
H_{{\mathrm{{phys}}}}(\omega)=\frac{{H_{{\mathrm{{sample}}}}(\omega)}}{{H_{{\mathrm{{std}}}}(\omega)}}.
\]
The final predicted sample spectrum is
\[
\hat{{E}}_{{\mathrm{{sam}}}}(\omega)=H_{{\mathrm{{phys}}}}(\omega)\hat{{E}}_{{\mathrm{{ref}}}}(\omega),
\]
and the model waveform is obtained by inverse FFT back to time domain. This is exactly why reference-standard choices matter so strongly in reflection: the fitted observable is not the raw reflected pulse, but the sample response relative to the chosen standard.

\subsection{{Drude and Two-Drude Materials}}
For a one-carrier Drude medium,
\[
\varepsilon(\omega)=\varepsilon_\infty-\frac{{\omega_p^2}}{{\omega(\omega+i\gamma)}}.
\]
The lecture studies frequently use relaxation time $\tau$ and DC conductivity $\sigma$ instead of $(\omega_p,\gamma)$:
\[
\gamma = \tau^{{-1}}, \qquad \sigma=\varepsilon_0\frac{{\omega_p^2}}{{\gamma}}.
\]

For the advanced epi stack the lecture uses a two-carrier extension,
\[
\varepsilon(\omega)=\varepsilon_\infty
-\frac{{\omega_{{p1}}^2}}{{\omega(\omega+i\gamma_1)}}
-\frac{{\omega_{{p2}}^2}}{{\omega(\omega+i\gamma_2)}}.
\]
This allows the epi layer to represent two conduction channels with different scattering times and conductivities.

\section{{How the Time-Domain Fit Works}}
The actual implementation has four stages:
\begin{{enumerate}}[leftmargin=2em]
\item preprocess the measured or generated traces by baseline subtraction, overlap alignment, and cropping,
\item build a parameterized optical stack with bounded fit variables,
\item simulate the sample waveform by propagating the measured reference pulse through that stack,
\item optimize the stack parameters to minimize a time-domain objective.
\end{{enumerate}}

The default weighted waveform objective used for the lecture studies is
\[
{_WEIGHTED_OBJECTIVE_LABEL}.
\]
The weights are generated from a smoothed amplitude envelope of the observed trace. This keeps the main pulse and the strongest informative echoes important while preventing long flat tails from dominating the fit.

The measured transmission example also uses additional techniques that are important in practical THz fitting:
\begin{{itemize}}[leftmargin=2em]
\item bounded optimization in physically meaningful parameter ranges,
\item staged objectives that mix waveform and transfer-function mismatch,
\item delay recovery when the waveform alignment is not exact,
\item finite internal-reflection sweeps so the fit does not hard-code one echo model,
\item covariance and correlation estimation from the local Jacobian around the optimum.
\end{{itemize}}

Figure~\ref{{fig:{measured_tx}}} shows the strong measured transmission example used in these notes. Figure~\ref{{fig:{measured_tx_corr}}} shows the corresponding local parameter-correlation matrix.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{measured_tx}.pdf}}
\caption{{Measured transmission example for A11013460. The waveform, residual, spectral amplitude, and spectral phase are all displayed because the time-domain fit must be judged by both waveform agreement and frequency-domain consistency.}}
\label{{fig:{measured_tx}}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.58\linewidth]{{{measured_tx_corr}.pdf}}
\caption{{Local fit-correlation matrix for the measured transmission example. This matrix is local: it tells us which fitted parameters trade off with one another near the optimum of \emph{{this}} fit.}}
\label{{fig:{measured_tx_corr}}}
\end{{figure}}

For real reflection data, these notes use the stronger measured mirror-reference pair from the A11013460 dataset. The observed wafer trace is fitted against a directly measured Au-mirror reference, while the model is allowed to absorb the mirror sign and setup mismatch through fitted scale, offset, angle, and polarization nuisance parameters. This produces a much cleaner teaching example than the earlier self-reference construction because the residual stays small enough for the waveform mismatch to be visually interpretable.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{measured_refl}.pdf}}
\caption{{Measured reflection example for A11013460 using an explicit Au-mirror reference pair. The top-left panel shows the prepared reference/target pair, while the other panels show the fitted reflection response, residual, and spectral agreement.}}
\label{{fig:{measured_refl}}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.58\linewidth]{{{measured_refl_corr}.pdf}}
\caption{{Local fit-correlation matrix for the measured A11013460 reflection example.}}
\label{{fig:{measured_refl_corr}}}
\end{{figure}}

\section{{Correlation Matrix vs Study Confidence Map}}
The fit-correlation matrix and the study confidence map are not the same object.

The correlation matrix is \emph{{local}}. It is computed from the covariance of one fit around one optimum. It answers: near this optimum, which fitted variables move together? Because the underlying optimizer works in the actual fitted variables, the matrix is naturally expressed in thickness, plasma frequency, damping rate, and delay parameters.

The study confidence map is \emph{{global}} over a chosen parameter plane. It answers: after simulating many truth points, adding noise, and refitting, which parts of the chosen parameter space recover reliably? The map is plotted in the physical parameters chosen for the study, such as $(\tau,\sigma)$ or $(\tau_1,\tau_2)$. A region can show a modest local correlation matrix at the nominal point and still produce a poor global confidence map once noise is added and the truth point moves away from nominal.

This is exactly why both are needed:
\begin{{itemize}}[leftmargin=2em]
\item the correlation matrix diagnoses local parameter trade-offs inside one fit,
\item the study map diagnoses parameter recoverability across a whole operating region.
\end{{itemize}}

\section{{How the Study Works}}
For each requested map point, the lecture study performs the following loop:
\begin{{enumerate}}[leftmargin=2em]
\item overwrite the truth parameters in the stack,
\item simulate the synthetic measurement with the same forward model used by the fit,
\item add white Gaussian noise to the simulated sample trace at a chosen dynamic range,
\item refit the noisy trace using the same model and fit settings,
\item compare the fitted parameters to the truth,
\item save the traces, summary rows, and local fit-correlation matrix.
\end{{enumerate}}

The color in the lecture heatmaps is not the objective itself. The main plotted quantity is the two-parameter recovery error
\[
\mathcal{{E}}_{{\mathrm{{rec}}}}=\sqrt{{\frac{{1}}{{2}}\sum_i\log_{{10}}^2\!\left(\hat{{p}}_i/p_i^\star\right)}},
\]
where $p_i^\star$ is the truth value and $\hat{{p}}_i$ is the recovered fitted value for the two axis parameters of the map. Low values therefore mean high confidence that the fit recovers the right physical parameters under the chosen noise level and measurement geometry.

The weighted waveform objective $\mathcal{{J}}_w$ is still saved for every case and discussed in the text, but the lecture plots use $\mathcal{{E}}_{{\mathrm{{rec}}}}$ because these notes are meant to answer a metrology question: how confidently do we recover the right parameters?

\section{{One-Layer Drude Study}}
The one-layer study keeps $\varepsilon_\infty=11.7$ fixed and fits a thin-bounded thickness together with the Drude parameters. The low-conductivity and high-conductivity regimes are split deliberately because they create different dynamic ranges and different geometry sensitivities.

"""
        + "".join(include_triptych(slug) for slug in one_layer_slugs)
        + rf"""
\section{{Advanced Epi Study}}
The advanced study represents a wafer-style stack with a conductive two-Drude epi layer, a thin oxide interlayer, and a finite silicon substrate. The \SI{{525}}{{\micro\meter}} substrate cases use a \SI{{10}}{{\micro\meter}} epi layer, while the \SI{{725}}{{\micro\meter}} substrate cases use a \SI{{50}}{{\micro\meter}} epi layer. This lets the lecture compare two practically different epi-thickness regimes inside the same measurement framework.

Reflection uses an explicit bare-substrate standard without the conductive epi layer. That is essential because the reflection observable is always a ratio to the chosen reference standard, not an isolated waveform in absolute units.

"""
        + "".join(include_triptych(slug) for slug in advanced_slugs)
        + rf"""
\section{{Industrial Outlook: THz-TDS Ellipsometry for Wafer Metrology}}
The practical end point of this workflow is a THz-TDS ellipsometer or reflecto-ellipsometric metrology head that can choose angle, polarization, and reference-standard strategy to maximize parameter confidence for a given wafer stack. In that setting, the fit engine provides the forward-model inversion, while the study maps provide the operating envelope: which geometry is likely to recover the right conductivity, scattering time, or thickness before the system is deployed on the production floor.

This is why both pieces matter in the wafer industry. The fit converts a measured waveform into physical quantities. The study tells us how much to trust those quantities under realistic noise and model assumptions. Together, they turn THz-TDS from a descriptive spectroscopy tool into a quantitative metrology workflow.

{bibliography}
\end{{document}}
""")


def _write_lecture_notebook():
    one_layer_profiles = _one_layer_map_specs(profile="full")
    advanced_profiles = _advanced_map_specs(profile="full")
    demo_one_layer = dict(one_layer_profiles[0])
    demo_advanced = dict(advanced_profiles[0])

    def _identifier(text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text))
        cleaned = cleaned.strip("_").lower()
        if not cleaned:
            cleaned = "section"
        if cleaned[0].isdigit():
            cleaned = f"section_{cleaned}"
        return cleaned

    def _study_axis_labels(spec: dict[str, object]) -> tuple[str, str]:
        if str(spec["slug"]).startswith("one_layer"):
            return r"$\tau$ (ps)", r"$\sigma$ (S/m)"
        if str(spec["map_kind"]) == "tau":
            return r"$\tau_1$ (ps)", r"$\tau_2$ (ps)"
        return r"$\sigma_1$ (S/m)", r"$\sigma_2$ (S/m)"

    import_source = textwrap.dedent(
        """
        from pathlib import Path
        import csv
        import json
        import math
        import sys

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        from IPython import get_ipython
        from IPython.display import display

        _ip = get_ipython()
        if _ip is not None:
            try:
                _ip.run_line_magic('matplotlib', 'inline')
            except Exception:
                pass

        repo_root = Path.cwd().resolve()
        search_roots = [repo_root, *repo_root.parents]
        for candidate in search_roots:
            if (candidate / 'docs' / 'generate_lecture_assets.py').exists():
                repo_root = candidate
                break
        else:
            raise RuntimeError('Could not locate the repo root containing docs/generate_lecture_assets.py')
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from docs.generate_lecture_assets import build_notebook_demo_bundle, run_lecture_map_from_spec
        from thzsim2 import (
            Drude,
            Fit,
            Layer,
            Measurement,
            ReferenceStandard,
            TraceData,
            drude_gamma_thz_from_tau_ps,
            drude_plasma_freq_thz_from_sigma_tau,
            prepare_reflection_first_peak_pair,
            prepare_trace_pair_for_fit,
            run_measured_fit,
            run_staged_measured_fit,
        )
        from thzsim2.core.fft import fft_t_to_w
        from thzsim2.io.trace_csv import write_trace_csv

        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "#fbfaf7",
                "axes.edgecolor": "#2a2a2a",
                "axes.labelcolor": "#1f1f1f",
                "axes.titleweight": "bold",
                "axes.titlesize": 13.0,
                "axes.labelsize": 11.0,
                "xtick.labelsize": 10.0,
                "ytick.labelsize": 10.0,
                "grid.color": "#b8b8b8",
                "grid.alpha": 0.20,
                "grid.linestyle": "--",
                "legend.frameon": False,
                "font.family": "STIXGeneral",
                "mathtext.fontset": "stix",
                "savefig.bbox": "tight",
            }
        )

        VALUE_LABELS = {
            "recovery_error": r"$\\mathcal{E}_{\\mathrm{rec}}$",
            "weighted_data_fit": r"$\\mathcal{J}_{w}$",
            "data_fit": r"$\\mathcal{J}$",
            "relative_l2": r"$\\mathcal{L}_{2,\\mathrm{rel}}$",
            "residual_rms": r"$\\mathrm{RMS}(r)$",
            "max_abs_residual": r"$\\max |r|$",
        }
        PARAMETER_LABELS = {
            "film_thickness_um": r"$d$ ($\\mu$m)",
            "film_plasma_freq_thz": r"$\\omega_{p}$ (THz)",
            "film_gamma_thz": r"$\\gamma$ (THz)",
            "epi_thickness_um": r"$d_{\\mathrm{epi}}$ ($\\mu$m)",
            "oxide_thickness_um": r"$d_{\\mathrm{ox}}$ ($\\mu$m)",
            "epi_plasma1_thz": r"$\\omega_{p1}$ (THz)",
            "epi_gamma1_thz": r"$\\gamma_1$ (THz)",
            "epi_plasma2_thz": r"$\\omega_{p2}$ (THz)",
            "epi_gamma2_thz": r"$\\gamma_2$ (THz)",
            "wafer_thickness_um": r"$d$ ($\\mu$m)",
            "wafer_plasma_freq_thz": r"$\\omega_p$ (THz)",
            "wafer_gamma_thz": r"$\\gamma$ (THz)",
            "delta_t_ps": r"$\\Delta t$ (ps)",
            "tau_ps": r"$\\tau$ (ps)",
            "sigma_s_per_m": r"$\\sigma$ (S/m)",
            "tau1_ps": r"$\\tau_1$ (ps)",
            "tau2_ps": r"$\\tau_2$ (ps)",
            "sigma1_s_per_m": r"$\\sigma_1$ (S/m)",
            "sigma2_s_per_m": r"$\\sigma_2$ (S/m)",
        }

        def metric_label(value_key):
            value_key = str(value_key)
            if value_key in VALUE_LABELS:
                return VALUE_LABELS[value_key]
            if value_key.startswith("signed_err_"):
                parameter_key = value_key[len("signed_err_") :]
                parameter_label = PARAMETER_LABELS.get(parameter_key, parameter_key)
                return rf"$({parameter_label})_{{\\mathrm{{true}}}} - ({parameter_label})_{{\\mathrm{{fit}}}}$"
            if value_key.startswith("abs_err_"):
                parameter_key = value_key[len("abs_err_") :]
                parameter_label = PARAMETER_LABELS.get(parameter_key, parameter_key)
                return rf"$|({parameter_label})_{{\\mathrm{{true}}}} - ({parameter_label})_{{\\mathrm{{fit}}}}|$"
            return value_key

        def print_parameter_mapping(title, mapping):
            print(title)
            if not mapping:
                print("  (none)")
                return
            for key in sorted(mapping):
                value = mapping[key]
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    print(f"  {key}: {float(value):.6g}")
                else:
                    print(f"  {key}: {value}")
        """
    ).strip()

    def _measured_tx_calc_source() -> str:
        return textwrap.dedent(
            """
            measured_tx_output = repo_root / 'docs' / 'lecture_build' / 'notebook_measured_transmission'
            measured_tx_output.mkdir(parents=True, exist_ok=True)
            measured_tx_data_dir = measured_tx_output / 'data'
            measured_tx_data_dir.mkdir(parents=True, exist_ok=True)

            def _measured_tx_json_ready(value):
                if isinstance(value, dict):
                    return {str(key): _measured_tx_json_ready(inner) for key, inner in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_measured_tx_json_ready(item) for item in value]
                if isinstance(value, Path):
                    return value.resolve().as_posix()
                if isinstance(value, np.ndarray):
                    return [_measured_tx_json_ready(item) for item in value.tolist()]
                if isinstance(value, np.integer):
                    return int(value)
                if isinstance(value, np.floating):
                    return float(value)
                if isinstance(value, np.bool_):
                    return bool(value)
                return value

            measured_tx_root = repo_root / 'Test_data_for_fitter' / 'A11013460_transmission'
            measured_tx_prepared = prepare_trace_pair_for_fit(
                measured_tx_root / 'REFERENCE.csv',
                measured_tx_root / 'SAMPLE1.csv',
                baseline_mode='auto_pre_pulse',
                baseline_window_samples=40,
                crop_mode='auto',
            )

            measured_tx_sigma_s_per_m = 100.0 / 67.0
            measured_tx_tau_ps = 0.25
            measured_tx_layers = [
                Layer(
                    name='wafer',
                    thickness_um=Fit(625.0, abs_min=600.0, abs_max=650.0, label='wafer_thickness_um'),
                    material=Drude(
                        eps_inf=11.7,
                        plasma_freq_thz=Fit(
                            drude_plasma_freq_thz_from_sigma_tau(measured_tx_sigma_s_per_m, measured_tx_tau_ps),
                            abs_min=0.05,
                            abs_max=3.5,
                            label='wafer_plasma_freq_thz',
                        ),
                        gamma_thz=Fit(
                            drude_gamma_thz_from_tau_ps(measured_tx_tau_ps),
                            abs_min=0.1,
                            abs_max=2.5,
                            label='wafer_gamma_thz',
                        ),
                    ),
                )
            ]

            measured_tx_staged = run_staged_measured_fit(
                measured_tx_prepared,
                measured_tx_layers,
                out_dir=measured_tx_output / 'runs' / 'measured_a11013460_tx',
                measurement=Measurement(
                    mode='transmission',
                    angle_deg=0.0,
                    polarization='s',
                    reference_standard=ReferenceStandard(kind='ambient_replacement'),
                ),
                weighting={'mode': 'trace_amplitude', 'floor': 0.03, 'power': 2.0, 'smooth_window_samples': 41},
                delay_options={'enabled': True, 'search_window_ps': 20.0},
                reflection_counts=(0, 2, 4, 8),
            )

            measured_tx_fit_result = measured_tx_staged['best_fit_result']
            measured_tx_time_ps = np.asarray(measured_tx_prepared.processed_sample.time_ps, dtype=np.float64)
            measured_tx_reference_trace = np.asarray(measured_tx_prepared.processed_reference.trace, dtype=np.float64)
            measured_tx_observed_trace = np.asarray(measured_tx_prepared.processed_sample.trace, dtype=np.float64)
            measured_tx_fit_trace = np.asarray(measured_tx_fit_result['fitted_simulation']['sample_trace'], dtype=np.float64)
            measured_tx_residual_trace = np.asarray(measured_tx_fit_result['residual_trace'], dtype=np.float64)

            write_trace_csv(
                measured_tx_data_dir / 'measured_transmission_reference_trace.csv',
                TraceData(
                    time_ps=measured_tx_time_ps.copy(),
                    trace=measured_tx_reference_trace.copy(),
                    source_kind='processed_reference',
                    metadata=_measured_tx_json_ready(measured_tx_prepared.processed_reference.metadata),
                ),
            )
            write_trace_csv(
                measured_tx_data_dir / 'measured_transmission_observed_trace.csv',
                TraceData(
                    time_ps=measured_tx_time_ps.copy(),
                    trace=measured_tx_observed_trace.copy(),
                    source_kind='processed_sample',
                    metadata=_measured_tx_json_ready(measured_tx_prepared.processed_sample.metadata),
                ),
            )
            write_trace_csv(
                measured_tx_data_dir / 'measured_transmission_fit_trace.csv',
                TraceData(
                    time_ps=measured_tx_time_ps.copy(),
                    trace=measured_tx_fit_trace.copy(),
                    source_kind='fit_sample',
                    metadata={},
                ),
            )
            write_trace_csv(
                measured_tx_data_dir / 'measured_transmission_residual_trace.csv',
                TraceData(
                    time_ps=measured_tx_time_ps.copy(),
                    trace=measured_tx_residual_trace.copy(),
                    source_kind='fit_residual',
                    metadata={},
                ),
            )

            measured_tx_corr_rows = []
            measured_tx_corr_matrix = measured_tx_fit_result.get('parameter_correlation')
            measured_tx_parameter_names = list(measured_tx_fit_result.get('parameter_names', []))
            if measured_tx_corr_matrix is not None and measured_tx_parameter_names:
                measured_tx_corr_matrix = np.asarray(measured_tx_corr_matrix, dtype=np.float64)
                for measured_tx_i, measured_tx_name_i in enumerate(measured_tx_parameter_names):
                    for measured_tx_j, measured_tx_name_j in enumerate(measured_tx_parameter_names):
                        measured_tx_corr_rows.append(
                            {
                                'param_i': str(measured_tx_name_i),
                                'param_j': str(measured_tx_name_j),
                                'correlation': float(measured_tx_corr_matrix[measured_tx_i, measured_tx_j]),
                            }
                        )

            with (measured_tx_data_dir / 'measured_transmission_correlation_rows.csv').open('w', newline='', encoding='utf-8') as measured_tx_handle:
                measured_tx_writer = csv.DictWriter(measured_tx_handle, fieldnames=['param_i', 'param_j', 'correlation'])
                measured_tx_writer.writeheader()
                for measured_tx_row in measured_tx_corr_rows:
                    measured_tx_writer.writerow(measured_tx_row)

            measured_tx_summary = {
                'dataset': 'A11013460_transmission',
                'sample': 'SAMPLE1.csv',
                'selection_reason': str(measured_tx_staged['selection_reason']),
                'selection_score': float(measured_tx_staged['selection_score']),
                'prepared_metadata': _measured_tx_json_ready(measured_tx_prepared.metadata),
                'residual_metrics': {
                    key: float(value)
                    for key, value in measured_tx_fit_result['residual_metrics'].items()
                    if np.isfinite(float(value))
                },
                'recovered_parameters': {
                    key: float(value)
                    for key, value in measured_tx_fit_result['recovered_parameters'].items()
                },
            }
            (measured_tx_data_dir / 'measured_transmission_summary.json').write_text(
                json.dumps(measured_tx_summary, indent=2, ensure_ascii=True),
                encoding='utf-8',
            )

            measured_tx_bundle = {
                'output_root': measured_tx_output.resolve().as_posix(),
                'prepared_traces': measured_tx_prepared,
                'fit_result': measured_tx_fit_result,
                'selection_reason': measured_tx_staged['selection_reason'],
                'selection_score': measured_tx_staged['selection_score'],
                'data_paths': {
                    'summary_json': (measured_tx_data_dir / 'measured_transmission_summary.json').resolve().as_posix(),
                    'reference_trace_csv': (measured_tx_data_dir / 'measured_transmission_reference_trace.csv').resolve().as_posix(),
                    'observed_trace_csv': (measured_tx_data_dir / 'measured_transmission_observed_trace.csv').resolve().as_posix(),
                    'fit_trace_csv': (measured_tx_data_dir / 'measured_transmission_fit_trace.csv').resolve().as_posix(),
                    'residual_trace_csv': (measured_tx_data_dir / 'measured_transmission_residual_trace.csv').resolve().as_posix(),
                    'correlation_rows_csv': (measured_tx_data_dir / 'measured_transmission_correlation_rows.csv').resolve().as_posix(),
                },
            }
            print_parameter_mapping('Measured transmission recovered parameters', measured_tx_fit_result.get('recovered_parameters', {}))
            print()
            print_parameter_mapping('Measured transmission fitted measurement', measured_tx_fit_result.get('fitted_measurement', {}))
            measured_tx_bundle['data_paths']
            """
        ).strip()

    def _measured_tx_plot_source() -> str:
        return textwrap.dedent(
            """
            measured_tx_plot_title = 'Measured transmission fit: A11013460'
            measured_tx_corr_title = 'Measured transmission fit: local parameter correlation'
            measured_tx_figure_root = Path(measured_tx_bundle['output_root']) / 'figures'
            measured_tx_figure_root.mkdir(parents=True, exist_ok=True)

            def _measured_tx_positive_spectrum(trace, time_ps):
                measured_tx_dt_s = float(np.median(np.diff(np.asarray(time_ps, dtype=np.float64)))) * 1e-12
                measured_tx_t0_s = float(np.asarray(time_ps, dtype=np.float64)[0]) * 1e-12
                measured_tx_omega, measured_tx_spectrum = fft_t_to_w(np.asarray(trace, dtype=np.float64), dt=measured_tx_dt_s, t0=measured_tx_t0_s)
                measured_tx_freq_thz = measured_tx_omega / (2.0 * np.pi * 1e12)
                measured_tx_mask = measured_tx_freq_thz > 0.0
                return np.asarray(measured_tx_freq_thz[measured_tx_mask], dtype=np.float64), np.asarray(measured_tx_spectrum[measured_tx_mask], dtype=np.complex128)

            def _measured_tx_relative_db(values, floor_db=-110.0):
                measured_tx_values = np.asarray(values, dtype=np.float64)
                measured_tx_reference = max(float(np.max(measured_tx_values)), 1e-30)
                measured_tx_floor = 10.0 ** (float(floor_db) / 20.0)
                return 20.0 * np.log10(np.maximum(measured_tx_values / measured_tx_reference, measured_tx_floor))

            measured_tx_observed = np.asarray(measured_tx_bundle['prepared_traces'].processed_sample.trace, dtype=np.float64)
            measured_tx_fitted = np.asarray(measured_tx_bundle['fit_result']['fitted_simulation']['sample_trace'], dtype=np.float64)
            measured_tx_residual = np.asarray(measured_tx_bundle['fit_result']['residual_trace'], dtype=np.float64)
            measured_tx_time_ps = np.asarray(measured_tx_bundle['prepared_traces'].processed_sample.time_ps, dtype=np.float64)

            measured_tx_freq_thz, measured_tx_observed_spec = _measured_tx_positive_spectrum(measured_tx_observed, measured_tx_time_ps)
            _, measured_tx_fitted_spec = _measured_tx_positive_spectrum(measured_tx_fitted, measured_tx_time_ps)
            measured_tx_observed_amp = _measured_tx_relative_db(np.abs(measured_tx_observed_spec))
            measured_tx_fitted_amp = _measured_tx_relative_db(np.abs(measured_tx_fitted_spec))
            measured_tx_observed_phase = np.unwrap(np.angle(measured_tx_observed_spec))
            measured_tx_fitted_phase = np.unwrap(np.angle(measured_tx_fitted_spec))

            measured_tx_overview_fig, measured_tx_overview_axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
            measured_tx_overview_axes[0, 0].plot(measured_tx_time_ps, measured_tx_observed, linewidth=1.9, color='#0f4c81', label='Observed')
            measured_tx_overview_axes[0, 0].plot(measured_tx_time_ps, measured_tx_fitted, linewidth=1.55, color='#d1495b', label='Fit')
            measured_tx_overview_axes[0, 0].set_title('Time-domain waveform')
            measured_tx_overview_axes[0, 0].set_xlabel(r'$t$ (ps)')
            measured_tx_overview_axes[0, 0].set_ylabel(r'$E(t)$')
            measured_tx_overview_axes[0, 0].legend(loc='upper left')

            measured_tx_overview_axes[0, 1].plot(measured_tx_time_ps, measured_tx_residual, linewidth=1.45, color='#3b7d3a')
            measured_tx_overview_axes[0, 1].axhline(0.0, color='#555555', linewidth=0.9, alpha=0.7)
            measured_tx_overview_axes[0, 1].set_title('Residual trace')
            measured_tx_overview_axes[0, 1].set_xlabel(r'$t$ (ps)')
            measured_tx_overview_axes[0, 1].set_ylabel(r'$r(t)=E_{fit}(t)-E_{data}(t)$')

            measured_tx_overview_axes[1, 0].plot(measured_tx_freq_thz, measured_tx_observed_amp, linewidth=1.85, color='#0f4c81', label='Observed')
            measured_tx_overview_axes[1, 0].plot(measured_tx_freq_thz, measured_tx_fitted_amp, linewidth=1.45, color='#d1495b', label='Fit')
            measured_tx_overview_axes[1, 0].set_title('Spectral amplitude')
            measured_tx_overview_axes[1, 0].set_xlabel(r'$f$ (THz)')
            measured_tx_overview_axes[1, 0].set_ylabel('Amplitude (dB)')
            measured_tx_overview_axes[1, 0].set_xlim(0.05, min(3.0, float(measured_tx_freq_thz[-1])))
            measured_tx_overview_axes[1, 0].legend(loc='upper right')

            measured_tx_overview_axes[1, 1].plot(measured_tx_freq_thz, measured_tx_observed_phase, linewidth=1.85, color='#0f4c81', label='Observed')
            measured_tx_overview_axes[1, 1].plot(measured_tx_freq_thz, measured_tx_fitted_phase, linewidth=1.45, color='#d1495b', label='Fit')
            measured_tx_overview_axes[1, 1].set_title('Unwrapped spectral phase')
            measured_tx_overview_axes[1, 1].set_xlabel(r'$f$ (THz)')
            measured_tx_overview_axes[1, 1].set_ylabel(r'$\\phi(f)$ (rad)')
            measured_tx_overview_axes[1, 1].set_xlim(0.05, min(3.0, float(measured_tx_freq_thz[-1])))
            measured_tx_overview_axes[1, 1].legend(loc='upper right')

            for measured_tx_axis in measured_tx_overview_axes.flat:
                measured_tx_axis.grid(True, alpha=0.18)

            measured_tx_overview_fig.suptitle(measured_tx_plot_title, fontsize=15, fontweight='bold')
            measured_tx_overview_fig.tight_layout()
            measured_tx_overview_png = measured_tx_figure_root / 'measured_transmission_overview_section.png'
            measured_tx_overview_pdf = measured_tx_figure_root / 'measured_transmission_overview_section.pdf'
            measured_tx_overview_fig.savefig(measured_tx_overview_png, dpi=220)
            measured_tx_overview_fig.savefig(measured_tx_overview_pdf)
            display(measured_tx_overview_fig)
            plt.close(measured_tx_overview_fig)

            measured_tx_corr_names = list(measured_tx_bundle['fit_result'].get('parameter_names', []))
            measured_tx_corr_matrix = measured_tx_bundle['fit_result'].get('parameter_correlation')
            if measured_tx_corr_matrix is None or not measured_tx_corr_names:
                raise ValueError('No parameter correlation matrix is available for the measured transmission example.')
            measured_tx_corr_matrix = np.asarray(measured_tx_corr_matrix, dtype=np.float64)

            measured_tx_corr_fig, measured_tx_corr_ax = plt.subplots(figsize=(4.8, 4.2))
            measured_tx_corr_image = measured_tx_corr_ax.imshow(measured_tx_corr_matrix, cmap='plasma', vmin=-1.0, vmax=1.0)
            measured_tx_corr_ax.set_title(measured_tx_corr_title)
            measured_tx_corr_ax.set_xticks(range(len(measured_tx_corr_names)))
            measured_tx_corr_ax.set_yticks(range(len(measured_tx_corr_names)))
            measured_tx_corr_ax.set_xticklabels(
                [PARAMETER_LABELS.get(name, name) for name in measured_tx_corr_names],
                rotation=45,
                ha='right',
            )
            measured_tx_corr_ax.set_yticklabels([PARAMETER_LABELS.get(name, name) for name in measured_tx_corr_names])
            measured_tx_corr_fig.colorbar(measured_tx_corr_image, ax=measured_tx_corr_ax, label=r'$\\rho_{ij}$')
            measured_tx_corr_fig.tight_layout()
            measured_tx_corr_png = measured_tx_figure_root / 'measured_transmission_correlation_section.png'
            measured_tx_corr_pdf = measured_tx_figure_root / 'measured_transmission_correlation_section.pdf'
            measured_tx_corr_fig.savefig(measured_tx_corr_png, dpi=220)
            measured_tx_corr_fig.savefig(measured_tx_corr_pdf)
            display(measured_tx_corr_fig)
            plt.close(measured_tx_corr_fig)

            measured_tx_bundle['figure_paths'] = {
                'overview_png': measured_tx_overview_png.resolve().as_posix(),
                'overview_pdf': measured_tx_overview_pdf.resolve().as_posix(),
                'correlation_png': measured_tx_corr_png.resolve().as_posix(),
                'correlation_pdf': measured_tx_corr_pdf.resolve().as_posix(),
            }
            measured_tx_bundle['figure_paths']
            """
        ).strip()

    def _measured_refl_calc_source() -> str:
        return textwrap.dedent(
            """
            measured_refl_output = repo_root / 'docs' / 'lecture_build' / 'notebook_measured_reflection'
            measured_refl_output.mkdir(parents=True, exist_ok=True)
            measured_refl_data_dir = measured_refl_output / 'data'
            measured_refl_data_dir.mkdir(parents=True, exist_ok=True)

            def _measured_refl_json_ready(value):
                if isinstance(value, dict):
                    return {str(key): _measured_refl_json_ready(inner) for key, inner in value.items()}
                if isinstance(value, (list, tuple)):
                    return [_measured_refl_json_ready(item) for item in value]
                if isinstance(value, Path):
                    return value.resolve().as_posix()
                if isinstance(value, np.ndarray):
                    return [_measured_refl_json_ready(item) for item in value.tolist()]
                if isinstance(value, np.integer):
                    return int(value)
                if isinstance(value, np.floating):
                    return float(value)
                if isinstance(value, np.bool_):
                    return bool(value)
                return value

            measured_refl_reference_path = (
                repo_root
                / 'Test_data_for_fitter'
                / 'A11013460_reflection'
                / 'reflection_setup_ref_after_with_AuMirror_A11013460_avg600_onDryAir10min_int56.csv'
            )
            measured_refl_sample_path = (
                repo_root
                / 'Test_data_for_fitter'
                / 'A11013460_reflection'
                / 'reflection_setup_sample_A11013460_avg600_onDryAir10min_int30.csv'
            )
            measured_refl_prepared = prepare_trace_pair_for_fit(
                measured_refl_reference_path,
                measured_refl_sample_path,
                baseline_mode='auto_pre_pulse',
                baseline_window_samples=40,
                crop_mode='auto',
            )

            measured_refl_sigma_s_per_m = 100.0 / 67.0
            measured_refl_tau_ps = 0.25
            measured_refl_layers = [
                Layer(
                    name='wafer',
                    thickness_um=Fit(625.0, abs_min=575.0, abs_max=675.0, label='wafer_thickness_um'),
                    material=Drude(
                        eps_inf=Fit(11.7, abs_min=4.0, abs_max=20.0, label='wafer_eps_inf'),
                        plasma_freq_thz=Fit(
                            drude_plasma_freq_thz_from_sigma_tau(measured_refl_sigma_s_per_m, measured_refl_tau_ps),
                            abs_min=0.01,
                            abs_max=4.0,
                            label='wafer_plasma_freq_thz',
                        ),
                        gamma_thz=Fit(
                            drude_gamma_thz_from_tau_ps(measured_refl_tau_ps),
                            abs_min=0.01,
                            abs_max=4.0,
                            label='wafer_gamma_thz',
                        ),
                    ),
                )
            ]

            measured_refl_result = run_measured_fit(
                measured_refl_prepared,
                measured_refl_layers,
                out_dir=measured_refl_output / 'runs' / 'measured_a11013460_reflection',
                measurement=Measurement(
                    mode='reflection',
                    angle_deg=Fit(10.0, abs_min=0.0, abs_max=45.0, label='measurement_angle_deg'),
                    polarization='mixed',
                    polarization_mix=Fit(0.5, abs_min=0.0, abs_max=1.0, label='measurement_polarization_mix'),
                    trace_scale=Fit(-1.0, abs_min=-2.5, abs_max=0.5, label='measurement_trace_scale'),
                    trace_offset=Fit(0.0, abs_min=-1.0, abs_max=1.0, label='measurement_trace_offset'),
                    reference_standard=ReferenceStandard(kind='identity'),
                ),
                optimizer={
                    'method': 'L-BFGS-B',
                    'options': {'maxiter': 140},
                    'global_options': {'maxiter': 10, 'popsize': 10, 'seed': 123},
                    'fd_rel_step': 1e-5,
                },
                max_internal_reflections=2,
                delay_options={'enabled': True, 'search_window_ps': 20.0, 'initial_ps': 0.0},
                weighting={'mode': 'trace_amplitude', 'floor': 0.03, 'power': 2.0, 'smooth_window_samples': 41},
                metric='weighted_data_fit',
            )

            measured_refl_fit_result = measured_refl_result.fit_result
            measured_refl_time_ps = np.asarray(measured_refl_prepared.processed_sample.time_ps, dtype=np.float64)
            measured_refl_reference_trace = np.asarray(measured_refl_prepared.processed_reference.trace, dtype=np.float64)
            measured_refl_observed_trace = np.asarray(measured_refl_prepared.processed_sample.trace, dtype=np.float64)
            measured_refl_fit_trace = np.asarray(measured_refl_fit_result['fitted_simulation']['sample_trace'], dtype=np.float64)
            measured_refl_residual_trace = np.asarray(measured_refl_fit_result['residual_trace'], dtype=np.float64)

            write_trace_csv(
                measured_refl_data_dir / 'measured_reflection_reference_trace.csv',
                TraceData(
                    time_ps=measured_refl_time_ps.copy(),
                    trace=measured_refl_reference_trace.copy(),
                    source_kind='processed_reference',
                    metadata=_measured_refl_json_ready(measured_refl_prepared.processed_reference.metadata),
                ),
            )
            write_trace_csv(
                measured_refl_data_dir / 'measured_reflection_observed_trace.csv',
                TraceData(
                    time_ps=measured_refl_time_ps.copy(),
                    trace=measured_refl_observed_trace.copy(),
                    source_kind='processed_sample',
                    metadata=_measured_refl_json_ready(measured_refl_prepared.processed_sample.metadata),
                ),
            )
            write_trace_csv(
                measured_refl_data_dir / 'measured_reflection_fit_trace.csv',
                TraceData(
                    time_ps=measured_refl_time_ps.copy(),
                    trace=measured_refl_fit_trace.copy(),
                    source_kind='fit_sample',
                    metadata={},
                ),
            )
            write_trace_csv(
                measured_refl_data_dir / 'measured_reflection_residual_trace.csv',
                TraceData(
                    time_ps=measured_refl_time_ps.copy(),
                    trace=measured_refl_residual_trace.copy(),
                    source_kind='fit_residual',
                    metadata={},
                ),
            )

            measured_refl_corr_rows = []
            measured_refl_corr_matrix = measured_refl_fit_result.get('parameter_correlation')
            measured_refl_parameter_names = list(measured_refl_fit_result.get('parameter_names', []))
            if measured_refl_corr_matrix is not None and measured_refl_parameter_names:
                measured_refl_corr_matrix = np.asarray(measured_refl_corr_matrix, dtype=np.float64)
                for measured_refl_i, measured_refl_name_i in enumerate(measured_refl_parameter_names):
                    for measured_refl_j, measured_refl_name_j in enumerate(measured_refl_parameter_names):
                        measured_refl_corr_rows.append(
                            {
                                'param_i': str(measured_refl_name_i),
                                'param_j': str(measured_refl_name_j),
                                'correlation': float(measured_refl_corr_matrix[measured_refl_i, measured_refl_j]),
                            }
                        )

            with (measured_refl_data_dir / 'measured_reflection_correlation_rows.csv').open('w', newline='', encoding='utf-8') as measured_refl_handle:
                measured_refl_writer = csv.DictWriter(measured_refl_handle, fieldnames=['param_i', 'param_j', 'correlation'])
                measured_refl_writer.writeheader()
                for measured_refl_row in measured_refl_corr_rows:
                    measured_refl_writer.writerow(measured_refl_row)

            measured_refl_summary = {
                'dataset': 'A11013460_reflection',
                'reference': measured_refl_reference_path.name,
                'sample': measured_refl_sample_path.name,
                'construction': 'mirror_reference_pair',
                'prepared_metadata': _measured_refl_json_ready(measured_refl_prepared.metadata),
                'residual_metrics': {
                    key: float(value)
                    for key, value in measured_refl_fit_result['residual_metrics'].items()
                    if np.isfinite(float(value))
                },
                'max_abs_residual': float(np.max(np.abs(measured_refl_residual_trace))),
                'fitted_measurement': {
                    key: (float(value) if value is not None else None)
                    for key, value in measured_refl_fit_result['fitted_measurement'].items()
                    if key in {'angle_deg', 'polarization_mix', 'trace_scale', 'trace_offset'}
                },
                'recovered_parameters': {
                    key: float(value)
                    for key, value in measured_refl_fit_result['recovered_parameters'].items()
                },
            }
            (measured_refl_data_dir / 'measured_reflection_summary.json').write_text(
                json.dumps(measured_refl_summary, indent=2, ensure_ascii=True),
                encoding='utf-8',
            )

            measured_refl_bundle = {
                'output_root': measured_refl_output.resolve().as_posix(),
                'prepared_traces': measured_refl_prepared,
                'fit_result': measured_refl_fit_result,
                'measured_fit_result': measured_refl_result,
                'data_paths': {
                    'summary_json': (measured_refl_data_dir / 'measured_reflection_summary.json').resolve().as_posix(),
                    'reference_trace_csv': (measured_refl_data_dir / 'measured_reflection_reference_trace.csv').resolve().as_posix(),
                    'observed_trace_csv': (measured_refl_data_dir / 'measured_reflection_observed_trace.csv').resolve().as_posix(),
                    'fit_trace_csv': (measured_refl_data_dir / 'measured_reflection_fit_trace.csv').resolve().as_posix(),
                    'residual_trace_csv': (measured_refl_data_dir / 'measured_reflection_residual_trace.csv').resolve().as_posix(),
                    'correlation_rows_csv': (measured_refl_data_dir / 'measured_reflection_correlation_rows.csv').resolve().as_posix(),
                },
            }
            print_parameter_mapping('Measured reflection recovered parameters', measured_refl_fit_result.get('recovered_parameters', {}))
            print()
            print_parameter_mapping('Measured reflection fitted measurement', measured_refl_fit_result.get('fitted_measurement', {}))
            measured_refl_bundle['data_paths']
            """
        ).strip()

    def _measured_refl_plot_source() -> str:
        return textwrap.dedent(
            """
            measured_refl_plot_title = 'Measured reflection fit: A11013460 mirror-reference pair'
            measured_refl_corr_title = 'Measured reflection fit: local parameter correlation'
            measured_refl_figure_root = Path(measured_refl_bundle['output_root']) / 'figures'
            measured_refl_figure_root.mkdir(parents=True, exist_ok=True)

            def _measured_refl_positive_spectrum(trace, time_ps):
                measured_refl_dt_s = float(np.median(np.diff(np.asarray(time_ps, dtype=np.float64)))) * 1e-12
                measured_refl_t0_s = float(np.asarray(time_ps, dtype=np.float64)[0]) * 1e-12
                measured_refl_omega, measured_refl_spectrum = fft_t_to_w(np.asarray(trace, dtype=np.float64), dt=measured_refl_dt_s, t0=measured_refl_t0_s)
                measured_refl_freq_thz = measured_refl_omega / (2.0 * np.pi * 1e12)
                measured_refl_mask = measured_refl_freq_thz > 0.0
                return np.asarray(measured_refl_freq_thz[measured_refl_mask], dtype=np.float64), np.asarray(measured_refl_spectrum[measured_refl_mask], dtype=np.complex128)

            def _measured_refl_relative_db(values, floor_db=-110.0):
                measured_refl_values = np.asarray(values, dtype=np.float64)
                measured_refl_reference = max(float(np.max(measured_refl_values)), 1e-30)
                measured_refl_floor = 10.0 ** (float(floor_db) / 20.0)
                return 20.0 * np.log10(np.maximum(measured_refl_values / measured_refl_reference, measured_refl_floor))

            measured_refl_reference = np.asarray(measured_refl_bundle['prepared_traces'].processed_reference.trace, dtype=np.float64)
            measured_refl_observed = np.asarray(measured_refl_bundle['prepared_traces'].processed_sample.trace, dtype=np.float64)
            measured_refl_fitted = np.asarray(measured_refl_bundle['fit_result']['fitted_simulation']['sample_trace'], dtype=np.float64)
            measured_refl_residual = np.asarray(measured_refl_bundle['fit_result']['residual_trace'], dtype=np.float64)
            measured_refl_time_ps = np.asarray(measured_refl_bundle['prepared_traces'].processed_sample.time_ps, dtype=np.float64)

            measured_refl_freq_thz, measured_refl_observed_spec = _measured_refl_positive_spectrum(measured_refl_observed, measured_refl_time_ps)
            _, measured_refl_fitted_spec = _measured_refl_positive_spectrum(measured_refl_fitted, measured_refl_time_ps)
            measured_refl_observed_amp = _measured_refl_relative_db(np.abs(measured_refl_observed_spec))
            measured_refl_fitted_amp = _measured_refl_relative_db(np.abs(measured_refl_fitted_spec))

            measured_refl_overview_fig, measured_refl_overview_axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
            measured_refl_overview_axes[0, 0].plot(measured_refl_time_ps, measured_refl_reference, linewidth=1.75, color='#5b84b1', label='Measured Au-mirror reference')
            measured_refl_overview_axes[0, 0].plot(measured_refl_time_ps, measured_refl_observed, linewidth=1.45, color='#d1495b', label='Observed wafer reflection')
            measured_refl_overview_axes[0, 0].set_title('Mirror-reference pair')
            measured_refl_overview_axes[0, 0].set_xlabel(r'$t$ (ps)')
            measured_refl_overview_axes[0, 0].set_ylabel(r'$E(t)$')
            measured_refl_overview_axes[0, 0].legend(loc='upper left')

            measured_refl_overview_axes[0, 1].plot(measured_refl_time_ps, measured_refl_observed, linewidth=1.8, color='#0f4c81', label='Observed')
            measured_refl_overview_axes[0, 1].plot(measured_refl_time_ps, measured_refl_fitted, linewidth=1.45, color='#d1495b', label='Fit')
            measured_refl_overview_axes[0, 1].set_title('Reflection fit')
            measured_refl_overview_axes[0, 1].set_xlabel(r'$t$ (ps)')
            measured_refl_overview_axes[0, 1].set_ylabel(r'$E(t)$')
            measured_refl_overview_axes[0, 1].legend(loc='upper left')

            measured_refl_overview_axes[1, 0].plot(measured_refl_time_ps, measured_refl_residual, linewidth=1.45, color='#3b7d3a')
            measured_refl_overview_axes[1, 0].axhline(0.0, color='#555555', linewidth=0.9, alpha=0.7)
            measured_refl_overview_axes[1, 0].set_title('Residual trace')
            measured_refl_overview_axes[1, 0].set_xlabel(r'$t$ (ps)')
            measured_refl_overview_axes[1, 0].set_ylabel(r'$r(t)=E_{fit}(t)-E_{data}(t)$')

            measured_refl_overview_axes[1, 1].plot(measured_refl_freq_thz, measured_refl_observed_amp, linewidth=1.85, color='#0f4c81', label='Observed')
            measured_refl_overview_axes[1, 1].plot(measured_refl_freq_thz, measured_refl_fitted_amp, linewidth=1.45, color='#d1495b', label='Fit')
            measured_refl_overview_axes[1, 1].set_title('Spectral amplitude')
            measured_refl_overview_axes[1, 1].set_xlabel(r'$f$ (THz)')
            measured_refl_overview_axes[1, 1].set_ylabel('Amplitude (dB)')
            measured_refl_overview_axes[1, 1].set_xlim(0.05, min(3.0, float(measured_refl_freq_thz[-1])))
            measured_refl_overview_axes[1, 1].legend(loc='upper right')

            for measured_refl_axis in measured_refl_overview_axes.flat:
                measured_refl_axis.grid(True, alpha=0.18)

            measured_refl_overview_fig.suptitle(measured_refl_plot_title, fontsize=15, fontweight='bold')
            measured_refl_overview_fig.tight_layout()
            measured_refl_overview_png = measured_refl_figure_root / 'measured_reflection_overview_section.png'
            measured_refl_overview_pdf = measured_refl_figure_root / 'measured_reflection_overview_section.pdf'
            measured_refl_overview_fig.savefig(measured_refl_overview_png, dpi=220)
            measured_refl_overview_fig.savefig(measured_refl_overview_pdf)
            display(measured_refl_overview_fig)
            plt.close(measured_refl_overview_fig)

            measured_refl_corr_names = list(measured_refl_bundle['fit_result'].get('parameter_names', []))
            measured_refl_corr_matrix = measured_refl_bundle['fit_result'].get('parameter_correlation')
            if measured_refl_corr_matrix is None or not measured_refl_corr_names:
                raise ValueError('No parameter correlation matrix is available for the measured reflection example.')
            measured_refl_corr_matrix = np.asarray(measured_refl_corr_matrix, dtype=np.float64)

            measured_refl_corr_fig, measured_refl_corr_ax = plt.subplots(figsize=(4.8, 4.2))
            measured_refl_corr_image = measured_refl_corr_ax.imshow(measured_refl_corr_matrix, cmap='plasma', vmin=-1.0, vmax=1.0)
            measured_refl_corr_ax.set_title(measured_refl_corr_title)
            measured_refl_corr_ax.set_xticks(range(len(measured_refl_corr_names)))
            measured_refl_corr_ax.set_yticks(range(len(measured_refl_corr_names)))
            measured_refl_corr_ax.set_xticklabels(
                [PARAMETER_LABELS.get(name, name) for name in measured_refl_corr_names],
                rotation=45,
                ha='right',
            )
            measured_refl_corr_ax.set_yticklabels([PARAMETER_LABELS.get(name, name) for name in measured_refl_corr_names])
            measured_refl_corr_fig.colorbar(measured_refl_corr_image, ax=measured_refl_corr_ax, label=r'$\\rho_{ij}$')
            measured_refl_corr_fig.tight_layout()
            measured_refl_corr_png = measured_refl_figure_root / 'measured_reflection_correlation_section.png'
            measured_refl_corr_pdf = measured_refl_figure_root / 'measured_reflection_correlation_section.pdf'
            measured_refl_corr_fig.savefig(measured_refl_corr_png, dpi=220)
            measured_refl_corr_fig.savefig(measured_refl_corr_pdf)
            display(measured_refl_corr_fig)
            plt.close(measured_refl_corr_fig)

            measured_refl_bundle['figure_paths'] = {
                'overview_png': measured_refl_overview_png.resolve().as_posix(),
                'overview_pdf': measured_refl_overview_pdf.resolve().as_posix(),
                'correlation_png': measured_refl_corr_png.resolve().as_posix(),
                'correlation_pdf': measured_refl_corr_pdf.resolve().as_posix(),
            }
            measured_refl_bundle['figure_paths']
            """
        ).strip()

    def _study_calc_source(
        var_name: str,
        spec: dict[str, object],
        *,
        profile: str,
        output_tag: str,
    ) -> str:
        return (
            f"{var_name}_spec = {json.dumps(spec, indent=2)}\n"
            f"{var_name}_output_root = repo_root / 'docs' / 'lecture_build' / 'notebook_section_runs' / {output_tag!r}\n"
            f"{var_name}_output_root.mkdir(parents=True, exist_ok=True)\n"
            f"{var_name}_result = run_lecture_map_from_spec({var_name}_spec, output_root={var_name}_output_root, profile={profile!r})\n"
            f"{var_name}_result['study_dir']"
        )

    def _study_plot_source(var_name: str, x_label: str, y_label: str) -> str:
        template = textwrap.dedent(
            """
            __VAR___study_dir = Path(__VAR___result['study_dir'])
            __VAR___plot_title = __VAR___result['title']
            __VAR___x_label = __X_LABEL__
            __VAR___y_label = __Y_LABEL__

            with (__VAR___study_dir / 'study_summary.json').open('r', encoding='utf-8') as __VAR___summary_handle:
                __VAR___summary_meta = json.load(__VAR___summary_handle)
            with (__VAR___study_dir / 'averaged_correlation.json').open('r', encoding='utf-8') as __VAR___corr_handle:
                __VAR___corr_meta = json.load(__VAR___corr_handle)

            __VAR___rows = []
            with (__VAR___study_dir / 'study_summary.csv').open('r', newline='', encoding='utf-8') as __VAR___summary_csv:
                __VAR___reader = csv.DictReader(__VAR___summary_csv)
                for __VAR___row in __VAR___reader:
                    __VAR___parsed = {}
                    for __VAR___key, __VAR___value in __VAR___row.items():
                        if __VAR___value is None:
                            __VAR___parsed[__VAR___key] = __VAR___value
                            continue
                        __VAR___text = str(__VAR___value).strip()
                        if __VAR___text == '':
                            __VAR___parsed[__VAR___key] = __VAR___text
                            continue
                        try:
                            __VAR___parsed[__VAR___key] = float(__VAR___text)
                        except ValueError:
                            __VAR___parsed[__VAR___key] = __VAR___text
                    __VAR___rows.append(__VAR___parsed)

            __VAR___x_key, __VAR___y_key = list(__VAR___summary_meta['recovery_keys'])
            __VAR___metric_options = {
                'data_fit': 'data_fit',
                'weighted_data_fit': 'weighted_data_fit',
                f'{__VAR___x_key}_true_minus_fit': f'signed_err_{__VAR___x_key}',
                f'{__VAR___y_key}_true_minus_fit': f'signed_err_{__VAR___y_key}',
                f'abs_{__VAR___x_key}_error': f'abs_err_{__VAR___x_key}',
                f'abs_{__VAR___y_key}_error': f'abs_err_{__VAR___y_key}',
            }
            __VAR___linear_value_key = __VAR___metric_options['data_fit']
            __VAR___log_value_key = __VAR___metric_options['data_fit']
            __VAR___x_values = sorted({float(__VAR___row[__VAR___x_key]) for __VAR___row in __VAR___rows})
            __VAR___y_values = sorted({float(__VAR___row[__VAR___y_key]) for __VAR___row in __VAR___rows})

            print('Available map metrics:')
            for __VAR___option_name, __VAR___option_key in __VAR___metric_options.items():
                print(f"  {__VAR___option_name}: {metric_label(__VAR___option_key)}")
            print(f"Current linear map: {metric_label(__VAR___linear_value_key)}")
            print(f"Current log map: {metric_label(__VAR___log_value_key)}")

            def __VAR___aggregate_grid(value_key):
                __VAR___grid = np.full((len(__VAR___y_values), len(__VAR___x_values)), np.nan, dtype=np.float64)
                for __VAR___iy, __VAR___y_value in enumerate(__VAR___y_values):
                    for __VAR___ix, __VAR___x_value in enumerate(__VAR___x_values):
                        __VAR___cell = [
                            float(__VAR___row[value_key])
                            for __VAR___row in __VAR___rows
                            if math.isclose(float(__VAR___row[__VAR___x_key]), __VAR___x_value, rel_tol=0.0, abs_tol=1e-12)
                            and math.isclose(float(__VAR___row[__VAR___y_key]), __VAR___y_value, rel_tol=0.0, abs_tol=1e-12)
                            and np.isfinite(float(__VAR___row[value_key]))
                        ]
                        if __VAR___cell:
                            __VAR___grid[__VAR___iy, __VAR___ix] = float(np.mean(__VAR___cell))
                return __VAR___grid

            def __VAR___positive_grid_for_log(grid):
                __VAR___grid = np.asarray(grid, dtype=np.float64).copy()
                __VAR___finite = __VAR___grid[np.isfinite(__VAR___grid)]
                __VAR___positive = __VAR___finite[__VAR___finite > 0.0]
                if __VAR___positive.size == 0:
                    __VAR___grid[np.isfinite(__VAR___grid)] = 1.0
                    return __VAR___grid
                __VAR___floor = max(float(np.min(__VAR___positive)) * 0.5, 1e-18)
                __VAR___grid[np.isfinite(__VAR___grid) & (__VAR___grid <= 0.0)] = __VAR___floor
                return __VAR___grid

            __VAR___linear_grid = __VAR___aggregate_grid(__VAR___linear_value_key)
            __VAR___linear_fig, __VAR___linear_ax = plt.subplots(figsize=(5.8, 4.6))
            __VAR___linear_finite = __VAR___linear_grid[np.isfinite(__VAR___linear_grid)]
            if str(__VAR___linear_value_key).startswith('signed_err_'):
                __VAR___linear_limit = max(float(np.max(np.abs(__VAR___linear_finite))), 1e-18)
                __VAR___linear_vmin = -__VAR___linear_limit
                __VAR___linear_vmax = __VAR___linear_limit
                __VAR___linear_levels = np.linspace(__VAR___linear_vmin, __VAR___linear_vmax, 256)
                __VAR___linear_cmap = 'plasma'
            else:
                __VAR___linear_vmin = float(np.min(__VAR___linear_finite))
                __VAR___linear_vmax = float(np.max(__VAR___linear_finite)) + 1e-12
                __VAR___linear_levels = np.linspace(__VAR___linear_vmin, __VAR___linear_vmax, 256)
                __VAR___linear_cmap = 'plasma'
            __VAR___linear_contour = __VAR___linear_ax.contourf(
                np.asarray(__VAR___x_values, dtype=np.float64),
                np.asarray(__VAR___y_values, dtype=np.float64),
                __VAR___linear_grid,
                levels=__VAR___linear_levels,
                cmap=__VAR___linear_cmap,
                extend='both',
            )
            __VAR___linear_ax.set_title(__VAR___plot_title + ' [linear]')
            __VAR___linear_ax.set_xlabel(__VAR___x_label)
            __VAR___linear_ax.set_ylabel(__VAR___y_label)
            __VAR___linear_cbar = __VAR___linear_fig.colorbar(__VAR___linear_contour, ax=__VAR___linear_ax)
            __VAR___linear_cbar.set_label(metric_label(__VAR___linear_value_key))
            __VAR___linear_fig.tight_layout()
            __VAR___linear_png = __VAR___study_dir / f"{__VAR___study_dir.name}__section_linear.png"
            __VAR___linear_pdf = __VAR___study_dir / f"{__VAR___study_dir.name}__section_linear.pdf"
            __VAR___linear_fig.savefig(__VAR___linear_png, dpi=220)
            __VAR___linear_fig.savefig(__VAR___linear_pdf)

            if str(__VAR___log_value_key).startswith('signed_err_'):
                __VAR___log_source_key = 'abs_err_' + str(__VAR___log_value_key)[len('signed_err_'):]
            else:
                __VAR___log_source_key = __VAR___log_value_key
            __VAR___log_grid = __VAR___positive_grid_for_log(__VAR___aggregate_grid(__VAR___log_source_key))
            __VAR___log_fig, __VAR___log_ax = plt.subplots(figsize=(5.8, 4.6))
            __VAR___log_finite = __VAR___log_grid[np.isfinite(__VAR___log_grid)]
            __VAR___log_positive = __VAR___log_finite[__VAR___log_finite > 0.0]
            __VAR___log_vmin = float(np.min(__VAR___log_positive))
            __VAR___log_vmax = float(np.max(__VAR___log_positive))
            if math.isclose(__VAR___log_vmin, __VAR___log_vmax):
                __VAR___log_vmax = __VAR___log_vmin * 1.01
            __VAR___log_levels = np.geomspace(__VAR___log_vmin, __VAR___log_vmax, 256)
            __VAR___log_contour = __VAR___log_ax.contourf(
                np.asarray(__VAR___x_values, dtype=np.float64),
                np.asarray(__VAR___y_values, dtype=np.float64),
                __VAR___log_grid,
                levels=__VAR___log_levels,
                cmap='plasma',
                norm=mcolors.LogNorm(vmin=__VAR___log_vmin, vmax=__VAR___log_vmax),
                extend='both',
            )
            __VAR___log_ax.set_title(__VAR___plot_title + ' [log]')
            __VAR___log_ax.set_xlabel(__VAR___x_label)
            __VAR___log_ax.set_ylabel(__VAR___y_label)
            __VAR___log_cbar = __VAR___log_fig.colorbar(__VAR___log_contour, ax=__VAR___log_ax)
            __VAR___log_cbar.set_label(metric_label(__VAR___log_source_key))
            __VAR___log_fig.tight_layout()
            __VAR___log_png = __VAR___study_dir / f"{__VAR___study_dir.name}__section_log.png"
            __VAR___log_pdf = __VAR___study_dir / f"{__VAR___study_dir.name}__section_log.pdf"
            __VAR___log_fig.savefig(__VAR___log_png, dpi=220)
            __VAR___log_fig.savefig(__VAR___log_pdf)

            __VAR___corr_rows = __VAR___corr_meta['rows']
            __VAR___corr_labels = sorted({str(__VAR___row['param_i']) for __VAR___row in __VAR___corr_rows} | {str(__VAR___row['param_j']) for __VAR___row in __VAR___corr_rows})
            __VAR___corr_index = {__VAR___label: __VAR___idx for __VAR___idx, __VAR___label in enumerate(__VAR___corr_labels)}
            __VAR___corr_matrix = np.full((len(__VAR___corr_labels), len(__VAR___corr_labels)), np.nan, dtype=np.float64)
            for __VAR___row in __VAR___corr_rows:
                __VAR___i = __VAR___corr_index[str(__VAR___row['param_i'])]
                __VAR___j = __VAR___corr_index[str(__VAR___row['param_j'])]
                __VAR___corr_matrix[__VAR___i, __VAR___j] = float(__VAR___row['correlation'])
                __VAR___corr_matrix[__VAR___j, __VAR___i] = float(__VAR___row['correlation'])
            for __VAR___diag in range(len(__VAR___corr_labels)):
                __VAR___corr_matrix[__VAR___diag, __VAR___diag] = 1.0

            __VAR___corr_fig, __VAR___corr_ax = plt.subplots(figsize=(4.8, 4.2))
            __VAR___corr_image = __VAR___corr_ax.imshow(__VAR___corr_matrix, cmap='plasma', vmin=-1.0, vmax=1.0)
            __VAR___corr_ax.set_title(__VAR___plot_title + ' [correlation]')
            __VAR___corr_ax.set_xticks(range(len(__VAR___corr_labels)))
            __VAR___corr_ax.set_yticks(range(len(__VAR___corr_labels)))
            __VAR___corr_ax.set_xticklabels([PARAMETER_LABELS.get(__VAR___label, __VAR___label) for __VAR___label in __VAR___corr_labels], rotation=45, ha='right')
            __VAR___corr_ax.set_yticklabels([PARAMETER_LABELS.get(__VAR___label, __VAR___label) for __VAR___label in __VAR___corr_labels])
            __VAR___corr_fig.colorbar(__VAR___corr_image, ax=__VAR___corr_ax, label=r'$\\rho_{ij}$')
            __VAR___corr_fig.tight_layout()
            __VAR___corr_png = __VAR___study_dir / f"{__VAR___study_dir.name}__section_corr.png"
            __VAR___corr_pdf = __VAR___study_dir / f"{__VAR___study_dir.name}__section_corr.pdf"
            __VAR___corr_fig.savefig(__VAR___corr_png, dpi=220)
            __VAR___corr_fig.savefig(__VAR___corr_pdf)

            __VAR___triptych_fig, __VAR___triptych_axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
            for __VAR___axis, __VAR___image_path, __VAR___panel_title in zip(
                __VAR___triptych_axes,
                (__VAR___linear_png, __VAR___log_png, __VAR___corr_png),
                ('Linear scale', 'Log scale', 'Average fit correlation'),
            ):
                __VAR___image = plt.imread(str(__VAR___image_path))
                __VAR___axis.imshow(__VAR___image)
                __VAR___axis.set_title(__VAR___panel_title)
                __VAR___axis.axis('off')
            __VAR___triptych_fig.suptitle(__VAR___plot_title, fontsize=14, fontweight='bold', y=1.02)
            __VAR___triptych_fig.tight_layout()
            __VAR___triptych_png = __VAR___study_dir / f"{__VAR___study_dir.name}__section_triptych.png"
            __VAR___triptych_pdf = __VAR___study_dir / f"{__VAR___study_dir.name}__section_triptych.pdf"
            __VAR___triptych_fig.savefig(__VAR___triptych_png, dpi=220)
            __VAR___triptych_fig.savefig(__VAR___triptych_pdf)

            display(__VAR___linear_fig)
            display(__VAR___log_fig)
            display(__VAR___corr_fig)
            display(__VAR___triptych_fig)
            plt.close(__VAR___linear_fig)
            plt.close(__VAR___log_fig)
            plt.close(__VAR___corr_fig)
            plt.close(__VAR___triptych_fig)

            __VAR___plot_paths = {
                'linear_png': __VAR___linear_png.resolve().as_posix(),
                'linear_pdf': __VAR___linear_pdf.resolve().as_posix(),
                'log_png': __VAR___log_png.resolve().as_posix(),
                'log_pdf': __VAR___log_pdf.resolve().as_posix(),
                'corr_png': __VAR___corr_png.resolve().as_posix(),
                'corr_pdf': __VAR___corr_pdf.resolve().as_posix(),
                'triptych_png': __VAR___triptych_png.resolve().as_posix(),
                'triptych_pdf': __VAR___triptych_pdf.resolve().as_posix(),
            }
            __VAR___plot_paths
            """
        ).strip()
        return (
            template.replace("__VAR__", var_name)
            .replace("__X_LABEL__", json.dumps(x_label))
            .replace("__Y_LABEL__", json.dumps(y_label))
        )

    cells = [
        _new_markdown_cell(
            "# THz-TDS Lecture Notebook\n\n"
            "This notebook is organized as truly section-local calculation and plotting blocks. "
            "Each section owns its own plotting code so you can customize one section without editing the others."
        ),
        _new_markdown_cell(
            "> Generated notebook notice\n"
            ">\n"
            "> This notebook is written by `docs/lecture_assets_v2.py` via `_write_lecture_notebook()`.\n"
            "> If you want your plot or cell edits to persist, update the generator source instead of only editing this `.ipynb`."
        ),
        _new_markdown_cell(
            "## Imports\n\n"
            "Only common imports and shared constants live here. The plotting logic itself is written inside each section."
        ),
        _new_code_cell(import_source),
        _new_markdown_cell(
            "## Smoke Build\n\n"
            "Use this first if you want a compact end-to-end validation of the lecture asset pipeline."
        ),
        _new_code_cell(
            "smoke_result = build_notebook_demo_bundle()\n"
            "smoke_result['output_root']",
            metadata={"skip_build_execution": True},
        ),
        _new_markdown_cell(
            "## Measured Transmission Example\n\n"
            "The next cell calculates the measured transmission example and saves the section data. "
            "The plotting cell after that contains the full figure code for this section only."
        ),
        _new_code_cell(_measured_tx_calc_source()),
        _new_markdown_cell(
            "Edit only this section's plotting cell if you want the measured transmission figures to look different from the other sections."
        ),
        _new_code_cell(_measured_tx_plot_source()),
        _new_markdown_cell(
            "## Measured Reflection Example\n\n"
            "The next cell calculates the measured reflection example and saves the section data. "
            "The plotting cell after that contains the full figure code for this section only."
        ),
        _new_code_cell(_measured_refl_calc_source()),
        _new_markdown_cell(
            "Edit only this section's plotting cell if you want the measured reflection figures to look different from the other sections."
        ),
        _new_code_cell(_measured_refl_plot_source()),
        _new_markdown_cell(
            "## Quick Study Demo Blocks\n\n"
            "Each demo has its own explicit calculation cell and its own explicit plotting cell."
        ),
    ]

    quick_specs = [
        ("Quick One-Layer Demo", demo_one_layer, "smoke", "quick"),
        ("Quick Advanced Demo", demo_advanced, "smoke", "quick"),
    ]
    for section_title, spec, profile, namespace in quick_specs:
        var_name = _identifier(f"{namespace}_{spec['slug']}")
        output_tag = f"{namespace}_{spec['slug']}"
        x_label, y_label = _study_axis_labels(spec)
        cells.extend(
            [
                _new_markdown_cell(f"### {section_title}"),
                _new_code_cell(_study_calc_source(var_name, spec, profile=profile, output_tag=output_tag)),
                _new_markdown_cell(
                    "The next cell contains the full plotting code for this section. "
                    "Change it here without affecting the other study sections."
                ),
                _new_code_cell(_study_plot_source(var_name, x_label, y_label)),
            ]
        )

    cells.append(
        _new_markdown_cell(
            "## Full-Resolution One-Layer Study Blocks\n\n"
            "Every block below contains its own calculation cell and its own complete plotting cell."
        )
    )
    for spec in one_layer_profiles:
        var_name = _identifier(f"full_{spec['slug']}")
        output_tag = f"full_{spec['slug']}"
        x_label, y_label = _study_axis_labels(spec)
        cells.extend(
            [
                _new_markdown_cell(f"### {spec['title']}"),
                _new_code_cell(
                    _study_calc_source(var_name, spec, profile="full", output_tag=output_tag),
                    metadata={"skip_build_execution": True},
                ),
                _new_markdown_cell(
                    "This plotting cell is section-local. You can rewrite this block without changing the other one-layer study plots."
                ),
                _new_code_cell(
                    _study_plot_source(var_name, x_label, y_label),
                    metadata={"skip_build_execution": True},
                ),
            ]
        )

    cells.append(
        _new_markdown_cell(
            "## Full-Resolution Advanced Epi Study Blocks\n\n"
            "Every block below contains its own calculation cell and its own complete plotting cell."
        )
    )
    for spec in advanced_profiles:
        var_name = _identifier(f"full_{spec['slug']}")
        output_tag = f"full_{spec['slug']}"
        x_label, y_label = _study_axis_labels(spec)
        cells.extend(
            [
                _new_markdown_cell(f"### {spec['title']}"),
                _new_code_cell(
                    _study_calc_source(var_name, spec, profile="full", output_tag=output_tag),
                    metadata={"skip_build_execution": True},
                ),
                _new_markdown_cell(
                    "This plotting cell is section-local. You can rewrite this block without changing the other advanced study plots."
                ),
                _new_code_cell(
                    _study_plot_source(var_name, x_label, y_label),
                    metadata={"skip_build_execution": True},
                ),
            ]
        )

    notebook = _new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return NOTEBOOK_PATH


def _render_notes_for_build(build_root: Path, manifest: dict):
    repo_source = _notes_tex_source(manifest, graphics_root="lecture_build/latest/figures/")
    NOTES_TEX_PATH.write_text(repo_source, encoding="utf-8")
    build_source = _notes_tex_source(manifest, graphics_root="../figures/")
    build_notes_path = build_root / "latex_source" / NOTES_TEX_PATH.name
    build_notes_path.parent.mkdir(parents=True, exist_ok=True)
    build_notes_path.write_text(build_source, encoding="utf-8")
    return {
        "repo_notes_tex": NOTES_TEX_PATH.resolve().as_posix(),
        "build_notes_tex": build_notes_path.resolve().as_posix(),
        "notebook": _write_lecture_notebook().resolve().as_posix(),
    }


def _compile_notes(build_root: Path, tex_path: Path):
    latex_dir = build_root / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={latex_dir.resolve().as_posix()}",
        tex_path.resolve().as_posix(),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    pdf_path = latex_dir / tex_path.with_suffix(".pdf").name
    return {"lecture_thz_tds_fit_study_notes": pdf_path.resolve().as_posix()}


def _execute_notebook(build_root: Path):
    notebook_outputs = build_root / "notebooks"
    notebook_outputs.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    execution_globals = {
        "__name__": "__main__",
        "__file__": NOTEBOOK_PATH.resolve().as_posix(),
    }
    execution_locals = execution_globals
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if bool(cell.get("metadata", {}).get("skip_build_execution", False)):
            continue
        source = str(cell.get("source", ""))
        if not source.strip():
            continue
        exec(compile(source, NOTEBOOK_PATH.resolve().as_posix(), "exec"), execution_globals, execution_locals)
    notebook.setdefault("metadata", {})["executed_by"] = "docs/lecture_assets_v2.py"
    executed_path = notebook_outputs / "THzTDS_Lecture_Fit_Study.executed.ipynb"
    executed_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return executed_path.resolve().as_posix()


def _profile_settings(profile: str) -> dict:
    profile = str(profile).strip().lower()
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'")
    return {
        "profile": profile,
        "fit_reflection_counts": (0, 2) if profile == "smoke" else (0, 2, 4, 8),
        "noise_db": 95.0,
        "study_optimizer": {
            "global_method": "none",
            "method": "L-BFGS-B",
            "options": {"maxiter": 60 if profile == "smoke" else 90},
            "fd_rel_step": 1e-5,
        },
    }


def _estimate_runtime_hours(profile_settings: dict) -> float:
    one_layer_map_count = len(_one_layer_map_specs(profile=profile_settings["profile"]))
    advanced_map_count = len(_advanced_map_specs(profile=profile_settings["profile"]))
    one_cases = len(_one_layer_map_specs(profile=profile_settings["profile"])[0]["tau_values"]) ** 2 * one_layer_map_count
    advanced_cases = len(_advanced_map_specs(profile=profile_settings["profile"])[0]["x_values"]) ** 2 * advanced_map_count
    return float((one_cases * 2.6 + advanced_cases * 3.8) / 3600.0)


def plot_saved_lecture_study_triptych(study_dir) -> dict[str, str]:
    study_dir = Path(study_dir)
    slug = study_dir.name
    linear_png = study_dir / f"{slug}__linear.png"
    log_png = study_dir / f"{slug}__log.png"
    corr_png = study_dir / f"{slug}__corr.png"
    title = json.loads((study_dir / "study_summary.json").read_text(encoding="utf-8"))["title"]
    figure = _plot_triptych(linear_png, log_png, corr_png, title=title)
    return _save_figure(figure, study_dir / f"{slug}__triptych_replot")


def _run_lecture_map_from_spec_internal(
    spec: dict,
    *,
    output_root: Path | None = None,
    profile: str = "full",
) -> dict:
    profile_settings = _profile_settings(profile)
    _lecture_style()
    if output_root is None:
        output_root = LECTURE_BUILD_ROOT / "notebook_custom_maps"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reference_result = _prepare_reference_from_config(
        _reference_input_config(),
        output_root=output_root / "runs",
        run_label=f"{spec['slug']}_reference",
    )
    if str(spec["slug"]).startswith("one_layer"):
        nominal_sigma = math.sqrt(min(spec["sigma_values"]) * max(spec["sigma_values"]))
        sample_result = _one_layer_fit_sample(
            reference_result,
            thickness_um=float(spec["thickness_um"]),
            sigma_nominal=nominal_sigma,
            tau_nominal=0.316,
            output_dir=output_root / "runs" / spec["slug"],
        )
        measurement = Measurement(
            mode=str(spec["mode"]),
            angle_deg=float(spec["angle_deg"]),
            polarization=str(spec["polarization"]),
            reference_standard=ReferenceStandard(kind="identity"),
        )

        def truth_builder(tau_ps, sigma_s_per_m, *, thickness_um=float(spec["thickness_um"])):
            return (
                {
                    "layers[0].thickness_um": thickness_um,
                    "layers[0].material.plasma_freq_thz": drude_plasma_freq_thz_from_sigma_tau(sigma_s_per_m, tau_ps),
                    "layers[0].material.gamma_thz": drude_gamma_thz_from_tau_ps(tau_ps),
                },
                {
                    "thickness_um": thickness_um,
                    "eps_inf": 11.7,
                    "tau_ps": tau_ps,
                    "sigma_s_per_m": sigma_s_per_m,
                },
            )

        lecture_spec = LectureGridStudySpec(
            slug=str(spec["slug"]),
            title=str(spec["title"]),
            section="one_layer",
            x_name="tau_ps",
            y_name="sigma_s_per_m",
            x_values=list(spec["tau_values"]),
            y_values=list(spec["sigma_values"]),
            x_label=r"$\tau$ (ps)",
            y_label=r"$\sigma$ (S/m)",
            measurement=measurement,
            noise_dynamic_range_db=float(profile_settings["noise_db"]),
            max_internal_reflections=8,
            recovery_keys=("tau_ps", "sigma_s_per_m"),
            fixed_note=rf"$d={float(spec['thickness_um']):.1f}\,\mu\mathrm{{m}}$, $\varepsilon_\infty=11.7$",
            truth_update_builder=truth_builder,
            fit_summary_builder=lambda stack: summarize_single_layer_drude_stack(stack),
            spec_payload=deepcopy(spec),
        )
    else:
        substrate_thickness_um = float(spec["substrate_thickness_um"])
        epi_thickness_um = float(spec["epi_thickness_um"])
        oxide_thickness_um = 10.0
        if spec["map_kind"] == "tau":
            sigma_fixed = float(spec["sigma_fixed"])
            sample_result = _advanced_fit_sample(
                reference_result,
                epi_thickness_um=epi_thickness_um,
                substrate_thickness_um=substrate_thickness_um,
                oxide_thickness_um=oxide_thickness_um,
                tau1_nominal=0.316,
                sigma1_nominal=sigma_fixed,
                tau2_nominal=0.316,
                sigma2_nominal=sigma_fixed,
                output_dir=output_root / "runs" / spec["slug"],
            )

            def truth_builder(value_x, value_y, *, sigma_fixed=sigma_fixed):
                return (
                    {
                        "layers[0].thickness_um": epi_thickness_um,
                        "layers[0].material.plasma_freq1_thz": drude_plasma_freq_thz_from_sigma_tau(sigma_fixed, value_x),
                        "layers[0].material.gamma1_thz": drude_gamma_thz_from_tau_ps(value_x),
                        "layers[0].material.plasma_freq2_thz": drude_plasma_freq_thz_from_sigma_tau(sigma_fixed, value_y),
                        "layers[0].material.gamma2_thz": drude_gamma_thz_from_tau_ps(value_y),
                        "layers[1].thickness_um": oxide_thickness_um,
                        "layers[2].thickness_um": substrate_thickness_um,
                    },
                    {
                        "epi_thickness_um": epi_thickness_um,
                        "oxide_thickness_um": oxide_thickness_um,
                        "substrate_thickness_um": substrate_thickness_um,
                        "eps_inf": 11.7,
                        "tau1_ps": value_x,
                        "sigma1_s_per_m": sigma_fixed,
                        "tau2_ps": value_y,
                        "sigma2_s_per_m": sigma_fixed,
                    },
                )

            x_name = "tau1_ps"
            y_name = "tau2_ps"
            x_label = r"$\tau_1$ (ps)"
            y_label = r"$\tau_2$ (ps)"
            recovery_keys = ("tau1_ps", "tau2_ps")
        else:
            tau_fixed = float(spec["tau_fixed"])
            sigma_nominal = math.sqrt(min(spec["x_values"]) * max(spec["x_values"]))
            sample_result = _advanced_fit_sample(
                reference_result,
                epi_thickness_um=epi_thickness_um,
                substrate_thickness_um=substrate_thickness_um,
                oxide_thickness_um=oxide_thickness_um,
                tau1_nominal=tau_fixed,
                sigma1_nominal=sigma_nominal,
                tau2_nominal=tau_fixed,
                sigma2_nominal=sigma_nominal,
                output_dir=output_root / "runs" / spec["slug"],
            )

            def truth_builder(value_x, value_y, *, tau_fixed=tau_fixed):
                return (
                    {
                        "layers[0].thickness_um": epi_thickness_um,
                        "layers[0].material.plasma_freq1_thz": drude_plasma_freq_thz_from_sigma_tau(value_x, tau_fixed),
                        "layers[0].material.gamma1_thz": drude_gamma_thz_from_tau_ps(tau_fixed),
                        "layers[0].material.plasma_freq2_thz": drude_plasma_freq_thz_from_sigma_tau(value_y, tau_fixed),
                        "layers[0].material.gamma2_thz": drude_gamma_thz_from_tau_ps(tau_fixed),
                        "layers[1].thickness_um": oxide_thickness_um,
                        "layers[2].thickness_um": substrate_thickness_um,
                    },
                    {
                        "epi_thickness_um": epi_thickness_um,
                        "oxide_thickness_um": oxide_thickness_um,
                        "substrate_thickness_um": substrate_thickness_um,
                        "eps_inf": 11.7,
                        "tau1_ps": tau_fixed,
                        "sigma1_s_per_m": value_x,
                        "tau2_ps": tau_fixed,
                        "sigma2_s_per_m": value_y,
                    },
                )

            x_name = "sigma1_s_per_m"
            y_name = "sigma2_s_per_m"
            x_label = r"$\sigma_1$ (S/m)"
            y_label = r"$\sigma_2$ (S/m)"
            recovery_keys = ("sigma1_s_per_m", "sigma2_s_per_m")

        if spec["mode"] == "reflection":
            reference_standard = ReferenceStandard(
                kind="stack",
                stack=_advanced_reflection_standard(
                    reference_result,
                    substrate_thickness_um=substrate_thickness_um,
                    oxide_thickness_um=oxide_thickness_um,
                    output_dir=output_root / "runs" / f"{spec['slug']}_standard",
                ),
            )
        else:
            reference_standard = ReferenceStandard(kind="identity")
        measurement = Measurement(
            mode=str(spec["mode"]),
            angle_deg=float(spec["angle_deg"]),
            polarization=str(spec["polarization"]),
            reference_standard=reference_standard,
        )
        lecture_spec = LectureGridStudySpec(
            slug=str(spec["slug"]),
            title=str(spec["title"]),
            section="advanced",
            x_name=x_name,
            y_name=y_name,
            x_values=list(spec["x_values"]),
            y_values=list(spec["y_values"]),
            x_label=x_label,
            y_label=y_label,
            measurement=measurement,
            noise_dynamic_range_db=float(profile_settings["noise_db"]),
            max_internal_reflections=4,
            recovery_keys=recovery_keys,
            fixed_note="Lecture custom study",
            truth_update_builder=truth_builder,
            fit_summary_builder=_summarize_advanced_stack,
            spec_payload=deepcopy(spec),
        )

    return _run_saved_grid_study(
        build_root=output_root,
        reference_result=reference_result,
        sample_result=sample_result,
        spec=lecture_spec,
        optimizer=dict(profile_settings["study_optimizer"]),
        weighting={"mode": "trace_amplitude", "floor": 0.03, "power": 2.0, "smooth_window_samples": 41},
    )


def run_lecture_map_from_spec(spec: dict, *, output_root: Path | None = None, profile: str = "full") -> dict:
    return _run_lecture_map_from_spec_internal(spec, output_root=output_root, profile=profile)


def build_lecture_bundle(
    *,
    profile: str,
    output_root: Path | None = None,
    update_latest: bool = True,
    compile_latex: bool = True,
    execute_notebook: bool = True,
) -> dict:
    _lecture_style()
    settings = _profile_settings(profile)
    if output_root is None:
        output_root = LECTURE_BUILD_ROOT / f"{_timestamp_slug()}__{profile}"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "profile": profile,
        "created_at": datetime.now().astimezone().isoformat(),
        "output_root": output_root.resolve().as_posix(),
        "estimated_runtime_hours": float(_estimate_runtime_hours(settings)),
        "stages": {},
        "key_figures": {},
        "key_figure_titles": {},
        "study_groups": {"one_layer": [], "advanced": []},
    }
    manifest_path = output_root / "manifest.json"
    _json_dump(manifest_path, manifest)

    stage_start = time.perf_counter()
    measured_figures = _run_measured_examples(output_root, settings)
    manifest["stages"]["measured_examples"] = {"duration_s": time.perf_counter() - stage_start}
    manifest["key_figures"].update(measured_figures)
    manifest["key_figure_titles"].update(
        {
            "fit_a11013460_overview": "Measured transmission fit: A11013460",
            "fit_a11013460_correlation": "Measured transmission correlation matrix",
            "fit_a11013460_reflection_overview": "Measured reflection fit: A11013460 mirror-reference pair",
            "fit_a11013460_reflection_correlation": "Measured reflection correlation matrix",
        }
    )
    _json_dump(manifest_path, manifest)

    stage_start = time.perf_counter()
    one_layer_results = _run_one_layer_studies(output_root, settings)
    manifest["stages"]["one_layer_studies"] = {
        "duration_s": time.perf_counter() - stage_start,
        "study_count": len(one_layer_results),
    }
    for slug, result in one_layer_results.items():
        manifest["key_figures"][slug] = {
            "png": result["figure_triptych_png"],
            "pdf": result["figure_triptych_pdf"],
        }
        manifest["key_figure_titles"][slug] = result["title"]
        manifest["study_groups"]["one_layer"].append(slug)
    _json_dump(manifest_path, manifest)

    stage_start = time.perf_counter()
    advanced_results = _run_advanced_studies(output_root, settings)
    manifest["stages"]["advanced_studies"] = {
        "duration_s": time.perf_counter() - stage_start,
        "study_count": len(advanced_results),
    }
    for slug, result in advanced_results.items():
        manifest["key_figures"][slug] = {
            "png": result["figure_triptych_png"],
            "pdf": result["figure_triptych_pdf"],
        }
        manifest["key_figure_titles"][slug] = result["title"]
        manifest["study_groups"]["advanced"].append(slug)
    _json_dump(manifest_path, manifest)

    manifest["sources"] = _render_notes_for_build(output_root, manifest)
    _json_dump(manifest_path, manifest)

    if update_latest:
        _copy_tree_latest(output_root, LATEST_BUILD_ROOT)
        manifest["latest_build"] = LATEST_BUILD_ROOT.resolve().as_posix()
        _json_dump(manifest_path, manifest)

    if compile_latex:
        stage_start = time.perf_counter()
        manifest["latex_outputs"] = _compile_notes(output_root, Path(manifest["sources"]["build_notes_tex"]))
        manifest["stages"]["latex"] = {"duration_s": time.perf_counter() - stage_start}
        _json_dump(manifest_path, manifest)

    if execute_notebook:
        stage_start = time.perf_counter()
        manifest["executed_notebook"] = _execute_notebook(output_root)
        manifest["stages"]["notebook"] = {"duration_s": time.perf_counter() - stage_start}
        _json_dump(manifest_path, manifest)

    return manifest


def build_notebook_demo_bundle(output_root: Path | None = None) -> dict:
    if output_root is None:
        output_root = LECTURE_BUILD_ROOT / "notebook_demo"
    return build_lecture_bundle(
        profile="smoke",
        output_root=output_root,
        update_latest=False,
        compile_latex=False,
        execute_notebook=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate the rewritten THz-TDS lecture notes, notebook, and study assets.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-root", default=None, help="Optional explicit output directory for the build.")
    parser.add_argument("--resume", default=None, help="Reserved for future checkpoint resume support.")
    args = parser.parse_args()

    if args.resume not in {None, "", "latest"}:
        raise ValueError("--resume currently supports only the placeholder value 'latest'")

    result = build_lecture_bundle(
        profile=args.profile,
        output_root=None if args.output_root is None else Path(args.output_root),
        update_latest=True,
        compile_latex=True,
        execute_notebook=True,
    )
    print(json.dumps({"output_root": result["output_root"], "estimated_runtime_hours": result["estimated_runtime_hours"]}, indent=2))


__all__ = [
    "build_lecture_bundle",
    "build_notebook_demo_bundle",
    "plot_saved_lecture_study_triptych",
    "run_lecture_map_from_spec",
    "main",
]
