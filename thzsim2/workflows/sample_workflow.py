from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from thzsim2.core.materials import evaluate_material_nk
from thzsim2.core.stack import validate_stack
from thzsim2.io.manifests import build_sample_manifest, update_run_manifest, write_json
from thzsim2.io.nk_csv import NKData, read_nk_csv, write_nk_csv
from thzsim2.io.run_folders import slugify
from thzsim2.io.summaries import write_sample_structure_txt
from thzsim2.models import (
    ConstantNK,
    Drude,
    DrudeLorentz,
    Fit,
    Layer,
    Lorentz,
    LorentzOscillator,
    NKFile,
    ReferenceResult,
    ResolvedFitParameter,
    SampleLayerResult,
    SampleResult,
)


def _coerce_freq_grid(freq_grid_thz):
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


def _select_freq_grid(reference, freq_grid_thz):
    if freq_grid_thz is not None:
        return _coerce_freq_grid(freq_grid_thz), "explicit_freq_grid_thz"
    if reference is None:
        raise ValueError("build_sample requires either reference=... or freq_grid_thz=...")
    if not isinstance(reference, ReferenceResult):
        raise TypeError("reference must be a ReferenceResult when provided")
    return _coerce_freq_grid(reference.spectrum.freq_thz), "reference.spectrum.freq_thz"


def _make_fit_key(label: str | None, path: str):
    return str(label) if label else slugify(path)


def _resolve_value(value, *, path: str, unit: str, layer_name: str, fit_registry: list[ResolvedFitParameter], used_keys: set[str]):
    if isinstance(value, Fit):
        if value.resolved_min is None or value.resolved_max is None:
            raise ValueError(
                f"Layer '{layer_name}' fit parameter '{path}' must define both lower and upper bounds"
            )
        initial = float(value.initial)
        bound_min = float(value.resolved_min)
        bound_max = float(value.resolved_max)
        if not (bound_min <= initial <= bound_max):
            raise ValueError(
                f"Layer '{layer_name}' fit parameter '{path}' must have initial value inside its bounds"
            )
        key = _make_fit_key(value.label, path)
        if key in used_keys:
            raise ValueError(f"Duplicate fit parameter key '{key}' is not allowed")
        used_keys.add(key)
        fit_parameter = ResolvedFitParameter(
            key=key,
            label=value.label or key,
            path=path,
            unit=unit,
            initial_value=initial,
            bound_min=bound_min,
            bound_max=bound_max,
            layer_name=layer_name,
        )
        fit_registry.append(fit_parameter)
        return initial, fit_parameter
    return float(value), None


def _fit_record_or_none(fit_parameter: ResolvedFitParameter | None):
    if fit_parameter is None:
        return None
    return {
        "key": fit_parameter.key,
        "label": fit_parameter.label,
        "path": fit_parameter.path,
        "bound_min": fit_parameter.bound_min,
        "bound_max": fit_parameter.bound_max,
    }


def _parameter_entry(name: str, value: float, unit: str, fit_parameter: ResolvedFitParameter | None):
    return {
        "name": name,
        "value": float(value),
        "unit": unit,
        "fit": _fit_record_or_none(fit_parameter),
    }


def _linear_interp_extrap(x, y, new_x):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    new_x = np.asarray(new_x, dtype=np.float64)
    if x.size == 1:
        return np.full(new_x.shape, float(y[0]), dtype=np.float64)

    values = np.interp(new_x, x, y)
    left = new_x < x[0]
    right = new_x > x[-1]

    left_slope = (y[1] - y[0]) / (x[1] - x[0])
    right_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])

    values[left] = y[0] + left_slope * (new_x[left] - x[0])
    values[right] = y[-1] + right_slope * (new_x[right] - x[-1])
    return values


def _resolve_imported_nk(material: NKFile, freq_grid_thz):
    nk_data = read_nk_csv(material.path)
    aligned_n = _linear_interp_extrap(nk_data.freq_thz, nk_data.n, freq_grid_thz)
    aligned_k = _linear_interp_extrap(nk_data.freq_thz, nk_data.k, freq_grid_thz)
    return nk_data, aligned_n, aligned_k


def _resolve_material(layer: Layer, layer_index: int, freq_grid_thz, fit_registry, used_keys):
    material = layer.material
    layer_path = f"layers[{layer_index}]"
    layer_fit_parameters = []

    if isinstance(material, NKFile):
        imported, aligned_n, aligned_k = _resolve_imported_nk(material, freq_grid_thz)
        parameters = []
        resolved_material = {
            "kind": "NKFile",
            "parameters": {},
            "source_nk_file": str(material.path.resolve()),
            "freq_thz": np.asarray(freq_grid_thz, dtype=np.float64).tolist(),
            "n": np.asarray(aligned_n, dtype=np.float64).tolist(),
            "k": np.asarray(aligned_k, dtype=np.float64).tolist(),
        }
        return (
            "NKFile",
            parameters,
            aligned_n,
            aligned_k,
            imported,
            layer_fit_parameters,
            resolved_material,
        )

    if isinstance(material, ConstantNK):
        n_value, n_fit = _resolve_value(
            material.n,
            path=f"{layer_path}.material.n",
            unit="",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        k_value, k_fit = _resolve_value(
            material.k,
            path=f"{layer_path}.material.k",
            unit="",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        if n_fit is not None:
            layer_fit_parameters.append(n_fit)
        if k_fit is not None:
            layer_fit_parameters.append(k_fit)
        resolved_material = ConstantNK(n=n_value, k=k_value)
        nk = evaluate_material_nk(freq_grid_thz, resolved_material)
        return (
            "ConstantNK",
            [
                _parameter_entry("n", n_value, "", n_fit),
                _parameter_entry("k", k_value, "", k_fit),
            ],
            np.real(nk),
            np.imag(nk),
            None,
            layer_fit_parameters,
            {
                "kind": "ConstantNK",
                "parameters": {"n": n_value, "k": k_value},
                "source_nk_file": None,
            },
        )

    if isinstance(material, Drude):
        eps_inf, eps_fit = _resolve_value(
            material.eps_inf,
            path=f"{layer_path}.material.eps_inf",
            unit="",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        plasma, plasma_fit = _resolve_value(
            material.plasma_freq_thz,
            path=f"{layer_path}.material.plasma_freq_thz",
            unit="THz",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        gamma, gamma_fit = _resolve_value(
            material.gamma_thz,
            path=f"{layer_path}.material.gamma_thz",
            unit="THz",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        for fit_parameter in (eps_fit, plasma_fit, gamma_fit):
            if fit_parameter is not None:
                layer_fit_parameters.append(fit_parameter)
        resolved_material = Drude(eps_inf=eps_inf, plasma_freq_thz=plasma, gamma_thz=gamma)
        nk = evaluate_material_nk(freq_grid_thz, resolved_material)
        return (
            "Drude",
            [
                _parameter_entry("eps_inf", eps_inf, "", eps_fit),
                _parameter_entry("plasma_freq_thz", plasma, "THz", plasma_fit),
                _parameter_entry("gamma_thz", gamma, "THz", gamma_fit),
            ],
            np.real(nk),
            np.imag(nk),
            None,
            layer_fit_parameters,
            {
                "kind": "Drude",
                "parameters": {
                    "eps_inf": eps_inf,
                    "plasma_freq_thz": plasma,
                    "gamma_thz": gamma,
                },
                "source_nk_file": None,
            },
        )

    if isinstance(material, Lorentz):
        eps_inf, eps_fit = _resolve_value(
            material.eps_inf,
            path=f"{layer_path}.material.eps_inf",
            unit="",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        delta_eps, delta_fit = _resolve_value(
            material.delta_eps,
            path=f"{layer_path}.material.delta_eps",
            unit="",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        resonance, resonance_fit = _resolve_value(
            material.resonance_thz,
            path=f"{layer_path}.material.resonance_thz",
            unit="THz",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        gamma, gamma_fit = _resolve_value(
            material.gamma_thz,
            path=f"{layer_path}.material.gamma_thz",
            unit="THz",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        for fit_parameter in (eps_fit, delta_fit, resonance_fit, gamma_fit):
            if fit_parameter is not None:
                layer_fit_parameters.append(fit_parameter)
        resolved_material = Lorentz(
            eps_inf=eps_inf,
            delta_eps=delta_eps,
            resonance_thz=resonance,
            gamma_thz=gamma,
        )
        nk = evaluate_material_nk(freq_grid_thz, resolved_material)
        return (
            "Lorentz",
            [
                _parameter_entry("eps_inf", eps_inf, "", eps_fit),
                _parameter_entry("delta_eps", delta_eps, "", delta_fit),
                _parameter_entry("resonance_thz", resonance, "THz", resonance_fit),
                _parameter_entry("gamma_thz", gamma, "THz", gamma_fit),
            ],
            np.real(nk),
            np.imag(nk),
            None,
            layer_fit_parameters,
            {
                "kind": "Lorentz",
                "parameters": {
                    "eps_inf": eps_inf,
                    "delta_eps": delta_eps,
                    "resonance_thz": resonance,
                    "gamma_thz": gamma,
                },
                "source_nk_file": None,
            },
        )

    if isinstance(material, DrudeLorentz):
        eps_inf, eps_fit = _resolve_value(
            material.eps_inf,
            path=f"{layer_path}.material.eps_inf",
            unit="",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        plasma, plasma_fit = _resolve_value(
            material.plasma_freq_thz,
            path=f"{layer_path}.material.plasma_freq_thz",
            unit="THz",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        gamma, gamma_fit = _resolve_value(
            material.gamma_thz,
            path=f"{layer_path}.material.gamma_thz",
            unit="THz",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )
        for fit_parameter in (eps_fit, plasma_fit, gamma_fit):
            if fit_parameter is not None:
                layer_fit_parameters.append(fit_parameter)

        resolved_oscillators = []
        parameter_entries = [
            _parameter_entry("eps_inf", eps_inf, "", eps_fit),
            _parameter_entry("plasma_freq_thz", plasma, "THz", plasma_fit),
            _parameter_entry("gamma_thz", gamma, "THz", gamma_fit),
        ]
        oscillator_entries = []
        for osc_index, oscillator in enumerate(material.oscillators):
            delta_eps, delta_fit = _resolve_value(
                oscillator.delta_eps,
                path=f"{layer_path}.material.oscillators[{osc_index}].delta_eps",
                unit="",
                layer_name=layer.name,
                fit_registry=fit_registry,
                used_keys=used_keys,
            )
            resonance, resonance_fit = _resolve_value(
                oscillator.resonance_thz,
                path=f"{layer_path}.material.oscillators[{osc_index}].resonance_thz",
                unit="THz",
                layer_name=layer.name,
                fit_registry=fit_registry,
                used_keys=used_keys,
            )
            osc_gamma, osc_gamma_fit = _resolve_value(
                oscillator.gamma_thz,
                path=f"{layer_path}.material.oscillators[{osc_index}].gamma_thz",
                unit="THz",
                layer_name=layer.name,
                fit_registry=fit_registry,
                used_keys=used_keys,
            )
            for fit_parameter in (delta_fit, resonance_fit, osc_gamma_fit):
                if fit_parameter is not None:
                    layer_fit_parameters.append(fit_parameter)
            resolved_oscillators.append(
                LorentzOscillator(
                    delta_eps=delta_eps,
                    resonance_thz=resonance,
                    gamma_thz=osc_gamma,
                )
            )
            oscillator_entries.append(
                {
                    "index": osc_index,
                    "delta_eps": _parameter_entry("delta_eps", delta_eps, "", delta_fit),
                    "resonance_thz": _parameter_entry("resonance_thz", resonance, "THz", resonance_fit),
                    "gamma_thz": _parameter_entry("gamma_thz", osc_gamma, "THz", osc_gamma_fit),
                }
            )

        resolved_material = DrudeLorentz(
            eps_inf=eps_inf,
            plasma_freq_thz=plasma,
            gamma_thz=gamma,
            oscillators=tuple(resolved_oscillators),
        )
        nk = evaluate_material_nk(freq_grid_thz, resolved_material)
        return (
            "DrudeLorentz",
            parameter_entries,
            np.real(nk),
            np.imag(nk),
            None,
            layer_fit_parameters,
            {
                "kind": "DrudeLorentz",
                "parameters": {
                    "eps_inf": eps_inf,
                    "plasma_freq_thz": plasma,
                    "gamma_thz": gamma,
                    "oscillators": [
                        {
                            "delta_eps": float(oscillator.delta_eps),
                            "resonance_thz": float(oscillator.resonance_thz),
                            "gamma_thz": float(oscillator.gamma_thz),
                        }
                        for oscillator in resolved_oscillators
                    ],
                },
                "source_nk_file": None,
            },
        )

    raise TypeError(f"Layer '{layer.name}' material has unsupported type {type(material).__name__}")


def _plot_sample_nk(path, layers: list[SampleLayerResult], *, overlay_imported: bool):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for index, layer in enumerate(layers):
        color = f"C{index % 10}"
        mask = layer.freq_thz > 0.0
        freq = layer.freq_thz[mask]
        n_values = layer.n[mask]
        k_values = layer.k[mask]
        axes[0].plot(freq, n_values, label=layer.name, color=color, linewidth=1.8)
        axes[1].plot(freq, k_values, label=layer.name, color=color, linewidth=1.8)

        if overlay_imported and layer.imported_freq_thz is not None:
            imported_mask = layer.imported_freq_thz > 0.0
            axes[0].plot(
                layer.imported_freq_thz[imported_mask],
                layer.imported_n[imported_mask],
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.8,
            )
            axes[1].plot(
                layer.imported_freq_thz[imported_mask],
                layer.imported_k[imported_mask],
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.8,
            )

    axes[0].set_ylabel("n")
    axes[1].set_ylabel("k")
    axes[1].set_xlabel("Frequency (THz)")
    axes[0].set_title("Sample n,k Curves")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    if layers:
        axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _relative_path(path: Path, root: Path):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_sample(
    layers,
    *,
    reference=None,
    freq_grid_thz=None,
    out_dir,
    n_in=1.0,
    n_out=1.0,
    overlay_imported=True,
):
    """Resolve, export, and summarize a sample definition on a target frequency grid."""
    validate_stack(layers, n_in=n_in, n_out=n_out)
    freq_grid, grid_source = _select_freq_grid(reference, freq_grid_thz)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().astimezone().isoformat()

    fit_registry: list[ResolvedFitParameter] = []
    used_keys: set[str] = set()
    resolved_layers: list[SampleLayerResult] = []
    manifest_layers: list[dict] = []
    resolved_stack_layers: list[dict] = []
    artifact_paths: dict[str, Path] = {}

    for index, layer in enumerate(layers):
        thickness_um, thickness_fit = _resolve_value(
            layer.thickness_um,
            path=f"layers[{index}].thickness_um",
            unit="um",
            layer_name=layer.name,
            fit_registry=fit_registry,
            used_keys=used_keys,
        )

        material_kind, parameter_entries, n_values, k_values, imported, layer_fit_parameters, resolved_material = _resolve_material(
            layer,
            index,
            freq_grid,
            fit_registry,
            used_keys,
        )

        layer_slug = slugify(layer.name)
        nk_csv_path = out_dir / f"layer_{index + 1:02d}_{layer_slug}_nk.csv"
        export_mask = freq_grid > 0.0
        write_nk_csv(
            nk_csv_path,
            NKData(
                freq_thz=freq_grid[export_mask],
                n=np.asarray(n_values, dtype=np.float64)[export_mask],
                k=np.asarray(k_values, dtype=np.float64)[export_mask],
            ),
        )
        artifact_paths[f"layer_{index + 1:02d}_{layer_slug}_nk_csv"] = nk_csv_path

        layer_result = SampleLayerResult(
            index=index,
            name=layer.name,
            thickness_um=thickness_um,
            material_kind=material_kind,
            parameters={
                "thickness_um": thickness_um,
                "parameter_entries": parameter_entries,
                "resolved_material": resolved_material,
            },
            freq_thz=freq_grid.copy(),
            n=np.asarray(n_values, dtype=np.float64),
            k=np.asarray(k_values, dtype=np.float64),
            fit_parameters=layer_fit_parameters,
            imported_freq_thz=None if imported is None else imported.freq_thz.copy(),
            imported_n=None if imported is None else imported.n.copy(),
            imported_k=None if imported is None else imported.k.copy(),
        )
        resolved_layers.append(layer_result)

        manifest_layer = {
            "index": index,
            "name": layer.name,
            "thickness_um": thickness_um,
            "thickness_fit": _fit_record_or_none(thickness_fit),
            "material_kind": material_kind,
            "parameters": parameter_entries,
            "source_nk_file": None if imported is None else str(layer.material.path.resolve()),
            "export_nk_csv": nk_csv_path.name,
        }
        manifest_layers.append(manifest_layer)
        resolved_stack_layers.append(
            {
                "name": layer.name,
                "thickness_um": thickness_um,
                "material_kind": material_kind,
                "material": resolved_material,
                "nk_csv_path": nk_csv_path,
            }
        )

    structure_txt_path = out_dir / "sample_structure.txt"
    sample_plot_path = out_dir / "sample_nk.png"
    manifest_path = out_dir / "sample_manifest.json"

    write_sample_structure_txt(
        structure_txt_path,
        freq_grid_thz=freq_grid,
        n_in=n_in,
        n_out=n_out,
        layers=manifest_layers,
    )
    _plot_sample_nk(sample_plot_path, resolved_layers, overlay_imported=overlay_imported)

    artifact_paths["sample_structure_txt"] = structure_txt_path
    artifact_paths["sample_nk_png"] = sample_plot_path
    artifact_paths["sample_manifest_json"] = manifest_path

    files = {
        "sample_structure_txt": structure_txt_path.name,
        "sample_nk_png": sample_plot_path.name,
        "layer_nk_csvs": [
            {"name": layer.name, "path": manifest_layer["export_nk_csv"]}
            for layer, manifest_layer in zip(resolved_layers, manifest_layers, strict=True)
        ],
    }

    manifest = build_sample_manifest(
        created_at=created_at,
        freq_grid_thz=freq_grid,
        n_in=n_in,
        n_out=n_out,
        layers=manifest_layers,
        fit_parameters=[asdict(parameter) for parameter in fit_registry],
        files=files,
        grid_source=grid_source,
    )
    write_json(manifest_path, manifest)

    run_manifest_path = out_dir.parent / "run_manifest.json"
    if run_manifest_path.exists():
        update_run_manifest(
            run_manifest_path,
            sample_manifest=_relative_path(manifest_path, run_manifest_path.parent),
        )

    return SampleResult(
        out_dir=out_dir,
        freq_grid_thz=freq_grid.copy(),
        n_in=float(n_in),
        n_out=float(n_out),
        layers=resolved_layers,
        fit_parameters=list(fit_registry),
        resolved_stack={
            "n_in": float(n_in),
            "n_out": float(n_out),
            "layers": resolved_stack_layers,
        },
        manifest=manifest,
        artifact_paths=artifact_paths,
    )
