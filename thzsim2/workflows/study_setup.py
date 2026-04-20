from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from thzsim2.models import (
    ConstantNK,
    Drude,
    DrudeLorentz,
    Fit,
    Layer,
    Lorentz,
    LorentzOscillator,
    Measurement,
    NKFile,
    ReferenceStandard,
)
from thzsim2.io.manifests import write_json
from thzsim2.workflows.reference import generate_reference_pulse, load_reference_csv, prepare_reference
from thzsim2.workflows.sample_workflow import build_sample
from thzsim2.workflows.study_workflow import run_study


_SETUP_SCHEMA_VERSION = 1
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _normalize_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_probably_url(text: str) -> bool:
    return bool(_URL_RE.match(text)) or text.startswith("git+")


def _relative_path_text(value: str | Path, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        rel = os.path.relpath(path.resolve(), base_dir.resolve())
    except ValueError:
        return path.resolve().as_posix()
    return Path(rel).as_posix()


def _resolve_path_text(value: str | Path, base_dir: Path) -> str:
    text = str(value)
    if _is_probably_url(text):
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve().as_posix()
    return (base_dir / path).resolve().as_posix()


def _transform_path_fields(value, *, base_dir: Path, to_relative: bool):
    if isinstance(value, dict):
        transformed = {}
        for key, item in value.items():
            if item is None:
                transformed[key] = None
                continue
            if isinstance(item, (dict, list, tuple)):
                transformed[key] = _transform_path_fields(item, base_dir=base_dir, to_relative=to_relative)
                continue
            if isinstance(item, Path):
                transformed[key] = (
                    _relative_path_text(item, base_dir) if to_relative else _resolve_path_text(item, base_dir)
                )
                continue
            if isinstance(item, str) and (
                key == "path" or key.endswith("_path") or key.endswith("_dir") or key == "output_root"
            ):
                transformed[key] = (
                    item
                    if _is_probably_url(item)
                    else (_relative_path_text(item, base_dir) if to_relative else _resolve_path_text(item, base_dir))
                )
                continue
            transformed[key] = _normalize_scalar(item)
        return transformed

    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_transform_path_fields(item, base_dir=base_dir, to_relative=to_relative) for item in value]
    return _normalize_scalar(value)


def _fit_to_config(fit: Fit):
    return {
        "kind": "Fit",
        "initial": float(fit.initial),
        "rel_min": None if fit.rel_min is None else float(fit.rel_min),
        "rel_max": None if fit.rel_max is None else float(fit.rel_max),
        "abs_min": None if fit.abs_min is None else float(fit.abs_min),
        "abs_max": None if fit.abs_max is None else float(fit.abs_max),
        "label": fit.label,
    }


def _parameter_to_config(value):
    value = _normalize_scalar(value)
    if isinstance(value, Fit):
        return _fit_to_config(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _material_to_config(material):
    if isinstance(material, NKFile):
        return {"kind": "NKFile", "path": str(material.path)}
    if isinstance(material, ConstantNK):
        return {
            "kind": "ConstantNK",
            "n": _parameter_to_config(material.n),
            "k": _parameter_to_config(material.k),
        }
    if isinstance(material, Drude):
        return {
            "kind": "Drude",
            "eps_inf": _parameter_to_config(material.eps_inf),
            "plasma_freq_thz": _parameter_to_config(material.plasma_freq_thz),
            "gamma_thz": _parameter_to_config(material.gamma_thz),
        }
    if isinstance(material, Lorentz):
        return {
            "kind": "Lorentz",
            "eps_inf": _parameter_to_config(material.eps_inf),
            "delta_eps": _parameter_to_config(material.delta_eps),
            "resonance_thz": _parameter_to_config(material.resonance_thz),
            "gamma_thz": _parameter_to_config(material.gamma_thz),
        }
    if isinstance(material, DrudeLorentz):
        return {
            "kind": "DrudeLorentz",
            "eps_inf": _parameter_to_config(material.eps_inf),
            "plasma_freq_thz": _parameter_to_config(material.plasma_freq_thz),
            "gamma_thz": _parameter_to_config(material.gamma_thz),
            "oscillators": [
                {
                    "delta_eps": _parameter_to_config(osc.delta_eps),
                    "resonance_thz": _parameter_to_config(osc.resonance_thz),
                    "gamma_thz": _parameter_to_config(osc.gamma_thz),
                }
                for osc in material.oscillators
            ],
        }
    if isinstance(material, dict):
        return deepcopy(material)
    raise TypeError(f"Unsupported material type {type(material).__name__}")


def _layer_to_config(layer):
    if isinstance(layer, Layer):
        return {
            "name": layer.name,
            "thickness_um": _parameter_to_config(layer.thickness_um),
            "material": _material_to_config(layer.material),
        }
    if isinstance(layer, dict):
        return deepcopy(layer)
    raise TypeError(f"Unsupported layer type {type(layer).__name__}")


def _sample_to_config(
    layers,
    *,
    n_in=1.0,
    n_out=1.0,
    overlay_imported=True,
    out_dir=None,
):
    if isinstance(layers, dict):
        config = deepcopy(layers)
        _layers_from_config(config.get("layers", []))
        config["layers"] = [_layer_to_config(layer) for layer in config.get("layers", [])]
        return config
    if layers and isinstance(next(iter(layers)), dict):
        _layers_from_config(layers)
    return {
        "layers": [_layer_to_config(layer) for layer in layers],
        "n_in": float(n_in),
        "n_out": float(n_out),
        "overlay_imported": bool(overlay_imported),
        "out_dir": None if out_dir is None else str(out_dir),
    }


def _reference_standard_to_config(reference_standard):
    if reference_standard is None:
        return {"kind": "identity"}
    if isinstance(reference_standard, ReferenceStandard):
        if reference_standard.kind == "identity":
            return {"kind": "identity"}
        stack = reference_standard.stack
    elif isinstance(reference_standard, dict):
        if str(reference_standard.get("kind", "identity")).strip().lower() == "identity":
            return {"kind": "identity"}
        stack = reference_standard.get("stack")
    else:
        raise TypeError("reference_standard must be a ReferenceStandard, dictionary, or None")

    if stack is None:
        raise ValueError("reference_standard.kind='stack' requires a stack configuration")
    if isinstance(stack, dict):
        return {"kind": "stack", "stack": _sample_to_config(stack)}
    if isinstance(stack, (list, tuple)):
        return {"kind": "stack", "stack": _sample_to_config(list(stack))}
    raise TypeError(
        "reference_standard.stack must be a sample-style dictionary or a list of Layer definitions when exporting"
    )


def _measurement_to_config(measurement):
    if measurement is None:
        return {
            "mode": "transmission",
            "angle_deg": 0.0,
            "polarization": "s",
            "reference_standard": {"kind": "identity"},
        }
    if isinstance(measurement, dict):
        payload = deepcopy(measurement)
        payload["angle_deg"] = _parameter_from_config(payload.get("angle_deg", 0.0), path="measurement.angle_deg")
        if "polarization_mix" in payload and payload["polarization_mix"] is not None:
            payload["polarization_mix"] = _parameter_from_config(
                payload["polarization_mix"],
                path="measurement.polarization_mix",
            )
        measurement = Measurement(**payload)
    if isinstance(measurement, Measurement):
        return {
            "mode": measurement.mode,
            "angle_deg": _parameter_to_config(measurement.angle_deg),
            "polarization": measurement.polarization,
            "polarization_mix": _parameter_to_config(measurement.polarization_mix)
            if measurement.polarization_mix is not None
            else None,
            "reference_standard": _reference_standard_to_config(measurement.reference_standard),
        }
    raise TypeError("measurement must be a Measurement, dictionary, or None")


def _reference_to_config(reference):
    if not isinstance(reference, dict):
        raise TypeError("reference must be a dictionary describing a measured CSV or generated pulse")
    config = deepcopy(reference)
    kind = str(config.get("kind", "")).strip().lower()
    if kind not in {"generated_pulse", "measured_csv"}:
        raise ValueError("reference.kind must be 'generated_pulse' or 'measured_csv'")

    if kind == "generated_pulse":
        generate = dict(config.get("generate", {}))
        if not generate:
            generate = {
                key: config[key]
                for key in (
                    "model",
                    "sample_count",
                    "dt_ps",
                    "time_center_ps",
                    "pulse_center_ps",
                    "tau_ps",
                    "f0_thz",
                    "amp",
                    "phi_rad",
                    "pad_factor",
                )
                if key in config
            }
        return {
            "kind": "generated_pulse",
            "generate": generate,
            "prepare": dict(config.get("prepare", {})),
        }

    return {
        "kind": "measured_csv",
        "path": str(config["path"]),
        "time_column": config.get("time_column"),
        "signal_column": config.get("signal_column"),
        "prepare": dict(config.get("prepare", {})),
    }


def build_study_setup(
    *,
    reference,
    layers,
    measurement=None,
    study=None,
    n_in=1.0,
    n_out=1.0,
    overlay_imported=True,
    sample_out_dir=None,
    notes=None,
):
    return {
        "meta": {
            "schema_version": _SETUP_SCHEMA_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
        },
        "reference": _reference_to_config(reference),
        "sample": _sample_to_config(
            layers,
            n_in=n_in,
            n_out=n_out,
            overlay_imported=overlay_imported,
            out_dir=sample_out_dir,
        ),
        "measurement": _measurement_to_config(measurement),
        "study": {} if study is None else deepcopy(study),
        "notes": None if notes is None else str(notes),
    }


def write_study_setup_csv(path, setup):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _transform_path_fields(deepcopy(setup), base_dir=path.parent, to_relative=True)
    rows = []
    for section in ("meta", "reference", "sample", "measurement", "study", "notes"):
        if section not in serializable:
            continue
        rows.append(
            {
                "section": section,
                "value_json": json.dumps(serializable[section], ensure_ascii=True, separators=(",", ":")),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "value_json"))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_study_setup_json(path, setup):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _transform_path_fields(deepcopy(setup), base_dir=path.parent, to_relative=True)
    write_json(path, serializable)
    return path


def load_study_setup_csv(path):
    path = Path(path)
    sections = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sections[row["section"]] = json.loads(row["value_json"])
    if "meta" not in sections:
        raise ValueError("study setup CSV is missing the 'meta' section")
    version = int(sections["meta"].get("schema_version", -1))
    if version != _SETUP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported study setup schema_version={version}; expected {_SETUP_SCHEMA_VERSION}"
        )
    return _transform_path_fields(sections, base_dir=path.parent, to_relative=False)


def load_study_setup_json(path):
    path = Path(path)
    sections = json.loads(path.read_text(encoding="utf-8"))
    if "meta" not in sections:
        raise ValueError("study setup JSON is missing the 'meta' section")
    version = int(sections["meta"].get("schema_version", -1))
    if version != _SETUP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported study setup schema_version={version}; expected {_SETUP_SCHEMA_VERSION}"
        )
    return _transform_path_fields(sections, base_dir=path.parent, to_relative=False)


def _parameter_from_config(value, *, path="value"):
    if isinstance(value, dict) and str(value.get("kind", "")).strip() == "Fit":
        payload = dict(value)
        payload.pop("kind", None)
        try:
            return Fit(**payload)
        except Exception as exc:
            raise ValueError(f"{path}: invalid Fit configuration ({exc})") from exc
    return value


def _material_from_config(config, *, path="material"):
    kind = str(config.get("kind", "")).strip()
    if kind == "NKFile":
        if "path" not in config:
            raise ValueError(f"{path}.path is required for kind='NKFile'")
        return NKFile(path=config["path"])
    if kind == "ConstantNK":
        if "n" not in config:
            raise ValueError(f"{path}.n is required for kind='ConstantNK'")
        return ConstantNK(
            n=_parameter_from_config(config["n"], path=f"{path}.n"),
            k=_parameter_from_config(config.get("k", 0.0), path=f"{path}.k"),
        )
    if kind == "Drude":
        for field_name in ("eps_inf", "plasma_freq_thz", "gamma_thz"):
            if field_name not in config:
                raise ValueError(f"{path}.{field_name} is required for kind='Drude'")
        return Drude(
            eps_inf=_parameter_from_config(config["eps_inf"], path=f"{path}.eps_inf"),
            plasma_freq_thz=_parameter_from_config(config["plasma_freq_thz"], path=f"{path}.plasma_freq_thz"),
            gamma_thz=_parameter_from_config(config["gamma_thz"], path=f"{path}.gamma_thz"),
        )
    if kind == "Lorentz":
        for field_name in ("eps_inf", "delta_eps", "resonance_thz", "gamma_thz"):
            if field_name not in config:
                raise ValueError(f"{path}.{field_name} is required for kind='Lorentz'")
        return Lorentz(
            eps_inf=_parameter_from_config(config["eps_inf"], path=f"{path}.eps_inf"),
            delta_eps=_parameter_from_config(config["delta_eps"], path=f"{path}.delta_eps"),
            resonance_thz=_parameter_from_config(config["resonance_thz"], path=f"{path}.resonance_thz"),
            gamma_thz=_parameter_from_config(config["gamma_thz"], path=f"{path}.gamma_thz"),
        )
    if kind == "DrudeLorentz":
        if "eps_inf" not in config:
            raise ValueError(f"{path}.eps_inf is required for kind='DrudeLorentz'")
        oscillators = tuple(
            LorentzOscillator(
                delta_eps=_parameter_from_config(osc["delta_eps"], path=f"{path}.oscillators[{index}].delta_eps"),
                resonance_thz=_parameter_from_config(
                    osc["resonance_thz"],
                    path=f"{path}.oscillators[{index}].resonance_thz",
                ),
                gamma_thz=_parameter_from_config(osc["gamma_thz"], path=f"{path}.oscillators[{index}].gamma_thz"),
            )
            for index, osc in enumerate(config.get("oscillators", []))
        )
        return DrudeLorentz(
            eps_inf=_parameter_from_config(config["eps_inf"], path=f"{path}.eps_inf"),
            plasma_freq_thz=_parameter_from_config(
                config.get("plasma_freq_thz", 0.0),
                path=f"{path}.plasma_freq_thz",
            ),
            gamma_thz=_parameter_from_config(config.get("gamma_thz", 0.0), path=f"{path}.gamma_thz"),
            oscillators=oscillators,
        )
    raise ValueError(f"{path}.kind='{kind}' is not supported")


def _layers_from_config(layer_configs):
    layers = []
    for index, config in enumerate(layer_configs):
        try:
            layers.append(
                Layer(
                    name=str(config["name"]),
                    thickness_um=_parameter_from_config(config["thickness_um"], path=f"layers[{index}].thickness_um"),
                    material=_material_from_config(dict(config["material"]), path=f"layers[{index}].material"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"layers[{index}] is missing required field '{exc.args[0]}'") from exc
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"layers[{index}] is invalid ({exc})") from exc
    return layers


def _build_sample_from_config(sample_config, *, reference_result, default_out_dir: Path):
    config = deepcopy(sample_config)
    out_dir = config.pop("out_dir", None)
    return build_sample(
        layers=_layers_from_config(config["layers"]),
        reference=reference_result,
        out_dir=default_out_dir if out_dir is None else Path(out_dir),
        n_in=float(config.get("n_in", 1.0)),
        n_out=float(config.get("n_out", 1.0)),
        overlay_imported=bool(config.get("overlay_imported", True)),
    )


def _measurement_from_config(measurement_config, *, reference_result):
    config = deepcopy(measurement_config)
    config["angle_deg"] = _parameter_from_config(config.get("angle_deg", 0.0), path="measurement.angle_deg")
    if "polarization_mix" in config and config["polarization_mix"] is not None:
        config["polarization_mix"] = _parameter_from_config(
            config["polarization_mix"],
            path="measurement.polarization_mix",
        )
    reference_standard_config = dict(config.get("reference_standard", {"kind": "identity"}))
    kind = str(reference_standard_config.get("kind", "identity")).strip().lower()
    if kind == "identity":
        reference_standard = ReferenceStandard(kind="identity")
    elif kind == "stack":
        standard_sample = _build_sample_from_config(
            reference_standard_config["stack"],
            reference_result=reference_result,
            default_out_dir=reference_result.run_dir / "reference_standard",
        )
        reference_standard = ReferenceStandard(kind="stack", stack=standard_sample)
    else:
        raise ValueError("measurement.reference_standard.kind must be 'identity' or 'stack'")

    return Measurement(
        mode=config.get("mode", "transmission"),
        angle_deg=config.get("angle_deg", 0.0),
        polarization=config.get("polarization", "s"),
        polarization_mix=config.get("polarization_mix"),
        reference_standard=reference_standard,
    )


def _prepare_reference_from_config(reference_config):
    config = deepcopy(reference_config)
    prepare_config = dict(config.get("prepare", {}))
    output_root = prepare_config.pop("output_root", "runs")
    run_label = prepare_config.pop("run_label", None)
    noise = prepare_config.pop("noise", None)

    kind = str(config.get("kind", "")).strip().lower()
    if kind == "generated_pulse":
        reference_input = generate_reference_pulse(**config["generate"])
    elif kind == "measured_csv":
        reference_input = load_reference_csv(
            config["path"],
            time_column=config.get("time_column"),
            signal_column=config.get("signal_column"),
        )
    else:
        raise ValueError(f"Unsupported reference kind '{kind}'")

    if prepare_config:
        unexpected = ", ".join(sorted(prepare_config))
        raise ValueError(f"Unsupported reference.prepare keys: {unexpected}")
    return prepare_reference(reference_input, noise=noise, output_root=output_root, run_label=run_label)


def run_study_from_setup_csv(path):
    setup = load_study_setup_csv(path)
    reference_result = _prepare_reference_from_config(setup["reference"])
    sample_result = _build_sample_from_config(
        setup["sample"],
        reference_result=reference_result,
        default_out_dir=reference_result.run_dir / "sample",
    )
    measurement = _measurement_from_config(setup["measurement"], reference_result=reference_result)
    study = deepcopy(setup["study"])
    out_dir = study.pop("out_dir", None)
    return run_study(
        reference_result,
        sample_result,
        study,
        measurement=measurement,
        out_dir=None if out_dir is None else Path(out_dir),
    )


def run_study_from_setup_json(path):
    setup = load_study_setup_json(path)
    reference_result = _prepare_reference_from_config(setup["reference"])
    sample_result = _build_sample_from_config(
        setup["sample"],
        reference_result=reference_result,
        default_out_dir=reference_result.run_dir / "sample",
    )
    measurement = _measurement_from_config(setup["measurement"], reference_result=reference_result)
    study = deepcopy(setup["study"])
    out_dir = study.pop("out_dir", None)
    return run_study(
        reference_result,
        sample_result,
        study,
        measurement=measurement,
        out_dir=None if out_dir is None else Path(out_dir),
    )

