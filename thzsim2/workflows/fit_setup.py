from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from thzsim2.io.manifests import write_json
from thzsim2.models import Measurement
from thzsim2.workflows.fit_workflow import prepare_trace_pair_for_fit, run_measured_fit
from thzsim2.workflows.study_setup import (
    _layers_from_config,
    _parameter_from_config,
    _measurement_to_config,
    _sample_to_config,
    _transform_path_fields,
)


_FIT_SETUP_SCHEMA_VERSION = 1


def _trace_input_to_config(source):
    if isinstance(source, dict):
        config = deepcopy(source)
    else:
        config = {"path": str(source)}
    if "path" not in config:
        raise ValueError("trace source config requires a 'path'")
    return {
        "path": str(config["path"]),
        "time_column": config.get("time_column"),
        "signal_column": config.get("signal_column"),
    }


def _preprocessing_to_config(preprocessing):
    if preprocessing is None:
        preprocessing = {}
    config = dict(preprocessing)
    crop_window = config.get("crop_time_window_ps")
    return {
        "baseline_subtract": bool(config.get("baseline_subtract", False)),
        "baseline_mode": config.get("baseline_mode"),
        "baseline_window_samples": int(config.get("baseline_window_samples", 50)),
        "crop_mode": config.get("crop_mode"),
        "crop_time_window_ps": None
        if crop_window is None
        else [float(crop_window[0]), float(crop_window[1])],
    }


def build_fit_setup(
    *,
    reference_trace,
    sample_trace,
    layers,
    preprocessing=None,
    measurement=None,
    optimizer=None,
    metric="mse",
    max_internal_reflections=0,
    n_in=1.0,
    n_out=1.0,
    overlay_imported=True,
    out_dir=None,
    notes=None,
):
    return {
        "meta": {
            "schema_version": _FIT_SETUP_SCHEMA_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
        },
        "traces": {
            "reference": _trace_input_to_config(reference_trace),
            "sample": _trace_input_to_config(sample_trace),
        },
        "preprocessing": _preprocessing_to_config(preprocessing),
        "sample": _sample_to_config(
            layers,
            n_in=n_in,
            n_out=n_out,
            overlay_imported=overlay_imported,
            out_dir=None,
        ),
        "measurement": _measurement_to_config(measurement),
        "fit": {
            "metric": str(metric),
            "optimizer": {} if optimizer is None else deepcopy(optimizer),
            "max_internal_reflections": int(max_internal_reflections),
            "out_dir": None if out_dir is None else str(out_dir),
        },
        "notes": None if notes is None else str(notes),
    }


def write_fit_setup_json(path, setup):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _transform_path_fields(deepcopy(setup), base_dir=path.parent, to_relative=True)
    write_json(path, serializable)
    return path


def load_fit_setup_json(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "meta" not in payload:
        raise ValueError("fit setup JSON is missing the 'meta' section")
    version = int(payload["meta"].get("schema_version", -1))
    if version != _FIT_SETUP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported fit setup schema_version={version}; expected {_FIT_SETUP_SCHEMA_VERSION}"
        )
    return _transform_path_fields(payload, base_dir=path.parent, to_relative=False)


def run_measured_fit_from_setup_json(path):
    setup = load_fit_setup_json(path)
    traces = dict(setup["traces"])
    preprocessing = dict(setup.get("preprocessing", {}))
    sample_config = dict(setup["sample"])
    fit_config = dict(setup.get("fit", {}))

    prepared_traces = prepare_trace_pair_for_fit(
        traces["reference"]["path"],
        traces["sample"]["path"],
        reference_time_column=traces["reference"].get("time_column"),
        reference_signal_column=traces["reference"].get("signal_column"),
        sample_time_column=traces["sample"].get("time_column"),
        sample_signal_column=traces["sample"].get("signal_column"),
        baseline_subtract=bool(preprocessing.get("baseline_subtract", False)),
        baseline_window_samples=int(preprocessing.get("baseline_window_samples", 50)),
        crop_time_window_ps=preprocessing.get("crop_time_window_ps"),
        baseline_mode=preprocessing.get("baseline_mode"),
        crop_mode=preprocessing.get("crop_mode"),
    )
    measurement_config = deepcopy(setup.get("measurement", {}))
    measurement_config["angle_deg"] = _parameter_from_config(
        measurement_config.get("angle_deg", 0.0),
        path="measurement.angle_deg",
    )
    if "polarization_mix" in measurement_config and measurement_config["polarization_mix"] is not None:
        measurement_config["polarization_mix"] = _parameter_from_config(
            measurement_config["polarization_mix"],
            path="measurement.polarization_mix",
        )
    measurement = Measurement(**measurement_config)
    layers = _layers_from_config(sample_config["layers"])

    out_dir = fit_config.get("out_dir")
    if out_dir is None:
        out_dir = Path(path).parent / "fit_outputs"

    return run_measured_fit(
        prepared_traces,
        layers=layers,
        out_dir=Path(out_dir),
        measurement=measurement,
        optimizer=fit_config.get("optimizer") or None,
        metric=fit_config.get("metric", "mse"),
        max_internal_reflections=int(fit_config.get("max_internal_reflections", 0)),
        n_in=float(sample_config.get("n_in", 1.0)),
        n_out=float(sample_config.get("n_out", 1.0)),
        overlay_imported=bool(sample_config.get("overlay_imported", True)),
    )
