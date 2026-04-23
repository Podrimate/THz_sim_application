from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from thzsim2.io.manifests import write_json
from thzsim2.models import Measurement
from thzsim2.workflows.deepdive_fit import run_staged_measured_fit
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
    metric="data_fit",
    metric_options=None,
    max_internal_reflections=0,
    fit_strategy="single_pass",
    reflection_counts=None,
    stage_sequence=None,
    delay_options=None,
    weighting=None,
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
            "fit_strategy": str(fit_strategy),
            "metric": str(metric),
            "metric_options": None if metric_options is None else deepcopy(metric_options),
            "optimizer": {} if optimizer is None else deepcopy(optimizer),
            "max_internal_reflections": int(max_internal_reflections),
            "reflection_counts": None if reflection_counts is None else [int(value) for value in reflection_counts],
            "stage_sequence": None if stage_sequence is None else deepcopy(stage_sequence),
            "delay_options": None if delay_options is None else deepcopy(delay_options),
            "weighting": None if weighting is None else deepcopy(weighting),
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


def _load_fit_runtime_inputs(path):
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
    measurement_config["trace_scale"] = _parameter_from_config(
        measurement_config.get("trace_scale", 1.0),
        path="measurement.trace_scale",
    )
    measurement_config["trace_offset"] = _parameter_from_config(
        measurement_config.get("trace_offset", 0.0),
        path="measurement.trace_offset",
    )
    measurement = Measurement(**measurement_config)
    layers = _layers_from_config(sample_config["layers"])

    out_dir = fit_config.get("out_dir")
    if out_dir is None:
        out_dir = Path(path).parent / "fit_outputs"

    return {
        "setup": setup,
        "prepared_traces": prepared_traces,
        "layers": layers,
        "measurement": measurement,
        "sample_config": sample_config,
        "fit_config": fit_config,
        "out_dir": Path(out_dir),
    }


def run_measured_fit_from_setup_json(path):
    runtime = _load_fit_runtime_inputs(path)
    sample_config = runtime["sample_config"]
    fit_config = runtime["fit_config"]

    return run_measured_fit(
        runtime["prepared_traces"],
        layers=runtime["layers"],
        out_dir=runtime["out_dir"],
        measurement=runtime["measurement"],
        optimizer=fit_config.get("optimizer") or None,
        metric=fit_config.get("metric", "data_fit"),
        metric_options=fit_config.get("metric_options") or None,
        max_internal_reflections=int(fit_config.get("max_internal_reflections", 0)),
        delay_options=fit_config.get("delay_options") or None,
        weighting=fit_config.get("weighting") or None,
        n_in=float(sample_config.get("n_in", 1.0)),
        n_out=float(sample_config.get("n_out", 1.0)),
        overlay_imported=bool(sample_config.get("overlay_imported", True)),
    )


def run_fit_from_setup_json(path):
    runtime = _load_fit_runtime_inputs(path)
    sample_config = runtime["sample_config"]
    fit_config = runtime["fit_config"]
    fit_strategy = str(fit_config.get("fit_strategy", "single_pass")).strip().lower()

    if fit_strategy in {"balanced_staged", "staged", "staged_balanced"}:
        reflection_counts = fit_config.get("reflection_counts")
        if reflection_counts is None:
            reflection_counts = [int(fit_config.get("max_internal_reflections", 0))]
        return run_staged_measured_fit(
            runtime["prepared_traces"],
            runtime["layers"],
            out_dir=runtime["out_dir"],
            measurement=runtime["measurement"],
            weighting=fit_config.get("weighting") or None,
            delay_options=fit_config.get("delay_options") or None,
            reflection_counts=tuple(int(value) for value in reflection_counts),
            stage_sequence=fit_config.get("stage_sequence") or None,
            n_in=float(sample_config.get("n_in", 1.0)),
            n_out=float(sample_config.get("n_out", 1.0)),
            overlay_imported=bool(sample_config.get("overlay_imported", True)),
        )
    return run_measured_fit_from_setup_json(path)
