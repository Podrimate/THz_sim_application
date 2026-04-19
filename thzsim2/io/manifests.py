from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from thzsim2.models.reference import ReferenceSummary, TraceData

UNITS = {
    "time": "ps",
    "frequency": "THz",
    "thickness": "um",
    "conductivity": "S/m",
}


def _json_default(value: Any):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def build_reference_manifest(
    *,
    run_id: str,
    created_at: str,
    trace_data: TraceData,
    summary: ReferenceSummary,
    files: dict[str, str],
):
    source = {"kind": trace_data.source_kind}
    if trace_data.source_path is not None:
        source["path"] = trace_data.source_path
    source.update(trace_data.metadata)

    return {
        "schema_name": "thzsim2.reference_manifest",
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at,
        "workflow": "reference",
        "units": dict(UNITS),
        "source": source,
        "grid": {
            "sample_count": trace_data.sample_count,
            "dt_ps": trace_data.dt_ps,
            "time_min_ps": trace_data.time_min_ps,
            "time_max_ps": trace_data.time_max_ps,
            "time_center_ps": trace_data.time_center_ps,
            "pad_factor": trace_data.pad_factor,
        },
        "trace_summary": {
            "amplitude_scale": summary.amplitude_scale,
            "pulse_center_ps": summary.pulse_center_ps,
        },
        "spectrum_summary": {
            "freq_min_thz": summary.freq_min_thz,
            "freq_max_thz": summary.freq_max_thz,
            "peak_freq_thz": summary.peak_freq_thz,
            "spectral_centroid_thz": summary.spectral_centroid_thz,
        },
        "files": files,
    }


def build_run_manifest(*, run_id: str, created_at: str, reference_manifest_path: str):
    return {
        "schema_name": "thzsim2.run_manifest",
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created_at,
        "workflow": "reference",
        "reference_manifest": reference_manifest_path,
    }


def build_sample_manifest(
    *,
    created_at: str,
    freq_grid_thz,
    n_in: float,
    n_out: float,
    layers: list[dict[str, Any]],
    fit_parameters: list[dict[str, Any]],
    files: dict[str, Any],
    grid_source: str,
):
    freq = np.asarray(freq_grid_thz, dtype=np.float64)
    return {
        "schema_name": "thzsim2.sample_manifest",
        "schema_version": "1.0",
        "created_at": created_at,
        "workflow": "sample",
        "units": dict(UNITS),
        "grid": {
            "source": grid_source,
            "sample_count": int(freq.size),
            "freq_min_thz": float(freq[0]),
            "freq_max_thz": float(freq[-1]),
        },
        "ambient_media": {
            "n_in": float(n_in),
            "n_out": float(n_out),
        },
        "layers": layers,
        "fit_parameters": fit_parameters,
        "files": files,
    }


def build_study_manifest(
    *,
    created_at: str,
    config: dict[str, Any],
    case_count: int,
    run_count: int,
    files: dict[str, Any],
):
    return {
        "schema_name": "thzsim2.study_manifest",
        "schema_version": "1.0",
        "created_at": created_at,
        "workflow": "study",
        "units": dict(UNITS),
        "config": config,
        "case_count": int(case_count),
        "run_count": int(run_count),
        "files": files,
    }


def update_run_manifest(path, **updates):
    path = Path(path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    write_json(path, payload)
