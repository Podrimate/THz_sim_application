from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np

from thzsim2.models.reference import TraceData

TRACE_FIELDNAMES = ("time_ps", "trace")

_TIME_UNITS_TO_PS = {
    "ps": 1.0,
    "ns": 1e3,
    "us": 1e6,
    "ms": 1e9,
    "s": 1e12,
    "fs": 1e-3,
}


def _coerce_real_trace(trace):
    arr = np.asarray(trace)
    if np.iscomplexobj(arr):
        scale = max(1.0, float(np.max(np.abs(arr))))
        if np.max(np.abs(arr.imag)) > 1e-12 * scale:
            raise ValueError("trace CSV export only supports real-valued time traces")
        return np.asarray(arr.real, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _normalize_header(text: str):
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _split_header_unit(header: str):
    header = str(header).strip()
    if "/" in header:
        name, unit = header.rsplit("/", 1)
        return name.strip(), unit.strip()
    if "(" in header and header.endswith(")"):
        name, unit = header[:-1].split("(", 1)
        return name.strip(), unit.strip()
    return header, None


def _infer_time_column(fieldnames, explicit: str | None):
    if explicit is not None:
        if explicit not in fieldnames:
            raise ValueError(f"time column '{explicit}' was not found in the trace CSV")
        return explicit

    preferred = [
        "time_ps",
        "Time_abs/ps",
        "time",
    ]
    for candidate in preferred:
        if candidate in fieldnames:
            return candidate

    for name in fieldnames:
        base, unit = _split_header_unit(name)
        normalized = _normalize_header(base)
        if normalized.startswith("time") and (unit is None or unit.lower() in _TIME_UNITS_TO_PS):
            return name
    raise ValueError("could not infer a time column from the trace CSV")


def _infer_signal_column(fieldnames, explicit: str | None):
    if explicit is not None:
        if explicit not in fieldnames:
            raise ValueError(f"signal column '{explicit}' was not found in the trace CSV")
        return explicit

    preferred = [
        "trace",
        "Signal/nA",
        "signal",
    ]
    for candidate in preferred:
        if candidate in fieldnames:
            return candidate

    for name in fieldnames:
        base, _ = _split_header_unit(name)
        normalized = _normalize_header(base)
        if normalized in {"trace", "signal", "amplitude"} or normalized.startswith("signal"):
            return name
    raise ValueError("could not infer a signal column from the trace CSV")


def _time_scale_to_ps(column_name: str):
    _, unit = _split_header_unit(column_name)
    if unit is None:
        return 1.0, "ps"
    unit_key = unit.lower()
    if unit_key not in _TIME_UNITS_TO_PS:
        raise ValueError(f"unsupported time unit '{unit}' in column '{column_name}'")
    return _TIME_UNITS_TO_PS[unit_key], unit


def _uniformity_stats(time_ps):
    time_ps = np.asarray(time_ps, dtype=np.float64)
    diffs = np.diff(time_ps)
    if np.any(diffs <= 0.0):
        raise ValueError("time axis must be strictly increasing")
    dt_nominal = float(np.median(diffs))
    tolerance = max(1e-12, abs(dt_nominal) * 1e-9)
    is_uniform = bool(np.allclose(diffs, dt_nominal, rtol=0.0, atol=tolerance))
    rounded = np.round(diffs, 6)
    values, counts = np.unique(rounded, return_counts=True)
    median_dt = float(np.median(rounded))
    max_count = int(np.max(counts))
    tied = values[counts == max_count]
    dominant_dt = float(tied[np.argmin(np.abs(tied - median_dt))])
    return {
        "is_uniform": is_uniform,
        "dt_nominal_ps": dt_nominal,
        "dominant_dt_ps": dominant_dt,
        "dt_min_ps": float(np.min(diffs)),
        "dt_max_ps": float(np.max(diffs)),
        "dt_range_ps": float(np.max(diffs) - np.min(diffs)),
    }


def _resample_to_uniform_grid(time_ps, trace):
    stats = _uniformity_stats(time_ps)
    dt_ps = float(stats["dominant_dt_ps"])
    uniform_time_ps = float(time_ps[0]) + np.arange(len(time_ps), dtype=np.float64) * dt_ps
    uniform_trace = np.interp(uniform_time_ps, np.asarray(time_ps, dtype=np.float64), np.asarray(trace, dtype=np.float64))
    return uniform_time_ps, uniform_trace, stats


def _baseline_metadata(trace):
    values = np.asarray(trace, dtype=np.float64)
    window = max(10, min(200, values.size // 20 or 10))
    return {
        "mean": float(np.mean(values)),
        "first_window_mean": float(np.mean(values[:window])),
        "last_window_mean": float(np.mean(values[-window:])),
        "window_size": int(window),
    }


def read_trace_csv(path, *, time_column: str | None = None, signal_column: str | None = None, resample: str = "strict") -> TraceData:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError("trace CSV is missing a header row")
        fieldnames = list(reader.fieldnames)
        if time_column is None and signal_column is None:
            missing = [name for name in TRACE_FIELDNAMES if name not in fieldnames]
            if not missing:
                time_column = "time_ps"
                signal_column = "trace"
            elif any(name in fieldnames for name in TRACE_FIELDNAMES):
                raise ValueError("trace CSV is missing required columns: time_ps, trace")

        time_column = _infer_time_column(fieldnames, time_column)
        signal_column = _infer_signal_column(fieldnames, signal_column)
        time_scale_to_ps, time_unit = _time_scale_to_ps(time_column)
        _, signal_unit = _split_header_unit(signal_column)

        time_ps = []
        trace = []
        for row in reader:
            time_ps.append(float(row[time_column]) * time_scale_to_ps)
            trace.append(float(row[signal_column]))

    time_ps = np.asarray(time_ps, dtype=np.float64)
    trace = np.asarray(trace, dtype=np.float64)
    stats = _uniformity_stats(time_ps)
    was_resampled = False
    if not stats["is_uniform"]:
        if resample == "strict":
            raise ValueError("time_ps must be uniformly spaced")
        if resample != "auto":
            raise ValueError("resample must be 'strict' or 'auto'")
        time_ps, trace, stats = _resample_to_uniform_grid(time_ps, trace)
        was_resampled = True

    metadata = {
        "import_columns": {"time": time_column, "signal": signal_column},
        "import_units": {"time": time_unit, "signal": signal_unit or "a.u."},
        "time_axis": {
            "resampled_to_uniform_grid": bool(was_resampled),
            **stats,
        },
        "baseline": _baseline_metadata(trace),
    }

    return TraceData(
        time_ps=time_ps,
        trace=trace,
        source_kind="csv",
        source_path=str(path.resolve()),
        metadata=metadata,
    )


def write_trace_csv(path, trace_data: TraceData):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = _coerce_real_trace(trace_data.trace)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        for time_ps, value in zip(trace_data.time_ps, trace, strict=True):
            writer.writerow(
                {
                    "time_ps": format(float(time_ps), ".16g"),
                    "trace": format(float(value), ".16g"),
                }
            )
