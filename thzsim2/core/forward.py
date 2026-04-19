from __future__ import annotations

from typing import Any

import numpy as np

from thzsim2.core.fft import fft_t_to_w, ifft_w_to_t
from thzsim2.core.transfer import stack_response_function
from thzsim2.models import Measurement, ReferenceResult, ReferenceStandard, SampleResult, TraceData


def normalize_measurement(measurement) -> Measurement:
    if measurement is None:
        measurement = Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            reference_standard=ReferenceStandard(kind="identity"),
        )
    elif isinstance(measurement, dict):
        measurement = Measurement(**measurement)
    elif not isinstance(measurement, Measurement):
        raise TypeError("measurement must be a Measurement, dictionary, or None")

    if measurement.reference_standard is None:
        if measurement.mode == "transmission":
            measurement.reference_standard = ReferenceStandard(kind="identity")
        else:
            raise ValueError("reflection measurements require an explicit reference_standard")
    return measurement


def _reference_standard_stack(reference_standard: ReferenceStandard | None):
    if reference_standard is None or reference_standard.kind == "identity":
        return None

    stack = reference_standard.stack
    if isinstance(stack, SampleResult):
        return stack.resolved_stack
    if isinstance(stack, dict):
        if {"n_in", "n_out", "layers"}.issubset(stack.keys()):
            return stack
        if "resolved_stack" in stack and isinstance(stack["resolved_stack"], dict):
            return stack["resolved_stack"]
    raise TypeError("reference_standard.kind='stack' requires a SampleResult or resolved stack dictionary")


def _measurement_record(measurement: Measurement):
    return {
        "mode": measurement.mode,
        "angle_deg": float(measurement.angle_deg),
        "polarization": measurement.polarization,
        "reference_standard_kind": measurement.reference_standard.kind if measurement.reference_standard else None,
    }


def _validate_reference_standard_response(response):
    values = np.asarray(response, dtype=np.complex128)
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("reference standard response must remain finite")
    scale = max(float(np.max(np.abs(values))), 1.0)
    if np.any(np.abs(values) <= 1e-12 * scale):
        raise ValueError("reference standard response is too small for stable normalization")


def simulate_sample_from_reference(
    reference: ReferenceResult,
    resolved_stack,
    *,
    max_internal_reflections=0,
    measurement=None,
):
    trace = reference.trace
    measurement = normalize_measurement(measurement)
    dt_s = trace.dt_ps * 1e-12
    t0_s = trace.time_ps[0] * 1e-12
    omega, reference_spectrum = fft_t_to_w(trace.trace, dt=dt_s, t0=t0_s)
    sample_response = stack_response_function(
        omega,
        resolved_stack,
        max_internal_reflections=max_internal_reflections,
        angle_deg=measurement.angle_deg,
        polarization=measurement.polarization,
        mode=measurement.mode,
    )

    reference_standard_stack = _reference_standard_stack(measurement.reference_standard)
    if reference_standard_stack is None:
        reference_standard_response = np.ones_like(sample_response, dtype=np.complex128)
    else:
        reference_standard_response = stack_response_function(
            omega,
            reference_standard_stack,
            max_internal_reflections=max_internal_reflections,
            angle_deg=measurement.angle_deg,
            polarization=measurement.polarization,
            mode=measurement.mode,
        )
        _validate_reference_standard_response(reference_standard_response)

    transfer = sample_response / reference_standard_response
    sample_spectrum = reference_spectrum * transfer
    sample_trace = ifft_w_to_t(sample_spectrum, dt=dt_s, t0=t0_s)
    return {
        "time_ps": trace.time_ps.copy(),
        "omega_rad_s": omega,
        "reference_trace": np.asarray(trace.trace, dtype=np.float64).copy(),
        "reference_spectrum": reference_spectrum,
        "sample_response": sample_response,
        "reference_standard_response": reference_standard_response,
        "transfer_function": transfer,
        "sample_trace": np.asarray(sample_trace.real, dtype=np.float64),
        "sample_spectrum": sample_spectrum,
        "measurement": _measurement_record(measurement),
        "trace_data": TraceData(
            time_ps=trace.time_ps.copy(),
            trace=np.asarray(sample_trace.real, dtype=np.float64),
            source_kind="simulation",
            metadata={"measurement": _measurement_record(measurement)},
        ),
    }
