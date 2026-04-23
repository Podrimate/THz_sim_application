from __future__ import annotations

from typing import Any

import numpy as np

from thzsim2.core.fft import fft_t_to_w, ifft_w_to_t
from thzsim2.core.transfer import stack_response_function
from thzsim2.models import Measurement, ReferenceResult, ReferenceStandard, SampleResult, TraceData
from thzsim2.models.fit import Fit


def _resolved_measurement_scalar(value, *, field_name: str):
    if isinstance(value, Fit):
        raise TypeError(f"{field_name} must be resolved to a numeric value before simulation")
    return float(value)


def normalize_measurement(measurement) -> Measurement:
    if measurement is None:
        measurement = Measurement(
            mode="transmission",
            angle_deg=0.0,
            polarization="s",
            trace_scale=1.0,
            trace_offset=0.0,
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
    measurement.angle_deg = _resolved_measurement_scalar(measurement.angle_deg, field_name="angle_deg")
    if measurement.polarization == "mixed":
        measurement.polarization_mix = _resolved_measurement_scalar(
            0.5 if measurement.polarization_mix is None else measurement.polarization_mix,
            field_name="polarization_mix",
        )
    elif measurement.polarization_mix is not None and isinstance(measurement.polarization_mix, Fit):
        raise TypeError("polarization_mix must be resolved to a numeric value before simulation")
    measurement.trace_scale = _resolved_measurement_scalar(measurement.trace_scale, field_name="trace_scale")
    measurement.trace_offset = _resolved_measurement_scalar(measurement.trace_offset, field_name="trace_offset")
    return measurement


def _ambient_replacement_stack(resolved_stack):
    ambient_n = float(resolved_stack["n_in"])
    layers = []
    for index, layer in enumerate(resolved_stack["layers"]):
        layers.append(
            {
                "name": f"{layer.get('name', f'layer_{index + 1}')}_ambient",
                "thickness_um": float(layer["thickness_um"]),
                "material_kind": "ConstantNK",
                "material": {
                    "kind": "ConstantNK",
                    "parameters": {"n": ambient_n, "k": 0.0},
                },
            }
        )
    return {
        "n_in": float(resolved_stack["n_in"]),
        "n_out": float(resolved_stack["n_out"]),
        "layers": layers,
    }


def _reference_standard_stack(reference_standard: ReferenceStandard | None, resolved_stack, measurement: Measurement):
    if reference_standard is None or reference_standard.kind == "identity":
        return None
    if reference_standard.kind == "ambient_replacement":
        if measurement.mode != "transmission":
            raise ValueError("reference_standard.kind='ambient_replacement' is only supported for transmission")
        return _ambient_replacement_stack(resolved_stack)

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
        "polarization_mix": None if measurement.polarization_mix is None else float(measurement.polarization_mix),
        "trace_scale": float(measurement.trace_scale),
        "trace_offset": float(measurement.trace_offset),
        "reference_standard_kind": measurement.reference_standard.kind if measurement.reference_standard else None,
    }


def _validate_reference_standard_response(response):
    values = np.asarray(response, dtype=np.complex128)
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("reference standard response must remain finite")
    inspect_values = values[1:] if values.size > 1 else values
    scale = max(float(np.max(np.abs(inspect_values))) if inspect_values.size else 0.0, 1.0)
    if inspect_values.size and np.any(np.abs(inspect_values) <= 1e-12 * scale):
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
        polarization_mix=measurement.polarization_mix,
        mode=measurement.mode,
    )

    reference_standard_stack = _reference_standard_stack(measurement.reference_standard, resolved_stack, measurement)
    if reference_standard_stack is None:
        reference_standard_response = np.ones_like(sample_response, dtype=np.complex128)
    else:
        reference_standard_response = stack_response_function(
            omega,
            reference_standard_stack,
            max_internal_reflections=max_internal_reflections,
            angle_deg=measurement.angle_deg,
            polarization=measurement.polarization,
            polarization_mix=measurement.polarization_mix,
            mode=measurement.mode,
        )
        _validate_reference_standard_response(reference_standard_response)

    physical_transfer = sample_response / reference_standard_response
    physical_sample_spectrum = reference_spectrum * physical_transfer
    physical_sample_trace = np.asarray(
        ifft_w_to_t(physical_sample_spectrum, dt=dt_s, t0=t0_s).real,
        dtype=np.float64,
    )
    sample_trace = physical_sample_trace * float(measurement.trace_scale) + float(measurement.trace_offset)
    if float(measurement.trace_scale) == 1.0 and float(measurement.trace_offset) == 0.0:
        sample_spectrum = physical_sample_spectrum
        transfer = physical_transfer
    else:
        _, sample_spectrum = fft_t_to_w(sample_trace, dt=dt_s, t0=t0_s)
        transfer = np.divide(
            sample_spectrum,
            reference_spectrum,
            out=np.zeros_like(sample_spectrum, dtype=np.complex128),
            where=np.abs(reference_spectrum) > 1e-30,
        )
    return {
        "time_ps": trace.time_ps.copy(),
        "omega_rad_s": omega,
        "reference_trace": np.asarray(trace.trace, dtype=np.float64).copy(),
        "reference_spectrum": reference_spectrum,
        "sample_response": sample_response,
        "reference_standard_response": reference_standard_response,
        "transfer_function_physical": physical_transfer,
        "transfer_function": transfer,
        "sample_trace_physical": physical_sample_trace,
        "sample_trace": np.asarray(sample_trace, dtype=np.float64),
        "sample_spectrum_physical": physical_sample_spectrum,
        "sample_spectrum": sample_spectrum,
        "measurement": _measurement_record(measurement),
        "trace_data": TraceData(
            time_ps=trace.time_ps.copy(),
            trace=np.asarray(sample_trace, dtype=np.float64),
            source_kind="simulation",
            metadata={"measurement": _measurement_record(measurement)},
        ),
    }
