from __future__ import annotations

import math
from collections.abc import Sequence

from thzsim2.models import Fit
from thzsim2.models.sample import ConstantNK, Drude, DrudeLorentz, Layer, Lorentz, LorentzOscillator, NKFile, TwoDrude


def _as_finite_float(value, *, what: str, layer_name: str, allow_nonnegative=False, allow_positive=False):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Layer '{layer_name}' {what} must be finite")
    if allow_positive and value <= 0.0:
        raise ValueError(f"Layer '{layer_name}' {what} must be > 0")
    if allow_nonnegative and value < 0.0:
        raise ValueError(f"Layer '{layer_name}' {what} must be >= 0")
    return value


def _validate_parameter_value(value, *, what: str, layer_name: str, allow_nonnegative=False, allow_positive=False):
    if isinstance(value, Fit):
        _as_finite_float(
            value.initial,
            what=f"{what} initial value",
            layer_name=layer_name,
            allow_nonnegative=allow_nonnegative,
            allow_positive=allow_positive,
        )
        if value.resolved_min is not None:
            _as_finite_float(
                value.resolved_min,
                what=f"{what} lower bound",
                layer_name=layer_name,
                allow_nonnegative=allow_nonnegative,
                allow_positive=allow_positive,
            )
        if value.resolved_max is not None:
            _as_finite_float(
                value.resolved_max,
                what=f"{what} upper bound",
                layer_name=layer_name,
                allow_nonnegative=allow_nonnegative,
                allow_positive=allow_positive,
            )
        return

    _as_finite_float(
        value,
        what=what,
        layer_name=layer_name,
        allow_nonnegative=allow_nonnegative,
        allow_positive=allow_positive,
    )


def _validate_nk_file(material: NKFile, layer_name: str):
    if not str(material.path).strip():
        raise ValueError(f"Layer '{layer_name}' NKFile path must be non-empty")


def _validate_constant_nk(material: ConstantNK, layer_name: str):
    _validate_parameter_value(material.n, what="ConstantNK.n", layer_name=layer_name, allow_positive=True)
    _validate_parameter_value(material.k, what="ConstantNK.k", layer_name=layer_name, allow_nonnegative=True)


def _validate_drude(material: Drude, layer_name: str):
    _validate_parameter_value(material.eps_inf, what="Drude.eps_inf", layer_name=layer_name, allow_positive=True)
    _validate_parameter_value(
        material.plasma_freq_thz,
        what="Drude.plasma_freq_thz",
        layer_name=layer_name,
        allow_nonnegative=True,
    )
    _validate_parameter_value(material.gamma_thz, what="Drude.gamma_thz", layer_name=layer_name, allow_nonnegative=True)


def _validate_lorentz(material: Lorentz, layer_name: str):
    _validate_parameter_value(material.eps_inf, what="Lorentz.eps_inf", layer_name=layer_name, allow_positive=True)
    _validate_parameter_value(material.delta_eps, what="Lorentz.delta_eps", layer_name=layer_name)
    _validate_parameter_value(
        material.resonance_thz,
        what="Lorentz.resonance_thz",
        layer_name=layer_name,
        allow_positive=True,
    )
    _validate_parameter_value(material.gamma_thz, what="Lorentz.gamma_thz", layer_name=layer_name, allow_nonnegative=True)


def _validate_two_drude(material: TwoDrude, layer_name: str):
    _validate_parameter_value(material.eps_inf, what="TwoDrude.eps_inf", layer_name=layer_name, allow_positive=True)
    _validate_parameter_value(
        material.plasma_freq1_thz,
        what="TwoDrude.plasma_freq1_thz",
        layer_name=layer_name,
        allow_nonnegative=True,
    )
    _validate_parameter_value(
        material.gamma1_thz,
        what="TwoDrude.gamma1_thz",
        layer_name=layer_name,
        allow_nonnegative=True,
    )
    _validate_parameter_value(
        material.plasma_freq2_thz,
        what="TwoDrude.plasma_freq2_thz",
        layer_name=layer_name,
        allow_nonnegative=True,
    )
    _validate_parameter_value(
        material.gamma2_thz,
        what="TwoDrude.gamma2_thz",
        layer_name=layer_name,
        allow_nonnegative=True,
    )


def _validate_drude_lorentz(material: DrudeLorentz, layer_name: str):
    _validate_parameter_value(
        material.eps_inf,
        what="DrudeLorentz.eps_inf",
        layer_name=layer_name,
        allow_positive=True,
    )
    _validate_parameter_value(
        material.plasma_freq_thz,
        what="DrudeLorentz.plasma_freq_thz",
        layer_name=layer_name,
        allow_nonnegative=True,
    )
    _validate_parameter_value(
        material.gamma_thz,
        what="DrudeLorentz.gamma_thz",
        layer_name=layer_name,
        allow_nonnegative=True,
    )
    for idx, oscillator in enumerate(material.oscillators):
        if not isinstance(oscillator, LorentzOscillator):
            raise TypeError(
                f"Layer '{layer_name}' DrudeLorentz.oscillators[{idx}] must be a LorentzOscillator"
            )
        _validate_parameter_value(
            oscillator.delta_eps,
            what=f"DrudeLorentz.oscillators[{idx}].delta_eps",
            layer_name=layer_name,
        )
        _validate_parameter_value(
            oscillator.resonance_thz,
            what=f"DrudeLorentz.oscillators[{idx}].resonance_thz",
            layer_name=layer_name,
            allow_positive=True,
        )
        _validate_parameter_value(
            oscillator.gamma_thz,
            what=f"DrudeLorentz.oscillators[{idx}].gamma_thz",
            layer_name=layer_name,
            allow_nonnegative=True,
        )


def validate_layer(layer: Layer, index: int):
    if not isinstance(layer, Layer):
        raise TypeError(f"Layer {index} must be a Layer instance")
    if not layer.name.strip():
        raise ValueError(f"Layer {index} name must be a non-empty string")

    _validate_parameter_value(
        layer.thickness_um,
        what="thickness_um",
        layer_name=layer.name,
        allow_nonnegative=True,
    )

    material = layer.material
    if material is None:
        raise ValueError(f"Layer '{layer.name}' material must be provided")

    if isinstance(material, NKFile):
        _validate_nk_file(material, layer.name)
        return
    if isinstance(material, ConstantNK):
        _validate_constant_nk(material, layer.name)
        return
    if isinstance(material, Drude):
        _validate_drude(material, layer.name)
        return
    if isinstance(material, TwoDrude):
        _validate_two_drude(material, layer.name)
        return
    if isinstance(material, Lorentz):
        _validate_lorentz(material, layer.name)
        return
    if isinstance(material, DrudeLorentz):
        _validate_drude_lorentz(material, layer.name)
        return

    raise TypeError(f"Layer '{layer.name}' material has unsupported type {type(material).__name__}")


def validate_stack(layers: Sequence[Layer], *, n_in=1.0, n_out=1.0):
    if not isinstance(layers, Sequence):
        raise TypeError("layers must be a sequence of Layer objects")

    n_in = float(n_in)
    n_out = float(n_out)
    if not math.isfinite(n_in) or n_in <= 0.0:
        raise ValueError("n_in must be a finite positive real number")
    if not math.isfinite(n_out) or n_out <= 0.0:
        raise ValueError("n_out must be a finite positive real number")

    names = set()
    for index, layer in enumerate(layers):
        validate_layer(layer, index)
        if layer.name in names:
            raise ValueError(f"Duplicate layer name '{layer.name}' is not allowed")
        names.add(layer.name)

    return tuple(layers)
