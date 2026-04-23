import numpy as np
import pytest

from thzsim2.core.stack import validate_stack
from thzsim2.models import ConstantNK, Drude, Layer


def test_stack_validation_accepts_valid_single_layer():
    layers = [Layer(name="film", thickness_um=50.0, material=ConstantNK(n=2.0, k=0.02))]

    validated = validate_stack(layers, n_in=1.0, n_out=1.0)

    assert len(validated) == 1
    assert validated[0].name == "film"


def test_stack_validation_rejects_missing_material_info():
    layers = [Layer(name="film", thickness_um=50.0, material=None)]

    with pytest.raises(ValueError, match="Layer 'film' material must be provided"):
        validate_stack(layers, n_in=1.0, n_out=1.0)


def test_stack_validation_rejects_negative_thickness():
    layers = [Layer(name="film", thickness_um=-5.0, material=ConstantNK(n=2.0, k=0.0))]

    with pytest.raises(ValueError, match="Layer 'film' thickness_um must be >= 0"):
        validate_stack(layers, n_in=1.0, n_out=1.0)


def test_stack_validation_rejects_duplicate_layer_names():
    layers = [
        Layer(name="film", thickness_um=50.0, material=ConstantNK(n=2.0, k=0.0)),
        Layer(name="film", thickness_um=20.0, material=Drude(eps_inf=3.0, plasma_freq_thz=1.0, gamma_thz=0.1)),
    ]

    with pytest.raises(ValueError, match="Duplicate layer name 'film'"):
        validate_stack(layers, n_in=1.0, n_out=1.0)
