"""Numerical core utilities preserved from the old modular repo."""

from .fft import fft_t_to_w, ifft_w_to_t, zero_pad_trace
from .forward import simulate_sample_from_reference
from .grids import make_grids, make_omega_grid, make_time_grid
from .materials import evaluate_material_nk, eps_drude, eps_drude_lorentz, eps_lorentz, nk_from_eps
from .metrics import data_fit, fit_sigma, mse, normalized_mse, relative_l2, residual_rms, snr_db
from .noise import add_white_gaussian_noise, noise_sigma_from_dynamic_range
from .pulses import gaussian_carrier_pulse, make_pulse, sech_carrier_pulse
from .stack import validate_layer, validate_stack
from .transfer import (
    fresnel_r,
    fresnel_r_oblique,
    fresnel_t,
    fresnel_t_oblique,
    multilayer_transfer,
    propagation_factor,
    single_layer_transfer,
    stack_response_function,
    stack_transfer_function,
)

__all__ = [
    "make_time_grid",
    "make_omega_grid",
    "make_grids",
    "fft_t_to_w",
    "ifft_w_to_t",
    "zero_pad_trace",
    "simulate_sample_from_reference",
    "gaussian_carrier_pulse",
    "sech_carrier_pulse",
    "make_pulse",
    "eps_drude",
    "eps_lorentz",
    "eps_drude_lorentz",
    "nk_from_eps",
    "evaluate_material_nk",
    "data_fit",
    "normalized_mse",
    "mse",
    "relative_l2",
    "residual_rms",
    "fit_sigma",
    "snr_db",
    "add_white_gaussian_noise",
    "noise_sigma_from_dynamic_range",
    "fresnel_t",
    "fresnel_r",
    "fresnel_t_oblique",
    "fresnel_r_oblique",
    "propagation_factor",
    "single_layer_transfer",
    "multilayer_transfer",
    "stack_response_function",
    "stack_transfer_function",
    "validate_layer",
    "validate_stack",
]
