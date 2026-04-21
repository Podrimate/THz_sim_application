import pytest

from thzsim2.models import Fit


def test_fit_resolves_relative_bounds_from_initial_value():
    fit = Fit(initial=10.0, rel_min=0.8, rel_max=1.3, label="thickness")

    assert fit.label == "thickness"
    assert fit.resolved_min == pytest.approx(8.0)
    assert fit.resolved_max == pytest.approx(13.0)
    assert fit.bounds == pytest.approx((8.0, 13.0))


def test_fit_rejects_zero_initial_with_relative_only_bounds():
    with pytest.raises(ValueError, match="initial is zero"):
        Fit(initial=0.0, rel_min=-0.1, rel_max=0.1)
