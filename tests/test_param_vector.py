"""Unit tests for kt_epimodel_hira.calibration.param_vector."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.calibration.param_vector import (
    N_VECTOR,
    REF_AGE_IDX,
    ParameterBounds,
    get_bounds_vector,
    get_param_names,
    initial_guess,
    params_to_vector,
    vector_to_params,
)
from kt_epimodel_hira.model.parameters import CalibrationParameters


def test_n_vector_is_21() -> None:
    assert N_VECTOR == 21


def test_ref_age_idx() -> None:
    assert REF_AGE_IDX == 5


def test_param_names_length() -> None:
    names = get_param_names()
    assert len(names) == N_VECTOR


def test_param_names_first_four_betas() -> None:
    names = get_param_names()
    assert names[0:4] == ["beta_h", "beta_w", "beta_s", "beta_o"]


def test_param_names_gamma_at_end() -> None:
    names = get_param_names()
    assert names[18] == "gamma_child"
    assert names[19] == "gamma_adult"
    assert names[20] == "gamma_elder"


def test_param_names_last_is_gamma_elder() -> None:
    assert get_param_names()[-1] == "gamma_elder"


def test_param_names_no_seasonality() -> None:
    names = get_param_names()
    for n in names:
        assert "seasonality" not in n


def test_param_names_excludes_phi_5() -> None:
    names = get_param_names()
    assert "phi_5" not in names
    for a in range(15):
        if a == REF_AGE_IDX:
            continue
        assert f"phi_{a}" in names


# ---------- params_to_vector ----------

def test_params_to_vector_shape() -> None:
    vec = params_to_vector(CalibrationParameters())
    assert vec.shape == (N_VECTOR,)


def test_params_to_vector_betas_at_start() -> None:
    cal = CalibrationParameters(
        beta_h=0.1, beta_w=0.2, beta_s=0.3, beta_o=0.4,
    )
    vec = params_to_vector(cal)
    np.testing.assert_allclose(vec[0:4], [0.1, 0.2, 0.3, 0.4])


def test_params_to_vector_gamma_triple() -> None:
    cal = CalibrationParameters(gamma_child=0.35, gamma_adult=0.15, gamma_elder=0.30)
    vec = params_to_vector(cal)
    assert vec[18] == 0.35
    assert vec[19] == 0.15
    assert vec[20] == 0.30


# ---------- vector_to_params ----------

def test_vector_to_params_returns_calibration() -> None:
    vec = initial_guess()
    out = vector_to_params(vec)
    assert isinstance(out, CalibrationParameters)


def test_vector_to_params_phi_ref_is_one() -> None:
    cal = vector_to_params(initial_guess())
    assert cal.phi[REF_AGE_IDX] == 1.0


def test_vector_to_params_gamma_triple() -> None:
    vec = initial_guess()
    vec[18] = 0.45
    vec[19] = 0.12
    vec[20] = 0.33
    cal = vector_to_params(vec)
    assert cal.gamma_child == 0.45
    assert cal.gamma_adult == 0.12
    assert cal.gamma_elder == 0.33


def test_vector_to_params_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        vector_to_params(np.zeros(19))
    with pytest.raises(ValueError):
        vector_to_params(np.zeros(23))


# ---------- roundtrip ----------

def test_roundtrip_default() -> None:
    cal = CalibrationParameters()
    vec = params_to_vector(cal)
    cal2 = vector_to_params(vec)
    assert cal2.beta_h == cal.beta_h
    np.testing.assert_array_equal(cal2.phi, cal.phi)
    assert cal2.gamma_child == cal.gamma_child
    assert cal2.gamma_adult == cal.gamma_adult
    assert cal2.gamma_elder == cal.gamma_elder


def test_roundtrip_custom() -> None:
    phi = np.linspace(0.5, 2.0, 15)
    phi[REF_AGE_IDX] = 1.0
    cal = CalibrationParameters(
        beta_h=0.1, beta_w=0.2, beta_s=0.3, beta_o=0.4,
        phi=phi, gamma_child=0.50, gamma_adult=0.10, gamma_elder=0.35,
    )
    cal2 = vector_to_params(params_to_vector(cal))
    np.testing.assert_allclose(cal2.phi, cal.phi)
    assert cal2.beta_h == 0.1
    assert cal2.gamma_child == 0.50
    assert cal2.gamma_adult == 0.10
    assert cal2.gamma_elder == 0.35


# ---------- bounds ----------

def test_bounds_length() -> None:
    bounds = get_bounds_vector()
    assert len(bounds) == N_VECTOR


def test_bounds_default() -> None:
    bounds = get_bounds_vector()
    for lo, hi in bounds:
        assert lo < hi


def test_bounds_custom() -> None:
    custom = ParameterBounds(beta_h=(0.005, 0.5))
    bounds = get_bounds_vector(custom)
    assert bounds[0] == (0.005, 0.5)


def test_bounds_beta() -> None:
    bounds = get_bounds_vector()
    for i in range(4):
        assert bounds[i] == (0.001, 5.0)


def test_bounds_phi() -> None:
    bounds = get_bounds_vector()
    for i in range(4, 18):
        assert bounds[i] == (0.1, 5.0)


def test_bounds_gamma_triple() -> None:
    bounds = get_bounds_vector()
    for i in (18, 19, 20):
        assert bounds[i] == (0.05, 0.60)


# ---------- initial_guess ----------

def test_initial_guess_default() -> None:
    vec = initial_guess()
    np.testing.assert_allclose(vec, params_to_vector(CalibrationParameters()))


def test_initial_guess_custom_base() -> None:
    base_c = CalibrationParameters(beta_h=0.2)
    vec = initial_guess(base_c)
    assert vec[0] == 0.2


def test_initial_guess_in_bounds() -> None:
    vec = initial_guess()
    bounds = get_bounds_vector()
    for i, (lo, hi) in enumerate(bounds):
        assert lo <= vec[i] <= hi
