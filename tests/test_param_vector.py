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
from kt_epimodel_hira.model.parameters import CalibrationParameters, DiseaseParameters


def test_n_vector_is_23() -> None:
    assert N_VECTOR == 23


def test_ref_age_idx() -> None:
    assert REF_AGE_IDX == 5


def test_param_names_length() -> None:
    names = get_param_names()
    assert len(names) == N_VECTOR


def test_param_names_first_four_betas() -> None:
    names = get_param_names()
    assert names[0:4] == ["beta_h", "beta_w", "beta_s", "beta_o"]


def test_param_names_gamma_at_18() -> None:
    assert get_param_names()[18] == "gamma_report"


def test_param_names_last_is_peak_day() -> None:
    assert get_param_names()[-1] == "seasonality_peak_day"


def test_param_names_amp_at_19() -> None:
    assert get_param_names()[19] == "seasonality_amp"


def test_param_names_base_at_20() -> None:
    assert get_param_names()[20] == "seasonality_base"


def test_param_names_sigma_at_21() -> None:
    assert get_param_names()[21] == "seasonality_sigma"


def test_param_names_excludes_phi_5() -> None:
    """phi_5 (25-29 reference) 는 vector 에 없음."""
    names = get_param_names()
    assert "phi_5" not in names
    # 다른 phi 는 모두 있음
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


def test_params_to_vector_gamma_at_18() -> None:
    cal = CalibrationParameters(gamma_report=0.7)
    vec = params_to_vector(cal)
    assert vec[18] == 0.7


def test_params_to_vector_amp_at_19() -> None:
    dis = DiseaseParameters(seasonality_amp=0.4)
    vec = params_to_vector(CalibrationParameters(), dis)
    assert vec[19] == 0.4


# ---------- vector_to_params ----------

def test_vector_to_params_returns_5_tuple() -> None:
    vec = np.ones(N_VECTOR)
    out = vector_to_params(vec)
    assert isinstance(out, tuple) and len(out) == 5


def test_vector_to_params_phi_ref_is_one() -> None:
    vec = np.ones(N_VECTOR)
    cal, _, _, _, _ = vector_to_params(vec)
    assert cal.phi[REF_AGE_IDX] == 1.0


def test_vector_to_params_amp_base_sigma_peak_extracted() -> None:
    vec = np.ones(N_VECTOR)
    vec[19] = 0.6   # amp
    vec[20] = 0.4   # base
    vec[21] = 35.0  # sigma
    vec[22] = 120.0 # peak_day
    _, amp, base, sigma, peak = vector_to_params(vec)
    assert amp == 0.6
    assert base == 0.4
    assert sigma == 35.0
    assert peak == 120.0


def test_vector_to_params_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        vector_to_params(np.zeros(21))
    with pytest.raises(ValueError):
        vector_to_params(np.zeros(22))


# ---------- roundtrip ----------

def test_roundtrip_default() -> None:
    cal = CalibrationParameters()
    dis = DiseaseParameters()
    vec = params_to_vector(cal, dis)
    cal2, amp2, base2, sigma2, peak2 = vector_to_params(vec)
    assert cal2.beta_h == cal.beta_h
    np.testing.assert_array_equal(cal2.phi, cal.phi)
    assert cal2.gamma_report == cal.gamma_report
    assert amp2 == dis.seasonality_amp
    assert base2 == dis.seasonality_base
    assert sigma2 == dis.seasonality_sigma
    assert peak2 == dis.seasonality_peak_day


def test_roundtrip_custom_all_seasonality() -> None:
    phi = np.linspace(0.5, 2.0, 15)
    phi[REF_AGE_IDX] = 1.0
    cal = CalibrationParameters(
        beta_h=0.1, beta_w=0.2, beta_s=0.3, beta_o=0.4,
        phi=phi, gamma_report=0.6,
    )
    dis = DiseaseParameters(
        seasonality_amp=0.7, seasonality_base=0.5, seasonality_sigma=25.0,
        seasonality_peak_day=115.0,
    )
    cal2, amp2, base2, sigma2, peak2 = vector_to_params(params_to_vector(cal, dis))
    np.testing.assert_allclose(cal2.phi, cal.phi)
    assert cal2.beta_h == 0.1
    assert amp2 == 0.7
    assert base2 == 0.5
    assert sigma2 == 25.0
    assert peak2 == 115.0


# ---------- bounds ----------

def test_bounds_length() -> None:
    bounds = get_bounds_vector()
    assert len(bounds) == N_VECTOR


def test_bounds_default() -> None:
    bounds = get_bounds_vector()
    # 모두 (lower, upper) tuple
    for lo, hi in bounds:
        assert lo < hi


def test_bounds_custom() -> None:
    custom = ParameterBounds(beta_h=(0.005, 0.5))
    bounds = get_bounds_vector(custom)
    assert bounds[0] == (0.005, 0.5)


def test_bounds_beta_h() -> None:
    """β default (Phase A 원복): (0.001, 5.0)."""
    bounds = get_bounds_vector()
    for i in range(4):    # beta_h, _w, _s, _o
        assert bounds[i] == (0.001, 5.0)


def test_bounds_phi() -> None:
    """φ default (Phase A 원복): (0.1, 5.0)."""
    bounds = get_bounds_vector()
    # phi entries at indices 4..17 (14 entries)
    for i in range(4, 18):
        assert bounds[i] == (0.1, 5.0)


def test_bounds_gamma_report() -> None:
    """γ_report default (Phase A 원복): (0.01, 1.0)."""
    bounds = get_bounds_vector()
    assert bounds[18] == (0.01, 1.0)


def test_bounds_seasonality_amp() -> None:
    """seasonality_amp default (Phase A 원복): (0.0, 3.0) — amp=0 허용."""
    bounds = get_bounds_vector()
    assert bounds[19] == (0.0, 3.0)


# ---------- initial_guess ----------

def test_initial_guess_default() -> None:
    """initial_guess() 가 peak_day=110 override (data 근접). DiseaseParameters() default 130 과 다름."""
    vec = initial_guess()
    np.testing.assert_allclose(
        vec, params_to_vector(
            CalibrationParameters(), DiseaseParameters(seasonality_peak_day=110.0),
        ),
    )


def test_initial_guess_custom_base() -> None:
    base_c = CalibrationParameters(beta_h=0.2)
    base_d = DiseaseParameters(
        seasonality_amp=0.55, seasonality_base=0.8, seasonality_sigma=30.0,
        seasonality_peak_day=125.0,
    )
    vec = initial_guess(base_c, base_d)
    assert vec[0] == 0.2
    assert vec[19] == 0.55
    assert vec[20] == 0.8
    assert vec[21] == 30.0
    assert vec[22] == 125.0


def test_seasonality_amp_bounds() -> None:
    bounds = get_bounds_vector()
    assert bounds[19] == (0.0, 3.0)


def test_seasonality_base_bounds() -> None:
    bounds = get_bounds_vector()
    assert bounds[20] == (0.0, 1.0)


def test_seasonality_sigma_bounds() -> None:
    bounds = get_bounds_vector()
    assert bounds[21] == (15.0, 80.0)


def test_seasonality_peak_day_bounds() -> None:
    bounds = get_bounds_vector()
    assert bounds[22] == (80.0, 150.0)


def test_initial_guess_in_bounds() -> None:
    vec = initial_guess()
    bounds = get_bounds_vector()
    for i in (19, 20, 21, 22):
        lo, hi = bounds[i]
        assert lo <= vec[i] <= hi
    # default amp=1.0, base=0.1, sigma=40, peak_day=110
    assert vec[19] == 1.0
    assert vec[20] == 0.1
    assert vec[21] == 40.0
    assert vec[22] == 110.0
