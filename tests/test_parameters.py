"""Unit tests for kt_epimodel_hira.model.parameters (Step A 개편)."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.model.parameters import (
    AGE_LABELS_15,
    CalibrationParameters,
    DiseaseParameters,
    EmploymentParameters,
    ModelParameters,
    PolicyParameters,
    REF_AGE_IDX,
    TimeVaryingParameters,
    VaccinationParameters,
)


# ---------- DiseaseParameters ----------

def test_disease_defaults() -> None:
    d = DiseaseParameters()
    assert d.sigma == 0.5
    assert d.gamma == 0.25
    assert d.latent_period == 2.0
    assert d.infectious_period == 4.0
    assert d.seasonality_amp == 0.7
    assert d.seasonality_peak_day == 105.0


def test_disease_kappa_length() -> None:
    d = DiseaseParameters()
    assert len(d.kappa) == 15
    assert d.kappa_array.shape == (15,)


def test_disease_kappa_values() -> None:
    """학생 0.42, 성인 0.60, 70+ 0.0."""
    d = DiseaseParameters()
    ka = d.kappa_array
    np.testing.assert_array_equal(ka[0:4], 0.42)
    np.testing.assert_array_equal(ka[4:14], 0.60)
    assert ka[14] == 0.0


def test_disease_negative_kappa_raises() -> None:
    bad_kappa = list(DiseaseParameters().kappa)
    bad_kappa[3] = -0.1
    with pytest.raises(ValueError):
        DiseaseParameters(kappa=tuple(bad_kappa))


def test_disease_invalid_kappa_length_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseParameters(kappa=tuple([0.5] * 14))


def test_disease_negative_sigma_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseParameters(sigma=-0.1)
    with pytest.raises(ValueError):
        DiseaseParameters(gamma=0.0)


def test_disease_seasonal_factor_at_peak_cosine() -> None:
    """Cosine mode: factor(peak) = base + amp."""
    d = DiseaseParameters(
        seasonality_mode="cosine",
        seasonality_amp=0.4, seasonality_base=1.0,
        seasonality_peak_day=100.0,
    )
    assert d.seasonal_factor(100.0) == pytest.approx(1.4)


def test_disease_seasonal_factor_half_period_offset_cosine() -> None:
    d = DiseaseParameters(
        seasonality_mode="cosine",
        seasonality_amp=0.4, seasonality_base=1.0,
        seasonality_peak_day=100.0, seasonality_period=365.0,
    )
    assert d.seasonal_factor(100.0 + 365.0 / 2) == pytest.approx(0.6)


def test_disease_seasonal_factor_amp_zero_returns_base() -> None:
    d = DiseaseParameters(seasonality_amp=0.0, seasonality_base=1.0)
    for day in (0, 50, 130, 200, 365):
        assert d.seasonal_factor(day) == pytest.approx(1.0)


def test_disease_invalid_amp_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_amp=-0.1)
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_amp=10.0)   # > 5


def test_disease_seasonality_base_default() -> None:
    d = DiseaseParameters()
    assert d.seasonality_base == 1.0


def test_disease_default_mode_cosine() -> None:
    d = DiseaseParameters()
    assert d.seasonality_mode == "cosine"
    assert d.seasonality_sigma == 40.0


def test_disease_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_mode="exponential")


def test_disease_seasonal_factor_gaussian_at_peak() -> None:
    d = DiseaseParameters(
        seasonality_mode="gaussian",
        seasonality_amp=0.8, seasonality_base=0.1,
        seasonality_peak_day=100.0, seasonality_sigma=30.0,
    )
    # at peak: g = 1 → factor = base + amp = 0.9
    assert d.seasonal_factor(100.0) == pytest.approx(0.9)


def test_disease_seasonal_factor_gaussian_far_decays() -> None:
    d = DiseaseParameters(
        seasonality_mode="gaussian",
        seasonality_amp=1.0, seasonality_base=0.1,
        seasonality_peak_day=100.0, seasonality_sigma=20.0,
    )
    # 5σ 거리: g ≈ 0 → factor ≈ base
    f_far = d.seasonal_factor(100.0 + 5 * 20.0)
    assert abs(f_far - 0.1) < 1e-3


def test_disease_seasonal_factor_cosine_still_works() -> None:
    d = DiseaseParameters(
        seasonality_mode="cosine",
        seasonality_amp=0.3, seasonality_base=1.0,
        seasonality_peak_day=100.0,
    )
    # at peak: cos(0) = 1 → 1.0 + 0.3 = 1.3
    assert d.seasonal_factor(100.0) == pytest.approx(1.3)


def test_disease_seasonality_sigma_invalid_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_sigma=0.0)
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_sigma=-5.0)


def test_disease_seasonality_base_invalid_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_base=-0.1)
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_base=float("nan"))


def test_disease_seasonal_factor_with_base_cosine() -> None:
    """cosine mode: factor = base + amp · cos."""
    d = DiseaseParameters(
        seasonality_mode="cosine",
        seasonality_base=0.5, seasonality_amp=0.3, seasonality_peak_day=100.0,
    )
    assert d.seasonal_factor(100.0) == pytest.approx(0.8)
    assert d.seasonal_factor(100.0 + 365.0 / 2) == pytest.approx(0.2)


def test_disease_seasonal_factor_clipped_to_zero_cosine() -> None:
    """cosine mode + base < amp: off-peak 음수 영역에서 0 clip."""
    d = DiseaseParameters(
        seasonality_mode="cosine",
        seasonality_base=0.3, seasonality_amp=1.0, seasonality_peak_day=100.0,
    )
    assert d.seasonal_factor(100.0 + 365.0 / 2) == 0.0


def test_disease_invalid_period_raises() -> None:
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_period=0.0)
    with pytest.raises(ValueError):
        DiseaseParameters(seasonality_period=-10.0)


# ---------- CalibrationParameters ----------

def test_calibration_4_betas_default() -> None:
    c = CalibrationParameters()
    assert c.beta_h == 0.05
    assert c.beta_w == 0.05
    assert c.beta_s == 0.05
    assert c.beta_o == 0.05


def test_calibration_betas_dict() -> None:
    c = CalibrationParameters(beta_h=0.1, beta_w=0.2, beta_s=0.3, beta_o=0.4)
    d = c.betas
    assert d == {"home": 0.1, "work": 0.2, "school": 0.3, "other": 0.4}


def test_calibration_phi_shape() -> None:
    c = CalibrationParameters()
    assert c.phi.shape == (15,)
    np.testing.assert_array_equal(c.phi, np.ones(15))


def test_calibration_invalid_phi_raises() -> None:
    with pytest.raises(ValueError):
        CalibrationParameters(phi=np.ones(14))
    bad = np.ones(15)
    bad[3] = -0.5
    with pytest.raises(ValueError):
        CalibrationParameters(phi=bad)


def test_calibration_negative_beta_raises() -> None:
    with pytest.raises(ValueError):
        CalibrationParameters(beta_h=-0.1)
    with pytest.raises(ValueError):
        CalibrationParameters(beta_w=float("nan"))


def test_calibration_gamma_defaults() -> None:
    c = CalibrationParameters()
    assert c.gamma_child == 0.40
    assert c.gamma_adult == 0.18
    assert c.gamma_elder == 0.25


def test_calibration_gamma_15_shape_and_values() -> None:
    c = CalibrationParameters(gamma_child=0.4, gamma_adult=0.2, gamma_elder=0.3)
    g = c.gamma_15
    assert g.shape == (15,)
    np.testing.assert_allclose(g[:4], 0.4)
    np.testing.assert_allclose(g[4:13], 0.2)
    np.testing.assert_allclose(g[13:], 0.3)


def test_calibration_invalid_gamma_raises() -> None:
    with pytest.raises(ValueError):
        CalibrationParameters(gamma_child=0.0)
    with pytest.raises(ValueError):
        CalibrationParameters(gamma_adult=1.5)
    with pytest.raises(ValueError):
        CalibrationParameters(gamma_elder=-0.1)


def test_calibration_reference_normalization_4_betas() -> None:
    phi = np.linspace(0.5, 2.0, 15)
    c = CalibrationParameters(
        beta_h=0.1, beta_w=0.2, beta_s=0.3, beta_o=0.4,
        phi=phi, gamma_child=0.4, gamma_adult=0.2, gamma_elder=0.3,
    )
    c2 = c.with_reference_normalized()
    assert c2.phi[REF_AGE_IDX] == pytest.approx(1.0)
    ratio = phi[REF_AGE_IDX]
    assert c2.beta_h == pytest.approx(c.beta_h * ratio)
    assert c2.beta_w == pytest.approx(c.beta_w * ratio)
    assert c2.beta_s == pytest.approx(c.beta_s * ratio)
    assert c2.beta_o == pytest.approx(c.beta_o * ratio)
    np.testing.assert_allclose(c2.beta_h * c2.phi, c.beta_h * c.phi, rtol=1e-12)
    assert c2.gamma_child == c.gamma_child
    assert c2.gamma_adult == c.gamma_adult
    assert c2.gamma_elder == c.gamma_elder


# ---------- PolicyParameters ----------

def test_policy_baseline() -> None:
    p = PolicyParameters.baseline()
    assert p.p_school == 1.0
    assert p.p_work == 1.0


def test_policy_school_closure() -> None:
    p = PolicyParameters.school_closure(attendance=0.05)
    assert p.p_school == 0.05
    assert p.p_work == 1.0


def test_policy_sick_leave_enhanced() -> None:
    p = PolicyParameters.sick_leave_enhanced(work_rate=0.3)
    assert p.p_school == 1.0
    assert p.p_work == 0.3


def test_policy_comprehensive() -> None:
    p = PolicyParameters.comprehensive(school_attendance=0.1, work_rate=0.5)
    assert p.p_school == 0.1
    assert p.p_work == 0.5


def test_policy_range_invalid_raises() -> None:
    with pytest.raises(ValueError):
        PolicyParameters(p_school=1.5)
    with pytest.raises(ValueError):
        PolicyParameters(p_work=-0.1)


# ---------- VaccinationParameters ----------

def test_vax_default_VE() -> None:
    v = VaccinationParameters()
    assert v.VE == 0.5
    assert v.annual_coverage.shape == (15,)
    assert v.peak_iso_week == 42
    assert v.spread_weeks == 4.0


def test_vax_invalid_VE_raises() -> None:
    with pytest.raises(ValueError):
        VaccinationParameters(VE=-0.1)
    with pytest.raises(ValueError):
        VaccinationParameters(VE=1.5)


def test_vax_rate_vector_shape() -> None:
    v = VaccinationParameters()
    rv = v.rate_vector(42)
    assert rv.shape == (15,)


def test_vax_rate_vector_peak_max() -> None:
    v = VaccinationParameters()
    peak_day = (v.peak_iso_week - 36) * 7
    rp = v.rate_vector(peak_day).max()
    rb = v.rate_vector(peak_day - 14).max()
    ra = v.rate_vector(peak_day + 14).max()
    assert rp > rb
    assert rp > ra


def test_vax_coverage_range_invalid() -> None:
    bad = np.full(15, 0.5)
    bad[0] = 1.2
    with pytest.raises(ValueError):
        VaccinationParameters(annual_coverage=bad)


# ---------- EmploymentParameters ----------

def test_employment_default_shape() -> None:
    e = EmploymentParameters()
    assert e.rho.shape == (1154, 15)
    assert e.n_admdong == 1154


def test_employment_custom_rho() -> None:
    rho = np.full((10, 15), 0.5)
    e = EmploymentParameters(rho=rho)
    assert e.n_admdong == 10


def test_employment_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        EmploymentParameters(rho=np.zeros((10, 14)))
    with pytest.raises(ValueError):
        EmploymentParameters(rho=np.zeros(15))


def test_employment_range_invalid_raises() -> None:
    bad = np.full((5, 15), 0.5)
    bad[2, 3] = 1.5
    with pytest.raises(ValueError):
        EmploymentParameters(rho=bad)
    bad2 = np.full((5, 15), 0.5)
    bad2[2, 3] = -0.1
    with pytest.raises(ValueError):
        EmploymentParameters(rho=bad2)


def test_employment_from_kt_data() -> None:
    e = EmploymentParameters.from_kt_data()
    assert e.rho.shape[1] == 15
    # 수도권 행정동 수
    assert 1000 < e.rho.shape[0] < 1200
    # 0-14세는 0 (build_rho_matrix fill_under_15=0)
    np.testing.assert_array_equal(e.rho[:, 0:3], 0.0)
    # 30-34세는 양수
    assert e.rho[:, 6].mean() > 0.5


# ---------- TimeVaryingParameters ----------

def test_time_varying_default_factors() -> None:
    tv = TimeVaryingParameters()
    assert tv.weekday_factor["home"] == 1.0
    assert tv.weekend_factor["work"] == 0.2
    assert tv.holiday_factor["school"] == 0.0
    assert tv.vacation_factor["school"] == 0.2


def test_time_varying_get_daytypes() -> None:
    tv = TimeVaryingParameters()
    assert tv.get("weekday_school")["home"] == 1.0
    assert tv.get("weekend")["work"] == 0.2
    assert tv.get("vacation_weekday")["school"] == 0.2
    assert tv.get("holiday")["school"] == 0.0


def test_time_varying_get_unknown_raises() -> None:
    with pytest.raises(ValueError):
        TimeVaryingParameters().get("typhoon")


def test_time_varying_missing_channel_raises() -> None:
    with pytest.raises(ValueError):
        TimeVaryingParameters(weekday_factor={"home": 1.0})


# ---------- ModelParameters ----------

def test_model_default_all_categories() -> None:
    m = ModelParameters()
    assert isinstance(m.disease, DiseaseParameters)
    assert isinstance(m.calibration, CalibrationParameters)
    assert isinstance(m.policy, PolicyParameters)
    assert isinstance(m.time_varying, TimeVaryingParameters)
    assert isinstance(m.vaccination, VaccinationParameters)
    # employment 은 default None (n_admdong 미정)
    assert m.employment is None


def test_model_with_policy_preserves_others() -> None:
    m = ModelParameters()
    new_pol = PolicyParameters.school_closure()
    m2 = m.with_policy(new_pol)
    assert m2.policy is new_pol
    assert m2.disease is m.disease
    assert m2.calibration is m.calibration
    assert m2.time_varying is m.time_varying
    assert m2.vaccination is m.vaccination
    assert m2.employment is m.employment


def test_model_with_calibration_preserves_others() -> None:
    m = ModelParameters()
    new_cal = CalibrationParameters(beta_h=0.2, beta_w=0.3, beta_s=0.4, beta_o=0.5)
    m2 = m.with_calibration(new_cal)
    assert m2.calibration is new_cal
    assert m2.policy is m.policy


def test_model_with_vaccination_preserves_others() -> None:
    m = ModelParameters()
    new_vax = VaccinationParameters(VE=0.7)
    m2 = m.with_vaccination(new_vax)
    assert m2.vaccination is new_vax
    assert m2.disease is m.disease


def test_model_with_disease() -> None:
    m = ModelParameters()
    new_d = DiseaseParameters(seasonality_amp=0.6)
    m2 = m.with_disease(new_d)
    assert m2.disease is new_d
    assert m2.calibration is m.calibration
    assert m2.policy is m.policy
    assert m2.vaccination is m.vaccination


def test_model_with_employment() -> None:
    m = ModelParameters()
    emp = EmploymentParameters(rho=np.full((50, 15), 0.5))
    m2 = m.with_employment(emp)
    assert m2.employment is emp
    assert m2.disease is m.disease
    assert m2.policy is m.policy


def test_age_labels_15_constant() -> None:
    assert len(AGE_LABELS_15) == 15
    assert AGE_LABELS_15[0] == "0-4"
    assert AGE_LABELS_15[14] == "70+"
