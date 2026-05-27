"""Unit tests for kt_epimodel_hira.model.foi (Step C — 4채널 분해)."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.model.compartments import IDX_I, N_AGE, initial_state
from kt_epimodel_hira.model.foi import (
    compute_foi,
    compute_foi_home,
    compute_foi_other,
    compute_foi_school,
    compute_foi_work,
    compute_phi_school,
    compute_phi_spillover,
    compute_phi_work,
)
from kt_epimodel_hira.model.mobility_tensor import (
    build_M_from_kt_array,
    build_M_home,
    build_M_school,
)
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters,
    DiseaseParameters,
    EmploymentParameters,
    ModelParameters,
    PolicyParameters,
)

N_ADM = 4


def _setup(seed_per: float = 50.0, beta: float = 0.05) -> dict:
    rng = np.random.default_rng(0)
    pop = rng.integers(500, 2000, size=(N_AGE, N_ADM)).astype(np.float64)
    pi_kt = rng.uniform(1.0, 30.0, size=(N_ADM, N_ADM, 7, 24))
    M_work = build_M_from_kt_array(pi_kt, "work")
    M_other = build_M_from_kt_array(pi_kt, "other")
    M_home = build_M_home(N_ADM)
    M_school = build_M_school(N_ADM)
    matrices = {
        "C_home":   np.full((N_AGE, N_AGE), 1.0),
        "C_work":   np.full((N_AGE, N_AGE), 1.0),
        "C_school": np.full((N_AGE, N_AGE), 1.0),
        "C_other":  np.full((N_AGE, N_AGE), 1.0),
    }
    state = initial_state(pop, seed_per_admdong=seed_per)

    rho = np.zeros((N_ADM, N_AGE))
    rho[:, 4:14] = 0.7        # 근로자 0.7 균등
    emp = EmploymentParameters(rho=rho)

    params = ModelParameters(
        calibration=CalibrationParameters(
            beta_h=beta, beta_w=beta, beta_s=beta, beta_o=beta,
            phi=np.ones(N_AGE), gamma_report=0.5,
        ),
    ).with_employment(emp)

    return {
        "pop": pop, "state": state, "M_work": M_work, "M_other": M_other,
        "M_home": M_home, "M_school": M_school, "matrices": matrices,
        "params": params, "rho": rho,
    }


# ---------- φ helpers ----------

def test_phi_school() -> None:
    phi = compute_phi_school(p_school=0.4)
    np.testing.assert_allclose(phi[0:4], 0.6)
    np.testing.assert_allclose(phi[4:], 0.0)


def test_phi_work() -> None:
    rho = np.full((N_ADM, N_AGE), 0.5)
    phi = compute_phi_work(p_work=0.2, rho=rho)
    # 근로자만
    np.testing.assert_allclose(phi[:, 4:14], 0.5 * 0.8)
    np.testing.assert_allclose(phi[:, 0:4], 0.0)
    np.testing.assert_allclose(phi[:, 14], 0.0)


def test_phi_spillover() -> None:
    rho = np.full((N_ADM, N_AGE), 0.6)
    phi = compute_phi_spillover(p_school=0.5, p_work=0.4, rho=rho)
    # 학생
    np.testing.assert_allclose(phi[:, 0:4], 0.5)
    # 근로자
    np.testing.assert_allclose(phi[:, 4:14], 0.6 * 0.6)
    # 70+
    np.testing.assert_allclose(phi[:, 14], 0.0)


# ---------- compute_foi_home ----------

def test_foi_home_shape() -> None:
    s = _setup()
    foi = compute_foi_home(
        s["state"], s["matrices"]["C_home"], s["pop"], s["rho"],
        s["params"].disease.kappa_array, 1.0, 1.0,
        s["params"].calibration.beta_h, s["params"].calibration.phi,
    )
    assert foi.shape == (N_AGE, N_ADM)


def test_foi_home_zero_I() -> None:
    s = _setup(seed_per=0.0)
    foi = compute_foi_home(
        s["state"], s["matrices"]["C_home"], s["pop"], s["rho"],
        s["params"].disease.kappa_array, 1.0, 1.0,
        s["params"].calibration.beta_h, s["params"].calibration.phi,
    )
    np.testing.assert_array_equal(foi, 0.0)


def test_foi_home_spillover_increases_with_p_school_drop() -> None:
    """p_school=0 (학교 휴교) → 학생들이 집에 머뭄 → home spillover 증가."""
    s = _setup()
    foi_open = compute_foi_home(
        s["state"], s["matrices"]["C_home"], s["pop"], s["rho"],
        s["params"].disease.kappa_array, p_school=1.0, p_work=1.0,
        beta_h=s["params"].calibration.beta_h, phi_susc=s["params"].calibration.phi,
    )
    foi_closed = compute_foi_home(
        s["state"], s["matrices"]["C_home"], s["pop"], s["rho"],
        s["params"].disease.kappa_array, p_school=0.0, p_work=1.0,
        beta_h=s["params"].calibration.beta_h, phi_susc=s["params"].calibration.phi,
    )
    assert foi_closed.sum() > foi_open.sum()


# ---------- compute_foi_school ----------

def test_foi_school_only_students() -> None:
    s = _setup()
    foi = compute_foi_school(
        s["state"], s["matrices"]["C_school"], s["pop"],
        p_school=1.0, beta_s=s["params"].calibration.beta_s,
        phi_susc=s["params"].calibration.phi,
    )
    assert foi[0:4].sum() > 0
    np.testing.assert_array_equal(foi[4:], 0.0)


def test_foi_school_p_school_zero() -> None:
    """학교 휴교 시 school 채널 0."""
    s = _setup()
    foi = compute_foi_school(
        s["state"], s["matrices"]["C_school"], s["pop"],
        p_school=0.0, beta_s=s["params"].calibration.beta_s,
        phi_susc=s["params"].calibration.phi,
    )
    np.testing.assert_array_equal(foi, 0.0)


# ---------- compute_foi_work ----------

def test_foi_work_only_workers() -> None:
    s = _setup()
    foi = compute_foi_work(
        s["state"], s["matrices"]["C_work"], s["pop"], s["M_work"],
        s["rho"], p_work=1.0, beta_w=s["params"].calibration.beta_w,
        phi_susc=s["params"].calibration.phi,
    )
    assert foi[4:14].sum() > 0
    np.testing.assert_array_equal(foi[0:4], 0.0)
    np.testing.assert_array_equal(foi[14:], 0.0)


def test_foi_work_p_work_zero() -> None:
    s = _setup()
    foi = compute_foi_work(
        s["state"], s["matrices"]["C_work"], s["pop"], s["M_work"],
        s["rho"], p_work=0.0, beta_w=s["params"].calibration.beta_w,
        phi_susc=s["params"].calibration.phi,
    )
    np.testing.assert_array_equal(foi, 0.0)


# ---------- compute_foi_other ----------

def test_foi_other_all_nonnegative() -> None:
    s = _setup()
    foi = compute_foi_other(
        s["state"], s["matrices"]["C_other"], s["pop"], s["M_other"],
        beta_o=s["params"].calibration.beta_o, phi_susc=s["params"].calibration.phi,
    )
    assert (foi >= 0).all()


def test_foi_other_zero_I() -> None:
    s = _setup(seed_per=0.0)
    foi = compute_foi_other(
        s["state"], s["matrices"]["C_other"], s["pop"], s["M_other"],
        beta_o=s["params"].calibration.beta_o, phi_susc=s["params"].calibration.phi,
    )
    np.testing.assert_array_equal(foi, 0.0)


# ---------- compute_foi (통합) ----------

def test_compute_foi_shape() -> None:
    s = _setup()
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    foi = compute_foi(s["state"], mob, s["matrices"], s["pop"], s["params"])
    assert foi.shape == (N_AGE, N_ADM)


def test_compute_foi_no_I() -> None:
    s = _setup(seed_per=0.0)
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    foi = compute_foi(s["state"], mob, s["matrices"], s["pop"], s["params"])
    np.testing.assert_array_equal(foi, 0.0)


def test_compute_foi_nonnegative() -> None:
    s = _setup()
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    foi = compute_foi(s["state"], mob, s["matrices"], s["pop"], s["params"])
    assert (foi >= 0).all()


def test_compute_foi_school_closure_reduces_student_foi() -> None:
    """학교 휴교 시 학생 FOI 감소 (school 채널 0 이지만 spillover 증가 가능 — net 감소 확인)."""
    s = _setup()
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    foi_base = compute_foi(
        s["state"], mob, s["matrices"], s["pop"], s["params"],
    )
    p_closed = s["params"].with_policy(PolicyParameters.school_closure(attendance=0.0))
    foi_closed = compute_foi(s["state"], mob, s["matrices"], s["pop"], p_closed)
    # 학생 연령 (0-3) 의 FOI 감소
    assert foi_closed[0:4].sum() < foi_base[0:4].sum()


def test_compute_foi_sick_leave_reduces_worker_foi() -> None:
    s = _setup()
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    foi_base = compute_foi(s["state"], mob, s["matrices"], s["pop"], s["params"])
    p_leave = s["params"].with_policy(PolicyParameters.sick_leave_enhanced(work_rate=0.0))
    foi_leave = compute_foi(s["state"], mob, s["matrices"], s["pop"], p_leave)
    assert foi_leave[4:14].sum() < foi_base[4:14].sum()


def test_compute_foi_proportional_beta() -> None:
    """β 2배 → FOI 2배."""
    s = _setup(beta=0.05)
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    foi1 = compute_foi(s["state"], mob, s["matrices"], s["pop"], s["params"])

    cal2 = CalibrationParameters(
        beta_h=0.10, beta_w=0.10, beta_s=0.10, beta_o=0.10,
        phi=np.ones(N_AGE), gamma_report=0.5,
    )
    p2 = s["params"].with_calibration(cal2)
    foi2 = compute_foi(s["state"], mob, s["matrices"], s["pop"], p2)
    np.testing.assert_allclose(foi2, 2 * foi1, rtol=1e-10)


def test_compute_foi_seasonal_factor_increases_with_amp_cosine() -> None:
    """Cosine: peak day factor=base+amp."""
    s = _setup()
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}

    # amp=0 baseline (factor=base=1.0)
    p0 = s["params"].with_disease(
        DiseaseParameters(seasonality_mode="cosine", seasonality_amp=0.0, seasonality_base=1.0)
    )
    foi0 = compute_foi(s["state"], mob, s["matrices"], s["pop"], p0, day_in_season=130)

    # amp=0.5, base=1.0 (factor at peak = 1.5)
    p1 = s["params"].with_disease(
        DiseaseParameters(seasonality_mode="cosine", seasonality_amp=0.5,
                          seasonality_base=1.0, seasonality_peak_day=130.0)
    )
    foi1 = compute_foi(s["state"], mob, s["matrices"], s["pop"], p1, day_in_season=130)
    np.testing.assert_allclose(foi1, 1.5 * foi0, rtol=1e-10)


def test_compute_foi_seasonal_at_offpeak_cosine() -> None:
    """Cosine: peak + half period → factor = base - amp."""
    s = _setup()
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    p0 = s["params"].with_disease(
        DiseaseParameters(seasonality_mode="cosine", seasonality_amp=0.0, seasonality_base=1.0)
    )
    foi0 = compute_foi(s["state"], mob, s["matrices"], s["pop"], p0, day_in_season=0)
    p1 = s["params"].with_disease(
        DiseaseParameters(seasonality_mode="cosine", seasonality_amp=0.4,
                          seasonality_base=1.0, seasonality_peak_day=130.0)
    )
    foi1 = compute_foi(s["state"], mob, s["matrices"], s["pop"], p1, day_in_season=130.0 + 182.5)
    np.testing.assert_allclose(foi1, 0.6 * foi0, rtol=1e-10)


def test_compute_foi_seasonal_gaussian_at_peak() -> None:
    """Gaussian: factor(peak) = base + amp."""
    s = _setup()
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    p0 = s["params"].with_disease(
        DiseaseParameters(seasonality_mode="gaussian", seasonality_amp=0.0, seasonality_base=1.0)
    )
    foi0 = compute_foi(s["state"], mob, s["matrices"], s["pop"], p0, day_in_season=130)
    p1 = s["params"].with_disease(
        DiseaseParameters(seasonality_mode="gaussian",
                          seasonality_amp=0.5, seasonality_base=1.0,
                          seasonality_peak_day=130.0, seasonality_sigma=30.0)
    )
    foi1 = compute_foi(s["state"], mob, s["matrices"], s["pop"], p1, day_in_season=130)
    np.testing.assert_allclose(foi1, 1.5 * foi0, rtol=1e-10)


def test_compute_foi_employment_required() -> None:
    s = _setup()
    p_no_emp = ModelParameters()   # employment=None
    mob = {"home": s["M_home"], "school": s["M_school"],
           "work": s["M_work"], "other": s["M_other"]}
    with pytest.raises(ValueError):
        compute_foi(s["state"], mob, s["matrices"], s["pop"], p_no_emp)
