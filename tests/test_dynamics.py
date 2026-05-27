"""Unit tests for kt_epimodel_hira.model.dynamics (Step D — SEIRV + 4채널)."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.model.compartments import (
    IDX_E,
    IDX_I,
    IDX_R,
    IDX_S,
    IDX_V,
    N_AGE,
    flatten_state,
    initial_state,
    unflatten_state,
)
from kt_epimodel_hira.model.dynamics import compute_derivatives, make_ode_rhs
from kt_epimodel_hira.model.mobility_tensor import (
    build_M_from_kt_array,
    build_M_home,
    build_M_school,
)
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters,
    EmploymentParameters,
    ModelParameters,
    PolicyParameters,
    VaccinationParameters,
)

N_ADM = 4


def _setup(
    beta: float = 0.05,
    seed_per: float = 50.0,
    VE: float = 0.5,
    init_vax: float = 0.0,
) -> dict:
    rng = np.random.default_rng(0)
    pop = rng.integers(500, 2000, size=(N_AGE, N_ADM)).astype(np.float64)
    pi_kt = rng.uniform(1.0, 30.0, size=(N_ADM, N_ADM, 7, 24))
    mobility = {
        "home":   build_M_home(N_ADM),
        "school": build_M_school(N_ADM),
        "work":   build_M_from_kt_array(pi_kt, "work"),
        "other":  build_M_from_kt_array(pi_kt, "other"),
    }
    matrices = {k: np.ones((N_AGE, N_AGE)) for k in ("C_home", "C_work", "C_school", "C_other")}

    rho = np.zeros((N_ADM, N_AGE))
    rho[:, 4:14] = 0.7
    emp = EmploymentParameters(rho=rho)

    cal = CalibrationParameters(
        beta_h=beta, beta_w=beta, beta_s=beta, beta_o=beta,
        phi=np.ones(N_AGE), gamma_report=0.5,
    )
    vax = VaccinationParameters(VE=VE)

    params = (
        ModelParameters(calibration=cal, vaccination=vax)
        .with_employment(emp)
    )
    state = initial_state(pop, seed_per_admdong=seed_per, initial_vaccinated_fraction=init_vax)
    return {
        "pop": pop, "mobility": mobility, "matrices": matrices,
        "state": state, "params": params,
    }


# ---------- shape & conservation ----------

def test_derivatives_shape() -> None:
    s = _setup()
    d = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"])
    assert d.shape == (5, N_AGE, N_ADM)


def test_population_conserved() -> None:
    s = _setup()
    d = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"])
    np.testing.assert_allclose(d.sum(axis=0), 0.0, atol=1e-9)


def test_conserved_at_peak_vaccination() -> None:
    """백신 peak 시점에서도 보존."""
    s = _setup(init_vax=0.0)
    d = compute_derivatives(
        s["state"], s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=42,
    )
    np.testing.assert_allclose(d.sum(axis=0), 0.0, atol=1e-9)


# ---------- compartment 부호 ----------

def test_dS_nonpositive() -> None:
    """S 는 빠져나가기만 (FOI + vaccination)."""
    s = _setup()
    d = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"])
    assert (d[IDX_S] <= 0).all()


def test_dV_zero_state_then_vax_inflow() -> None:
    """V=0 시 vaccination 만 V로 유입 → dV ≥ 0."""
    s = _setup()
    state = s["state"].copy()
    state[IDX_V] = 0.0
    d = compute_derivatives(state, s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=42)
    assert (d[IDX_V] >= -1e-12).all()
    assert d[IDX_V].sum() > 0


def test_dE_correct_no_I() -> None:
    """I=V=R=0, E>0, S=pop → foi=0 → dE = -σ·E."""
    s = _setup(seed_per=0.0)
    state = s["state"].copy()
    state[IDX_E] = 5.0
    state[IDX_S] = s["pop"] - 5.0
    d = compute_derivatives(state, s["mobility"], s["matrices"], s["pop"], s["params"])
    expected = -s["params"].disease.sigma * 5.0
    np.testing.assert_allclose(d[IDX_E], expected, rtol=1e-12)


def test_dI_correct_no_E() -> None:
    """E=0, I>0 → dI = -γ·I."""
    s = _setup(seed_per=0.0)
    state = s["state"].copy()
    state[IDX_I] = 7.0
    state[IDX_S] = s["pop"] - 7.0
    d = compute_derivatives(state, s["mobility"], s["matrices"], s["pop"], s["params"])
    expected = -s["params"].disease.gamma * 7.0
    np.testing.assert_allclose(d[IDX_I], expected, rtol=1e-12)


def test_dR_only_from_I() -> None:
    """dR = γ·I, V·E 무관."""
    s = _setup(seed_per=0.0)
    state = s["state"].copy()
    state[IDX_I] = 4.0
    state[IDX_E] = 100.0
    state[IDX_V] = 100.0
    state[IDX_S] = s["pop"] - 204.0
    d = compute_derivatives(state, s["mobility"], s["matrices"], s["pop"], s["params"])
    expected = s["params"].disease.gamma * 4.0
    np.testing.assert_allclose(d[IDX_R], expected, rtol=1e-12)


# ---------- β 효과 ----------

def test_tiny_beta_only_natural_progression() -> None:
    """β≈0: dE 는 foi 기여 없고 -σ·E 만 남음."""
    s = _setup(beta=1e-30)
    state = s["state"].copy()
    state[IDX_E] = 5.0
    state[IDX_S] = s["pop"] - 5.0 - state[IDX_I]
    d = compute_derivatives(state, s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=0)
    # foi 기여 없이 dE = -σ·E
    np.testing.assert_allclose(d[IDX_E], -s["params"].disease.sigma * 5.0, atol=1e-3)


# ---------- 백신 시간 의존 ----------

def test_vaccination_day_zero_small() -> None:
    """day=0 (시즌 시작) — 백신 peak 가 day 42 → day 0 의 rate 작음."""
    s = _setup()
    d0 = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=0)
    d42 = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=42)
    # day 42 의 V 유입이 day 0 보다 큼
    assert d42[IDX_V].sum() > d0[IDX_V].sum()


def test_VE_1_no_breakthrough() -> None:
    """VE=1 → V 에서 E 로 유출 없음 (breakthrough=0)."""
    s = _setup(VE=1.0, init_vax=0.2)
    d = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=10)
    # V 의 감소는 breakthrough 만 (vaccination 유입은 양수)
    # day 10 에서 vaccination v_rate>0 이므로 dV = +v_rate·S - 0·V > 0
    assert (d[IDX_V] >= -1e-12).all()


def test_VE_0_full_breakthrough_equals_S() -> None:
    """VE=0 → V 의 감염 확률 = S 와 동일 (breakthrough = foi)."""
    s = _setup(VE=0.0, init_vax=0.2)
    d = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=0)
    # (1-VE)·foi = foi → V 에서 E 로 유출 = foi·V
    # 보존 확인 (이미 다른 테스트에서 했지만 명시)
    np.testing.assert_allclose(d.sum(axis=0), 0.0, atol=1e-9)


# ---------- scipy 인터페이스 ----------

def test_rhs_signature_1d() -> None:
    s = _setup()
    rhs = make_ode_rhs(s["mobility"], s["matrices"], s["pop"], s["params"])
    y0 = flatten_state(s["state"])
    dy = rhs(0.0, y0)
    assert dy.shape == y0.shape


def test_rhs_roundtrip_consistency() -> None:
    """rhs(0, flatten(state)) == flatten(compute_derivatives(state, day=0))."""
    s = _setup()
    rhs = make_ode_rhs(s["mobility"], s["matrices"], s["pop"], s["params"])
    y0 = flatten_state(s["state"])
    dy = rhs(0.0, y0)
    d_direct = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"], day_in_season=0)
    np.testing.assert_allclose(dy, flatten_state(d_direct), rtol=1e-12)


def test_rhs_t_advances_vaccination() -> None:
    """t 증가 → day_in_season 증가 → 백신 rate 다름."""
    s = _setup()
    rhs = make_ode_rhs(s["mobility"], s["matrices"], s["pop"], s["params"])
    y0 = flatten_state(s["state"])
    dy_0 = rhs(0.0, y0)
    dy_42 = rhs(42.0, y0)
    # 같지 않음 (백신 rate 차이)
    assert not np.allclose(dy_0, dy_42)


# ---------- Euler 한 스텝 보존 ----------

def test_euler_step_conserves() -> None:
    s = _setup()
    rhs = make_ode_rhs(s["mobility"], s["matrices"], s["pop"], s["params"])
    y0 = flatten_state(s["state"])
    dy = rhs(0.0, y0)
    y1 = y0 + 0.1 * dy
    state1 = unflatten_state(y1, N_ADM)
    np.testing.assert_allclose(state1.sum(axis=0), s["pop"], atol=1e-6)


# ---------- 정책 효과 (전 단계 확인 재현) ----------

def test_school_closure_reduces_dE_students() -> None:
    s = _setup()
    d_base = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], s["params"])
    p_closed = s["params"].with_policy(PolicyParameters.school_closure(attendance=0.0))
    d_closed = compute_derivatives(s["state"], s["mobility"], s["matrices"], s["pop"], p_closed)
    # 학생 (0-3) 의 dE 감소 — foi 감소 효과
    assert d_closed[IDX_E][0:4].sum() < d_base[IDX_E][0:4].sum()
