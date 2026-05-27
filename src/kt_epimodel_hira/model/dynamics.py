"""SEIRV ODE 우변 (Step D — J 제거, V 추가, 4채널 FOI).

수식:
    dS/dt = -λ·S - v(t)·S
    dV/dt = +v(t)·S - (1-VE)·λ·V
    dE/dt = +λ·S + (1-VE)·λ·V - σ·E
    dI/dt = +σ·E - γ·I
    dR/dt = +γ·I

λ = compute_foi(state, mobility, matrices, pop, params)  (15, n_admdong)
v(t) = VaccinationParameters.rate_vector(day_in_season)  (15,)

인구 보존: dS + dV + dE + dI + dR = 0.
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable

import numpy as np

from kt_epimodel_hira.model.compartments import (
    IDX_E,
    IDX_I,
    IDX_R,
    IDX_S,
    IDX_V,
    flatten_state,
    unflatten_state,
)
from kt_epimodel_hira.model.foi import compute_foi
from kt_epimodel_hira.model.parameters import ModelParameters


def compute_derivatives(
    state: np.ndarray,
    mobility: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
    pop_15: np.ndarray,
    params: ModelParameters,
    day_in_season: int = 0,
) -> np.ndarray:
    """ODE 우변.

    Args:
        state: (5, 15, n_admdong).
        mobility: {'home', 'school', 'work', 'other'} → (15, n, n).
        matrices: NIMS contact matrices.
        pop_15: (15, n_admdong).
        params: ModelParameters (employment 필수).
        day_in_season: 시즌 시작부터 일수 (백신 시간 의존).

    Returns:
        dstate: (5, 15, n_admdong).
    """
    foi = compute_foi(
        state, mobility, matrices, pop_15, params, day_in_season=day_in_season,
    )   # (15, n)

    S = state[IDX_S]
    V = state[IDX_V]
    E = state[IDX_E]
    I = state[IDX_I]

    sigma = params.disease.sigma
    gamma = params.disease.gamma
    VE = params.vaccination.VE

    v_rate = params.vaccination.rate_vector(day_in_season)[:, None]   # (15, 1)
    breakthrough = (1.0 - VE) * foi                                   # (15, n)

    dS = -foi * S - v_rate * S
    dV = v_rate * S - breakthrough * V
    dE = foi * S + breakthrough * V - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return np.stack([dS, dV, dE, dI, dR], axis=0)


def make_ode_rhs(
    mobility: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
    pop_15: np.ndarray,
    params: ModelParameters,
    day_in_season_offset: float = 0.0,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """scipy.solve_ivp 호환 f(t, y) → dy/dt (시간 불변 mobility/matrices).

    Args:
        day_in_season_offset: t=0 의 day_in_season 값.
    """
    n_adm = pop_15.shape[1]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        state = unflatten_state(y, n_adm)
        day = int(t + day_in_season_offset)
        dstate = compute_derivatives(state, mobility, matrices, pop_15, params, day)
        return flatten_state(dstate)

    return rhs


def make_time_varying_ode_rhs(
    mobility_by_daytype: dict[str, dict[str, np.ndarray]],
    matrices: dict[str, np.ndarray],
    pop_15: np.ndarray,
    params: ModelParameters,
    start_date: int,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """날짜에 따라 daytype 가 바뀌는 f(t, y).

    Args:
        mobility_by_daytype: {'weekday': {home, school, work, other}, 'weekend': {...}}.
        start_date: yyyymmdd — t=0 의 날짜 + 시즌 day_in_season=0 가정.
    """
    from kt_data.data.load_calendar import classify_date

    required = {"weekday", "weekend"}
    missing = required - set(mobility_by_daytype.keys())
    if missing:
        raise ValueError(f"mobility_by_daytype missing keys: {missing}")

    n_adm = pop_15.shape[1]
    s_y, s_m, s_d = start_date // 10000, (start_date // 100) % 100, start_date % 100
    start_dt = _dt.date(s_y, s_m, s_d)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        current_dt = start_dt + _dt.timedelta(days=int(t))
        yyyymmdd = current_dt.year * 10000 + current_dt.month * 100 + current_dt.day
        daytype = classify_date(yyyymmdd)

        mob_key = "weekend" if daytype in ("weekend", "holiday") else "weekday"
        mobility = mobility_by_daytype[mob_key]

        factor = params.time_varying.get(daytype)
        matrices_scaled = {
            "C_home":   factor["home"]   * matrices["C_home"],
            "C_work":   factor["work"]   * matrices["C_work"],
            "C_school": factor["school"] * matrices["C_school"],
            "C_other":  factor["other"]  * matrices["C_other"],
        }

        state = unflatten_state(y, n_adm)
        dstate = compute_derivatives(
            state, mobility, matrices_scaled, pop_15, params, int(t),
        )
        return flatten_state(dstate)

    return rhs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from kt_data import (
        get_population_matrix,
        load_contact_matrices,
        load_population_15groups,
    )

    from kt_epimodel_hira.model.compartments import initial_state
    from kt_epimodel_hira.model.mobility_tensor import (
        build_M_home,
        build_M_other,
        build_M_school,
        build_M_work,
    )
    from kt_epimodel_hira.model.parameters import EmploymentParameters

    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)
    n = pop.shape[1]

    matrices = load_contact_matrices()
    mobility = {
        "home":   build_M_home(n),
        "school": build_M_school(n),
        "work":   build_M_work("202301", admdong_codes=codes, pop_15=pop),
        "other":  build_M_other("202301", admdong_codes=codes, pop_15=pop),
    }

    params = ModelParameters().with_employment(EmploymentParameters.from_kt_data(codes))
    state = initial_state(pop, seed_per_admdong=10)

    dstate = compute_derivatives(state, mobility, matrices, pop, params, day_in_season=0)
    print("=== Derivatives at t=0 (시즌 시작) ===")
    print(f"  dstate shape: {dstate.shape}")
    print(f"  dS sum: {dstate[IDX_S].sum():>14,.2f}")
    print(f"  dV sum: {dstate[IDX_V].sum():>14,.2f}")
    print(f"  dE sum: {dstate[IDX_E].sum():>14,.2f}")
    print(f"  dI sum: {dstate[IDX_I].sum():>14,.2f}")
    print(f"  dR sum: {dstate[IDX_R].sum():>14,.2f}")
    print(f"  총 변화 (보존):  {dstate.sum():.3e}")

    dstate_peak = compute_derivatives(
        state, mobility, matrices, pop, params, day_in_season=42,
    )
    print("\n=== Derivatives at peak vaccination (day 42) ===")
    print(f"  dS sum: {dstate_peak[IDX_S].sum():>14,.2f}")
    print(f"  dV sum: {dstate_peak[IDX_V].sum():>14,.2f}  (백신 유입 → 양수 예상)")
    print(f"  dE sum: {dstate_peak[IDX_E].sum():>14,.2f}")
    print(f"  보존:           {dstate_peak.sum():.3e}")

    rhs = make_ode_rhs(mobility, matrices, pop, params)
    y0 = flatten_state(state)
    dy = rhs(0.0, y0)
    print("\n=== ODE RHS (scipy 호환) ===")
    print(f"  dy shape: {dy.shape}")
    print(f"  dy sum:   {dy.sum():.3e}")
