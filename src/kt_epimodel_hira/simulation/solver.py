"""ODE solver wrapper (Step E — SEIRV + mobility dict + matrices dict).

scipy.integrate.solve_ivp 래퍼:
- run_simulation: 시간 불변 mobility/matrices.
- run_simulation_time_varying: daytype 변동.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from kt_epimodel_hira.model.compartments import (
    IDX_E,
    IDX_I,
    IDX_R,
    IDX_S,
    IDX_V,
    N_AGE,
    N_COMPARTMENTS,
    flatten_state,
    unflatten_state,
)
from kt_epimodel_hira.model.dynamics import (
    make_ode_rhs,
    make_time_varying_ode_rhs,
)
from kt_epimodel_hira.model.parameters import ModelParameters


@dataclass
class SimulationResult:
    """시뮬레이션 결과 (SEIRV)."""
    t: np.ndarray              # (n_t,)
    states: np.ndarray         # (n_t, 5, 15, n_admdong)
    success: bool
    message: str
    n_admdong: int

    # --- access ---
    def get_compartment(self, idx: int) -> np.ndarray:
        return self.states[:, idx, :, :]

    def total_by_compartment(self) -> np.ndarray:
        """(n_t, 5)."""
        return self.states.sum(axis=(2, 3))

    def total_by_age(self) -> np.ndarray:
        """(n_t, 5, 15)."""
        return self.states.sum(axis=3)

    # --- attack rate ---
    def attack_rate(self, population: np.ndarray) -> np.ndarray:
        """(I + R) / N. V 분자 제외. (n_t, 15, n_admdong)."""
        infected = self.states[:, IDX_I] + self.states[:, IDX_R]
        pop_b = population[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(pop_b > 0, infected / pop_b, 0.0)

    # --- incidence ---
    def daily_new_infection(self) -> np.ndarray:
        """일일 신규 감염 (S→E flux 근사). (n_t-1,).

        근사: Δ(E + I + R) — 새로 감염 경로(E)로 들어온 누적량 변화.
        −ΔS 와 달리 백신 흐름(S→V) 을 제외.
        """
        E = self.states[:, IDX_E].sum(axis=(1, 2))
        I = self.states[:, IDX_I].sum(axis=(1, 2))
        R = self.states[:, IDX_R].sum(axis=(1, 2))
        return np.diff(E + I + R)

    def daily_new_infection_by_age(self) -> np.ndarray:
        """일일 신규 감염 — 연령별 (15군). (n_t-1, 15).

        동일 원리: Δ(E + I + R) per age group. 백신 흐름(S→V) 제외.
        """
        E = self.states[:, IDX_E, :, :].sum(axis=-1)   # (n_t, 15)
        I = self.states[:, IDX_I, :, :].sum(axis=-1)
        R = self.states[:, IDX_R, :, :].sum(axis=-1)
        return np.diff(E + I + R, axis=0)

    # --- vaccination ---
    def vaccinated_count(self) -> np.ndarray:
        """V compartment 시계열. (n_t, 15, n_admdong)."""
        return self.states[:, IDX_V]

    def vaccinated_total(self) -> np.ndarray:
        """전체 V 시계열. (n_t,)."""
        return self.states[:, IDX_V].sum(axis=(1, 2))


def _stack_solution(sol_y: np.ndarray, n_t: int, n_admdong: int) -> np.ndarray:
    states = np.zeros((n_t, N_COMPARTMENTS, N_AGE, n_admdong), dtype=np.float64)
    for k in range(n_t):
        states[k] = unflatten_state(sol_y[:, k], n_admdong)
    return states


def run_simulation(
    initial_state_arr: np.ndarray,
    mobility: dict[str, np.ndarray],
    matrices: dict[str, np.ndarray],
    pop_15: np.ndarray,
    params: ModelParameters,
    t_span: tuple[float, float] = (0.0, 224.0),
    t_eval: np.ndarray | None = None,
    day_in_season_offset: float = 0.0,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> SimulationResult:
    """시간 불변 mobility/matrices SEIRV 시뮬레이션."""
    n_admdong = initial_state_arr.shape[2]
    if t_eval is None:
        t_eval = np.arange(t_span[0], t_span[1] + 1, 1.0)

    rhs = make_ode_rhs(mobility, matrices, pop_15, params, day_in_season_offset)
    y0 = flatten_state(initial_state_arr)

    sol = solve_ivp(
        rhs, t_span, y0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    states = _stack_solution(sol.y, len(sol.t), n_admdong)
    return SimulationResult(
        t=sol.t,
        states=states,
        success=bool(sol.success),
        message=str(sol.message),
        n_admdong=n_admdong,
    )


def run_simulation_time_varying(
    initial_state_arr: np.ndarray,
    mobility_by_daytype: dict[str, dict[str, np.ndarray]],
    matrices: dict[str, np.ndarray],
    pop_15: np.ndarray,
    params: ModelParameters,
    start_date: int,
    n_days: int = 224,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> SimulationResult:
    """daytype 가변 시뮬레이션."""
    n_admdong = initial_state_arr.shape[2]
    rhs = make_time_varying_ode_rhs(
        mobility_by_daytype, matrices, pop_15, params, start_date,
    )
    y0 = flatten_state(initial_state_arr)
    t_eval = np.arange(0, n_days + 1, 1.0)

    sol = solve_ivp(
        rhs, (0.0, float(n_days)), y0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    states = _stack_solution(sol.y, len(sol.t), n_admdong)
    return SimulationResult(
        t=sol.t,
        states=states,
        success=bool(sol.success),
        message=str(sol.message),
        n_admdong=n_admdong,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from kt_data.data.load_contact import load_contact_matrices
    from kt_data.data.load_population import (
        get_population_matrix,
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
    state = initial_state(pop, seed_per_admdong=5)

    print("=== 시뮬레이션 시작 ===")
    print(f"인구:       {pop.sum():,.0f}")
    print(f"초기 I:     {state[IDX_I].sum():,.0f}")
    print(f"행정동:     {n}")

    t0 = time.perf_counter()
    result = run_simulation(state, mobility, matrices, pop, params, t_span=(0, 30))
    elapsed = time.perf_counter() - t0

    print(f"\n=== 완료 ({elapsed:.1f}초) success={result.success} ===")
    totals = result.total_by_compartment()
    print(f"\n{'Day':>4} {'S':>14} {'V':>12} {'E':>10} {'I':>10} {'R':>10}")
    for k in (0, 1, 5, 10, 20, 30):
        if k < len(result.t):
            print(
                f"{result.t[k]:>4.0f} "
                f"{totals[k, 0]:>14,.0f} {totals[k, 1]:>12,.0f} "
                f"{totals[k, 2]:>10,.0f} {totals[k, 3]:>10,.0f} {totals[k, 4]:>10,.0f}"
            )

    total_t = totals.sum(axis=1)
    print(f"\n총 인구 (보존): Δ = {total_t.max() - total_t.min():.3e}")

    ar = (totals[-1, IDX_I] + totals[-1, IDX_R]) / pop.sum()
    vax = totals[-1, IDX_V] / pop.sum()
    print(f"30일 attack rate:    {ar:.4%}")
    print(f"30일 vaccinated:     {vax:.4%}")
