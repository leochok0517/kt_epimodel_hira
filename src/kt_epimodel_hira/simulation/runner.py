"""High-level workflow runner (Step E — SEIRV + 4채널 mobility).

데이터 로드 → mobility 4채널 빌드 → C 합성 → 초기 상태 → 시뮬레이션.
정책 시나리오 batch 실행 + 비교 헬퍼 포함.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import polars as pl

from kt_data.data.load_contact import load_contact_matrices
from kt_data.data.load_population import (
    get_population_matrix,
    load_population_15groups,
)

from kt_epimodel_hira.model.compartments import (
    IDX_I,
    IDX_R,
    IDX_V,
    initial_state,
)
from kt_epimodel_hira.model.mobility_tensor import (
    build_M_home,
    build_M_other,
    build_M_school,
    build_M_work,
)
from kt_epimodel_hira.model.parameters import (
    EmploymentParameters,
    ModelParameters,
    PolicyParameters,
)
from kt_epimodel_hira.simulation.solver import SimulationResult, run_simulation


@lru_cache(maxsize=8)
def _cached_inputs(yyyymm: str, daytype: str) -> dict[str, Any]:
    """반복 호출 캐시 — pop, codes, mobility 4채널, matrices."""
    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)
    n = pop.shape[1]
    mobility = {
        "home":   build_M_home(n),
        "school": build_M_school(n),
        "work":   build_M_work(yyyymm, daytype=daytype, admdong_codes=codes, pop_15=pop),
        "other":  build_M_other(yyyymm, daytype=daytype, admdong_codes=codes, pop_15=pop),
    }
    matrices = load_contact_matrices()
    return {"pop": pop, "codes": codes, "mobility": mobility, "matrices": matrices}


def run_single_season(
    yyyymm: str = "202301",
    daytype: str = "weekday",
    params: ModelParameters | None = None,
    seed_per_admdong: float = 5.0,
    seed_age: str = "population_weighted",
    initial_immunity: float = 0.0,
    initial_vaccinated_fraction: float = 0.0,
    t_span: tuple[float, float] = (0.0, 224.0),
    day_in_season_offset: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """원샷 SEIRV 시뮬레이션 — 데이터 자동 로드 + 4채널 mobility."""
    inputs = _cached_inputs(yyyymm, daytype)
    pop = inputs["pop"]
    codes = inputs["codes"]
    mobility = inputs["mobility"]
    matrices = inputs["matrices"]

    if params is None:
        params = ModelParameters()
    if params.employment is None:
        params = params.with_employment(EmploymentParameters.from_kt_data(codes))

    state = initial_state(
        pop,
        seed_per_admdong=seed_per_admdong,
        seed_age_distribution=seed_age,
        initial_immunity=initial_immunity,
        initial_vaccinated_fraction=initial_vaccinated_fraction,
    )

    if verbose:
        print(f"Run: {yyyymm}/{daytype}, t_span={t_span}")
        print(f"  pop:        {pop.sum():,.0f}")
        print(f"  initial I:  {state[IDX_I].sum():,.0f}")
        print(f"  initial R:  {state[IDX_R].sum():,.0f}")
        print(f"  initial V:  {state[IDX_V].sum():,.0f}")

    result = run_simulation(
        state, mobility, matrices, pop, params,
        t_span=t_span, day_in_season_offset=day_in_season_offset,
    )

    if verbose:
        ar = float(result.attack_rate(pop)[-1].mean())
        vax = float(result.vaccinated_total()[-1] / pop.sum())
        print(f"  success:    {result.success}")
        print(f"  final AR:   {ar:.4%}")
        print(f"  final vax:  {vax:.4%}")

    return {
        "result": result,
        "params": params,
        "admdong_codes": codes,
        "pop_15": pop,
        "metadata": {
            "yyyymm": yyyymm,
            "daytype": daytype,
            "t_span": t_span,
            "seed_per_admdong": seed_per_admdong,
            "seed_age": seed_age,
            "initial_immunity": initial_immunity,
            "initial_vaccinated_fraction": initial_vaccinated_fraction,
        },
    }


def run_scenarios(
    scenarios: dict[str, PolicyParameters],
    yyyymm: str = "202301",
    base_params: ModelParameters | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """다 정책 시나리오 일괄 실행."""
    if base_params is None:
        base_params = ModelParameters()

    results: dict[str, dict[str, Any]] = {}
    for name, policy in scenarios.items():
        if verbose:
            print(f"\n=== Scenario: {name} ===")
        params = base_params.with_policy(policy)
        results[name] = run_single_season(
            yyyymm=yyyymm, params=params, verbose=verbose, **kwargs,
        )
    return results


def compare_scenarios(results: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """시나리오 비교 표."""
    rows = []
    for name, data in results.items():
        result: SimulationResult = data["result"]
        pop: np.ndarray = data["pop_15"]
        totals = result.total_by_compartment()       # (n_t, 5)
        total_pop = float(pop.sum())

        cum_infected_final = float(totals[-1, IDX_I] + totals[-1, IDX_R])
        I_series = totals[:, IDX_I]
        peak_idx = int(np.argmax(I_series))

        rows.append({
            "scenario": name,
            "final_attack_rate": cum_infected_final / total_pop,
            "final_R": float(totals[-1, IDX_R]),
            "final_vaccinated": float(totals[-1, IDX_V]) / total_pop,
            "peak_infectious": float(I_series[peak_idx]),
            "peak_day": float(result.t[peak_idx]),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from kt_epimodel_hira.model.parameters import CalibrationParameters

    # 정책 효과 가시화 위해 β 강화
    base = ModelParameters(
        calibration=CalibrationParameters(
            beta_h=0.3, beta_w=0.3, beta_s=0.3, beta_o=0.3,
            phi=np.ones(15), gamma_report=0.5,
        ),
    )

    scenarios = {
        "baseline":       PolicyParameters.baseline(),
        "school_closure": PolicyParameters.school_closure(),
        "sick_leave":     PolicyParameters.sick_leave_enhanced(),
        "comprehensive":  PolicyParameters.comprehensive(),
    }
    results = run_scenarios(
        scenarios, base_params=base, t_span=(0.0, 60.0), seed_per_admdong=5.0,
    )

    print("\n=== Summary ===")
    print(compare_scenarios(results))
