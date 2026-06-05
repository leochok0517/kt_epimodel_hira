"""Aggregated (1-admdong) 모델 — calibration 인프라 (HIRA 버전).

행정동 1,154 → 단일 admdong 으로 합산 (수도권 전체).
mobility 는 모두 identity (15, 1, 1).
ρ 는 시도별 인구 가중 평균 → (1, 15).

ILI 버전 (kt_epimodel/calibration/simple_model.py) 대비 차이:
    - ``estimate_initial_infected_from_ili`` → ``estimate_initial_infected_from_hira``
      (수식: per-1000 / 인구 분모 제거, 카운트 직접 사용).
    - 그 외 함수 (``build_aggregated_inputs``, ``_build_initial_state_with_age_seed``,
      ``simulate_aggregated``) 는 ILI 무관 → ILI 버전 그대로.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from kt_data import (
    load_contact_matrices,
    load_employment_rate,
    load_population_15groups,
)
from kt_data.data.load_population import get_population_matrix

from kt_epimodel_hira.calibration.hira_target import HIRA_GROUP_TO_NIMS_WEIGHTED
from kt_epimodel_hira.model.compartments import (
    IDX_E,
    IDX_I,
    IDX_R,
    IDX_S,
    IDX_V,
    N_COMPARTMENTS,
    initial_state,
)
from kt_epimodel_hira.model.parameters import (
    EmploymentParameters,
    ModelParameters,
)
from kt_epimodel_hira.simulation.solver import SimulationResult, run_simulation


def build_aggregated_inputs() -> dict[str, Any]:
    """수도권 합산 inputs (1 admdong, 15 ages).

    ILI 버전과 동일 (인구·접촉·이동·고용만 사용 — fit target 무관).
    """
    df_pop = load_population_15groups()
    N_mat, _codes, _ = get_population_matrix(df_pop)
    pop_aggregated = N_mat.sum(axis=0).reshape(15, 1).astype(np.float64)

    matrices = load_contact_matrices()

    mobility_eye = np.ones((15, 1, 1), dtype=np.float64)
    mobility = {
        "home":   mobility_eye.copy(),
        "school": mobility_eye.copy(),
        "work":   mobility_eye.copy(),
        "other":  mobility_eye.copy(),
    }

    df_emp = load_employment_rate()
    sudogwon_short = ["서울", "경기", "인천"]
    sudogwon_full = {"서울": "서울특별시", "경기": "경기도", "인천": "인천광역시"}
    sudogwon_emp = df_emp.filter(pl.col("sido_nm").is_in(sudogwon_short))

    sido_pops: dict[str, float] = {}
    for short in sudogwon_short:
        full = sudogwon_full[short]
        sido_pops[short] = float(
            df_pop.filter(pl.col("sido_nm") == full).get_column("pop").sum()
        )
    total_sudogwon_pop = sum(sido_pops.values())
    if total_sudogwon_pop == 0:
        raise RuntimeError("Sudogwon population zero — check load_population_15groups")

    rho_by_age = np.zeros(15, dtype=np.float64)
    for row in sudogwon_emp.iter_rows(named=True):
        a = int(row["age_idx"])
        if 0 <= a < 15:
            w = sido_pops[row["sido_nm"]] / total_sudogwon_pop
            rho_by_age[a] += float(row["employment_rate"]) * w
    rho = rho_by_age.reshape(1, 15)

    return {
        "pop_15": pop_aggregated,
        "mobility": mobility,
        "matrices": matrices,
        "rho": rho,
        "admdong_codes": ["SUDOGWON"],
    }


def estimate_initial_infected_from_hira(
    season: str,
    pop_15: np.ndarray,
    sido_codes: list[int] | None = None,
    setting: str = "outpatient_inpatient",
    week_zero_average_n: int = 3,
    gamma_report_assumed: float | None = None,
    gamma_15_assumed: np.ndarray | None = None,
) -> np.ndarray:
    """HIRA 시즌 초반 baseline 으로 NIMS 15군 초기 I 추정.

    수식: ``I = hira_baseline_count / gamma_assumed``

    Args:
        season: '2019-2020' 등.
        pop_15: (15,) 또는 (15, n_admdong) — 가중 분배에만 사용.
        sido_codes: HIRA 시도 필터. None 이면 수도권 3 시도.
        setting: 'outpatient_inpatient' or 'inpatient_only'.
        week_zero_average_n: 처음 N 주 평균 (단일 주 노이즈 회피).
        gamma_report_assumed: scalar fallback (deprecated, use gamma_15_assumed).
        gamma_15_assumed: (15,) age-dependent gamma. Takes priority.
            If neither given, uses CalibrationParameters defaults.

    Returns:
        (15,) — 연령별 초기 I 인원수.
    """
    from kt_epimodel_hira.model.parameters import CalibrationParameters

    if gamma_15_assumed is not None:
        gamma_15 = np.asarray(gamma_15_assumed, dtype=np.float64)
    elif gamma_report_assumed is not None:
        gamma_15 = np.full(15, gamma_report_assumed, dtype=np.float64)
    else:
        gamma_15 = CalibrationParameters().gamma_15

    if (gamma_15 <= 0).any():
        raise ValueError(f"gamma values must be > 0, got min={gamma_15.min()}")

    from kt_data.data.load_hira import (
        SUDOGWON_SIDO_CODES,
        aggregate_hira_weekly,
        extract_hira_season,
        load_hira_episodes,
    )

    pop_arr = np.asarray(pop_15, dtype=np.float64)
    if pop_arr.ndim == 2:
        pop_total_15 = pop_arr.sum(axis=1) if pop_arr.shape[0] == 15 else pop_arr.flatten()
    else:
        pop_total_15 = pop_arr
    if pop_total_15.shape != (15,):
        raise ValueError(f"pop_15 must reduce to (15,), got {pop_total_15.shape}")

    if sido_codes is None:
        sido_codes = list(SUDOGWON_SIDO_CODES)
    season_start_year = int(season.split("-")[0])

    df_daily = load_hira_episodes(setting=setting, sido_codes=sido_codes)
    df_weekly = aggregate_hira_weekly(df_daily, sum_over=("sido_code", "sex"))
    df_season = extract_hira_season(df_weekly, season_start_year=season_start_year)
    if df_season.height == 0:
        raise ValueError(f"no HIRA data in season {season!r}")

    # 시즌 초반 N 주 평균 (그룹별)
    hira_baseline: dict[str, float] = {}
    for ag in HIRA_GROUP_TO_NIMS_WEIGHTED:
        sub = (
            df_season.filter(
                (pl.col("age_group") == ag)
                & (pl.col("week_in_season") < int(week_zero_average_n))
            )
            .sort("week_in_season")
        )
        if sub.height == 0:
            hira_baseline[ag] = 0.0
        else:
            counts = sub["episodes"].to_numpy().astype(np.float64)
            hira_baseline[ag] = float(np.nanmean(counts))

    # I = baseline_count / gamma_weighted, distributed by pop within HIRA group
    I_initial_15 = np.zeros(15, dtype=np.float64)
    for ag, weights in HIRA_GROUP_TO_NIMS_WEIGHTED.items():
        total_weighted_pop = sum(
            w * float(pop_total_15[idx]) for idx, w in weights.items()
        )
        if total_weighted_pop < 1e-10:
            continue
        # Weighted average gamma for this HIRA group
        gamma_avg = sum(
            w * float(pop_total_15[idx]) * float(gamma_15[idx])
            for idx, w in weights.items()
        ) / total_weighted_pop
        gamma_avg = max(gamma_avg, 1e-10)
        I_group = hira_baseline[ag] / gamma_avg
        for nims_idx, w in weights.items():
            share = (w * float(pop_total_15[nims_idx])) / total_weighted_pop
            I_initial_15[nims_idx] += I_group * share
    return I_initial_15


# ---------------------------------------------------------------------------
# Initial state + simulate — ILI 버전 그대로 (target 무관)
# ---------------------------------------------------------------------------

# Age-specific initial immunity profile R(0) — step function.
# Derived from POLYMOD contact accumulation (Mossong et al. 2008) and elderly
# cross-reactive immunity from past pandemics (Vandegrift et al. 2014).
# See docs/PRIOR_SPECIFICATION.md Appendix A for full rationale.
R0_IMMUNITY_PROFILE: np.ndarray = np.array(
    [0.10] * 4    # 0-19   children: limited prior exposure
    + [0.30] * 6  # 20-49  young/middle adults: baseline accumulated
    + [0.45] * 3  # 50-64  middle-aged: routine vax + accumulation
    + [0.65] * 2, # 65+    elderly: vax (82%) + cross-immunity
    dtype=np.float64,
)
assert R0_IMMUNITY_PROFILE.shape == (15,), \
    f"R0_IMMUNITY_PROFILE must be (15,), got {R0_IMMUNITY_PROFILE.shape}"


def _build_initial_state_with_age_seed(
    pop_15: np.ndarray,
    seed_by_age: np.ndarray,
    seed_e_factor: float,
    initial_immunity,
    initial_vaccinated_fraction,
) -> np.ndarray:
    """연령별 seed 로 SEIRV 초기 state 직접 구성.

    상태 텐서: (5, 15, n_admdong). 행정동 분배는 인구 비례.

    Args:
        initial_immunity: scalar (uniform R(0)) or (15,) array (age-specific).
            Scalar accepted for backward compat with existing code.
        initial_vaccinated_fraction: scalar or (15,) array.
    """
    pop_arr = np.asarray(pop_15, dtype=np.float64)
    if pop_arr.ndim == 1:
        pop_2d = pop_arr.reshape(15, 1)
    else:
        pop_2d = pop_arr
    n_age, n_adm = pop_2d.shape

    seed = np.asarray(seed_by_age, dtype=np.float64)
    if seed.shape != (15,):
        raise ValueError(f"seed_by_age must be (15,), got {seed.shape}")
    if (seed < 0).any():
        raise ValueError("seed_by_age must be nonneg")

    # Broadcast scalar -> (15,) for backward compat; pass-through (15,) arrays
    imm = np.broadcast_to(
        np.asarray(initial_immunity, dtype=np.float64), (n_age,)
    )
    vac = np.broadcast_to(
        np.asarray(initial_vaccinated_fraction, dtype=np.float64), (n_age,)
    )
    if (imm < 0).any() or (imm >= 1).any():
        raise ValueError(f"initial_immunity must be in [0, 1), got {imm}")
    if (vac < 0).any() or (vac >= 1).any():
        raise ValueError(f"initial_vaccinated_fraction in [0, 1), got {vac}")
    if (imm + vac >= 1).any():
        raise ValueError("initial_immunity + initial_vaccinated_fraction must be < 1 per age")

    state = np.zeros((N_COMPARTMENTS, n_age, n_adm), dtype=np.float64)
    for a in range(n_age):
        row = pop_2d[a]
        n_a = row.sum()
        if n_a <= 0:
            continue
        share = row / n_a

        i_seed = float(seed[a])
        e_seed = i_seed * float(seed_e_factor)
        r_init = float(imm[a]) * n_a
        v_init = float(vac[a]) * n_a
        s_init = n_a - i_seed - e_seed - r_init - v_init
        if s_init < 0:
            avail = max(n_a - r_init - v_init, 0.0)
            total_ie = i_seed + e_seed
            if total_ie > 0:
                scale = avail / total_ie
                i_seed *= scale
                e_seed *= scale
                s_init = 0.0
            else:
                s_init = 0.0

        state[IDX_S, a] = s_init * share
        state[IDX_V, a] = v_init * share
        state[IDX_E, a] = e_seed * share
        state[IDX_I, a] = i_seed * share
        state[IDX_R, a] = r_init * share

    return state


def simulate_aggregated(
    params: ModelParameters,
    inputs: dict[str, Any],
    seed_total: float = 100.0,
    seed_by_age: np.ndarray | None = None,
    seed_e_factor: float = 0.5,
    initial_immunity=0.0,
    initial_vaccinated_fraction=0.0,
    t_span: tuple[float, float] = (0.0, 224.0),
    day_in_season_offset: float = 0.0,
) -> SimulationResult:
    """집계 모델 시뮬레이션 (ILI 버전과 동일)."""
    if params.employment is None:
        params = params.with_employment(EmploymentParameters(rho=inputs["rho"]))

    if seed_by_age is not None:
        state = _build_initial_state_with_age_seed(
            inputs["pop_15"], seed_by_age, seed_e_factor,
            initial_immunity, initial_vaccinated_fraction,
        )
    else:
        state = initial_state(
            inputs["pop_15"],
            seed_per_admdong=seed_total,
            seed_age_distribution="population_weighted",
            initial_immunity=initial_immunity,
            initial_vaccinated_fraction=initial_vaccinated_fraction,
        )

    return run_simulation(
        state,
        inputs["mobility"],
        inputs["matrices"],
        inputs["pop_15"],
        params,
        t_span=t_span,
        day_in_season_offset=day_in_season_offset,
    )
