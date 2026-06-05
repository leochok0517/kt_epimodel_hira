"""HIRA calibration target (kt_epimodel_hira).

국민건강보험공단 인플루엔자 진료에피소드 카운트를 모델 fit target 으로 변환.
모델 일일 incidence → 주별 HIRA count 변환 + Poisson log-likelihood.

ILI 버전 (kt_epimodel/calibration/ili_target.py) 대비 차이:
    - 단위가 비율(per-1000) 이 아닌 **카운트** → 인구 분모 곱셈/나눗셈 제거.
    - 7 그룹 매핑 (ILI_GROUP_TO_NIMS_WEIGHTED) → 6 그룹 매핑
      (HIRA_GROUP_TO_NIMS_WEIGHTED).
    - "외래환자 1000명당" 환산 코드 (× 1000.0) 제거.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import polars as pl

from kt_data.data.load_hira import (
    HIRA_AGE_GROUPS,
    SUDOGWON_SIDO_CODES,
    aggregate_hira_weekly,
    extract_hira_season,
    load_hira_episodes,
)

N_WEEKS: int = 52


# ---------------------------------------------------------------------------
# 6 → NIMS 15 가중 매핑
# ---------------------------------------------------------------------------

#: HIRA 6 연령 그룹 → NIMS 15군 인구비례 가중 매핑.
#: 각 NIMS idx (0..14) 에서 전 HIRA 그룹 가중치 합 = 1.0 (test 검증).
#:
#: 매핑 도출 (5세 단위 균등 분포 가정):
#:     - "0-5"   (6 ages)  → NIMS 0 (0-4, 5 ages 전부) + NIMS 1 (5세 → 1/5)
#:     - "6-11"  (6 ages)  → NIMS 1 (6-9, 4/5) + NIMS 2 (10-11, 2/5)
#:     - "12-17" (6 ages)  → NIMS 2 (12-14, 3/5) + NIMS 3 (15-17, 3/5)
#:     - "18-44" (27 ages) → NIMS 3 (18-19, 2/5) + NIMS 4-8 (20-44 전부)
#:     - "45-64" (20 ages) → NIMS 9-12 (45-64 전부)
#:     - "65+"             → NIMS 13 (65-69) + NIMS 14 (70+)
HIRA_GROUP_TO_NIMS_WEIGHTED: dict[str, dict[int, float]] = {
    "0-5":   {0: 1.0, 1: 0.2},
    "6-11":  {1: 0.8, 2: 0.4},
    "12-17": {2: 0.6, 3: 0.6},
    "18-44": {3: 0.4, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0},
    "45-64": {9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
    "65+":   {13: 1.0, 14: 1.0},
}


def _verify_weighted_coverage() -> dict[int, float]:
    """각 NIMS 인덱스에서 전 HIRA 그룹의 weight 합 (검증용 — 1.0 기대)."""
    total: dict[int, float] = {i: 0.0 for i in range(15)}
    for weights in HIRA_GROUP_TO_NIMS_WEIGHTED.values():
        for nims_idx, w in weights.items():
            total[nims_idx] += w
    return total


# ---------------------------------------------------------------------------
# Target loaders
# ---------------------------------------------------------------------------

def season_start_date(season: str) -> int:
    """시즌명 → ISO 36주 월요일 yyyymmdd.

    예: '2019-2020' → 20190902.
    """
    start_year = int(season.split("-")[0])
    d = _dt.date.fromisocalendar(start_year, 36, 1)
    return d.year * 10000 + d.month * 100 + d.day


def load_hira_target(
    season: str,
    sido_codes: list[int] | None = None,
    setting: str = "outpatient_inpatient",
    first_peak_only: bool = False,
    first_peak_end_week: int = 26,
) -> dict:
    """한 시즌 HIRA 카운트 → calibration용 dict (연령 합산 전체).

    Args:
        season: '2019-2020' 등.
        sido_codes: 시도 코드 필터. None 이면 ``SUDOGWON_SIDO_CODES``.
        setting: ``"outpatient_inpatient"`` 또는 ``"inpatient_only"``.
        first_peak_only: True 면 첫 봉우리 (week < first_peak_end_week) 만 weight 1.
        first_peak_end_week: default 26.

    Returns:
        dict {season, week_in_season, hira_count, is_valid, weights, n_weeks}.
        ILI 버전과 키 통일성 위해 ``ili_rate`` 대신 ``hira_count`` 사용.
        ``weights``: NLL 가중치 — default 는 is_valid float, first_peak_only=True
        면 후반부 0.
    """
    if sido_codes is None:
        sido_codes = list(SUDOGWON_SIDO_CODES)
    season_start_year = int(season.split("-")[0])

    # 일별 로드 → 주별 (시도/성별/연령 모두 합산)
    df_daily = load_hira_episodes(setting=setting, sido_codes=sido_codes)
    df_weekly_by_age = aggregate_hira_weekly(df_daily, sum_over=("sido_code", "sex"))
    # 연령도 합산: group_by week 만
    df_weekly_total = (
        df_weekly_by_age.group_by(["week_start_date", "iso_year", "iso_week"])
        .agg(pl.col("episodes").sum().cast(pl.Int64))
        .sort("week_start_date")
    )
    df_season = extract_hira_season(df_weekly_total, season_start_year=season_start_year)
    if df_season.height == 0:
        raise ValueError(f"no HIRA data in season {season!r}")

    weeks = df_season["week_in_season"].to_numpy()
    counts = df_season["episodes"].to_numpy().astype(np.float64)
    is_valid = np.ones_like(weeks, dtype=bool)
    weights = is_valid.astype(np.float64)
    if first_peak_only:
        weights[weeks >= first_peak_end_week] = 0.0

    return {
        "season": season,
        "week_in_season": weeks,
        "hira_count": counts,
        "is_valid": is_valid,
        "weights": weights,
        "n_weeks": int(len(weeks)),
    }


def load_hira_target_by_age(
    season: str,
    sido_codes: list[int] | None = None,
    setting: str = "outpatient_inpatient",
    first_peak_only: bool = False,
    first_peak_end_week: int = 26,
) -> dict:
    """6 HIRA 연령 그룹 target.

    Returns:
        {'season', 'age_groups', 'hira_counts', 'weights', 'is_valid',
         'week_in_season', 'n_weeks'}
        hira_counts/weights/is_valid 는 {age_group: (n_weeks,)} dict.
    """
    if sido_codes is None:
        sido_codes = list(SUDOGWON_SIDO_CODES)
    season_start_year = int(season.split("-")[0])

    # 일별 → 주별 (시도/성별 합산, 연령 유지)
    df_daily = load_hira_episodes(setting=setting, sido_codes=sido_codes)
    df_weekly = aggregate_hira_weekly(df_daily, sum_over=("sido_code", "sex"))
    df_season = extract_hira_season(df_weekly, season_start_year=season_start_year)
    if df_season.height == 0:
        raise ValueError(f"no HIRA data in season {season!r}")

    result: dict = {
        "season": season,
        "age_groups": list(HIRA_AGE_GROUPS),
        "hira_counts": {},
        "weights": {},
        "is_valid": {},
        "week_in_season": None,
        "n_weeks": N_WEEKS,
    }

    for ag in HIRA_AGE_GROUPS:
        sub = df_season.filter(pl.col("age_group") == ag).sort("week_in_season")
        if sub.height == 0:
            raise ValueError(f"no HIRA data for age group {ag!r} in season {season!r}")
        weeks = sub["week_in_season"].to_numpy()
        counts = sub["episodes"].to_numpy().astype(np.float64)

        is_valid = np.ones_like(weeks, dtype=bool)
        weights = is_valid.astype(np.float64)
        if first_peak_only:
            weights[weeks >= first_peak_end_week] = 0.0

        result["hira_counts"][ag] = counts
        result["weights"][ag] = weights
        result["is_valid"][ag] = is_valid
        if result["week_in_season"] is None:
            result["week_in_season"] = weeks

    return result


# ---------------------------------------------------------------------------
# Simulation → HIRA count conversion
# ---------------------------------------------------------------------------

def simulation_to_hira(
    daily_incidence_by_age: np.ndarray,
    gamma_15: np.ndarray,
    n_weeks: int = N_WEEKS,
) -> np.ndarray:
    """모델 출력 → 주별 HIRA 카운트 (연령 합산).

    Args:
        daily_incidence_by_age: (n_days, 15) — NIMS 15군 일일 신규 감염.
        gamma_15: (15,) age-dependent reporting fraction.
        n_weeks: 출력 주차 수.
    """
    arr = np.asarray(daily_incidence_by_age, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 15:
        raise ValueError(
            f"daily_incidence_by_age must be (n_days, 15), got {arr.shape}"
        )
    gamma_15 = np.asarray(gamma_15, dtype=np.float64)

    n_complete = arr.shape[0] // 7
    weekly = arr[: n_complete * 7].reshape(n_complete, 7, 15).sum(axis=1)
    reported = weekly * gamma_15[np.newaxis, :]
    hira = reported.sum(axis=1)

    if len(hira) >= n_weeks:
        return hira[:n_weeks]
    return np.concatenate([hira, np.zeros(n_weeks - len(hira))])


def simulation_to_hira_by_age(
    daily_incidence_by_age: np.ndarray,
    gamma_15: np.ndarray,
    n_weeks: int = N_WEEKS,
) -> dict[str, np.ndarray]:
    """모델 출력 → 6 HIRA 연령 그룹 주별 카운트.

    gamma_15 is applied at NIMS-15 level before HIRA group mapping,
    so cross-boundary groups (e.g. 18-44 spanning child/adult) get
    the correct per-age gamma.

    Args:
        daily_incidence_by_age: (n_days, 15) — NIMS 15군 일일 신규 감염.
        gamma_15: (15,) age-dependent reporting fraction.
        n_weeks: 출력 주차 수.
    """
    arr = np.asarray(daily_incidence_by_age, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 15:
        raise ValueError(
            f"daily_incidence_by_age must be (n_days, 15), got {arr.shape}"
        )
    gamma_15 = np.asarray(gamma_15, dtype=np.float64)

    n_days = arr.shape[0]
    n_complete = n_days // 7
    weekly = arr[: n_complete * 7].reshape(n_complete, 7, 15).sum(axis=1)
    reported = weekly * gamma_15[np.newaxis, :]

    def _pad(x: np.ndarray) -> np.ndarray:
        if len(x) >= n_weeks:
            return x[:n_weeks]
        return np.concatenate([x, np.zeros(n_weeks - len(x))])

    out: dict[str, np.ndarray] = {}
    for ag, weights in HIRA_GROUP_TO_NIMS_WEIGHTED.items():
        group_count = np.zeros(n_complete)
        for nims_idx, w in weights.items():
            group_count += w * reported[:, nims_idx]
        out[ag] = _pad(group_count)
    return out


# ---------------------------------------------------------------------------
# Poisson NLL
# ---------------------------------------------------------------------------

def poisson_log_likelihood(
    observed: np.ndarray,
    predicted: np.ndarray,
    is_valid: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    min_rate: float = 0.01,
) -> float:
    """Weighted Poisson NLL = Σ w_i [y_pred − y_obs · log(y_pred)].

    Args:
        weights: (n_weeks,) 가중치. None 이면 1.0. weight=0 인 주는 NLL 기여 0.
        min_rate: predicted 의 lower floor.
            # HIRA-D: min_rate 0.01 (was 1.0). 카운트 데이터 스케일에 맞춰 낮춤.
            # 너무 낮으면 baseline 구간 (low count) 에서 NLL 발산 위험 — fit 후 점검.
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError(
            f"shape mismatch: observed {observed.shape} vs predicted {predicted.shape}"
        )

    mask = np.ones_like(observed, dtype=bool) if is_valid is None else is_valid.astype(bool)
    mask &= ~np.isnan(observed)
    mask &= ~np.isnan(predicted)

    if weights is None:
        w_arr = np.ones_like(observed, dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape != observed.shape:
            raise ValueError(
                f"weights shape {w_arr.shape} != observed {observed.shape}"
            )
        mask &= w_arr > 0

    obs = observed[mask]
    pred = np.maximum(predicted[mask], min_rate)
    w = w_arr[mask]
    if obs.size == 0:
        return float("inf")
    return float(np.sum(w * (pred - obs * np.log(pred))))
