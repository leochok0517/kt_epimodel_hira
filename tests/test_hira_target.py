"""Tests for kt_epimodel_hira.calibration.hira_target."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.calibration.hira_target import (
    HIRA_GROUP_TO_NIMS_WEIGHTED,
    N_WEEKS,
    _verify_weighted_coverage,
    poisson_log_likelihood,
    season_start_date,
    simulation_to_hira,
    simulation_to_hira_by_age,
)
from kt_data.data import DATA_ROOT
from kt_data.data.load_hira import _HIRA_XLSX_NAME

# Skip data-dependent tests if xlsx missing
_XLSX_PATH = DATA_ROOT / "external" / "hira" / _HIRA_XLSX_NAME
_HAS_FILE = _XLSX_PATH.exists()
requires_file = pytest.mark.skipif(not _HAS_FILE, reason=f"HIRA xlsx not at {_XLSX_PATH}")


# ---------- mapping ----------

def test_mapping_keys_match_hira_age_groups() -> None:
    from kt_data import HIRA_AGE_GROUPS
    assert set(HIRA_GROUP_TO_NIMS_WEIGHTED.keys()) == set(HIRA_AGE_GROUPS)


def test_weighted_coverage_sums_to_one() -> None:
    """각 NIMS idx (0..14) 의 가중치 합 = 1.0."""
    coverage = _verify_weighted_coverage()
    for nims_idx in range(15):
        assert coverage[nims_idx] == pytest.approx(1.0, abs=1e-9), (
            f"NIMS {nims_idx} coverage = {coverage[nims_idx]} != 1.0"
        )


def test_weights_in_unit_interval() -> None:
    for ag, weights in HIRA_GROUP_TO_NIMS_WEIGHTED.items():
        for nims_idx, w in weights.items():
            assert 0.0 <= w <= 1.0 + 1e-12, f"weight out of [0, 1]: {ag}/{nims_idx} = {w}"


def test_weighted_nims_indices_in_range() -> None:
    for weights in HIRA_GROUP_TO_NIMS_WEIGHTED.values():
        for nims_idx in weights:
            assert 0 <= nims_idx <= 14


def test_distribution_then_aggregation_recovers_input() -> None:
    """6 그룹 count → 15군 분배 → 다시 6 그룹 합산 = 원본 (인구비례 가중에서 일관성)."""
    # 임의 6 그룹 카운트
    src = {"0-5": 100.0, "6-11": 80.0, "12-17": 60.0, "18-44": 200.0, "45-64": 150.0, "65+": 120.0}
    # NIMS 15 분배 시: 가중치 w 가 'NIMS idx 의 어느 비율을 HIRA 그룹이 가져가는가' 이므로,
    # 다시 합산하려면 같은 w 로 가중평균 → 원본 그룹값이 나옴.
    # 검증: w 가 NIMS 그룹에 share 라서 (1-share)는 다른 HIRA 그룹이 차지.
    # 그룹 → NIMS 분배: HIRA[g] * (w_{g,n} * pop_n / Σ_n w_{g,n}*pop_n)
    # 다시 NIMS → HIRA: Σ_n share_{g,n}*nims_inc = HIRA[g] (인구 동일 가정 단순화)
    # 여기선 분배·합산 alg 자체를 검증하기보단 보존만 확인.
    # 매핑 dict 가 cover all of NIMS 0..14 (sum=1.0) → 임의 NIMS 분포에 대해 group 별 합산
    # = NIMS 카운트 합과 일치해야 함 (각 NIMS idx 가 정확히 sum=1 로 분포됨).
    nims_arbitrary = np.arange(1, 16, dtype=float)  # 1..15
    out_per_group = {}
    for ag, weights in HIRA_GROUP_TO_NIMS_WEIGHTED.items():
        out_per_group[ag] = sum(w * nims_arbitrary[idx] for idx, w in weights.items())
    total_group = sum(out_per_group.values())
    total_nims = float(nims_arbitrary.sum())
    assert total_group == pytest.approx(total_nims, abs=1e-9)


# ---------- season_start_date ----------

def test_season_start_2018_2019() -> None:
    # ISO 36 of 2018 = Mon 2018-09-03
    assert season_start_date("2018-2019") == 20180903


def test_season_start_2019_2020() -> None:
    # ISO 36 of 2019 = Mon 2019-09-02
    assert season_start_date("2019-2020") == 20190902


# ---------- simulation_to_hira (count units, no per-1000) ----------

def test_simulation_to_hira_is_count_not_rate() -> None:
    """ILI 버전과 차이 검증: 인구 분모 / per-1000 곱셈 없음."""
    daily = np.full(7 * 4, 100.0)   # 4 주 × 7 일 × 100/day
    gamma_report = 0.2
    hira = simulation_to_hira(daily, gamma_report, n_weeks=4)
    # 각 주당 일일 incidence 합 = 700, gamma_report = 0.2 → 주별 = 140 (카운트)
    np.testing.assert_allclose(hira, [140.0, 140.0, 140.0, 140.0])


def test_simulation_to_hira_pads_to_n_weeks() -> None:
    daily = np.full(7 * 3, 10.0)
    out = simulation_to_hira(daily, gamma_report=0.1, n_weeks=10)
    assert out.shape == (10,)
    np.testing.assert_allclose(out[:3], [7.0, 7.0, 7.0])
    np.testing.assert_allclose(out[3:], 0.0)


def test_simulation_to_hira_truncates_to_n_weeks() -> None:
    daily = np.full(7 * 60, 1.0)
    out = simulation_to_hira(daily, gamma_report=0.5, n_weeks=52)
    assert out.shape == (52,)


# ---------- simulation_to_hira_by_age ----------

def test_simulation_to_hira_by_age_shape_and_keys() -> None:
    from kt_data import HIRA_AGE_GROUPS
    daily = np.ones((7 * 5, 15), dtype=float) * 10.0
    out = simulation_to_hira_by_age(daily, gamma_report=0.2, n_weeks=5)
    assert set(out.keys()) == set(HIRA_AGE_GROUPS)
    for ag in HIRA_AGE_GROUPS:
        assert out[ag].shape == (5,)


def test_simulation_to_hira_by_age_preserves_total() -> None:
    """6 그룹 합 ≈ gamma_report × (전체 incidence 주별 합)."""
    rng = np.random.default_rng(0)
    daily = rng.uniform(1.0, 100.0, size=(7 * 8, 15))
    gamma_report = 0.3
    out = simulation_to_hira_by_age(daily, gamma_report=gamma_report, n_weeks=8)
    total_by_group = sum(out[ag].sum() for ag in out)
    expected_total = gamma_report * daily.sum()
    assert total_by_group == pytest.approx(expected_total, rel=1e-9)


def test_simulation_to_hira_by_age_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="must be"):
        simulation_to_hira_by_age(np.zeros((10,)), gamma_report=0.2)
    with pytest.raises(ValueError, match="must be"):
        simulation_to_hira_by_age(np.zeros((10, 7)), gamma_report=0.2)


# ---------- poisson_log_likelihood ----------

def test_poisson_perfect_prediction_minimal_nll() -> None:
    obs = np.array([10.0, 20.0, 30.0, 40.0])
    nll_perfect = poisson_log_likelihood(obs, obs.copy())
    nll_constant = poisson_log_likelihood(obs, np.full_like(obs, obs.mean()))
    assert nll_perfect < nll_constant


def test_poisson_weight_zero_excludes() -> None:
    """weight=0 인 주는 NLL 기여 0 — partial = 명시 weighted 부분합."""
    obs = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([5.0, 25.0, 30.0, 40.0])
    weights_first_2_off = np.array([0.0, 0.0, 1.0, 1.0])
    weights_last_2_off = np.array([1.0, 1.0, 0.0, 0.0])
    nll_all = poisson_log_likelihood(obs, pred)
    nll_part_first = poisson_log_likelihood(obs, pred, weights=weights_last_2_off)
    nll_part_last = poisson_log_likelihood(obs, pred, weights=weights_first_2_off)
    # weighted 합은 두 부분합과 정확히 같아야 함
    assert nll_all == pytest.approx(nll_part_first + nll_part_last, abs=1e-9)


def test_poisson_all_zero_weights_returns_inf() -> None:
    obs = np.array([10.0, 20.0, 30.0])
    pred = np.array([10.0, 20.0, 30.0])
    nll = poisson_log_likelihood(obs, pred, weights=np.zeros(3))
    assert np.isinf(nll)


def test_poisson_min_rate_default_hira_scale() -> None:
    """min_rate default = 1.0 (HIRA count scale, ILI 0.1 보다 큼)."""
    # 0 predicted 라도 min_rate=1.0 으로 floored → finite NLL
    obs = np.array([5.0])
    pred = np.array([0.0])
    nll = poisson_log_likelihood(obs, pred)
    assert np.isfinite(nll)


# ---------- live data: load_hira_target_by_age ----------

@requires_file
def test_load_hira_target_by_age_schema_2019_2020() -> None:
    from kt_data import HIRA_AGE_GROUPS
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    t = load_hira_target_by_age("2019-2020", first_peak_only=True, first_peak_end_week=26)
    assert set(t["age_groups"]) == set(HIRA_AGE_GROUPS)
    assert t["n_weeks"] == N_WEEKS
    for ag in HIRA_AGE_GROUPS:
        assert ag in t["hira_counts"]
        assert t["hira_counts"][ag].shape == (52,)
        assert t["weights"][ag].shape == (52,)
        # first_peak_only → week >= 26 weight = 0
        assert t["weights"][ag][26:].sum() == 0.0
        assert t["weights"][ag][:26].sum() > 0


@requires_file
def test_load_hira_target_by_age_count_sum_matches_known_total() -> None:
    """수도권 2019-2020 시즌 전체 합산이 ~817K 근처 (날짜 경계 차로 정확히 일치 X)."""
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    t = load_hira_target_by_age("2019-2020", first_peak_only=False)
    total = sum(t["hira_counts"][ag].sum() for ag in t["age_groups"])
    # ISO 36 starts 2019-09-02, ends 2020-08-30 → 52 weeks
    # 사용자 verified total 817,914 used date range 2019-09-01 ~ 2020-08-31 (35건 차이 있음)
    assert 800_000 < total < 850_000, (
        f"수도권 시즌 합산 {total:,.0f} — 예상 817K 근처"
    )
