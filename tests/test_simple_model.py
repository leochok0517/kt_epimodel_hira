"""Tests for kt_epimodel_hira.calibration.simple_model."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs,
    estimate_initial_infected_from_hira,
    simulate_aggregated,
)
from kt_epimodel_hira.model.parameters import ModelParameters
from kt_data.data import DATA_ROOT
from kt_data.data.load_hira import _HIRA_XLSX_NAME

_XLSX_PATH = DATA_ROOT / "external" / "hira" / _HIRA_XLSX_NAME
requires_file = pytest.mark.skipif(
    not _XLSX_PATH.exists(), reason=f"HIRA xlsx not at {_XLSX_PATH}"
)


# ---------- build_aggregated_inputs (ILI 무관, 동일 검증) ----------

def test_build_inputs_shapes() -> None:
    inp = build_aggregated_inputs()
    assert inp["pop_15"].shape == (15, 1)
    assert inp["rho"].shape == (1, 15)
    assert set(inp["mobility"].keys()) == {"home", "school", "work", "other"}
    for ch in ("home", "school", "work", "other"):
        assert inp["mobility"][ch].shape == (15, 1, 1)


def test_build_inputs_mobility_all_identity() -> None:
    inp = build_aggregated_inputs()
    for ch in ("home", "school", "work", "other"):
        assert (inp["mobility"][ch] == 1.0).all()


def test_build_inputs_sudogwon_population_realistic() -> None:
    inp = build_aggregated_inputs()
    total = float(inp["pop_15"].sum())
    # 수도권 약 2.6천만
    assert 20_000_000 < total < 30_000_000


# ---------- estimate_initial_infected_from_hira ----------

@requires_file
def test_estimate_initial_seed_shape_and_nonneg() -> None:
    inp = build_aggregated_inputs()
    seed = estimate_initial_infected_from_hira(
        "2019-2020", inp["pop_15"].flatten(), gamma_report_assumed=0.2,
    )
    assert seed.shape == (15,)
    assert (seed >= 0).all()


@requires_file
def test_estimate_initial_seed_scales_inversely_with_gamma_assumed() -> None:
    """γ_assumed 2배 → seed 절반."""
    inp = build_aggregated_inputs()
    seed_02 = estimate_initial_infected_from_hira(
        "2019-2020", inp["pop_15"].flatten(), gamma_report_assumed=0.2,
    )
    seed_04 = estimate_initial_infected_from_hira(
        "2019-2020", inp["pop_15"].flatten(), gamma_report_assumed=0.4,
    )
    # 단순 합 비교
    np.testing.assert_allclose(seed_04.sum() * 2.0, seed_02.sum(), rtol=1e-9)


@requires_file
def test_estimate_initial_seed_rejects_nonpositive_gamma() -> None:
    inp = build_aggregated_inputs()
    with pytest.raises(ValueError, match="must be > 0"):
        estimate_initial_infected_from_hira(
            "2019-2020", inp["pop_15"].flatten(), gamma_report_assumed=0.0,
        )


@requires_file
def test_estimate_initial_seed_total_reasonable() -> None:
    """default γ=0.2, 시즌 시작 시 수도권 baseline → seed total 수백 ~ 수만 수준."""
    inp = build_aggregated_inputs()
    seed = estimate_initial_infected_from_hira(
        "2019-2020", inp["pop_15"].flatten(), gamma_report_assumed=0.2,
    )
    total = float(seed.sum())
    # 시즌 시작 baseline 매우 낮은 시점 (week 0-2) 평균 → 작은 값 기대
    assert 0 <= total < 1_000_000, f"seed total {total:,.0f} 비현실적"


# ---------- simulate_aggregated (ILI 무관, smoke) ----------

def test_simulate_aggregated_smoke() -> None:
    inp = build_aggregated_inputs()
    params = ModelParameters()
    r = simulate_aggregated(params, inp, seed_total=100, t_span=(0, 7))
    assert r.success
    # 인구 보존
    totals = r.total_by_compartment()
    pop_total = totals.sum(axis=1)
    assert pop_total.max() - pop_total.min() < 1e-3


def test_simulate_aggregated_with_seed_by_age() -> None:
    inp = build_aggregated_inputs()
    params = ModelParameters()
    seed = np.zeros(15)
    seed[5] = 50.0  # 25-29세에 50 seed
    r = simulate_aggregated(
        params, inp, seed_by_age=seed, seed_e_factor=0.5,
        initial_immunity=0.0, t_span=(0, 5),
    )
    assert r.success
