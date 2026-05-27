"""Unit tests for kt_epimodel_hira.simulation.runner (Step E)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from kt_epimodel_hira.model.compartments import IDX_I, IDX_R, IDX_V
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters,
    ModelParameters,
    PolicyParameters,
)
from kt_epimodel_hira.simulation.runner import (
    compare_scenarios,
    run_scenarios,
    run_single_season,
)


# ---------- single season ----------

def test_run_single_season_default() -> None:
    data = run_single_season(t_span=(0, 3), verbose=False)
    assert {"result", "params", "admdong_codes", "pop_15", "metadata"}.issubset(data.keys())
    assert data["result"].success


def test_run_single_season_short_states_shape() -> None:
    data = run_single_season(t_span=(0, 3), verbose=False)
    r = data["result"]
    n_adm = data["pop_15"].shape[1]
    assert r.states.shape == (4, 5, 15, n_adm)


def test_run_single_season_employment_auto_built() -> None:
    """params.employment=None 이어도 자동 빌드."""
    data = run_single_season(t_span=(0, 1), verbose=False)
    assert data["params"].employment is not None
    assert data["params"].employment.rho.shape == (data["pop_15"].shape[1], 15)


def test_run_single_season_with_immunity() -> None:
    data = run_single_season(
        t_span=(0, 1), initial_immunity=0.2, verbose=False,
    )
    r = data["result"]
    pop_total = data["pop_15"].sum()
    R0 = r.states[0, IDX_R].sum()
    # 초기 R = 0.2 · (pop - I)
    assert R0 > 0.18 * pop_total


def test_run_single_season_with_vaccinated() -> None:
    data = run_single_season(
        t_span=(0, 1), initial_vaccinated_fraction=0.3, verbose=False,
    )
    r = data["result"]
    pop_total = data["pop_15"].sum()
    V0 = r.states[0, IDX_V].sum()
    assert V0 > 0.28 * pop_total


def test_run_single_season_population_conserved() -> None:
    data = run_single_season(t_span=(0, 5), verbose=False)
    r = data["result"]
    target = data["pop_15"].sum()
    totals = r.states.sum(axis=(1, 2, 3))
    np.testing.assert_allclose(totals, target, rtol=1e-5)


def test_run_single_season_custom_beta() -> None:
    """β 강화 → 더 많은 신규 감염."""
    p_low = ModelParameters(
        calibration=CalibrationParameters(
            beta_h=0.01, beta_w=0.01, beta_s=0.01, beta_o=0.01,
            phi=np.ones(15), gamma_report=0.5,
        ),
    )
    p_high = ModelParameters(
        calibration=CalibrationParameters(
            beta_h=0.5, beta_w=0.5, beta_s=0.5, beta_o=0.5,
            phi=np.ones(15), gamma_report=0.5,
        ),
    )
    d1 = run_single_season(t_span=(0, 7), params=p_low, verbose=False)
    d2 = run_single_season(t_span=(0, 7), params=p_high, verbose=False)
    r1_final = d1["result"].states[-1, IDX_R].sum()
    r2_final = d2["result"].states[-1, IDX_R].sum()
    assert r2_final > r1_final


# ---------- scenarios ----------

def test_run_scenarios_multiple() -> None:
    scenarios = {
        "baseline":       PolicyParameters.baseline(),
        "school_closure": PolicyParameters.school_closure(),
    }
    results = run_scenarios(scenarios, t_span=(0, 3), verbose=False)
    assert set(results.keys()) == {"baseline", "school_closure"}
    for d in results.values():
        assert d["result"].success


def test_run_scenarios_school_closure_reduces_student_infection() -> None:
    """학교 휴교 → 학생 (0-3) 감염 감소."""
    base = ModelParameters(
        calibration=CalibrationParameters(
            beta_h=0.5, beta_w=0.5, beta_s=0.5, beta_o=0.5,
            phi=np.ones(15), gamma_report=0.5,
        ),
    )
    scenarios = {
        "baseline":       PolicyParameters.baseline(),
        "school_closure": PolicyParameters.school_closure(attendance=0.0),
    }
    results = run_scenarios(
        scenarios, base_params=base, t_span=(0, 14), seed_per_admdong=10.0, verbose=False,
    )
    R_base = results["baseline"]["result"].states[-1, IDX_R][0:4].sum()
    R_closed = results["school_closure"]["result"].states[-1, IDX_R][0:4].sum()
    assert R_closed < R_base


# ---------- compare_scenarios ----------

def test_compare_scenarios_returns_dataframe() -> None:
    scenarios = {"baseline": PolicyParameters.baseline()}
    results = run_scenarios(scenarios, t_span=(0, 3), verbose=False)
    df = compare_scenarios(results)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1


def test_compare_scenarios_columns_include_vaccinated() -> None:
    scenarios = {
        "a": PolicyParameters.baseline(),
        "b": PolicyParameters.school_closure(),
    }
    results = run_scenarios(scenarios, t_span=(0, 3), verbose=False)
    df = compare_scenarios(results)
    required = {
        "scenario", "final_attack_rate", "final_R",
        "final_vaccinated", "peak_infectious", "peak_day",
    }
    assert required.issubset(set(df.columns))
    assert df.height == 2
