"""Tests for kt_epimodel_hira.calibration.optimizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kt_epimodel_hira.calibration.optimizer import (
    CalibrationResult,
    _resolve_initial_vec,
    load_result,
    optimize_calibration,
    optimize_calibration_by_age,
    save_result,
)
from kt_epimodel_hira.model.parameters import CalibrationParameters
from kt_data.data import DATA_ROOT
from kt_data.data.load_hira import _HIRA_XLSX_NAME

_XLSX_PATH = DATA_ROOT / "external" / "hira" / _HIRA_XLSX_NAME
requires_file = pytest.mark.skipif(
    not _XLSX_PATH.exists(), reason=f"HIRA xlsx not at {_XLSX_PATH}"
)


# ---------- _resolve_initial_vec (data 무관) ----------

def test_resolve_initial_vec_default_uses_initial_guess() -> None:
    from kt_epimodel_hira.calibration.param_vector import initial_guess

    vec = _resolve_initial_vec(None, None)
    np.testing.assert_array_equal(vec, initial_guess())


def test_resolve_initial_vec_explicit() -> None:
    custom = np.full(23, 0.1)
    custom[18] = 0.3
    vec = _resolve_initial_vec(custom, None)
    np.testing.assert_array_equal(vec, custom)


def test_resolve_initial_vec_warm_start_clips_to_bounds() -> None:
    """warm-start vector 가 bounds 밖이면 clip."""
    from kt_epimodel_hira.calibration.param_vector import (
        get_bounds_vector,
        initial_guess,
    )
    base_cal = CalibrationParameters()
    fake = CalibrationResult(
        season="dummy", method="L-BFGS-B", success=True, nll=0.0, nll_initial=0.0,
        calibration=base_cal,
        seasonality_amp=0.0, seasonality_base=1.0,
        seasonality_sigma=40.0, seasonality_peak_day=110.0,
        seasonality_mode="gaussian",
        vector=initial_guess().copy(),
        n_evaluations=0, elapsed_seconds=0.0, message="",
        seed_total=100.0, initial_immunity=0.0, initial_vaccinated_fraction=0.0,
    )
    # 일부 element 를 bound 밖으로
    bounds = get_bounds_vector()
    fake.vector[0] = bounds[0][0] - 10.0
    fake.vector[18] = bounds[18][1] + 10.0

    resolved = _resolve_initial_vec(None, fake)
    for i, (lo, hi) in enumerate(bounds):
        assert lo <= resolved[i] <= hi


def test_resolve_initial_vec_conflict_raises() -> None:
    from kt_epimodel_hira.calibration.param_vector import initial_guess
    fake = CalibrationResult(
        season="d", method="L-BFGS-B", success=True, nll=0.0, nll_initial=0.0,
        calibration=CalibrationParameters(),
        seasonality_amp=0.0, seasonality_base=1.0,
        seasonality_sigma=40.0, seasonality_peak_day=110.0,
        seasonality_mode="gaussian",
        vector=initial_guess().copy(),
        n_evaluations=0, elapsed_seconds=0.0, message="",
        seed_total=100.0, initial_immunity=0.0, initial_vaccinated_fraction=0.0,
    )
    with pytest.raises(ValueError, match="at most one"):
        _resolve_initial_vec(initial_guess(), fake)


# ---------- save_result / load_result (data 무관) ----------

def test_save_load_roundtrip(tmp_path: Path) -> None:
    """모킹 result → 저장 → 로드 → 모든 필드 일치."""
    from kt_epimodel_hira.calibration.param_vector import initial_guess
    cal = CalibrationParameters(
        beta_h=0.5, beta_w=0.3, beta_s=0.7, beta_o=0.2,
        phi=np.linspace(0.2, 2.0, 15), gamma_report=0.25,
    )
    original = CalibrationResult(
        season="2019-2020_by_age",
        method="L-BFGS-B",
        success=True,
        nll=-123.45,
        nll_initial=999.99,
        calibration=cal,
        seasonality_amp=0.15,
        seasonality_base=0.1,
        seasonality_sigma=20.0,
        seasonality_peak_day=140.0,
        seasonality_mode="gaussian",
        vector=initial_guess().copy(),
        n_evaluations=1234,
        elapsed_seconds=300.0,
        message="CONVERGED",
        seed_total=500.0,
        initial_immunity=0.3,
        initial_vaccinated_fraction=0.0,
        first_peak_only=True,
        first_peak_end_week=26,
        use_data_seed=True,
        seed_by_age=[1.0] * 15,
        gamma_report_assumed=0.2,
    )
    out_path = tmp_path / "fake_result.json"
    save_result(original, out_path)
    assert out_path.exists()

    loaded = load_result(out_path)
    assert loaded.season == original.season
    assert loaded.method == original.method
    assert loaded.nll == original.nll
    assert loaded.gamma_report_assumed == 0.2
    assert loaded.use_data_seed is True
    assert loaded.first_peak_only is True
    np.testing.assert_array_equal(loaded.vector, original.vector)
    np.testing.assert_array_equal(loaded.calibration.phi, cal.phi)
    assert loaded.calibration.gamma_report == 0.25


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    from kt_epimodel_hira.calibration.param_vector import initial_guess
    nested = tmp_path / "a" / "b" / "c" / "r.json"
    original = CalibrationResult(
        season="d", method="L-BFGS-B", success=True, nll=0.0, nll_initial=0.0,
        calibration=CalibrationParameters(),
        seasonality_amp=0.0, seasonality_base=1.0,
        seasonality_sigma=40.0, seasonality_peak_day=110.0,
        seasonality_mode="gaussian",
        vector=initial_guess().copy(),
        n_evaluations=0, elapsed_seconds=0.0, message="",
        seed_total=100.0, initial_immunity=0.0, initial_vaccinated_fraction=0.0,
    )
    save_result(original, nested)
    assert nested.exists()


# ---------- live runs (smoke) ----------

@requires_file
def test_optimize_by_age_runs_smoke() -> None:
    """HIRA 짧은 fit (max_iter=2) — finite NLL + 정상 dataclass 반환."""
    r = optimize_calibration_by_age(
        season="2019-2020", method="Nelder-Mead",
        max_iterations=2, verbose=False,
    )
    assert isinstance(r, CalibrationResult)
    assert r.season.endswith("_by_age")
    assert np.isfinite(r.nll_initial)
    assert np.isfinite(r.nll)
    assert r.gamma_report_assumed == 0.2  # default
    assert r.use_data_seed is True
    assert r.first_peak_only is True


@requires_file
def test_optimize_by_age_method_validates() -> None:
    with pytest.raises(ValueError, match="method must"):
        optimize_calibration_by_age(
            season="2019-2020", method="GradientDescent",
            max_iterations=1, verbose=False,
        )


@requires_file
def test_optimize_single_runs_smoke() -> None:
    r = optimize_calibration(
        season="2019-2020", method="Nelder-Mead",
        max_iterations=2, verbose=False,
    )
    assert isinstance(r, CalibrationResult)
    assert np.isfinite(r.nll)
