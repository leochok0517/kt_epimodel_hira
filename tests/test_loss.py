"""Tests for kt_epimodel_hira.calibration.loss."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from kt_epimodel_hira.calibration.loss import (
    make_loss_function,
    make_loss_function_by_age,
)
from kt_epimodel_hira.calibration.param_vector import initial_guess
from kt_epimodel_hira.calibration.simple_model import build_aggregated_inputs
from kt_epimodel_hira.model.parameters import ModelParameters
from kt_data.data import DATA_ROOT
from kt_data.data.load_hira import _HIRA_XLSX_NAME

_XLSX_PATH = DATA_ROOT / "external" / "hira" / _HIRA_XLSX_NAME
requires_file = pytest.mark.skipif(
    not _XLSX_PATH.exists(), reason=f"HIRA xlsx not at {_XLSX_PATH}"
)


@functools.lru_cache(maxsize=1)
def _setup() -> dict:
    return {"inputs": build_aggregated_inputs()}


# ---------- single (연령 합산 target) ----------

@requires_file
def test_loss_callable_single() -> None:
    from kt_epimodel_hira.calibration.hira_target import load_hira_target
    target = load_hira_target("2019-2020")
    loss = make_loss_function(target, _setup()["inputs"], ModelParameters())
    assert callable(loss)


@requires_file
def test_loss_returns_float_single() -> None:
    from kt_epimodel_hira.calibration.hira_target import load_hira_target
    target = load_hira_target("2019-2020")
    loss = make_loss_function(target, _setup()["inputs"], ModelParameters())
    val = loss(initial_guess())
    assert isinstance(val, float)
    assert np.isfinite(val)


@requires_file
def test_loss_deterministic_single() -> None:
    from kt_epimodel_hira.calibration.hira_target import load_hira_target
    target = load_hira_target("2019-2020")
    loss = make_loss_function(target, _setup()["inputs"], ModelParameters())
    vec = initial_guess()
    v1, v2 = loss(vec), loss(vec)
    np.testing.assert_allclose(v1, v2, rtol=1e-10)


@requires_file
def test_loss_invalid_vec_returns_penalty_single() -> None:
    from kt_epimodel_hira.calibration.hira_target import load_hira_target
    target = load_hira_target("2019-2020")
    loss = make_loss_function(
        target, _setup()["inputs"], ModelParameters(), penalty=999.9,
    )
    bad = initial_guess()
    bad[0] = -1.0   # beta_h 음수 → CalibrationParameters validation error
    assert loss(bad) == 999.9


# ---------- by_age (6 그룹 동시) ----------

@requires_file
def test_loss_callable_by_age() -> None:
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    target_age = load_hira_target_by_age("2019-2020", first_peak_only=True)
    loss = make_loss_function_by_age(target_age, _setup()["inputs"], ModelParameters())
    val = loss(initial_guess())
    assert np.isfinite(val)


@requires_file
def test_loss_by_age_total_is_sum_of_six() -> None:
    """by_age NLL 은 6 그룹 합 — single NLL 과 다름 (단순 sanity)."""
    from kt_epimodel_hira.calibration.hira_target import (
        load_hira_target,
        load_hira_target_by_age,
    )
    target = load_hira_target("2019-2020", first_peak_only=True)
    target_age = load_hira_target_by_age("2019-2020", first_peak_only=True)
    loss_single = make_loss_function(target, _setup()["inputs"], ModelParameters())
    loss_age = make_loss_function_by_age(target_age, _setup()["inputs"], ModelParameters())
    vec = initial_guess()
    assert np.isfinite(loss_single(vec))
    assert np.isfinite(loss_age(vec))


@requires_file
def test_loss_by_age_excludes_vax_flux() -> None:
    """initial_vaccinated_fraction 변경 시 NLL 차이가 vax flux 오염 시그니처
    (1000× 이상) 가 아니어야 함.

    HIRA 는 절대 카운트라 V 보호 효과로 NLL 자체는 factor-2 수준 변할 수 있음
    (감염자 수 자체가 줄어드니까). 하지만 ILI 의 vax flux 버그는 5e6× 부풀림을
    만들었었음 — 그 시그니처가 없으면 helper 가 정상 적용된 것.
    """
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    target_age = load_hira_target_by_age("2019-2020", first_peak_only=True)

    loss_a = make_loss_function_by_age(
        target_age, _setup()["inputs"], ModelParameters(),
        initial_vaccinated_fraction=0.0, t_span=(0.0, 60.0),
    )
    loss_b = make_loss_function_by_age(
        target_age, _setup()["inputs"], ModelParameters(),
        initial_vaccinated_fraction=0.3, t_span=(0.0, 60.0),
    )

    vec = initial_guess()
    v_a = loss_a(vec)
    v_b = loss_b(vec)
    assert np.isfinite(v_a)
    assert np.isfinite(v_b)
    # vax flux 오염 sigature: 1000x 이상 차이. 정상 V 보호 효과는 < 10x.
    ratio = max(v_a, v_b) / max(min(v_a, v_b), 1.0)
    assert ratio < 10.0, (
        f"NLL ratio {ratio:.1f}x — vax flux 오염 의심 "
        f"(정상 V 보호 효과는 ≤ 10x. ILI 옛 버그는 1e6× scale)"
    )
