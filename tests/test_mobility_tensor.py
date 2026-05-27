"""Unit tests for kt_epimodel_hira.model.mobility_tensor (Step C 갱신)."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.model.mobility_tensor import (
    KT_TO_NIMS_OTHER,
    KT_TO_NIMS_WORK,
    N_AGE,
    STATIC_OTHER,
    STATIC_WORK,
    build_M_from_kt_array,
    build_M_home,
    build_M_identity,
    build_M_other,
    build_M_school,
    build_pi_from_kt_array,
)

N_ADM = 5


def _toy_kt(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pi_kt = rng.uniform(1.0, 50.0, size=(N_ADM, N_ADM, 7, 24))
    pop = rng.integers(100, 1000, size=(N_AGE, N_ADM)).astype(np.float64)
    return pi_kt, pop


# ---------- identity helpers ----------

def test_M_identity_shape() -> None:
    M = build_M_identity(N_ADM)
    assert M.shape == (N_AGE, N_ADM, N_ADM)


def test_M_identity_per_age() -> None:
    M = build_M_identity(N_ADM)
    eye = np.eye(N_ADM)
    for a in range(N_AGE):
        np.testing.assert_array_equal(M[a], eye)


def test_M_home_is_identity() -> None:
    M = build_M_home(N_ADM)
    np.testing.assert_array_equal(M, build_M_identity(N_ADM))


def test_M_school_is_identity() -> None:
    M = build_M_school(N_ADM)
    np.testing.assert_array_equal(M, build_M_identity(N_ADM))


# ---------- M_work ----------

def test_M_work_shape() -> None:
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "work")
    assert M.shape == (N_AGE, N_ADM, N_ADM)


def test_M_work_row_sum_one() -> None:
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "work")
    row_sums = M.sum(axis=2)                       # (15, n)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_M_work_static_ages_identity() -> None:
    """0-3 (학생), 14 (70+) 는 identity."""
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "work")
    eye = np.eye(N_ADM)
    for a in STATIC_WORK:
        np.testing.assert_array_equal(M[a], eye)


def test_M_work_workers_mobile() -> None:
    """4-13 (근로자) 는 KT 패턴 (대각 < 1, off-diagonal 존재)."""
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "work")
    for a in range(4, 14):
        diag = np.diag(M[a])
        assert (diag < 1.0).all()
        off = M[a] - np.diag(diag)
        assert off.sum() > 0


def test_M_work_kt_pair_identical() -> None:
    """KT 한 그룹 → NIMS 두 연령 동일 패턴."""
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "work")
    for k, (a1, a2) in KT_TO_NIMS_WORK.items():
        np.testing.assert_array_equal(M[a1], M[a2])


# ---------- M_other ----------

def test_M_other_shape() -> None:
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "other")
    assert M.shape == (N_AGE, N_ADM, N_ADM)


def test_M_other_row_sum_one() -> None:
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "other")
    np.testing.assert_allclose(M.sum(axis=2), 1.0, atol=1e-12)


def test_M_other_static_only_kids_and_70() -> None:
    """0-9, 70+ 만 identity (10-19 는 학원 mobile)."""
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "other")
    eye = np.eye(N_ADM)
    for a in STATIC_OTHER:
        np.testing.assert_array_equal(M[a], eye)
    # 10-19 는 mobile
    for a in (2, 3):
        assert not np.array_equal(M[a], eye)


def test_M_other_teens_use_kt_age_10_19() -> None:
    """KT idx 0 (10-19) → NIMS [2, 3]."""
    pi_kt, _ = _toy_kt()
    M = build_M_from_kt_array(pi_kt, "other")
    np.testing.assert_array_equal(M[2], M[3])


def test_M_invalid_channel_raises() -> None:
    pi_kt, _ = _toy_kt()
    with pytest.raises(ValueError):
        build_M_from_kt_array(pi_kt, "lunch")


# ---------- legacy ----------

def test_pi_legacy_shape() -> None:
    pi_kt, pop = _toy_kt()
    pi = build_pi_from_kt_array(pi_kt, pop)
    assert pi.shape == (N_AGE, N_ADM, N_ADM)


def test_pi_legacy_row_sum() -> None:
    pi_kt, pop = _toy_kt()
    pi = build_pi_from_kt_array(pi_kt, pop)
    np.testing.assert_allclose(pi.sum(axis=2), 1.0, atol=1e-12)
