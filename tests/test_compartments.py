"""Unit tests for kt_epimodel_hira.model.compartments (Step B — J 제거)."""

from __future__ import annotations

import numpy as np
import pytest

from kt_epimodel_hira.model import compartments as comp_mod
from kt_epimodel_hira.model.compartments import (
    IDX_E,
    IDX_I,
    IDX_R,
    IDX_S,
    IDX_V,
    N_AGE,
    N_COMPARTMENTS,
    attack_rate,
    attack_rate_by_age,
    attack_rate_total,
    check_conservation,
    flatten_state,
    get_E,
    get_I,
    get_R,
    get_S,
    get_V,
    initial_state,
    total_infectious,
    total_population,
    unflatten_state,
    vaccinated_fraction,
    validate_state,
)

N_ADM = 7


def _toy_pop(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(100, 1000, size=(N_AGE, N_ADM)).astype(np.float64)


# ---------- constants ----------

def test_idx_constants() -> None:
    assert IDX_S == 0
    assert IDX_V == 1
    assert IDX_E == 2
    assert IDX_I == 3
    assert IDX_R == 4


def test_n_compartments_five() -> None:
    assert N_COMPARTMENTS == 5


def test_no_idx_j_exported() -> None:
    """IDX_J 는 더 이상 export 되지 않음."""
    assert not hasattr(comp_mod, "IDX_J")
    assert not hasattr(comp_mod, "get_J")
    assert not hasattr(comp_mod, "total_infectious_active")


# ---------- initial_state ----------

def test_initial_state_shape() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    assert s.shape == (5, N_AGE, N_ADM)


def test_initial_total_equals_population() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=10)
    np.testing.assert_allclose(total_population(s), pop, atol=1e-9)


def test_initial_S_majority() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=10)
    # 시드 ~70 / 인구 ~30000 → S 가 압도적
    assert s[IDX_S].sum() > 0.99 * pop.sum()


def test_initial_E_zero() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    assert (s[IDX_E] == 0).all()


def test_initial_V_default_zero() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    assert (s[IDX_V] == 0).all()


def test_initial_R_default_zero() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    assert (s[IDX_R] == 0).all()


def test_initial_uniform_distribution() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=15, seed_age_distribution="uniform")
    # 각 admdong 별 I 합 = 15
    np.testing.assert_allclose(s[IDX_I].sum(axis=0), 15.0, rtol=1e-12)
    # 균등 분포 → 모든 연령 동일
    np.testing.assert_allclose(s[IDX_I], 1.0, rtol=1e-12)


def test_initial_population_weighted_distribution() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=10, seed_age_distribution="population_weighted")
    # admdong 별 I 합 = 10
    np.testing.assert_allclose(s[IDX_I].sum(axis=0), 10.0, atol=1e-9)


def test_initial_immunity_creates_R() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=0.0, initial_immunity=0.2)
    np.testing.assert_allclose(s[IDX_R], 0.2 * pop, rtol=1e-12)
    # 보존
    np.testing.assert_allclose(total_population(s), pop, atol=1e-9)


def test_initial_vaccinated_creates_V() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=0.0, initial_vaccinated_fraction=0.3)
    np.testing.assert_allclose(s[IDX_V], 0.3 * pop, rtol=1e-12)
    np.testing.assert_allclose(total_population(s), pop, atol=1e-9)


def test_initial_unknown_distribution_raises() -> None:
    with pytest.raises(ValueError):
        initial_state(_toy_pop(), seed_age_distribution="random_nonsense")


def test_initial_invalid_immunity_raises() -> None:
    with pytest.raises(ValueError):
        initial_state(_toy_pop(), initial_immunity=1.5)
    with pytest.raises(ValueError):
        initial_state(_toy_pop(), initial_immunity=-0.1)


def test_initial_immunity_plus_vax_lt_1() -> None:
    with pytest.raises(ValueError):
        initial_state(
            _toy_pop(), initial_immunity=0.7, initial_vaccinated_fraction=0.5,
        )


def test_initial_zero_pop_admdong_safe() -> None:
    pop = _toy_pop()
    pop[:, 3] = 0
    s = initial_state(pop, seed_per_admdong=10)
    assert np.isfinite(s).all()
    assert (s[IDX_I][:, 3] == 0).all()


# ---------- accessors ----------

def test_accessor_shapes() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10, initial_vaccinated_fraction=0.1)
    for getter in (get_S, get_V, get_E, get_I, get_R):
        assert getter(s).shape == (N_AGE, N_ADM)


def test_total_population_shape() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    assert total_population(s).shape == (N_AGE, N_ADM)


def test_total_infectious_returns_I() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    np.testing.assert_array_equal(total_infectious(s), s[IDX_I])


# ---------- flatten / unflatten ----------

def test_flatten_shape() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    flat = flatten_state(s)
    assert flat.shape == (5 * N_AGE * N_ADM,)


def test_flatten_unflatten_roundtrip() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    flat = flatten_state(s)
    restored = unflatten_state(flat, n_admdong=N_ADM)
    np.testing.assert_array_equal(s, restored)


def test_unflatten_wrong_size_raises() -> None:
    with pytest.raises(ValueError):
        unflatten_state(np.zeros(100), n_admdong=N_ADM)


def test_flatten_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        flatten_state(np.zeros((6, N_AGE, N_ADM)))  # 6 compartments → 잘못


# ---------- validation ----------

def test_validate_wrong_shape() -> None:
    with pytest.raises(ValueError):
        validate_state(np.zeros((6, N_AGE, N_ADM)))   # 이전 6 compartment
    with pytest.raises(ValueError):
        validate_state(np.zeros((4, N_AGE, N_ADM)))


def test_validate_nan() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    s[IDX_S][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_state(s)


def test_validate_negative_above_tol() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    s[IDX_S][0, 0] = -1.0
    with pytest.raises(ValueError):
        validate_state(s)


def test_validate_small_negative_allowed() -> None:
    s = initial_state(_toy_pop(), seed_per_admdong=10)
    s[IDX_S][0, 0] = -1e-12
    validate_state(s)   # 통과


def test_check_conservation_pass() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=10)
    assert check_conservation(s, pop)


def test_check_conservation_fail() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=10)
    s[IDX_S][0, 0] += 5000.0   # 인위 위반
    assert not check_conservation(s, pop)


# ---------- stats ----------

def test_attack_rate_initial_small() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=10)
    ar = attack_rate_total(s, pop)
    # 시드 70/admdong 7 = 70명 / 큰 인구 → 작은 값
    assert 0 < ar < 0.01


def test_attack_rate_full_R() -> None:
    pop = _toy_pop()
    s = np.zeros((5, N_AGE, N_ADM), dtype=np.float64)
    s[IDX_R] = pop
    assert attack_rate_total(s, pop) == pytest.approx(1.0)
    np.testing.assert_allclose(attack_rate(s, pop), 1.0)
    np.testing.assert_allclose(attack_rate_by_age(s, pop), 1.0)


def test_attack_rate_excludes_V() -> None:
    """V 는 attack rate 분자에서 제외."""
    pop = _toy_pop()
    s = np.zeros((5, N_AGE, N_ADM), dtype=np.float64)
    s[IDX_V] = pop      # 전부 V
    # I=R=0 이므로 attack rate=0
    assert attack_rate_total(s, pop) == 0.0


def test_attack_rate_zero_pop_safe() -> None:
    pop = _toy_pop()
    pop[:, 0] = 0
    s = initial_state(pop, seed_per_admdong=0)
    ar = attack_rate(s, pop)
    assert np.isfinite(ar).all()


def test_vaccinated_fraction() -> None:
    pop = _toy_pop()
    s = initial_state(pop, seed_per_admdong=0, initial_vaccinated_fraction=0.25)
    assert vaccinated_fraction(s, pop) == pytest.approx(0.25)


def test_vaccinated_fraction_zero_pop() -> None:
    pop = np.zeros((N_AGE, N_ADM))
    s = np.zeros((5, N_AGE, N_ADM))
    assert vaccinated_fraction(s, pop) == 0.0
