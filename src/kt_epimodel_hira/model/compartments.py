"""SEIRV compartments (Step B — J 제거).

상태 텐서: shape (5, 15, n_admdong)

- 0: S (Susceptible)
- 1: V (Vaccinated, 부분 면역 — (1-VE) λ 확률로 감염)
- 2: E (Exposed, latent)
- 3: I (Infectious)
- 4: R (Recovered)

이전 J (격리) compartment 는 채널별 β + p_work + ρ 로 흡수됨.
"""

from __future__ import annotations

import numpy as np

# Compartment 인덱스 (J 제거)
IDX_S: int = 0
IDX_V: int = 1
IDX_E: int = 2
IDX_I: int = 3
IDX_R: int = 4
N_COMPARTMENTS: int = 5
N_AGE: int = 15

_NEG_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initial_state(
    population: np.ndarray,
    seed_per_admdong: float = 1.0,
    seed_age_distribution: str = "population_weighted",
    initial_immunity: float = 0.0,
    initial_vaccinated_fraction: float = 0.0,
) -> np.ndarray:
    """시즌 시작 상태 (SEIRV).

    Args:
        population: (15, n_admdong) 거주 인구.
        seed_per_admdong: 행정동당 초기 I 총 수.
        seed_age_distribution: 'uniform' | 'population_weighted'.
        initial_immunity: 시즌 시작 R 비율 (이전 시즌 면역 잔존). [0, 1).
        initial_vaccinated_fraction: 시즌 시작 V 비율. [0, 1).

    Returns:
        state: (5, 15, n_admdong) float64. Σ_c = population.
    """
    if population.ndim != 2 or population.shape[0] != N_AGE:
        raise ValueError(f"population shape must be (15, n_admdong), got {population.shape}")
    if (population < 0).any():
        raise ValueError("population must be nonnegative")
    if not (0.0 <= initial_immunity < 1.0):
        raise ValueError(f"initial_immunity must be in [0, 1), got {initial_immunity}")
    if not (0.0 <= initial_vaccinated_fraction < 1.0):
        raise ValueError(
            f"initial_vaccinated_fraction must be in [0, 1), got {initial_vaccinated_fraction}"
        )
    if initial_immunity + initial_vaccinated_fraction >= 1.0:
        raise ValueError("initial_immunity + initial_vaccinated_fraction must be < 1")
    if seed_per_admdong < 0:
        raise ValueError(f"seed_per_admdong must be nonneg, got {seed_per_admdong}")

    pop = population.astype(np.float64, copy=True)
    n_age, n_adm = pop.shape

    R0 = initial_immunity * pop
    V0 = initial_vaccinated_fraction * pop

    if seed_age_distribution == "uniform":
        I0 = np.full((n_age, n_adm), seed_per_admdong / n_age, dtype=np.float64)
    elif seed_age_distribution == "population_weighted":
        pop_sum = pop.sum(axis=0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(pop_sum > 0, pop / pop_sum, 0.0)
        I0 = seed_per_admdong * share
    else:
        raise ValueError(
            f"seed_age_distribution must be 'uniform' or 'population_weighted', "
            f"got {seed_age_distribution!r}"
        )

    # I0 를 (pop - R0 - V0) 이내로 클리핑 (음수 S 방지, 보존 유지)
    available_for_I = pop - R0 - V0
    I0 = np.minimum(I0, np.maximum(available_for_I, 0.0))

    state = np.zeros((N_COMPARTMENTS, n_age, n_adm), dtype=np.float64)
    state[IDX_S] = pop - R0 - V0 - I0
    state[IDX_V] = V0
    state[IDX_I] = I0
    state[IDX_R] = R0
    # E = 0
    return state


# ---------------------------------------------------------------------------
# 1D <-> 3D
# ---------------------------------------------------------------------------

def flatten_state(state: np.ndarray) -> np.ndarray:
    """(5, 15, n) → (5·15·n,)."""
    if state.ndim != 3 or state.shape[0] != N_COMPARTMENTS or state.shape[1] != N_AGE:
        raise ValueError(f"state shape {state.shape} invalid")
    return state.reshape(-1)


def unflatten_state(flat: np.ndarray, n_admdong: int, n_ages: int = N_AGE) -> np.ndarray:
    """(5·15·n,) → (5, 15, n)."""
    expected = N_COMPARTMENTS * n_ages * n_admdong
    if flat.shape != (expected,):
        raise ValueError(f"flat shape {flat.shape} != ({expected},)")
    return flat.reshape(N_COMPARTMENTS, n_ages, n_admdong)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_S(state: np.ndarray) -> np.ndarray:
    return state[IDX_S]


def get_V(state: np.ndarray) -> np.ndarray:
    return state[IDX_V]


def get_E(state: np.ndarray) -> np.ndarray:
    return state[IDX_E]


def get_I(state: np.ndarray) -> np.ndarray:
    return state[IDX_I]


def get_R(state: np.ndarray) -> np.ndarray:
    return state[IDX_R]


def total_population(state: np.ndarray) -> np.ndarray:
    """모든 compartment 합 = 인구. (15, n_admdong)."""
    return state.sum(axis=0)


def total_infectious(state: np.ndarray) -> np.ndarray:
    """전염원 (I)."""
    return state[IDX_I]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_state(
    state: np.ndarray,
    n_ages: int = N_AGE,
    n_admdong: int | None = None,
) -> None:
    if state.ndim != 3:
        raise ValueError(f"state must be 3D, got ndim={state.ndim}")
    if state.shape[0] != N_COMPARTMENTS:
        raise ValueError(
            f"state.shape[0] must be {N_COMPARTMENTS}, got {state.shape[0]}"
        )
    if state.shape[1] != n_ages:
        raise ValueError(f"state.shape[1] must be {n_ages}, got {state.shape[1]}")
    if n_admdong is not None and state.shape[2] != n_admdong:
        raise ValueError(f"state.shape[2] {state.shape[2]} != {n_admdong}")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or inf")
    if (state < -_NEG_TOL).any():
        raise ValueError(f"state has negative values below tolerance {-_NEG_TOL}")


def check_conservation(
    state: np.ndarray,
    population: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    return bool(np.allclose(state.sum(axis=0), population, atol=tol, rtol=tol))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def attack_rate(state: np.ndarray, population: np.ndarray) -> np.ndarray:
    """(I + R) / N. V 는 분자 제외. (15, n_admdong)."""
    infected = state[IDX_I] + state[IDX_R]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(population > 0, infected / population, 0.0)


def attack_rate_by_age(state: np.ndarray, population: np.ndarray) -> np.ndarray:
    infected = (state[IDX_I] + state[IDX_R]).sum(axis=1)
    pop_age = population.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pop_age > 0, infected / pop_age, 0.0)


def attack_rate_total(state: np.ndarray, population: np.ndarray) -> float:
    infected = state[IDX_I].sum() + state[IDX_R].sum()
    pop = population.sum()
    return float(infected / pop) if pop > 0 else 0.0


def vaccinated_fraction(state: np.ndarray, population: np.ndarray) -> float:
    pop_total = population.sum()
    return float(state[IDX_V].sum() / pop_total) if pop_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from kt_data.data.load_population import (
        get_population_matrix,
        load_population_15groups,
    )

    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)

    print(f"인구:    {pop.shape}, 총 {pop.sum():,.0f}")

    state = initial_state(
        pop,
        seed_per_admdong=10,
        initial_immunity=0.2,
        initial_vaccinated_fraction=0.0,
    )
    print(f"\n초기 상태 shape: {state.shape}")
    print(f"  S: {state[IDX_S].sum():>14,.0f}")
    print(f"  V: {state[IDX_V].sum():>14,.0f}")
    print(f"  E: {state[IDX_E].sum():>14,.0f}")
    print(f"  I: {state[IDX_I].sum():>14,.0f}")
    print(f"  R: {state[IDX_R].sum():>14,.0f}")

    validate_state(state)
    print(f"\n인구 보존:        {check_conservation(state, pop)}")

    flat = flatten_state(state)
    print(f"Flat shape:        {flat.shape}")
    restored = unflatten_state(flat, n_admdong=pop.shape[1])
    print(f"Round-trip equal:  {np.array_equal(state, restored)}")

    print(f"\nInitial attack rate:     {attack_rate_total(state, pop):.4%}")
    print(f"Initial vaccinated frac: {vaccinated_fraction(state, pop):.4%}")
