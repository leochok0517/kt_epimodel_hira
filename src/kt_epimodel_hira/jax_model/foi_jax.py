"""JAX rewrite of FOI (M1a).

1:1 numerical translation of kt_epimodel_hira/model/foi.py using jax.numpy.
Pure functions, all params passed explicitly (no dataclasses inside trace).
For regression test against the numpy version.

Signature differences vs numpy version:
- jnp arrays only
- ModelParameters split into explicit args
- Contact matrices (C_*) and mobility tensors (M_*) passed pre-converted

Cosine seasonality (amp, peak, period, base) is hardcoded — Gaussian is not
needed for current calibration (HIRA uses cosine fixed).
"""
from __future__ import annotations
import jax.numpy as jnp

N_AGE = 15
STUDENT_SLICE = slice(0, 4)
WORKER_SLICE = slice(4, 14)
_EPS = 1e-10


def seasonal_factor_cosine(
    t: float,
    amp: float = 0.7,
    base: float = 0.0,
    peak_day: float = 105.0,
    period: float = 365.0,
) -> float:
    """factor(t) = max(base + amp*cos(2π(t-peak)/period), 0)."""
    g = jnp.cos(2.0 * jnp.pi * (t - peak_day) / period)
    raw = base + amp * g
    return jnp.maximum(raw, 0.0)


def compute_phi_school(p_school: float) -> jnp.ndarray:
    phi = jnp.zeros(N_AGE, dtype=jnp.float64)
    return phi.at[STUDENT_SLICE].set(1.0 - p_school)


def compute_phi_work(p_work: float, rho: jnp.ndarray) -> jnp.ndarray:
    """rho shape (n_admdong, 15). Returns same shape; workers (4:14) only."""
    phi = jnp.zeros_like(rho)
    return phi.at[:, WORKER_SLICE].set(rho[:, WORKER_SLICE] * (1.0 - p_work))


def compute_phi_spillover(
    p_school: float, p_work: float, rho: jnp.ndarray,
) -> jnp.ndarray:
    """(n_admdong, 15)."""
    phi = jnp.zeros_like(rho)
    phi = phi.at[:, STUDENT_SLICE].set(1.0 - p_school)
    phi = phi.at[:, WORKER_SLICE].set(rho[:, WORKER_SLICE] * (1.0 - p_work))
    return phi


def compute_foi_home_jax(
    state: jnp.ndarray,        # (5, 15, n_admdong)
    C_home: jnp.ndarray,
    pop_15: jnp.ndarray,
    rho: jnp.ndarray,
    kappa: jnp.ndarray,
    p_school: float,
    p_work: float,
    beta_h: float,
    phi_susc: jnp.ndarray,
    seasonal_factor: float = 1.0,
) -> jnp.ndarray:
    I = state[3]  # IDX_I
    N = pop_15
    N_safe = jnp.maximum(N, _EPS)
    phi_spill = compute_phi_spillover(p_school, p_work, rho)
    spill_factor = 1.0 + kappa[None, :] * phi_spill            # (n, 15)
    I_eff = I * spill_factor.T                                 # (15, n)
    contact_pressure = C_home @ (I_eff / N_safe)               # (15, n)
    foi_h = (beta_h * seasonal_factor) * phi_susc[:, None] * contact_pressure
    return jnp.where(N > _EPS, foi_h, 0.0)


def compute_foi_school_jax(
    state: jnp.ndarray,
    C_school: jnp.ndarray,
    pop_15: jnp.ndarray,
    p_school: float,
    beta_s: float,
    phi_susc: jnp.ndarray,
    seasonal_factor: float = 1.0,
) -> jnp.ndarray:
    I = state[3]
    N = pop_15
    N_safe = jnp.maximum(N, _EPS)
    n_adm = N.shape[1]

    I_eff = jnp.zeros_like(I)
    I_eff = I_eff.at[STUDENT_SLICE].set(p_school * I[STUDENT_SLICE])

    contact_pressure = C_school @ (I_eff / N_safe)             # (15, n)

    foi_s_all = (
        (beta_s * seasonal_factor)
        * phi_susc[:, None] * contact_pressure
    )
    foi_s = jnp.zeros((N_AGE, n_adm), dtype=jnp.float64)
    foi_s = foi_s.at[STUDENT_SLICE].set(foi_s_all[STUDENT_SLICE])
    return jnp.where(N > _EPS, foi_s, 0.0)


def compute_foi_work_jax(
    state: jnp.ndarray,
    C_work: jnp.ndarray,
    pop_15: jnp.ndarray,
    M_work: jnp.ndarray,
    rho: jnp.ndarray,
    p_work: float,
    beta_w: float,
    phi_susc: jnp.ndarray,
    seasonal_factor: float = 1.0,
) -> jnp.ndarray:
    I = state[3]
    N = pop_15
    n_adm = N.shape[1]

    rho_T = rho.T                                              # (15, n)
    weighted_I = rho_T * I
    weighted_N = rho_T * N

    I_at_j = jnp.einsum("akj,ak->aj", M_work, weighted_I)
    N_at_j = jnp.einsum("akj,ak->aj", M_work, weighted_N)
    N_at_j_safe = jnp.maximum(N_at_j, _EPS)

    ratio_at_j = p_work * I_at_j / N_at_j_safe
    contact_pressure_at_j = C_work @ ratio_at_j

    pressure_at_i = jnp.einsum("aij,aj->ai", M_work, contact_pressure_at_j)
    foi_w_all = (beta_w * seasonal_factor) * phi_susc[:, None] * rho_T * pressure_at_i

    foi_w = jnp.zeros((N_AGE, n_adm), dtype=jnp.float64)
    foi_w = foi_w.at[WORKER_SLICE].set(foi_w_all[WORKER_SLICE])
    return jnp.where(N > _EPS, foi_w, 0.0)


def compute_foi_other_jax(
    state: jnp.ndarray,
    C_other: jnp.ndarray,
    pop_15: jnp.ndarray,
    M_other: jnp.ndarray,
    beta_o: float,
    phi_susc: jnp.ndarray,
    seasonal_factor: float = 1.0,
) -> jnp.ndarray:
    I = state[3]
    N = pop_15

    I_at_j = jnp.einsum("akj,ak->aj", M_other, I)
    N_at_j = jnp.einsum("akj,ak->aj", M_other, N)
    N_at_j_safe = jnp.maximum(N_at_j, _EPS)

    ratio_at_j = I_at_j / N_at_j_safe
    contact_pressure_at_j = C_other @ ratio_at_j
    pressure_at_i = jnp.einsum("aij,aj->ai", M_other, contact_pressure_at_j)

    foi_o = (beta_o * seasonal_factor) * phi_susc[:, None] * pressure_at_i
    return jnp.where(N > _EPS, foi_o, 0.0)


def compute_foi_jax(
    state: jnp.ndarray,
    *,
    C_home: jnp.ndarray, C_school: jnp.ndarray, C_work: jnp.ndarray, C_other: jnp.ndarray,
    M_home: jnp.ndarray, M_school: jnp.ndarray, M_work: jnp.ndarray, M_other: jnp.ndarray,
    pop_15: jnp.ndarray,
    rho: jnp.ndarray,
    kappa: jnp.ndarray,
    p_school: float, p_work: float,
    beta_h: float, beta_w: float, beta_s: float, beta_o: float,
    phi_susc: jnp.ndarray,
    seasonal_factor: float = 1.0,
) -> jnp.ndarray:
    """4-channel FOI sum. Same return shape (15, n_admdong) as numpy."""
    foi_h = compute_foi_home_jax(
        state, C_home, pop_15, rho, kappa, p_school, p_work,
        beta_h, phi_susc, seasonal_factor,
    )
    foi_s = compute_foi_school_jax(
        state, C_school, pop_15, p_school, beta_s, phi_susc, seasonal_factor,
    )
    foi_w = compute_foi_work_jax(
        state, C_work, pop_15, M_work, rho, p_work, beta_w, phi_susc, seasonal_factor,
    )
    foi_o = compute_foi_other_jax(
        state, C_other, pop_15, M_other, beta_o, phi_susc, seasonal_factor,
    )
    return foi_h + foi_s + foi_w + foi_o
