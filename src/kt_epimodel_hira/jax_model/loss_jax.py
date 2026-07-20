"""JAX HIRA conversion + Poisson NLL + multi-season loss (M1c).

Mirrors:
- hira_target.py::simulation_to_hira_by_age
- hira_target.py::poisson_log_likelihood
- loss.py::make_loss_function_by_age (in functional form)

Differences vs numpy:
- HIRA_GROUP_TO_NIMS_WEIGHTED dict -> static (6, 15) matrix (one-time build)
- No masking via boolean indexing (zero weights handled by w * term)
- Assumes no NaN in observations (production targets are clean)

All functions are pure; no closures over Python objects beyond the static
contact/mobility tensors.
"""
from __future__ import annotations
from typing import Sequence
import jax
import jax.numpy as jnp
import numpy as np

from kt_epimodel_hira.calibration.hira_target import (
    HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)

HIRA_AGE_GROUPS = list(HIRA_GROUP_TO_NIMS_WEIGHTED.keys())
N_AGE = 15
N_HIRA = len(HIRA_AGE_GROUPS)


def build_hira_mapping_matrix() -> jnp.ndarray:
    """6×15 weight matrix: H[i, j] = weight of NIMS j in HIRA group i.

    sum_i H[i, j] == 1.0 for each j (verified by existing tests).
    """
    H = np.zeros((N_HIRA, N_AGE), dtype=np.float64)
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for nims_idx, w in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, nims_idx] = w
    # Validate
    col_sums = H.sum(axis=0)
    assert np.allclose(col_sums, 1.0), f"NIMS column sums != 1: {col_sums}"
    return jnp.asarray(H)


HIRA_MAPPING = build_hira_mapping_matrix()  # (6, 15) static


def simulation_to_hira_by_age_jax(
    daily_inc_15: jnp.ndarray,        # (n_days, 15)
    gamma_15: jnp.ndarray,            # (15,)
    n_weeks: int = 52,
    hira_mapping: jnp.ndarray = HIRA_MAPPING,
) -> jnp.ndarray:
    """(n_days, 15) -> (n_weeks, 6) HIRA counts.

    Replicates hira_target.simulation_to_hira_by_age but stacked as array.
    Returns (n_weeks, 6). Order of HIRA groups matches HIRA_AGE_GROUPS.
    """
    n_complete = daily_inc_15.shape[0] // 7
    # weekly aggregation: (n_complete, 7, 15) -> (n_complete, 15)
    weekly = daily_inc_15[: n_complete * 7].reshape(n_complete, 7, N_AGE).sum(axis=1)
    reported = weekly * gamma_15[None, :]   # (n_complete, 15)
    # HIRA mapping: (n_complete, 15) @ (15, 6) -> (n_complete, 6)
    hira = reported @ hira_mapping.T
    # Pad with zeros if needed
    if n_complete < n_weeks:
        pad = jnp.zeros((n_weeks - n_complete, N_HIRA))
        hira = jnp.concatenate([hira, pad], axis=0)
    else:
        hira = hira[:n_weeks]
    return hira


def poisson_nll_jax(
    obs: jnp.ndarray,        # (n_weeks, n_groups)
    pred: jnp.ndarray,       # (n_weeks, n_groups)
    weights: jnp.ndarray,    # (n_weeks, n_groups)
    min_rate: float = 0.01,
) -> jnp.ndarray:
    """sum(w * (pred_clip - obs * log(pred_clip))). weight=0 zeros contribution.

    Numpy version masks NaN; we assume clean obs (production targets are).
    """
    pred_c = jnp.maximum(pred, min_rate)
    nll_pointwise = pred_c - obs * jnp.log(pred_c)
    return jnp.sum(weights * nll_pointwise)


def nb_nll_jax(
    obs: jnp.ndarray,        # (n_weeks, n_groups)
    pred: jnp.ndarray,       # (n_weeks, n_groups)
    weights: jnp.ndarray,    # (n_weeks, n_groups)
    concentration: jnp.ndarray,    # scalar k (NegativeBinomial2 dispersion)
    min_rate: float = 0.01,
) -> jnp.ndarray:
    """Weighted Negative-Binomial-2 NLL.

    Parametrization: mean = pred, var = pred + pred² / k. Larger k → Poisson.
    log P(y | μ, k) = lgamma(y+k) - lgamma(k) - lgamma(y+1)
                    + k log(k/(k+μ)) + y log(μ/(k+μ))
    """
    pred_c = jnp.maximum(pred, min_rate)
    k = concentration
    log_p_fail = jnp.log(k) - jnp.log(k + pred_c)        # log(k/(k+μ))
    log_p_succ = jnp.log(pred_c) - jnp.log(k + pred_c)   # log(μ/(k+μ))
    logpmf = (
        jax.lax.lgamma(obs + k)
        - jax.lax.lgamma(k)
        - jax.lax.lgamma(obs + 1.0)
        + k * log_p_fail
        + obs * log_p_succ
    )
    return -jnp.sum(weights * logpmf)


def single_season_nll_jax(
    initial_state: jnp.ndarray,
    obs_hira: jnp.ndarray,           # (n_weeks, 6)
    weights_hira: jnp.ndarray,       # (n_weeks, 6)
    *,
    # solver args
    sim_kwargs: dict,
    # calibration (already unpacked: beta_4 + phi_14 + gamma_3 in sim_kwargs)
    n_weeks: int = 52,
    min_rate: float = 0.01,
    discretize_time: bool = False,
) -> jnp.ndarray:
    """Forward sim + HIRA conversion + Poisson NLL for one season."""
    states = simulate_jax(
        initial_state, **sim_kwargs,
        discretize_time=discretize_time,
    )
    inc_15 = daily_new_infection_by_age_jax(states)  # (n_days-1, 15)
    # gamma_15 from gamma triple, replicate CalibrationParameters.gamma_15:
    g_child = sim_kwargs.get("_gamma_child")
    g_adult = sim_kwargs.get("_gamma_adult")
    g_elder = sim_kwargs.get("_gamma_elder")
    gamma_15 = jnp.concatenate([
        jnp.full(4, g_child),
        jnp.full(9, g_adult),
        jnp.full(2, g_elder),
    ])
    pred_hira = simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)
    return poisson_nll_jax(obs_hira, pred_hira, weights_hira, min_rate=min_rate)


def gamma_triple_to_15(gamma_child, gamma_adult, gamma_elder) -> jnp.ndarray:
    """Replicate CalibrationParameters.gamma_15 indexing.

    GAMMA_AGE_GROUPS = {child: [0,1,2,3], adult: [4..12], elder: [13,14]}.
    """
    return jnp.concatenate([
        jnp.full(4, gamma_child),
        jnp.full(9, gamma_adult),
        jnp.full(2, gamma_elder),
    ])


def make_multi_season_loss_fn_nb(
    *,
    initial_states: Sequence[jnp.ndarray],
    obs_hira_list: Sequence[jnp.ndarray],
    weights_hira_list: Sequence[jnp.ndarray],
    shared_static: dict,
    n_weeks: int = 52,
    min_rate: float = 0.01,
    discretize_time: bool = False,
):
    """Multi-season NB loss. Closure signature: ``(vec_33, concentration) → NLL``.

    Identical forward sim and HIRA conversion as the Poisson factory; only the
    observation likelihood is swapped for Negative-Binomial-2. The dispersion
    `concentration` is supplied externally so the numpyro model can sample it.
    """
    n_seasons = len(initial_states)

    def loss(vec_33, concentration):
        phi = vec_33[:14]
        gamma_3 = vec_33[14:17]
        phi_full = jnp.concatenate([phi[:5], jnp.array([1.0]), phi[5:]])
        gamma_15 = gamma_triple_to_15(gamma_3[0], gamma_3[1], gamma_3[2])

        total = 0.0
        for i in range(n_seasons):
            beta_4 = vec_33[17 + i*4 : 21 + i*4]
            kw = dict(shared_static)
            kw["beta_h"] = beta_4[0]
            kw["beta_w"] = beta_4[1]
            kw["beta_s"] = beta_4[2]
            kw["beta_o"] = beta_4[3]
            kw["phi_susc"] = phi_full
            states = simulate_jax(initial_states[i], **kw,
                                   discretize_time=discretize_time)
            inc_15 = daily_new_infection_by_age_jax(states)
            pred_hira = simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)
            total = total + nb_nll_jax(
                obs_hira_list[i], pred_hira, weights_hira_list[i],
                concentration=concentration, min_rate=min_rate,
            )
        return total

    return loss


def make_shared_pi_joint_loss_nb(
    *,
    initial_states: Sequence[jnp.ndarray],
    obs_hira_list: Sequence[jnp.ndarray],
    weights_hira_list: Sequence[jnp.ndarray],
    shared_static: dict,
    ngm_eigval_fn,
    phi_full: jnp.ndarray,
    gamma_15: jnp.ndarray,
    n_weeks: int = 52,
    min_rate: float = 0.01,
    discretize_time: bool = False,
):
    """Multi-season NB joint loss with a SHARED channel mix π and per-season R0.

    Physical structure: the channel ratio π = (home, work, school, other) is a
    virus/contact-structure property held common across seasons, while the size
    axis R0_s varies per season (immunity/drift). β_{c,s} = scale_s · π_c with
    scale_s = R0_s / ρ(K(π, φ)) via 1-homogeneous NGM inversion — same channel
    ratios, per-season magnitude.

    φ, γ, κ (inside ``shared_static``) are fixed inputs (not fit here).

    Closure signature: ``(log_R0_vec, logit_pi, phi_nb) -> joint NLL (Σ_s)``
      - log_R0_vec: (n_seasons,) per-season log R0
      - logit_pi:   (4,) shared channel-mix logits (softmax inside)
      - phi_nb:     scalar NB concentration
    """
    n_seasons = len(initial_states)

    def loss(log_R0_vec, logit_pi, phi_nb):
        pi = jax.nn.softmax(logit_pi)                       # shared (4,)
        rho_pi = ngm_eigval_fn(pi[0], pi[1], pi[2], pi[3], phi_full)
        total = 0.0
        for i in range(n_seasons):
            R0_s = jnp.exp(log_R0_vec[i])
            beta_4 = (R0_s / rho_pi) * pi                   # 1-homogeneous inversion
            kw = dict(shared_static)
            kw["beta_h"] = beta_4[0]
            kw["beta_w"] = beta_4[1]
            kw["beta_s"] = beta_4[2]
            kw["beta_o"] = beta_4[3]
            kw["phi_susc"] = phi_full
            states = simulate_jax(initial_states[i], **kw,
                                   discretize_time=discretize_time)
            inc_15 = daily_new_infection_by_age_jax(states)
            pred_hira = simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)
            total = total + nb_nll_jax(
                obs_hira_list[i], pred_hira, weights_hira_list[i],
                concentration=phi_nb, min_rate=min_rate,
            )
        return total

    return loss


def make_multi_season_loss_fn(
    *,
    initial_states: Sequence[jnp.ndarray],         # 4 seasons of (5, 15, 1)
    obs_hira_list: Sequence[jnp.ndarray],          # 4 seasons of (n_weeks, 6)
    weights_hira_list: Sequence[jnp.ndarray],      # 4 seasons of (n_weeks, 6)
    shared_static: dict,                           # contact, mobility, kappa, sigma, gamma, ...
    n_weeks: int = 52,
    min_rate: float = 0.01,
    discretize_time: bool = False,
):
    """Build a closure: (vec_33) -> scalar NLL across 4 seasons.

    vec_33 layout: [phi(14), gamma(3), beta_2017(4), beta_2018(4), beta_2019(4), beta_2022(4)]
    matches multi_season_loss in scripts.
    """
    n_seasons = len(initial_states)

    def loss(vec_33):
        phi = vec_33[:14]
        gamma_3 = vec_33[14:17]
        phi_full = jnp.concatenate([phi[:5], jnp.array([1.0]), phi[5:]])  # insert reference phi_5=1
        gamma_15 = gamma_triple_to_15(gamma_3[0], gamma_3[1], gamma_3[2])

        total = 0.0
        for i in range(n_seasons):
            beta_4 = vec_33[17 + i*4 : 21 + i*4]
            kw = dict(shared_static)
            kw["beta_h"] = beta_4[0]
            kw["beta_w"] = beta_4[1]
            kw["beta_s"] = beta_4[2]
            kw["beta_o"] = beta_4[3]
            kw["phi_susc"] = phi_full
            states = simulate_jax(initial_states[i], **kw,
                                   discretize_time=discretize_time)
            inc_15 = daily_new_infection_by_age_jax(states)
            pred_hira = simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)
            total = total + poisson_nll_jax(
                obs_hira_list[i], pred_hira, weights_hira_list[i],
                min_rate=min_rate,
            )
        return total

    return loss
