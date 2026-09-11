"""Joint likelihood and NUTS model for the influenza-A refit.

Mirrors ``nuts_seasonpop.make_joint_loss`` and
``nuts_seasonpop_v7_reparam.build_model_v7`` with two differences, both
structural rather than physical:

* the season list is taken from the setup dict instead of a module constant,
  so the fit is not limited to the three seasons of v5;
* the observations come from :mod:`fluA_setup`, i.e. claims rescaled to
  influenza A.

Every fixed quantity - the susceptibility profile ``PHI``, the reporting
fractions ``GAMMA``, the crowding vector ``KAP``, the baseline retention
``BASE``, the presymptomatic weight ``W`` and its NGM factor, and the channel
pin (``LOGIT_REF`` / ``SIGMA_PIN``) - is imported from ``nuts_seasonpop``
rather than restated, so this module cannot drift from the published fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from kt_epimodel_hira.jax_model.erlang_presymp import (
    daily_new_onset_by_age_erlang_presymp,
    simulate_jax_erlang_presymp,
    split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    nb_nll_jax,
    simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex

import nuts_seasonpop as NS


def make_joint_loss_fluA(C):
    """Joint negative log-likelihood over ``C['seasons']``.

    Args:
        C: the dict returned by ``fluA_setup.build_fluA_setup``.

    Returns:
        ``loss(log_R0_vec, logit_pi, phi_nb) -> scalar``, with ``log_R0_vec``
        of length ``len(C['seasons'])``.
    """
    seasons = C["seasons"]
    phi_j = jnp.asarray(NS.PHI)
    gamma_j = jnp.asarray(NS.GAMMA)
    kap_j = jnp.asarray(NS.KAP)
    nw = C["nw"]

    obs_l = [jnp.asarray(o) for o in C["obs"]]
    w_l = [jnp.asarray(w) for w in C["w"]]
    init_states = [split_seed_to_erlang(C["st_by_s"][s]) for s in seasons]
    ngm_by_s = [C["ngm3_by_s"][s] for s in seasons]
    shared_by_s = [C["shared_by_s"][s] for s in seasons]
    n_sea = len(seasons)

    def loss_joint(log_R0_vec, logit_pi, phi_nb):
        pi = jax.nn.softmax(logit_pi)
        total = 0.0
        for i in range(n_sea):
            R0_s = jnp.exp(log_R0_vec[i])
            beta_s = derive_beta_from_R0_simplex(
                ngm_by_s[i], R0_s, pi, phi_j
            ) / NS.NGM_F
            kw = dict(shared_by_s[i])
            kw["beta_h"] = beta_s[0]
            kw["beta_w"] = beta_s[1]
            kw["beta_s"] = beta_s[2]
            kw["beta_o"] = beta_s[3]
            kw["phi_susc"] = phi_j
            kw["kappa"] = kap_j
            kw["p_school"] = NS.BASE
            kw["p_work"] = NS.BASE
            st = simulate_jax_erlang_presymp(
                init_states[i], w_presymp=NS.W, **kw, discretize_time=False
            )
            inc = daily_new_onset_by_age_erlang_presymp(st)
            pred = simulation_to_hira_by_age_jax(inc, gamma_j, n_weeks=nw)
            total = total + nb_nll_jax(
                obs_l[i], pred, w_l[i], concentration=phi_nb, min_rate=0.01
            )
        return total

    return loss_joint


def build_model_fluA(C, reparam: bool = True, with_likelihood: bool = True):
    """NUTS model in the v7 coordinate system, for ``len(C['seasons'])`` seasons.

    ``reparam=True`` samples three free logit components and pins the fourth
    at zero. ``pi = softmax(logit_pi)`` is shift-invariant, so this removes a
    flat direction without changing the inferred object.
    """
    n_sea = len(C["seasons"])
    logit_ref_j = jnp.asarray(NS.LOGIT_REF)
    sigma_pin_j = jnp.asarray(NS.SIGMA_PIN)
    loss_fn = make_joint_loss_fluA(C) if with_likelihood else None

    def model():
        log_R0 = numpyro.sample(
            "log_R0", dist.Normal(jnp.log(2.0) * jnp.ones(n_sea), 0.5)
        )
        if reparam:
            logit_pi_free = numpyro.sample(
                "logit_pi_free", dist.Uniform(jnp.full(3, -10.0), jnp.full(3, 10.0))
            )
            logit_pi = jnp.concatenate([logit_pi_free, jnp.zeros(1)])
            numpyro.deterministic("logit_pi", logit_pi)
        else:
            logit_pi = numpyro.sample(
                "logit_pi", dist.Uniform(jnp.full(4, -10.0), jnp.full(4, 10.0))
            )
        centered = logit_pi - jnp.mean(logit_pi)
        numpyro.factor(
            "pin_prior",
            -0.5 * jnp.sum(((centered - logit_ref_j) / sigma_pin_j) ** 2),
        )
        log_phi_nb = numpyro.sample("log_phi_nb", dist.Normal(jnp.log(10.0), 1.5))
        phi_nb = jnp.exp(log_phi_nb)
        numpyro.deterministic("phi_nb", phi_nb)
        numpyro.deterministic("pi", jax.nn.softmax(logit_pi))
        if loss_fn is not None:
            numpyro.factor("data_loglik", -loss_fn(log_R0, logit_pi, phi_nb))

    return model
