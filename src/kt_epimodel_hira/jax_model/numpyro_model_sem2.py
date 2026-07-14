"""SEM-2 (seminar feedback) — 4→2 channel collapse numpyro model.

Hypothesis tested: HIRA data does not support the 4-channel decomposition
(home/work/other are non-identifiable). Collapsing home + work + other into a
single "home_total" channel while keeping school separate should fit as well
as the production 4-channel model. Winter break ramp is preserved through the
school channel (β_s still scaled by school_mult in dynamics_jax).

Construction:
- π is a 2-simplex (π_h_total, π_school); β_w = β_o = 0.
- C_home_eff = C_h + C_w + C_o is passed in via shared_static (script level).
- C_school is unchanged.
- NGM closure receives the collapsed matrices so β is derived consistently.

Existing functions in numpyro_model.py are not modified; this file only adds a
new model variant per the SEM-2 constraint (no edits to shared core).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from kt_epimodel_hira.calibration.gamma_registry import (
    get_active_gamma, get_active_source,
    CHILD_IDX, ADULT_IDX, ELDER_IDX,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    derive_beta_from_R0_simplex,
)


def hira_model_nb_2ch(
    loss_fn_nb,
    *,
    ngm_eigval_fn,
    n_seasons: int = 4,
    gamma_3_override=None,
):
    """NB model with the 4-channel simplex collapsed to (home_total, school).

    The script that calls this builds C_home_eff = C_h + C_w + C_o (passed into
    ``shared_static`` of loss_fn_nb) and zeroes C_work, C_other. β is derived
    via NGM with β_w = β_o = 0 so the spectral radius depends only on β_h, β_s.

    Sampled:
        - log_R0: (n_seasons,) per-season log R0.
        - logit_pi2: (n_seasons, 2) per-season (home_total, school) logit.
        - phi_nb: scalar NB dispersion.
    """
    if gamma_3_override is None:
        gamma_active = get_active_gamma()
        gamma_3_const = jnp.array([
            float(gamma_active[CHILD_IDX[0]]),
            float(gamma_active[ADULT_IDX[0]]),
            float(gamma_active[ELDER_IDX[0]]),
        ])
    else:
        gamma_3_const = jnp.asarray(gamma_3_override, dtype=jnp.float64)

    def model():
        log_R0 = numpyro.sample(
            "log_R0",
            dist.TruncatedNormal(
                jnp.log(2.0), 0.4,
                low=jnp.log(0.8), high=jnp.log(3.0),
            ).expand([n_seasons]).to_event(1),
        )
        R0 = jnp.exp(log_R0)

        # 2-simplex per season: (home_total, school). σ=0.3 keeps near uniform
        # (same prior tightness as production 4-channel base).
        logit_pi2 = numpyro.sample(
            "logit_pi2",
            dist.Normal(0.0, 0.3).expand([n_seasons, 2]).to_event(2),
        )
        pi2 = jax.nn.softmax(logit_pi2, axis=-1)
        # Embed into (h, w, s, o) layout with w = o = 0
        pi_h = pi2[..., 0:1]
        pi_s = pi2[..., 1:2]
        zero = jnp.zeros_like(pi_h)
        pi_full = jnp.concatenate([pi_h, zero, pi_s, zero], axis=-1)  # (n_seasons, 4)

        phi = jnp.ones(14)
        phi_full = jnp.ones(15)

        phi_nb = numpyro.sample("phi_nb", dist.HalfNormal(10.0))

        def derive_one(r, p):
            return derive_beta_from_R0_simplex(ngm_eigval_fn, r, p, phi_full)
        beta_per_season = jax.vmap(derive_one)(R0, pi_full)  # (n_seasons, 4)
        beta_16 = beta_per_season.reshape(-1)

        numpyro.deterministic("R0", R0)
        numpyro.deterministic("pi", pi_full)
        numpyro.deterministic("pi2", pi2)
        numpyro.deterministic("beta", beta_16)

        vec_33 = jnp.concatenate([phi, gamma_3_const, beta_16])
        nll = loss_fn_nb(vec_33, phi_nb)
        numpyro.factor("likelihood_nb_2ch", -nll)

    model.gamma_source = get_active_source()
    return model
