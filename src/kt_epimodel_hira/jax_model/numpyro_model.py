"""Numpyro probabilistic model for HIRA multi-season calibration.

γ is fixed externally (see calibration/gamma_registry.py + GAMMA_STRATEGY.md):
β · φ · γ multiplicative non-identifiability is irresolvable from data alone,
so γ is taken from a citable external source. Only φ and β are sampled.

Sampled parameters:
- phi (14): LogNormal(0, 0.5) + smoothing factor λ_phi · Σ(Δφ)²
- beta (16): HalfNormal(σ=1.0)  -- PRIOR_SPECIFICATION.md §2.3

Fixed:
- gamma_child, gamma_adult, gamma_elder = from gamma_registry.get_active_gamma()
  (compressed to per-group constants in the model body)

Likelihood: Poisson NLL via numpyro.factor("likelihood", -nll). The
multi-season NLL is provided by make_multi_season_loss_fn (M1c).
"""
from __future__ import annotations
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from kt_epimodel_hira.calibration.gamma_registry import (
    get_active_gamma, get_active_source,
    CHILD_IDX, ADULT_IDX, ELDER_IDX,
)


def vec_to_param_dict(vec_33):
    """Map 33-dim warm-start vec to numpyro param dict for init_to_value.

    Even though γ is now fixed, the helper still extracts γ from a 33-dim
    vec (used in init_params keying just for phi + beta).

    Layout: [phi(14), gamma(3) -- ignored --, beta(16)]
    """
    return {
        "phi": jnp.asarray(vec_33[:14]),
        "beta": jnp.asarray(vec_33[17:33]),
    }


def hira_model(
    loss_fn,
    *,
    lambda_phi: float = 0.1,
):
    """Build a parameter-less numpyro model.

    γ is *fixed* at the active registry source. The model samples only
    φ (14) and β (16). NLL is computed against the same loss_fn used in
    M1c regression, with γ injected into the 33-dim vector at call time.

    Use with NUTS:
        from kt_epimodel_hira.jax_model.numpyro_model import hira_model
        model = hira_model(loss_fn)
        kernel = NUTS(model)
        mcmc = MCMC(kernel, ...)
        mcmc.run(rng_key)
    """
    # Snapshot active γ at build time. The model body uses these as constants.
    gamma_active = get_active_gamma()  # (15,)
    # Extract per-group scalar (all values within each group are identical)
    gamma_child_fixed = float(gamma_active[CHILD_IDX[0]])
    gamma_adult_fixed = float(gamma_active[ADULT_IDX[0]])
    gamma_elder_fixed = float(gamma_active[ELDER_IDX[0]])

    def model():
        # phi: 14 ages, truncated normal centered at reference (φ_25-29 = 1.0).
        # Tightened per Cauchemez et al. (household ~15% child-vs-adult diff).
        # Previous σ=0.3 [0.1, 3.0] led to posterior mean 2.0 (β-φ ridge);
        # σ=0.15 [0.5, 1.5] blocks that mode. See PRIOR_SPECIFICATION §2.2.5-6.
        phi = numpyro.sample(
            "phi",
            dist.TruncatedNormal(1.0, 0.15, low=0.5, high=1.5)
                .expand([14]).to_event(1),
        )
        numpyro.factor("phi_smooth", -lambda_phi * jnp.sum(jnp.diff(phi) ** 2))

        # beta: 16 channels (4 ch × 4 seasons). TruncatedNormal derived from
        # R0 inversion: R0_peak ≈ 36.9 × β (uniform), so β ∈ [0.027, 0.068]
        # covers R0 ∈ [1.0, 2.5] (seasonal influenza range). Per-channel,
        # widen to high=0.20 to allow single-channel dominance (observed in
        # L-BFGS fits, e.g. β_h=0.20). σ=0.04 keeps simultaneous-large-β
        # configurations rare (4-channel epidemic explosion prevention).
        beta = numpyro.sample(
            "beta",
            dist.TruncatedNormal(0.04, 0.04, low=0.001, high=0.15)
                .expand([16]).to_event(1),
        )

        # gamma: FIXED at active registry source (deterministic, not sampled)
        gamma_3 = jnp.array([
            gamma_child_fixed, gamma_adult_fixed, gamma_elder_fixed,
        ])
        # Track for posterior summaries (these are constants, but appearing
        # in idata makes provenance explicit).
        numpyro.deterministic("gamma_child", gamma_3[0])
        numpyro.deterministic("gamma_adult", gamma_3[1])
        numpyro.deterministic("gamma_elder", gamma_3[2])

        # Assemble 33-dim vec for loss_fn signature compatibility
        vec_33 = jnp.concatenate([phi, gamma_3, beta])

        nll = loss_fn(vec_33)
        numpyro.factor("likelihood", -nll)

    # Attach provenance for downstream logging
    model.gamma_source = get_active_source()
    return model
