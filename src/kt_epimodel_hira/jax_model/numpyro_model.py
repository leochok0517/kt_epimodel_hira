"""Numpyro probabilistic model for HIRA multi-season calibration.

Reparam A (v7+): the β · φ multiplicative ridge — formally broken by the
φ_5≡1 anchor in loss_jax but causing strong soft anisotropy (chain 1 vs 2
step sizes 1.79e-03 vs 7.81e-05 in v6) — is now aligned to the coordinate
axes:

- per-season log_R0 (4): primary axis, the data-visible scale
- per-season channel simplex (4 logits per season, softmax → 4-simplex):
  h/w/s/o mix
- φ (14 ages): age susceptibility multipliers (φ_5≡1.0 anchor in loss_fn)

β (16) is *derived* per season: β_ch = (R0 / ρ(K(π_s, φ))) · π_s_ch, where
ρ is the NGM spectral radius. This makes R0 exactly identified by data and
removes the scale ridge from the sampler's geometry. γ stays fixed
(see gamma_registry + GAMMA_STRATEGY.md).

Likelihood: Poisson NLL via numpyro.factor("likelihood", -nll).
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


def vec_to_param_dict(vec_33):
    """Map 33-dim warm-start vec to (phi, beta) for init_to_value.

    Layout: [phi(14), gamma(3) -- ignored --, beta(16)]. Reparam v7+ does not
    sample β directly, so this helper is now used only for legacy regression
    tests. Use vec_to_reparam_init for v7+ init.
    """
    return {
        "phi": jnp.asarray(vec_33[:14]),
        "beta": jnp.asarray(vec_33[17:33]),
    }


def make_ngm_eigvalue_fn(
    *,
    pop_15,          # (15, n_admdong)
    rho,             # (n_admdong, 15) or (15, n_admdong) per data convention
    C_home, C_work, C_school, C_other,
    R0_immunity,     # (15,) initial immunity profile
    gamma: float,
    seasonal_factor: float = 1.7,
):
    """Build a JAX NGM spectral-radius closure.

    Returns ``ngm_eigval(beta_h, beta_w, beta_s, beta_o, phi_full)`` → scalar R0.

    Mirrors compute_R0_from_sample in scripts/m2_smoke_prior.py but pure JAX
    so it is autodiff- and jit-friendly inside the numpyro model.
    """
    pop = jnp.asarray(pop_15).reshape(-1)            # (15,)
    N_safe = jnp.maximum(pop, 1e-10)
    rho_flat = jnp.asarray(rho).reshape(-1)
    # Match numpy convention in compute_R0_from_sample: rho is (15,) flattened
    if rho_flat.shape[0] != 15:
        # If aggregated input has (n_admdong, 15), reduce to per-age rho
        rho_arr = jnp.asarray(rho)
        rho_flat = rho_arr.mean(axis=0) if rho_arr.shape[-1] == 15 else rho_arr.reshape(-1)[:15]
    rho_ok = (rho_flat > 0).astype(jnp.float64)
    S_frac = 1.0 - jnp.asarray(R0_immunity)
    work_mask = jnp.zeros(15, dtype=jnp.float64).at[4:14].set(1.0)
    school_mask = jnp.zeros((15, 15), dtype=jnp.float64).at[:4, :4].set(1.0)
    sf_gamma = seasonal_factor / gamma

    Ch = jnp.asarray(C_home)
    Cw = jnp.asarray(C_work)
    Cs = jnp.asarray(C_school)
    Co = jnp.asarray(C_other)

    def ngm_eigval(beta_h, beta_w, beta_s, beta_o, phi_full):
        # C_eff(a, a'): per-channel additive structure (mirrors compute_R0_from_sample)
        C_eff = beta_h * Ch
        C_eff = C_eff + beta_s * Cs * school_mask
        # work: row-a addition = beta_w * Cw[a, j] * rho[a] * rho_ok[j], for a in [4, 13]
        C_eff = C_eff + beta_w * Cw * (rho_flat * work_mask)[:, None] * rho_ok[None, :]
        C_eff = C_eff + beta_o * Co
        # K = (sf/γ) · diag(pop) · diag(φ) · diag(S) · C_eff · diag(1/N)
        # Equivalent vectorized form (avoids 15×15 diag matmuls):
        K = sf_gamma * (pop * phi_full * S_frac)[:, None] * C_eff * (1.0 / N_safe)[None, :]
        eigvals = jnp.linalg.eigvals(K)
        return jnp.max(jnp.real(eigvals))

    return ngm_eigval


def derive_beta_from_R0_simplex(ngm_eigval_fn, R0, pi_4, phi_full):
    """Derive (β_h, β_w, β_s, β_o) from (R0, simplex, φ) so that NGM ρ = R0.

    Uses 1-homogeneity of the spectral radius in β: ρ(K(s·π, φ)) = s·ρ(K(π, φ)),
    so the scale s = R0 / ρ(K(π, φ)) inverts exactly.
    """
    rho_pi = ngm_eigval_fn(pi_4[0], pi_4[1], pi_4[2], pi_4[3], phi_full)
    scale = R0 / rho_pi
    return scale * pi_4


def hira_model(
    loss_fn,
    *,
    ngm_eigval_fn,
    n_seasons: int = 4,
    lambda_phi: float = 0.1,
):
    """Build a parameter-less numpyro model with reparam A.

    Args:
        loss_fn: jax-compiled multi-season Poisson NLL (signature: vec_33 → scalar).
        ngm_eigval_fn: closure from ``make_ngm_eigvalue_fn``. Used to derive β.
        n_seasons: number of seasons (4 in current setup).
        lambda_phi: smoothing penalty on log φ differences.

    Sampled (primary axes):
        - log_R0: (n_seasons,) per-season log basic reproduction number.
        - logit_pi: (n_seasons, 4) per-season channel mix (logits, softmax inside).
        - phi: (14,) age susceptibility (LogNormal, φ_5≡1.0 anchor in loss).

    Deterministic:
        - R0 = exp(log_R0), pi = softmax(logit_pi), beta (16) = derive(R0, pi, φ)
        - gamma_{child,adult,elder} from registry.
    """
    gamma_active = get_active_gamma()
    gamma_child_fixed = float(gamma_active[CHILD_IDX[0]])
    gamma_adult_fixed = float(gamma_active[ADULT_IDX[0]])
    gamma_elder_fixed = float(gamma_active[ELDER_IDX[0]])

    def model():
        # Primary axis: log_R0 per season. TN centered at log 1.5 (typical
        # seasonal flu R0); [log 0.8, log 2.8] = [-0.22, 1.03] covers full range.
        log_R0 = numpyro.sample(
            "log_R0",
            dist.TruncatedNormal(
                jnp.log(1.5), 0.3,
                low=jnp.log(0.8), high=jnp.log(2.8),
            ).expand([n_seasons]).to_event(1),
        )
        R0 = jnp.exp(log_R0)

        # Channel mix per season: softmax over 4 logits (no reference fix; σ=0.5
        # gives shape mass ≈ unimodal at uniform mix). Symmetric so all channels
        # are exchangeable a priori.
        # σ=0.3 keeps simplex mass near uniform (no single channel >~0.6 with
        # high prior weight). σ=0.5 in v7a caused NUTS to probe near-degenerate
        # mixes (one channel ≈1.0) → β explosion → ODE max_steps overrun.
        logit_pi = numpyro.sample(
            "logit_pi",
            dist.Normal(0.0, 0.3).expand([n_seasons, 4]).to_event(2),
        )
        pi = jax.nn.softmax(logit_pi, axis=-1)

        # φ FIXED at uniform 1.0 (per-contact susceptibility neutral). v7b/v8/v9
        # showed φ is genuinely non-identifiable: prior tightening (σ 0.15→0.05)
        # didn't help (v8 r_hat 4.43), and unified init (v9) still produced
        # chain-split posterior φ ranging 1.75–2.29 → likelihood surface itself
        # is multimodal in φ. Per-age transmission differences are already
        # captured by the 4-channel contacts + step R(0) profile. φ is held
        # constant at 1.0; the φ_5≡1.0 anchor in loss_fn is now trivially
        # consistent. phi_smooth penalty is dropped (no longer sampled).
        phi = jnp.ones(14)
        phi_full = jnp.ones(15)

        # Derive β per season via 1-homogeneous NGM inversion
        def derive_one(r, p):
            return derive_beta_from_R0_simplex(ngm_eigval_fn, r, p, phi_full)

        beta_per_season = jax.vmap(derive_one)(R0, pi)  # (n_seasons, 4)
        beta_16 = beta_per_season.reshape(-1)            # [β_2017_4, ...]

        # γ: fixed from registry
        gamma_3 = jnp.array([
            gamma_child_fixed, gamma_adult_fixed, gamma_elder_fixed,
        ])
        numpyro.deterministic("gamma_child", gamma_3[0])
        numpyro.deterministic("gamma_adult", gamma_3[1])
        numpyro.deterministic("gamma_elder", gamma_3[2])
        numpyro.deterministic("R0", R0)
        numpyro.deterministic("pi", pi)
        numpyro.deterministic("beta", beta_16)

        vec_33 = jnp.concatenate([phi, gamma_3, beta_16])
        nll = loss_fn(vec_33)
        numpyro.factor("likelihood", -nll)

    model.gamma_source = get_active_source()
    return model


def hira_model_nb_chprior(
    loss_fn_nb,
    *,
    ngm_eigval_fn,
    n_seasons: int = 4,
    target_pi,                      # (4,) field-knowledge centered target
    sigma_per_channel,              # (4,) prior σ on centered logit per channel
    gamma_3_override=None,
):
    """NB model with Gaussian prior on per-season channel logit toward a target.

    The 4-channel simplex is free in principle (Normal(0, 2.0) base prior so
    the optimizer can explore), and a centered-logit penalty pulls it toward
    the field-knowledge target. σ_per_channel lets school/other stay loose
    while home/work are pinned.
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

    target_arr = jnp.asarray(target_pi, dtype=jnp.float64)
    logit_target = jnp.log(jnp.clip(target_arr, 1e-6, None))
    logit_target = logit_target - jnp.mean(logit_target)
    inv_var = 1.0 / (jnp.asarray(sigma_per_channel) ** 2)

    def model():
        log_R0 = numpyro.sample(
            "log_R0",
            dist.TruncatedNormal(
                jnp.log(2.0), 0.4,
                low=jnp.log(0.8), high=jnp.log(3.0),
            ).expand([n_seasons]).to_event(1),
        )
        R0 = jnp.exp(log_R0)

        # Wide base prior so the optimizer can move; channel prior is added below
        logit_pi = numpyro.sample(
            "logit_pi",
            dist.Normal(0.0, 2.0).expand([n_seasons, 4]).to_event(2),
        )
        # Centered-logit Gaussian prior pulling toward target
        centered = logit_pi - jnp.mean(logit_pi, axis=-1, keepdims=True)
        dev = centered - logit_target[None, :]
        prior_logp = -0.5 * jnp.sum(dev * dev * inv_var[None, :])
        numpyro.factor("ch_prior", prior_logp)

        pi = jax.nn.softmax(logit_pi, axis=-1)

        phi = jnp.ones(14)
        phi_full = jnp.ones(15)
        phi_nb = numpyro.sample("phi_nb", dist.HalfNormal(10.0))

        def derive_one(r, p):
            return derive_beta_from_R0_simplex(ngm_eigval_fn, r, p, phi_full)
        beta_per_season = jax.vmap(derive_one)(R0, pi)
        beta_16 = beta_per_season.reshape(-1)

        numpyro.deterministic("R0", R0)
        numpyro.deterministic("pi", pi)
        numpyro.deterministic("beta", beta_16)

        vec_33 = jnp.concatenate([phi, gamma_3_const, beta_16])
        nll = loss_fn_nb(vec_33, phi_nb)
        numpyro.factor("likelihood_nb_chprior", -nll)

    model.gamma_source = get_active_source()
    return model


def hira_model_nb_wo(
    loss_fn_nb,
    *,
    ngm_eigval_fn,
    n_seasons: int = 4,
    work_ratio: float = 17.03 / (17.03 + 31.77),  # = 0.349 (NIMS row-sum)
    gamma_3_override=None,
):
    """NB model with the work:other split fixed by NIMS contact row-sums.

    work and other are nearly non-identifiable from HIRA's 6-age aggregation
    (D4 NLL gap ~6.5K vs noise scale ~80K). To resolve, we collapse the simplex
    to 3 free dims (home, school, work+other combined) and split the combined
    mass by a fixed channel-contact ratio (NIMS row-sums work 17.03 vs other
    31.77 → work_ratio ≈ 0.349). γ can be overridden (e.g. level=0.6 × CDC).
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
                jnp.log(1.5), 0.3,
                low=jnp.log(0.8), high=jnp.log(2.8),
            ).expand([n_seasons]).to_event(1),
        )
        R0 = jnp.exp(log_R0)

        # 3-simplex per season: (home, school, work+other combined)
        logit_pi3 = numpyro.sample(
            "logit_pi3",
            dist.Normal(0.0, 0.3).expand([n_seasons, 3]).to_event(2),
        )
        pi3 = jax.nn.softmax(logit_pi3, axis=-1)
        pi_home = pi3[..., 0:1]
        pi_school = pi3[..., 1:2]
        pi_wo = pi3[..., 2:3]
        pi_work = pi_wo * work_ratio
        pi_other = pi_wo * (1.0 - work_ratio)
        # reassemble to (h, w, s, o) matching loss layout
        pi_full = jnp.concatenate([pi_home, pi_work, pi_school, pi_other], axis=-1)

        phi = jnp.ones(14)
        phi_full = jnp.ones(15)

        phi_nb = numpyro.sample("phi_nb", dist.HalfNormal(10.0))

        def derive_one(r, p):
            return derive_beta_from_R0_simplex(ngm_eigval_fn, r, p, phi_full)
        beta_per_season = jax.vmap(derive_one)(R0, pi_full)
        beta_16 = beta_per_season.reshape(-1)

        numpyro.deterministic("R0", R0)
        numpyro.deterministic("pi", pi_full)
        numpyro.deterministic("beta", beta_16)
        numpyro.deterministic("gamma_3", gamma_3_const)

        vec_33 = jnp.concatenate([phi, gamma_3_const, beta_16])
        nll = loss_fn_nb(vec_33, phi_nb)
        numpyro.factor("likelihood_nb_wo", -nll)

    model.gamma_source = get_active_source()
    return model


def hira_model_nb(
    loss_fn_nb,
    *,
    ngm_eigval_fn,
    n_seasons: int = 4,
):
    """Reparam A model with Negative-Binomial observation noise.

    Adds a single global concentration parameter ``phi_nb`` (NB dispersion):
        obs ~ NegativeBinomial2(mean=pred, concentration=phi_nb).
    ``loss_fn_nb`` is the NB-weighted NLL closure with signature
    ``(vec_33, concentration) → scalar``.
    """
    gamma_active = get_active_gamma()
    gamma_child_fixed = float(gamma_active[CHILD_IDX[0]])
    gamma_adult_fixed = float(gamma_active[ADULT_IDX[0]])
    gamma_elder_fixed = float(gamma_active[ELDER_IDX[0]])

    def model():
        log_R0 = numpyro.sample(
            "log_R0",
            dist.TruncatedNormal(
                jnp.log(1.5), 0.3,
                low=jnp.log(0.8), high=jnp.log(2.8),
            ).expand([n_seasons]).to_event(1),
        )
        R0 = jnp.exp(log_R0)

        logit_pi = numpyro.sample(
            "logit_pi",
            dist.Normal(0.0, 0.3).expand([n_seasons, 4]).to_event(2),
        )
        pi = jax.nn.softmax(logit_pi, axis=-1)

        # φ FIXED at 1.0 (v7b/v8/v9 confirmed non-identifiability)
        phi = jnp.ones(14)
        phi_full = jnp.ones(15)

        # NB dispersion. Larger phi_nb → closer to Poisson. HalfNormal(10) is
        # weakly informative; data picks the right level.
        phi_nb = numpyro.sample("phi_nb", dist.HalfNormal(10.0))

        # Derive β per season via 1-homogeneous NGM inversion
        def derive_one(r, p):
            return derive_beta_from_R0_simplex(ngm_eigval_fn, r, p, phi_full)

        beta_per_season = jax.vmap(derive_one)(R0, pi)
        beta_16 = beta_per_season.reshape(-1)

        gamma_3 = jnp.array([
            gamma_child_fixed, gamma_adult_fixed, gamma_elder_fixed,
        ])
        numpyro.deterministic("R0", R0)
        numpyro.deterministic("pi", pi)
        numpyro.deterministic("beta", beta_16)

        vec_33 = jnp.concatenate([phi, gamma_3, beta_16])
        nll = loss_fn_nb(vec_33, phi_nb)
        numpyro.factor("likelihood_nb", -nll)

    model.gamma_source = get_active_source()
    return model
