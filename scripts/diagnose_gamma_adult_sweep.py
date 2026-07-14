"""γ_adult sweep at fixed γ-level=0.6 — channel rationalization test.

For 2019-2020 season, with φ=1.0 fixed, gamma_level=0.6 anchor (Jung q=0.67
minus asymptomatic/non-claiming), sweep the adult relative detection ratio.
At each adult_rel, re-optimize (log_R0, logit_pi) by L-BFGS-B and inspect the
resulting channel simplex.

The question: does home/work β > 0 emerge when γ_adult drops (i.e. when the
calibration is allowed to fit fewer adult HIRA counts not by zeroing β_h/β_w
but by lowering reporting)?
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, make_multi_season_loss_fn,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)


SEASON = "2019-2020"
GAMMA_LEVEL = 0.6                     # Jung q=0.67 minus asymptomatic
CHILD_REL_BASE = 1.45                 # CDC normalized to unweighted mean=1
ELDER_REL_BASE = 0.90
ADULT_REL_GRID = [0.65, 0.50, 0.40, 0.30, 0.20]


def build_gamma_15(adult_rel: float) -> np.ndarray:
    """level=0.6, shape normalized to unweighted mean(child_rel, adult_rel, elder_rel)=1.

    Returns (15,) γ matching the loss_jax gamma slicing
    (4 child + 9 adult + 2 elder).
    """
    K = 3.0 / (CHILD_REL_BASE + adult_rel + ELDER_REL_BASE)
    g_c = GAMMA_LEVEL * K * CHILD_REL_BASE
    g_a = GAMMA_LEVEL * K * adult_rel
    g_e = GAMMA_LEVEL * K * ELDER_REL_BASE
    g_15 = np.concatenate([np.full(4, g_c), np.full(9, g_a), np.full(2, g_e)])
    if (g_15 > 1).any():
        print(f"  [warn] adult_rel={adult_rel}: γ exceeds 1 — clipping. raw: "
              f"child={g_c:.3f} adult={g_a:.3f} elder={g_e:.3f}")
        g_15 = np.clip(g_15, 0.0, 1.0)
    return g_15


def main():
    print("=" * 70)
    print(f"γ_adult sweep — season {SEASON}, level {GAMMA_LEVEL}, φ=1.0 fixed")
    print("=" * 70)

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    rho_emp = inputs["rho"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy

    shared = dict(
        C_home=jnp.asarray(matrices["C_home"]),
        C_school=jnp.asarray(matrices["C_school"]),
        C_work=jnp.asarray(matrices["C_work"]),
        C_other=jnp.asarray(matrices["C_other"]),
        M_home=jnp.asarray(mobility["home"]),
        M_school=jnp.asarray(mobility["school"]),
        M_work=jnp.asarray(mobility["work"]),
        M_other=jnp.asarray(mobility["other"]),
        pop_15=jnp.asarray(pop_15),
        rho=jnp.asarray(rho_emp),
        kappa=jnp.asarray(disease.kappa_array),
        sigma=disease.sigma, gamma=disease.gamma,
        p_school=policy.p_school, p_work=policy.p_work,
        VE=vax.VE,
        annual_coverage=jnp.asarray(vax.annual_coverage),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=disease.seasonality_amp,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )

    seed_15 = estimate_initial_infected_from_hira(
        SEASON, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    tgt = load_hira_target_by_age(
        SEASON, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    nw = tgt["n_weeks"]
    obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]

    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.7,
    )

    # Build a JAX-grad-able objective:  (log_R0, logit_pi[4]) → NLL  given γ
    phi_full = jnp.ones(15)
    phi_14 = jnp.ones(14)

    def make_obj(gamma_15_arr):
        """Returns (obj_jit, grad_jit). gamma_3 baked in via vec_33."""
        # gamma_triple_to_15 uses (child, adult, elder) which the loss replicates
        # → use the three values directly.
        gamma_3 = jnp.array([gamma_15_arr[0], gamma_15_arr[4], gamma_15_arr[13]])
        loss_fn = make_multi_season_loss_fn(
            initial_states=[state0],
            obs_hira_list=[jnp.asarray(obs)],
            weights_hira_list=[jnp.asarray(w)],
            shared_static=shared,
            n_weeks=nw, min_rate=0.01, discretize_time=False,
        )

        def obj(theta):
            log_R0 = theta[0]
            logit_pi = theta[1:5]
            R0 = jnp.exp(log_R0)
            pi = jax.nn.softmax(logit_pi)
            beta_4 = derive_beta_from_R0_simplex(ngm, R0, pi, phi_full)
            beta_16 = jnp.concatenate([beta_4, jnp.zeros(12)])
            vec_33 = jnp.concatenate([phi_14, gamma_3, beta_16])
            return loss_fn(vec_33)

        obj_jit = jax.jit(obj)
        grad_jit = jax.jit(jax.grad(obj))
        return obj_jit, grad_jit

    # init: log_R0 = log(1.93), logit_pi balanced
    theta0 = np.array([np.log(1.93), 0.0, 0.0, 0.0, 0.0])

    print(f"{'adult_rel':>10} {'γ_adult':>9} {'γ_child':>9} {'γ_elder':>9} "
          f"{'R0':>6} {'home':>6} {'work':>6} {'school':>6} {'other':>6}  "
          f"{'NLL':>14}")
    print("-" * 110)

    results = []
    for adult_rel in ADULT_REL_GRID:
        gamma_15 = build_gamma_15(adult_rel)
        g_c, g_a, g_e = gamma_15[0], gamma_15[4], gamma_15[13]
        obj_jit, grad_jit = make_obj(gamma_15)

        def f_and_g(x):
            x_jnp = jnp.asarray(x)
            return float(obj_jit(x_jnp)), np.asarray(grad_jit(x_jnp))

        res = minimize(
            f_and_g, theta0, jac=True, method="L-BFGS-B",
            bounds=[(np.log(0.5), np.log(3.5))] + [(-3.0, 3.0)] * 4,
            options={"maxiter": 100, "ftol": 1e-8, "gtol": 1e-5},
        )
        log_R0_opt = res.x[0]
        logit_pi_opt = res.x[1:5]
        R0_opt = float(np.exp(log_R0_opt))
        pi_opt = np.asarray(jax.nn.softmax(jnp.asarray(logit_pi_opt)))
        nll = float(res.fun)

        print(f"{adult_rel:>10.2f} {g_a:>9.3f} {g_c:>9.3f} {g_e:>9.3f} "
              f"{R0_opt:>6.3f} "
              f"{pi_opt[0]:>6.3f} {pi_opt[1]:>6.3f} {pi_opt[2]:>6.3f} {pi_opt[3]:>6.3f}  "
              f"{nll:>14,.0f}")

        results.append({
            "adult_rel": adult_rel,
            "gamma_child": float(g_c), "gamma_adult": float(g_a), "gamma_elder": float(g_e),
            "R0": R0_opt, "pi": pi_opt.tolist(),
            "nll": nll, "n_iter": res.nit,
        })

    print()
    print("Interpretation key:")
    print("  - home/work π rising from ≈0 → channel rationalization with lower γ_adult")
    print("  - NLL flat across rows → γ-level is non-identifiable (R0 absorbs)")
    print("  - NLL monotone (lower for some adult_rel) → data prefers that detection shape")

    # save
    import json
    out = "outputs/calibration/gamma_adult_sweep.json"
    json.dump({
        "season": SEASON, "level": GAMMA_LEVEL,
        "child_rel_base": CHILD_REL_BASE, "elder_rel_base": ELDER_REL_BASE,
        "results": results,
    }, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
