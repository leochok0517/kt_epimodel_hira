"""Final test: 4-season joint fit — does cross-season variation revive home/work?

Single-season 2019-20 said home π = 0 robustly. The only remaining info
source (HIRA age resolution is fixed) is between-season variation. If
different seasons have different channel-mix signatures, the joint fit
should be able to identify home/work better than any single season.

Setup: winter break ON + realloc=1, amp=0.9, γ CDC, φ=1.0, free 4ch per
season. Multi-start (5 inits) under L-BFGS on the 20-dim vector
(log_R0×4 + logit_pi×4×4).
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import json
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


SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
AMP = 0.9
# Winter-break dates (school year — approx Korean academic calendar, identical
# for all 4 seasons within a week, day-of-season starts Sep 1)
HOLIDAY_PARAMS = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)


def main():
    print("=" * 70)
    print(f"4-season joint calibration — final home/work identifiability test")
    print("=" * 70)
    print(f"  seasons: {SEASONS}")
    print(f"  amp={AMP}, holiday realloc=1.0, γ CDC, φ=1.0")

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
        vax_peak_iso_week=vax.peak_iso_week,
        vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=float(AMP),
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared.update(HOLIDAY_PARAMS)

    # build seasonal data
    initial_states, obs_list, w_list = [], [], []
    for s in SEASONS:
        tgt = load_hira_target_by_age(
            s, sido_codes=list(SUDOGWON_SIDO_CODES),
            first_peak_only=True, first_peak_end_week=26,
        )
        seed = estimate_initial_infected_from_hira(
            s, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
            gamma_15_assumed=CalibrationParameters().gamma_15,
        )
        state0 = _build_initial_state_with_age_seed(
            pop_15, seed, seed_e_factor=0.5,
            initial_immunity=R0_IMMUNITY_PROFILE,
            initial_vaccinated_fraction=0.0,
        )
        initial_states.append(jnp.asarray(state0))
        nw = tgt["n_weeks"]
        obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            obs[:, i] = tgt["hira_counts"][ag]
            w[:, i] = tgt["weights"][ag]
        obs_list.append(jnp.asarray(obs))
        w_list.append(jnp.asarray(w))

    loss_fn = make_multi_season_loss_fn(
        initial_states=initial_states, obs_hira_list=obs_list,
        weights_hira_list=w_list, shared_static=shared,
        n_weeks=tgt["n_weeks"], min_rate=0.01, discretize_time=False,
    )
    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )
    phi_full = jnp.ones(15); phi_14 = jnp.ones(14)
    gamma_3 = jnp.array([float(GAMMA_CDC[0]), float(GAMMA_CDC[4]), float(GAMMA_CDC[13])])

    # 20-dim θ: [log_R0×4, logit4_per_season×4 (4 logits per season)]
    def unpack(theta):
        log_R0 = theta[:4]
        logit_blocks = theta[4:].reshape(4, 4)
        return log_R0, logit_blocks

    def obj(theta):
        log_R0, logit_blocks = unpack(theta)
        R0 = jnp.exp(log_R0)             # (4,)
        pi_per_season = jax.nn.softmax(logit_blocks, axis=-1)   # (4, 4)
        betas_list = []
        for si in range(4):
            b4 = derive_beta_from_R0_simplex(ngm, R0[si], pi_per_season[si], phi_full)
            betas_list.append(b4)
        beta_16 = jnp.concatenate(betas_list)
        vec_33 = jnp.concatenate([phi_14, gamma_3, beta_16])
        return loss_fn(vec_33)
    obj_jit = jax.jit(obj); grad_jit = jax.jit(jax.grad(obj))

    def fg(x):
        xj = jnp.asarray(x)
        return float(obj_jit(xj)), np.asarray(grad_jit(xj))

    INITS = {
        "balanced":    [0.0, 0.0, 0.0, 0.0],
        "home_high":   [1.5, -1.0, -1.0, 0.5],
        "work_high":   [-1.0, 1.5, -1.0, 0.0],
        "school_high": [-1.0, -1.0, 1.5, 0.0],
        "other_high":  [-1.0, -1.0, 0.0, 1.5],
    }

    bounds = ([(np.log(0.7), np.log(3.5))] * 4
              + [(-4.0, 4.0)] * 16)

    print(f"\n{'init':>12}  {'NLL':>14}  {'R0 per season':>30}")
    multi = []
    for name, base in INITS.items():
        theta0 = np.concatenate([
            np.array([np.log(1.9)] * 4),
            np.tile(base, 4),
        ])
        res = minimize(fg, theta0, jac=True, method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6})
        log_R0, logit_blocks = unpack(res.x)
        R0_arr = np.exp(np.asarray(log_R0))
        pi_arr = np.asarray(jax.nn.softmax(jnp.asarray(logit_blocks), axis=-1))
        multi.append({"init": name, "nll": float(res.fun),
                       "R0": R0_arr.tolist(),
                       "pi": pi_arr.tolist()})
        print(f"  {name:>10}  {res.fun:>14,.0f}  "
              f"{[round(float(x), 3) for x in R0_arr]}")

    best = min(multi, key=lambda r: r["nll"])
    print(f"\n  best init: {best['init']}  NLL {best['nll']:,.0f}")
    print(f"\n[per-season channel mix at best fit]")
    print(f"  {'season':>8}  {'R0':>5}  {'π (h, w, s, o)':>34}")
    for si, s in enumerate(SEASONS):
        pi = best["pi"][si]
        print(f"  {s:>8}  {best['R0'][si]:>5.3f}  "
              f"{str([round(x, 3) for x in pi]):>34}")

    # multi-start home π range per season
    print(f"\n[multi-start home π range across inits]")
    for si, s in enumerate(SEASONS):
        homes = [m["pi"][si][0] for m in multi]
        works = [m["pi"][si][1] for m in multi]
        print(f"  {s}: home {min(homes):.3f}–{max(homes):.3f}  "
              f"work {min(works):.3f}–{max(works):.3f}")

    nlls = [m["nll"] for m in multi]
    print(f"\n  NLL spread: {max(nlls) - min(nlls):.0f}  "
          f"({(max(nlls)-min(nlls))/abs(np.mean(nlls))*100:.4f}%)")

    out = "outputs/metapop/joint_4season_diagnostic.json"
    json.dump({"seasons": SEASONS, "amp": AMP, "holiday": HOLIDAY_PARAMS,
                "multi_start": multi, "best": best},
               open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
