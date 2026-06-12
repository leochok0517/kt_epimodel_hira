"""Sweep realloc + amp to test how much winter break can revive home π.

Same setup as diagnose_break_realloc but sweeps (amp, realloc) to find the
regime where home channel recovers most strongly. realloc=1 routes school
contact loss into p_school (triggers existing spillover κ); amp controls
break depth (sf_min = 1 − amp).

For each combo, multi-start (3 inits: balanced, work_high, home_high) under
γ CDC, φ=1.0, free 4-channel, finds the lowest-NLL home value.
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


SEASON = "2019-2020"
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
HOLIDAY_DATES = dict(
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
)

AMP_GRID = [0.5, 0.7, 0.9]
REALLOC_GRID = [0.0, 0.5, 1.0]
INITS = {
    "balanced":  [0.0, 0.0, 0.0, 0.0],
    "home_high": [1.5, -1.0, -1.0, 0.5],
    "work_high": [-1.0, 1.5, -1.0, 0.0],
}


def main():
    print("=" * 70)
    print("Sweep realloc × amp — recover home π under winter break "
          f"({SEASON}, γ CDC)")
    print("=" * 70)

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    rho_emp = inputs["rho"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy

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
    obs = np.zeros((nw, 6)); w_obs = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w_obs[:, i] = tgt["weights"][ag]

    phi_full = jnp.ones(15); phi_14 = jnp.ones(14)
    gamma_3 = jnp.array([float(GAMMA_CDC[0]), float(GAMMA_CDC[4]), float(GAMMA_CDC[13])])

    def shared_with(amp, realloc):
        d = dict(
            C_home=jnp.asarray(matrices["C_home"]),
            C_school=jnp.asarray(matrices["C_school"]),
            C_work=jnp.asarray(matrices["C_work"]),
            C_other=jnp.asarray(matrices["C_other"]),
            M_home=jnp.asarray(mobility["home"]),
            M_school=jnp.asarray(mobility["school"]),
            M_work=jnp.asarray(mobility["work"]),
            M_other=jnp.asarray(mobility["other"]),
            pop_15=jnp.asarray(pop_15), rho=jnp.asarray(rho_emp),
            kappa=jnp.asarray(disease.kappa_array),
            sigma=disease.sigma, gamma=disease.gamma,
            p_school=policy.p_school, p_work=policy.p_work,
            VE=vax.VE,
            annual_coverage=jnp.asarray(vax.annual_coverage),
            vax_peak_iso_week=vax.peak_iso_week,
            vax_spread_weeks=vax.spread_weeks,
            seasonality_amp=float(amp),
            seasonality_base=disease.seasonality_base,
            seasonality_peak_day=disease.seasonality_peak_day,
            seasonality_period=disease.seasonality_period,
            school_holiday_amp=0.7,                # depth of school drop kept at 0.7
            school_holiday_realloc=float(realloc),
        )
        d.update(HOLIDAY_DATES)
        return d

    def fit(shared, amp, init):
        loss_fn = make_multi_season_loss_fn(
            initial_states=[state0],
            obs_hira_list=[jnp.asarray(obs)],
            weights_hira_list=[jnp.asarray(w_obs)],
            shared_static=shared, n_weeks=nw,
            min_rate=0.01, discretize_time=False,
        )
        ngm = make_ngm_eigvalue_fn(
            pop_15=pop_15, rho=rho_emp,
            C_home=matrices["C_home"], C_work=matrices["C_work"],
            C_school=matrices["C_school"], C_other=matrices["C_other"],
            R0_immunity=R0_IMMUNITY_PROFILE,
            gamma=disease.gamma, seasonal_factor=1.0 + float(amp),
        )
        def obj(theta):
            log_R0 = theta[0]; logit4 = theta[1:5]
            R0 = jnp.exp(log_R0)
            pi = jax.nn.softmax(logit4)
            beta_4 = derive_beta_from_R0_simplex(ngm, R0, pi, phi_full)
            beta_16 = jnp.concatenate([beta_4, jnp.zeros(12)])
            return loss_fn(jnp.concatenate([phi_14, gamma_3, beta_16]))
        obj_jit = jax.jit(obj); grad_jit = jax.jit(jax.grad(obj))

        def fg(x):
            xj = jnp.asarray(x)
            return float(obj_jit(xj)), np.asarray(grad_jit(xj))
        theta0 = np.array([np.log(1.9)] + list(init))
        res = minimize(fg, theta0, jac=True, method="L-BFGS-B",
                        bounds=[(np.log(0.7), np.log(3.5))] + [(-4.0, 4.0)] * 4,
                        options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-6})
        return {"nll": float(res.fun), "R0": float(np.exp(res.x[0])),
                "pi": [float(x) for x in jax.nn.softmax(jnp.asarray(res.x[1:5]))]}

    print(f"\n  {'amp':>4} {'realloc':>7}  best home π  best NLL  "
          f"{'best π (h, w, s, o)':>32}")
    grid_results = []
    for amp in AMP_GRID:
        for realloc in REALLOC_GRID:
            shared = shared_with(amp, realloc)
            best = None
            for init_name, init in INITS.items():
                r = fit(shared, amp, init)
                if best is None or r["nll"] < best["nll"]:
                    best = r; best["init"] = init_name
            grid_results.append({"amp": amp, "realloc": realloc, **best})
            print(f"  {amp:>4.1f}  {realloc:>5.1f}  {best['pi'][0]:>10.3f}  "
                  f"{best['nll']:>12,.0f}  "
                  f"{str([round(x, 3) for x in best['pi']]):>32}  "
                  f"(from {best['init']})")

    print("\n[interpretation]")
    best_home = max(grid_results, key=lambda r: r["pi"][0])
    print(f"  max home π across grid: {best_home['pi'][0]:.3f} "
          f"at amp={best_home['amp']}, realloc={best_home['realloc']}")
    print(f"  NLL there: {best_home['nll']:,.0f}")

    # save
    out = "outputs/metapop/break_realloc_sweep.json"
    json.dump({"season": SEASON, "amp_grid": AMP_GRID,
                "realloc_grid": REALLOC_GRID, "results": grid_results,
                "best_home": best_home}, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
