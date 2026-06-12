"""Test whether contact reallocation during winter break recovers home π.

Reuses the existing spillover κ (slide 15: students 0.42, adults 0.60, 70+ 0).
No new parameter — implemented as: school_holiday_realloc ∈ [0, 1] routes
the school-channel loss through p_school (which triggers existing home
spillover) instead of destroying it via β_s.

3-config fit on 2019-20 (γ CDC, φ=1.0, free 4-channel, amp=0.7):
- no_holiday:          school_holiday_amp = 0
- holiday_school_only: amp = 0.7, realloc = 0.0 (current implementation)
- holiday_realloc:     amp = 0.7, realloc = 1.0 (NEW — contacts conserve)

Expected: home π rises under realloc if the missing signal was "students
home during break drive household transmission."
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
AMP = 0.7

HOLIDAY_BASE = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
)


def main():
    print("=" * 70)
    print(f"winter-break realloc test ({SEASON}, γ CDC, φ=1.0, amp={AMP})")
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

    def shared_with(holiday_amp, realloc):
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
            seasonality_amp=float(AMP),
            seasonality_base=disease.seasonality_base,
            seasonality_peak_day=disease.seasonality_peak_day,
            seasonality_period=disease.seasonality_period,
        )
        d.update(HOLIDAY_BASE)
        d["school_holiday_amp"] = float(holiday_amp)
        d["school_holiday_realloc"] = float(realloc)
        return d

    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )
    gamma_3 = jnp.array([GAMMA_CDC[0], GAMMA_CDC[4], GAMMA_CDC[13]])

    def fit_free_4ch(shared, init_logit=None):
        loss_fn = make_multi_season_loss_fn(
            initial_states=[state0],
            obs_hira_list=[jnp.asarray(obs)],
            weights_hira_list=[jnp.asarray(w_obs)],
            shared_static=shared, n_weeks=nw,
            min_rate=0.01, discretize_time=False,
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

        if init_logit is None:
            init_logit = [0.0, 0.0, 0.0, 0.0]
        theta0 = np.array([np.log(1.9)] + list(init_logit))
        res = minimize(fg, theta0, jac=True, method="L-BFGS-B",
                        bounds=[(np.log(0.7), np.log(3.5))] + [(-4.0, 4.0)] * 4,
                        options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-6})
        return {"nll": float(res.fun), "R0": float(np.exp(res.x[0])),
                "pi": [float(x) for x in jax.nn.softmax(jnp.asarray(res.x[1:5]))]}

    print(f"\n{'config':>22}  {'NLL':>14}  {'R0':>6}  {'π (h, w, s, o)':>32}")
    print("-" * 88)
    CONFIGS = [
        ("no_holiday",          shared_with(0.0, 0.0)),
        ("holiday_school_only", shared_with(0.7, 0.0)),
        ("holiday_realloc",     shared_with(0.7, 1.0)),
    ]
    results = []
    for name, sh in CONFIGS:
        r = fit_free_4ch(sh)
        results.append({"config": name, **r})
        print(f"  {name:>20}  {r['nll']:>14,.0f}  {r['R0']:>6.3f}  "
              f"{str([round(x, 3) for x in r['pi']]):>32}")

    # Multi-start the realloc config to test home π stability
    print(f"\n[multi-start under realloc] free 4-channel, 5 inits, γ CDC")
    sh_realloc = shared_with(0.7, 1.0)
    INITS = {
        "balanced":    [0.0, 0.0, 0.0, 0.0],
        "home_high":   [1.5, -1.0, -1.0, 0.5],
        "school_high": [-1.0, -1.0, 1.5, 0.0],
        "work_high":   [-1.0, 1.5, -1.0, 0.0],
        "other_high":  [-1.0, -1.0, 0.0, 1.5],
    }
    multi = []
    print(f"  {'init':>12}  {'NLL':>14}  {'R0':>6}  {'π (h, w, s, o)':>32}")
    for name, t0 in INITS.items():
        r = fit_free_4ch(sh_realloc, init_logit=t0)
        multi.append({"init": name, **r})
        print(f"  {name:>12}  {r['nll']:>14,.0f}  {r['R0']:>6.3f}  "
              f"{str([round(x, 3) for x in r['pi']]):>32}")
    nlls = [m["nll"] for m in multi]
    print(f"  NLL spread: {max(nlls) - min(nlls):.0f}  "
          f"({(max(nlls)-min(nlls))/abs(np.mean(nlls))*100:.4f}%)")
    home_pis = [m["pi"][0] for m in multi]
    print(f"  home π range across inits: {min(home_pis):.3f} – {max(home_pis):.3f}")

    # save
    out = "outputs/metapop/break_realloc_diagnostic.json"
    json.dump({"season": SEASON, "amp": AMP, "configs": results,
               "multi_start_realloc": multi}, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
