"""TODO-4: test winter-break (school contact time-dependent) on 2019-20.

For each (amp ∈ {0.3, 0.5, 0.7}, holiday ∈ {OFF, ON}) refit (log_R0,
logit_pi3) under the work:other 0.349 reparam at γ_level=0.6, φ=1.0.
Tests three predictions:
  (1) school π drops when holiday is modelled
  (2) data-preferred amp rises (holiday absorbs the natural-experiment drop
      that amp was reaching for)
  (3) NLL improves under holiday-on
And re-evaluates the school_closure cliff under the holiday model.
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
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


SEASON = "2019-2020"
GAMMA_LEVEL = 0.6
CHILD_REL, ADULT_REL, ELDER_REL = 1.45, 0.65, 0.90
WORK_RATIO = 17.03 / (17.03 + 31.77)

# Korean winter break for the 2019-20 season (t=0 ≈ Sept 1, 2019)
HOLIDAY_PARAMS = dict(
    school_holiday_start_day=113.0,     # Dec 23, 2019
    school_holiday_min_start_day=127.0, # Jan 6, 2020
    school_holiday_min_end_day=162.0,   # Feb 10, 2020
    school_holiday_end_day=183.0,       # Mar 2, 2020
)
HOLIDAY_DROP = 0.7   # 1.0 → 0.3 during plateau (residual 30% via after-school/care)


def build_gamma_15():
    K = 3.0 / (CHILD_REL + ADULT_REL + ELDER_REL)
    return np.concatenate([
        np.full(4, GAMMA_LEVEL * K * CHILD_REL),
        np.full(9, GAMMA_LEVEL * K * ADULT_REL),
        np.full(2, GAMMA_LEVEL * K * ELDER_REL),
    ])


def main():
    print("=" * 70)
    print(f"TODO-4 — winter-break test ({SEASON}, holiday drop {HOLIDAY_DROP})")
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
    gamma_15 = build_gamma_15()
    gamma_3 = jnp.array([float(gamma_15[0]), float(gamma_15[4]), float(gamma_15[13])])

    def make_shared(amp, holiday):
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
            vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
            seasonality_amp=float(amp),
            seasonality_base=disease.seasonality_base,
            seasonality_peak_day=disease.seasonality_peak_day,
            seasonality_period=disease.seasonality_period,
            school_holiday_amp=float(HOLIDAY_DROP if holiday else 0.0),
        )
        if holiday:
            d.update(HOLIDAY_PARAMS)
        return d

    def fit_one(amp, holiday):
        shared = make_shared(amp, holiday)
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
            log_R0 = theta[0]; logit3 = theta[1:4]
            R0 = jnp.exp(log_R0)
            pi3 = jax.nn.softmax(logit3)
            pi_full = jnp.array([
                pi3[0],
                pi3[2] * WORK_RATIO,
                pi3[1],
                pi3[2] * (1.0 - WORK_RATIO),
            ])
            beta_4 = derive_beta_from_R0_simplex(ngm, R0, pi_full, phi_full)
            beta_16 = jnp.concatenate([beta_4, jnp.zeros(12)])
            vec_33 = jnp.concatenate([phi_14, gamma_3, beta_16])
            return loss_fn(vec_33)

        obj_jit = jax.jit(obj)
        grad_jit = jax.jit(jax.grad(obj))

        def fg(x):
            xj = jnp.asarray(x)
            return float(obj_jit(xj)), np.asarray(grad_jit(xj))

        res = minimize(
            fg, np.array([np.log(1.9), 0.0, 0.0, 0.0]),
            jac=True, method="L-BFGS-B",
            bounds=[(np.log(0.7), np.log(3.5))] + [(-3.0, 3.0)] * 3,
            options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-5},
        )
        R0_opt = float(np.exp(res.x[0]))
        pi3_opt = np.asarray(jax.nn.softmax(jnp.asarray(res.x[1:4])))
        pi_full = [pi3_opt[0], pi3_opt[2] * WORK_RATIO,
                   pi3_opt[1], pi3_opt[2] * (1 - WORK_RATIO)]
        return {
            "amp": amp, "holiday": holiday, "nll": float(res.fun),
            "R0": R0_opt, "pi": [float(x) for x in pi_full],
            "theta": res.x.tolist(), "shared": shared, "ngm_built": ngm,
        }

    # ─── Sweep ───
    print(f"\n{'amp':>5} {'holiday':>8}  {'NLL':>14}  {'R0':>6}  "
          f"{'π (h, w, s, o)':>32}")
    print("-" * 80)
    fits = []
    for amp in (0.3, 0.5, 0.7):
        for hol in (False, True):
            r = fit_one(amp, hol)
            fits.append(r)
            print(f"{amp:>5.1f} {('ON' if hol else 'OFF'):>8}  {r['nll']:>14,.0f}  "
                  f"{r['R0']:>6.3f}  "
                  f"{str([round(x, 3) for x in r['pi']]):>32}")

    # ─── Per-amp comparison ───
    print("\n[Step 3] ΔNLL (holiday ON − OFF), Δπ_school, ΔR0")
    print(f"  {'amp':>5}  {'ΔNLL':>12}  {'Δπ_school':>11}  {'Δπ_work':>9}  {'ΔR0':>7}")
    summary = []
    for amp in (0.3, 0.5, 0.7):
        off = [f for f in fits if f["amp"] == amp and not f["holiday"]][0]
        on  = [f for f in fits if f["amp"] == amp and f["holiday"]][0]
        d_nll = on["nll"] - off["nll"]
        d_sch = on["pi"][2] - off["pi"][2]
        d_wrk = on["pi"][1] - off["pi"][1]
        d_R0 = on["R0"] - off["R0"]
        print(f"  {amp:>5.1f}  {d_nll:>+12,.0f}  {d_sch:>+11.3f}  "
              f"{d_wrk:>+9.3f}  {d_R0:>+7.3f}")
        summary.append({"amp": amp, "delta_nll": d_nll, "delta_pi_school": d_sch,
                         "delta_pi_work": d_wrk, "delta_R0": d_R0})

    # ─── Best amp under each regime ───
    nll_off = [f["nll"] for f in fits if not f["holiday"]]
    nll_on  = [f["nll"] for f in fits if f["holiday"]]
    amps = [0.3, 0.5, 0.7]
    print(f"\n  best amp (NLL): OFF → {amps[int(np.argmin(nll_off))]}   "
          f"ON → {amps[int(np.argmin(nll_on))]}")
    print(f"  NLL spread OFF: {max(nll_off) - min(nll_off):.0f}  "
          f"ON: {max(nll_on) - min(nll_on):.0f}")

    # ─── Step 4: school_closure cliff with holiday model (at amp=0.7, holiday ON) ───
    print("\n[Step 4] school_closure (p=0.5) under holiday model "
          "(amp=0.7, holiday ON best fit)")
    best_on = sorted([f for f in fits if f["holiday"]], key=lambda r: r["nll"])[0]
    print(f"  best holiday-ON fit: amp={best_on['amp']}, "
          f"R0={best_on['R0']:.3f}, π={[round(x, 3) for x in best_on['pi']]}")
    ngm = best_on["ngm_built"]
    pi_full = jnp.asarray(best_on["pi"])
    beta_4 = derive_beta_from_R0_simplex(ngm, best_on["R0"], pi_full, phi_full)
    shared_base = best_on["shared"]

    def fwd(p_school):
        kw = dict(shared_base)
        kw["beta_h"] = float(beta_4[0]); kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2]); kw["beta_o"] = float(beta_4[3])
        kw["phi_susc"] = phi_full
        kw["p_school"] = float(p_school); kw["p_work"] = 1.0
        st = simulate_jax(state0, **kw, discretize_time=False)
        return float(daily_new_infection_by_age_jax(st).sum())

    inc_base = fwd(1.0)
    cliff_table = []
    for p in (0.3, 0.5, 0.7, 0.9):
        inc = fwd(p)
        averted = (inc_base - inc) / max(inc_base, 1.0) * 100
        print(f"  p_school = {p}: total {inc:>10,.0f}  averted {averted:>5.1f}%")
        cliff_table.append({"p_school": p, "total": inc, "averted_pct": averted})

    out = "outputs/metapop/winter_break_diagnostic.json"
    json.dump({
        "season": SEASON, "holiday_drop": HOLIDAY_DROP,
        "holiday_params": HOLIDAY_PARAMS,
        "fits": [{k: v for k, v in f.items() if k not in ("shared", "ngm_built")} for f in fits],
        "summary": summary,
        "cliff_table_under_holiday": cliff_table,
    }, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
