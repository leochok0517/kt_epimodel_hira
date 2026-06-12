"""TODO-4 diagnostic: seasonality amp source, identifiability, and cliff.

D1 source: amp=0.7 is a FIXED constant in DiseaseParameters (line 53,
"cosine amplitude (fixed)") and param_vector.py:14 ("Seasonality fixed
(cosine, amp=0.7, peak=105)"). numpyro_model does NOT sample amp.
→ amp is a hardcoded prior, not data-fit.

D2 shape: sf(t) = 1.0 + 0.7·cos(2π(t-105)/365). Range [0.3, 1.7].
D3 identifiability: refit (R0, simplex) at amp ∈ {0.3, 0.5, 0.7, 0.9}.
   Flat NLL → amp non-id (β absorbs); sharp minimum → data identifies amp.
D4 cliff: forward school_closure at each amp.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def build_gamma_15():
    K = 3.0 / (CHILD_REL + ADULT_REL + ELDER_REL)
    return np.concatenate([
        np.full(4, GAMMA_LEVEL * K * CHILD_REL),
        np.full(9, GAMMA_LEVEL * K * ADULT_REL),
        np.full(2, GAMMA_LEVEL * K * ELDER_REL),
    ])


def sf_curve(amp, base=1.0, peak_day=105.0, period=365.0):
    t = np.arange(period)
    return np.maximum(base + amp * np.cos(2.0 * np.pi * (t - peak_day) / period), 0.0)


def main():
    print("=" * 70)
    print(f"TODO-4 — seasonality amp diagnostic (season {SEASON})")
    print("=" * 70)

    print("\n[D1 source] amp = 0.7 is FIXED (model/parameters.py:53 comment "
          "'cosine amplitude (fixed)'); not sampled in numpyro_model.")

    print("\n[D2 shape] sf(t) profile at different amp (base=1.0, peak day=105)")
    amps = [0.3, 0.5, 0.7, 0.9]
    print(f"  {'amp':>5}  {'sf_min':>7}  {'sf_max':>7}  {'days sf>1.0':>11}  "
          f"{'days R_eff>1 at R0=1.9':>22}")
    R0_BASE = 1.90
    summary_d2 = []
    for amp in amps:
        sf = sf_curve(amp)
        days_sf_gt_1 = int((sf > 1.0).sum())
        # R_eff(t) = R0 * sf(t) / sf_peak; sf_peak = 1 + amp
        sf_peak = 1.0 + amp
        R_eff = R0_BASE * sf / sf_peak
        days_R_eff_gt_1 = int((R_eff > 1.0).sum())
        print(f"  {amp:>5.1f}  {sf.min():>7.3f}  {sf.max():>7.3f}  {days_sf_gt_1:>11}  "
              f"{days_R_eff_gt_1:>22}")
        summary_d2.append({"amp": amp, "sf_min": float(sf.min()),
                            "sf_max": float(sf.max()),
                            "days_sf_gt_1": days_sf_gt_1,
                            "days_R_eff_gt_1": days_R_eff_gt_1})

    # ─── setup loss + ngm at FIXED gamma ───
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

    def make_loss_and_ngm(amp_value):
        shared = dict(
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
            seasonality_amp=float(amp_value),
            seasonality_base=disease.seasonality_base,
            seasonality_peak_day=disease.seasonality_peak_day,
            seasonality_period=disease.seasonality_period,
        )
        loss_fn = make_multi_season_loss_fn(
            initial_states=[state0],
            obs_hira_list=[jnp.asarray(obs)],
            weights_hira_list=[jnp.asarray(w_obs)],
            shared_static=shared, n_weeks=nw,
            min_rate=0.01, discretize_time=False,
        )
        # Set NGM seasonal_factor = 1.0 + amp (peak) so R0 keeps physical meaning
        ngm = make_ngm_eigvalue_fn(
            pop_15=pop_15, rho=rho_emp,
            C_home=matrices["C_home"], C_work=matrices["C_work"],
            C_school=matrices["C_school"], C_other=matrices["C_other"],
            R0_immunity=R0_IMMUNITY_PROFILE,
            gamma=disease.gamma, seasonal_factor=1.0 + float(amp_value),
        )
        return loss_fn, ngm, shared

    # ─── D3: refit (R0, logit_pi3 with fixed work_ratio) at each amp ───
    print("\n[D3 identifiability] refit (log_R0, logit_pi3) at each amp "
          "(γ_level=0.6, φ=1.0, work_ratio=0.349)")

    def make_obj(loss_fn, ngm):
        # theta = [log_R0, logit3_home, logit3_school, logit3_wo]
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
        return obj_jit, grad_jit

    theta0 = np.array([np.log(1.9), 0.0, 0.0, 0.0])
    fit_results = []
    print(f"  {'amp':>5}  {'NLL':>14}  {'R0':>6}  {'π (h, w, s, o)':>40}")
    for amp in amps:
        loss_fn, ngm, shared = make_loss_and_ngm(amp)
        obj_jit, grad_jit = make_obj(loss_fn, ngm)
        def f_and_g(x):
            return float(obj_jit(jnp.asarray(x))), np.asarray(grad_jit(jnp.asarray(x)))
        res = minimize(
            f_and_g, theta0, jac=True, method="L-BFGS-B",
            bounds=[(np.log(0.7), np.log(3.5))] + [(-3.0, 3.0)] * 3,
            options={"maxiter": 200, "ftol": 1e-8, "gtol": 1e-5},
        )
        R0_opt = float(np.exp(res.x[0]))
        pi3_opt = np.asarray(jax.nn.softmax(jnp.asarray(res.x[1:4])))
        pi_full = [pi3_opt[0], pi3_opt[2] * WORK_RATIO,
                   pi3_opt[1], pi3_opt[2] * (1 - WORK_RATIO)]
        fit_results.append({
            "amp": amp, "nll": float(res.fun), "R0": R0_opt,
            "pi": [float(x) for x in pi_full],
            "theta": res.x.tolist(),
        })
        print(f"  {amp:>5.1f}  {res.fun:>14,.0f}  {R0_opt:>6.3f}  "
              f"{str([round(x, 2) for x in pi_full]):>40}")
    nlls = np.array([r["nll"] for r in fit_results])
    print(f"  NLL range across amps: {nlls.min():.0f} ~ {nlls.max():.0f}  "
          f"(spread {nlls.max() - nlls.min():.0f}, "
          f"{(nlls.max() - nlls.min())/abs(nlls.mean())*100:.3f}%)")
    print(f"  best amp: {fit_results[int(nlls.argmin())]['amp']}")

    # ─── D4: school_closure at each amp ───
    print("\n[D4 cliff sensitivity] school_closure (p_school=0.5) averted at each amp")
    print(f"  {'amp':>5}  {'R0_fit':>7}  {'baseline':>10}  {'school0.5':>10}  "
          f"{'averted':>8}  {'R_eff_peak':>10}")
    d4_table = []
    for r in fit_results:
        amp = r["amp"]
        loss_fn, ngm, shared = make_loss_and_ngm(amp)
        pi_full = jnp.asarray(r["pi"])
        beta_4 = derive_beta_from_R0_simplex(ngm, r["R0"], pi_full, phi_full)
        kw_base = dict(shared); kw_base["beta_h"] = float(beta_4[0]); kw_base["beta_w"] = float(beta_4[1])
        kw_base["beta_s"] = float(beta_4[2]); kw_base["beta_o"] = float(beta_4[3])
        kw_base["phi_susc"] = phi_full
        kw_base["p_school"] = 1.0; kw_base["p_work"] = 1.0
        st_base = simulate_jax(state0, **kw_base, discretize_time=False)
        inc_base = float(daily_new_infection_by_age_jax(st_base).sum())
        kw_sch = dict(kw_base); kw_sch["p_school"] = 0.5
        st_sch = simulate_jax(state0, **kw_sch, discretize_time=False)
        inc_sch = float(daily_new_infection_by_age_jax(st_sch).sum())
        averted = (inc_base - inc_sch) / max(inc_base, 1.0) * 100
        # R_eff_peak at school 0.5 = ρ_NGM with β_s × 0.5
        R_eff_pk = float(ngm(beta_4[0], beta_4[1], 0.5 * beta_4[2], beta_4[3], phi_full))
        print(f"  {amp:>5.1f}  {r['R0']:>7.3f}  {inc_base:>10,.0f}  {inc_sch:>10,.0f}  "
              f"{averted:>7.1f}%  {R_eff_pk:>10.3f}")
        d4_table.append({"amp": amp, "R0": r["R0"], "baseline_total": inc_base,
                          "school_total": inc_sch, "averted_pct": averted,
                          "R_eff_peak_sch05": R_eff_pk})

    # ─── plots ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t_year = np.arange(365)
    for amp in amps:
        axes[0].plot(t_year, sf_curve(amp), label=f"amp={amp}", lw=2)
    axes[0].axhline(1.0, color="grey", ls="--", lw=1)
    axes[0].set_xlabel("day of year"); axes[0].set_ylabel("seasonal factor sf(t)")
    axes[0].set_title("Seasonality cosine — sf(t) = 1.0 + amp · cos(2π(t-105)/365)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    amp_arr = np.array([r["amp"] for r in d4_table])
    av_arr = np.array([r["averted_pct"] for r in d4_table])
    axes[1].plot(amp_arr, av_arr, "o-", color="#1a5490", lw=2, markersize=8)
    for r in d4_table:
        axes[1].annotate(f" R_eff={r['R_eff_peak_sch05']:.2f}",
                          (r["amp"], r["averted_pct"]), fontsize=9)
    axes[1].set_xlabel("seasonality amp")
    axes[1].set_ylabel("school_closure (p=0.5) averted %")
    axes[1].set_title("school_closure cliff sensitivity to amp")
    axes[1].set_ylim(0, 105); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = "presentations/figures/seasonality_amp_diagnostic.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")

    import json
    json.dump({
        "season": SEASON,
        "D1_source": "FIXED at 0.7 in model/parameters.py:53 (cosine amplitude, fixed). "
                      "param_vector.py:14: 'Seasonality fixed (cosine, amp=0.7, peak=105)'. "
                      "Not sampled by numpyro_model.",
        "D2_shape": summary_d2,
        "D3_identifiability_NLL": fit_results,
        "D4_cliff": d4_table,
    }, open("outputs/metapop/seasonality_diagnostic.json", "w"), indent=2, default=float)
    print(f"saved outputs/metapop/seasonality_diagnostic.json")


if __name__ == "__main__":
    main()
