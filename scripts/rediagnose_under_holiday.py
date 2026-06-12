"""Re-diagnose identifiability under the winter-break (holiday) model.

Holiday ON improved NLL by ~400K and flipped amp/channel preferences. The
fixes we put in (γ level=0.6, work:other 0.349, amp 0.7) were diagnosed
without modeling the holiday. Question: which of those fixes were patching
the holiday-misspec symptom?

D1+D3. Free 4-channel simplex (work:other unfixed), multi-start
       (4 inits) under holiday ON → tests work-other identifiability
       and home-π robustness.
D2. Holiday ON × γ regime: CDC absolute (0.40/0.18/0.25) vs level=0.6
    (0.87/0.39/0.54). Compares home π and NLL.
D4. Holiday ON × amp free (L-BFGS over log_R0, logit_pi3, log_amp).
    amp prior was identified in TODO-4 (preferred 0.7 under holiday).
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
HOLIDAY_PARAMS = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
)


def gamma_level06_relative():
    """level 0.6 × CDC relative shape (unweighted mean 1)."""
    K = 3.0 / (1.45 + 0.65 + 0.90)
    g_c = 0.6 * K * 1.45
    g_a = 0.6 * K * 0.65
    g_e = 0.6 * K * 0.90
    return np.concatenate([np.full(4, g_c), np.full(9, g_a), np.full(2, g_e)])


def gamma_cdc_absolute():
    """CDC absolute (0.40, 0.18, 0.25)."""
    return np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])


def main():
    print("=" * 70)
    print(f"Re-diagnostic under winter-break model ({SEASON})")
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

    def make_shared(amp):
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
        )
        d.update(HOLIDAY_PARAMS)
        return d

    def make_loss(gamma_15, shared):
        loss_fn = make_multi_season_loss_fn(
            initial_states=[state0],
            obs_hira_list=[jnp.asarray(obs)],
            weights_hira_list=[jnp.asarray(w_obs)],
            shared_static=shared, n_weeks=nw,
            min_rate=0.01, discretize_time=False,
        )
        gamma_3 = jnp.array([float(gamma_15[0]), float(gamma_15[4]), float(gamma_15[13])])
        return loss_fn, gamma_3

    def ngm_at(amp):
        return make_ngm_eigvalue_fn(
            pop_15=pop_15, rho=rho_emp,
            C_home=matrices["C_home"], C_work=matrices["C_work"],
            C_school=matrices["C_school"], C_other=matrices["C_other"],
            R0_immunity=R0_IMMUNITY_PROFILE,
            gamma=disease.gamma, seasonal_factor=1.0 + float(amp),
        )

    def obj_4ch_softmax(theta, loss_fn, ngm, gamma_3):
        """theta = [log_R0, logit4_h, logit4_w, logit4_s, logit4_o].
        Full 4-channel free simplex."""
        log_R0 = theta[0]
        logit4 = theta[1:5]
        R0 = jnp.exp(log_R0)
        pi = jax.nn.softmax(logit4)
        beta_4 = derive_beta_from_R0_simplex(ngm, R0, pi, phi_full)
        beta_16 = jnp.concatenate([beta_4, jnp.zeros(12)])
        vec_33 = jnp.concatenate([phi_14, gamma_3, beta_16])
        return loss_fn(vec_33)

    def fit_4ch(amp, gamma_15, theta0):
        shared = make_shared(amp)
        loss_fn, gamma_3 = make_loss(gamma_15, shared)
        ngm = ngm_at(amp)
        obj_jit = jax.jit(lambda t: obj_4ch_softmax(t, loss_fn, ngm, gamma_3))
        grad_jit = jax.jit(jax.grad(lambda t: obj_4ch_softmax(t, loss_fn, ngm, gamma_3)))

        def fg(x):
            xj = jnp.asarray(x)
            return float(obj_jit(xj)), np.asarray(grad_jit(xj))

        res = minimize(
            fg, theta0, jac=True, method="L-BFGS-B",
            bounds=[(np.log(0.7), np.log(3.5))] + [(-4.0, 4.0)] * 4,
            options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-6},
        )
        R0_opt = float(np.exp(res.x[0]))
        pi_opt = np.asarray(jax.nn.softmax(jnp.asarray(res.x[1:5])))
        return {
            "amp": amp, "nll": float(res.fun), "R0": R0_opt,
            "pi": [float(x) for x in pi_opt],
        }

    # ─── D1+D3: free 4-channel × multi-start ───
    print("\n[D1+D3] free 4-channel simplex (work:other unfixed), "
          "multi-start (4 inits), holiday ON, γ level=0.6")
    gamma_15 = gamma_level06_relative()
    INITS = {
        "balanced":    [np.log(1.85), 0.0, 0.0, 0.0, 0.0],
        "home_high":   [np.log(1.85), 1.5, -1.0, -1.0, 0.5],
        "school_high": [np.log(1.85), -1.0, -1.0, 1.5, 0.0],
        "work_high":   [np.log(1.85), -1.0, 1.5, -1.0, 0.0],
        "other_high":  [np.log(1.85), -1.0, -1.0, 0.0, 1.5],
    }
    print(f"  {'init':>12}  {'NLL':>14}  {'R0':>6}  {'π (h, w, s, o)':>32}")
    multi = []
    for name, t0 in INITS.items():
        r = fit_4ch(0.7, gamma_15, np.array(t0))
        multi.append({"init": name, **r})
        print(f"  {name:>12}  {r['nll']:>14,.0f}  {r['R0']:>6.3f}  "
              f"{str([round(x, 3) for x in r['pi']]):>32}")
    nlls_multi = [m["nll"] for m in multi]
    print(f"  NLL spread across inits: {max(nlls_multi)-min(nlls_multi):.0f}  "
          f"({(max(nlls_multi)-min(nlls_multi))/abs(np.mean(nlls_multi))*100:.3f}%)")
    best_multi = sorted(multi, key=lambda r: r["nll"])[0]
    print(f"  best init: {best_multi['init']}, π = {[round(x, 3) for x in best_multi['pi']]}")

    # ─── D2: γ regime under holiday ON ───
    print("\n[D2] γ regime (holiday ON, amp=0.7, work:other FREE)")
    print(f"  {'γ regime':>15}  {'γ (c, a, e)':>20}  {'NLL':>14}  {'R0':>6}  "
          f"{'π (h, w, s, o)':>32}")
    d2 = []
    for name, g_15 in [("CDC absolute", gamma_cdc_absolute()),
                      ("level=0.6 × CDC", gamma_level06_relative())]:
        r = fit_4ch(0.7, g_15, np.array([np.log(1.85), 0.0, 0.0, 0.0, 0.0]))
        d2.append({"regime": name, "gamma": [float(g_15[0]), float(g_15[4]), float(g_15[13])], **r})
        print(f"  {name:>15}  {str([round(float(g_15[i]), 2) for i in (0, 4, 13)]):>20}  "
              f"{r['nll']:>14,.0f}  {r['R0']:>6.3f}  "
              f"{str([round(x, 3) for x in r['pi']]):>32}")

    # ─── D4: amp free (holiday ON, γ level=0.6, work:other free) ───
    print("\n[D4] amp free (holiday ON, work:other free, γ level=0.6)")
    # We can't easily JIT make_loss_fn with variable amp, so reuse:
    # fit at a grid of amp, take min — coarse but tells the answer
    print(f"  {'amp':>5}  {'NLL':>14}  {'R0':>6}  {'π (h, w, s, o)':>32}")
    amp_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    d4 = []
    for amp in amp_grid:
        r = fit_4ch(amp, gamma_15, np.array([np.log(1.85), 0.0, 0.0, 0.0, 0.0]))
        d4.append({"amp": amp, **r})
        print(f"  {amp:>5.1f}  {r['nll']:>14,.0f}  {r['R0']:>6.3f}  "
              f"{str([round(x, 3) for x in r['pi']]):>32}")
    nlls_d4 = [r["nll"] for r in d4]
    best_amp = amp_grid[int(np.argmin(nlls_d4))]
    print(f"  best amp under holiday ON + work:other FREE: {best_amp}  "
          f"(NLL {min(nlls_d4):,.0f})")
    print(f"  NLL spread across amp: {max(nlls_d4) - min(nlls_d4):.0f}")

    out = "outputs/metapop/rediagnose_under_holiday.json"
    json.dump({
        "season": SEASON,
        "D1_D3_multi_start_free4ch": multi,
        "D2_gamma_regime": d2,
        "D4_amp_free_grid": d4,
        "best_amp_under_holiday": best_amp,
    }, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
