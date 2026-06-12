"""Channel-ratio field-knowledge prior — 4 combos (NIMS vs literature × strong/weak).

Closes the home/work non-id by adding a Gaussian prior on the per-season
4-channel logit, centered on a target π derived from external knowledge.

Two evidence bases (R0-contribution shares):
- A (NIMS): contact row-sum → corrected by per-channel unit R0
  (contact share home 0.270, work 0.197, school 0.165, other 0.368;
  unit R0 h=8.70, w=6.21, s=25.40, o=9.33 at φ=1, sf=1.7).
  → π_A ∝ contact_share / unit_R0 → π_A ≈ [0.286, 0.292, 0.060, 0.363]
- B (literature, Italy 2009 H1N1 etc.): R0-contribution {home 0.40,
  work 0.10, school 0.27, other 0.23}. unit R0 conversion gives
  π_B ≈ [0.472, 0.165, 0.109, 0.253].

Two strengths:
- strong: tight (σ_logit = 0.1) on all 4 channels
- weak:   tight on home/work (σ=0.1), loose on school/other (σ=1.0)

Per-season fit (4 seasons joint) under holiday ON + realloc=1, amp=0.9,
γ CDC, φ=1.0.
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


SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
AMP = 0.9
HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)

# Unit R0 per channel (φ=1, sf=1.7) — from earlier diagnostic
UNIT_R0 = np.array([8.70, 6.21, 25.40, 9.33])   # (h, w, s, o)


def build_target_A():
    contact = np.array([23.26, 17.03, 14.24, 31.77])
    contact_share = contact / contact.sum()
    beta_share = contact_share / UNIT_R0
    pi = beta_share / beta_share.sum()
    return pi, contact_share


def build_target_B():
    R0_contrib = np.array([0.40, 0.10, 0.27, 0.23])
    R0_contrib = R0_contrib / R0_contrib.sum()
    beta_share = R0_contrib / UNIT_R0
    pi = beta_share / beta_share.sum()
    return pi, R0_contrib


def logit_target(pi):
    lp = np.log(np.clip(pi, 1e-6, None))
    return lp - lp.mean()


def main():
    pi_A, share_A = build_target_A()
    pi_B, share_B = build_target_B()
    print("=" * 70)
    print("Channel-ratio prior — 4 combos × policy forward")
    print("=" * 70)
    print(f"  target A (NIMS-derived R0 contrib): π = "
          f"{[round(float(x), 3) for x in pi_A]}  (R0 share {[round(float(x), 3) for x in share_A]})")
    print(f"  target B (literature R0 contrib):    π = "
          f"{[round(float(x), 3) for x in pi_B]}  (R0 share {[round(float(x), 3) for x in share_B]})")

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
    shared.update(HOLIDAY)

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

    def make_obj(target_pi, sigma_per_channel):
        logit_t = jnp.asarray(logit_target(target_pi))
        sigma = jnp.asarray(sigma_per_channel)
        inv_var = 1.0 / (sigma ** 2)

        def obj(theta):
            log_R0 = theta[:4]
            logit_blocks = theta[4:].reshape(4, 4)
            R0 = jnp.exp(log_R0)
            pi_per_season = jax.nn.softmax(logit_blocks, axis=-1)
            betas = []
            for si in range(4):
                betas.append(derive_beta_from_R0_simplex(
                    ngm, R0[si], pi_per_season[si], phi_full))
            beta_16 = jnp.concatenate(betas)
            vec_33 = jnp.concatenate([phi_14, gamma_3, beta_16])
            data_nll = loss_fn(vec_33)
            # Penalty: centered-logit deviations from target per season per channel
            centered = logit_blocks - jnp.mean(logit_blocks, axis=-1, keepdims=True)
            dev = centered - logit_t[None, :]                       # (4, 4)
            penalty = 0.5 * jnp.sum(dev * dev * inv_var[None, :])   # broadcast
            return data_nll + penalty

        return jax.jit(obj), jax.jit(jax.grad(obj))

    def fit_combo(target_pi, sigmas):
        obj_jit, grad_jit = make_obj(target_pi, sigmas)
        def fg(x):
            xj = jnp.asarray(x)
            return float(obj_jit(xj)), np.asarray(grad_jit(xj))
        lt = logit_target(target_pi)
        theta0 = np.concatenate([
            np.array([np.log(2.0)] * 4),
            np.tile(lt, 4),
        ])
        bounds = ([(np.log(0.7), np.log(3.5))] * 4
                  + [(-4.0, 4.0)] * 16)
        res = minimize(fg, theta0, jac=True, method="L-BFGS-B",
                        bounds=bounds, options={"maxiter": 500,
                                                "ftol": 1e-9, "gtol": 1e-6})
        log_R0 = res.x[:4]; logit_blocks = res.x[4:].reshape(4, 4)
        R0 = np.exp(log_R0)
        pi = np.asarray(jax.nn.softmax(jnp.asarray(logit_blocks), axis=-1))
        return {"total_obj": float(res.fun), "R0": R0.tolist(), "pi": pi.tolist()}

    # σ on centered logits — much tighter than before; previous σ=0.1 was
    # totally dominated by the data NLL (penalty ~400 vs data ~28M).
    SIG_STRONG = [0.01, 0.01, 0.01, 0.01]               # ≈ pin all channels at target
    SIG_WEAK   = [0.01, 0.01, 0.30, 0.30]               # pin home/work, free school/other
    COMBOS = {
        "A_strong": (pi_A, SIG_STRONG),
        "A_weak":   (pi_A, SIG_WEAK),
        "B_strong": (pi_B, SIG_STRONG),
        "B_weak":   (pi_B, SIG_WEAK),
    }
    results = {}
    print(f"\n[fits] (per-season π averaged across 4 seasons shown)")
    print(f"  {'combo':>10}  {'home':>5} {'work':>5} {'sch':>5} {'oth':>5}  "
          f"{'R0 mean':>8}  {'total obj':>14}")
    for name, (target, sigmas) in COMBOS.items():
        r = fit_combo(target, sigmas)
        pi_arr = np.array(r["pi"])     # (4 season, 4 channel)
        pi_mean = pi_arr.mean(axis=0)
        R0_mean = float(np.mean(r["R0"]))
        results[name] = r
        print(f"  {name:>10}  {pi_mean[0]:>5.2f} {pi_mean[1]:>5.2f} "
              f"{pi_mean[2]:>5.2f} {pi_mean[3]:>5.2f}  {R0_mean:>8.3f}  "
              f"{r['total_obj']:>14,.0f}")

    # ─── Policy forward at 2019-20 for each combo ───
    print(f"\n[policy forward] 2019-20 baseline / sick_leave (w=0.4) / school (p=0.5)")
    state0_2019 = initial_states[2]

    def fwd(beta_4, p_school, p_work):
        kw = dict(shared)
        kw["beta_h"] = float(beta_4[0]); kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2]); kw["beta_o"] = float(beta_4[3])
        kw["phi_susc"] = phi_full
        kw["p_school"] = float(p_school); kw["p_work"] = float(p_work)
        st = simulate_jax(state0_2019, **kw, discretize_time=False)
        return float(daily_new_infection_by_age_jax(st).sum())

    print(f"  {'combo':>10}  {'baseline':>10}  {'sick_av':>8}  {'school_av':>10}")
    pol_results = []
    for name, r in results.items():
        pi_2019 = jnp.asarray(r["pi"][2])
        R0_2019 = float(r["R0"][2])
        beta = derive_beta_from_R0_simplex(ngm, R0_2019, pi_2019, phi_full)
        inc_b = fwd(beta, 1.0, 1.0)
        inc_si = fwd(beta, 1.0, 0.4)
        inc_sc = fwd(beta, 0.5, 1.0)
        si_av = (inc_b - inc_si) / max(inc_b, 1.0) * 100
        sc_av = (inc_b - inc_sc) / max(inc_b, 1.0) * 100
        pol_results.append({"combo": name, "baseline": inc_b,
                             "sick_av": si_av, "school_av": sc_av,
                             "R0_2019": R0_2019,
                             "pi_2019": list(map(float, pi_2019.tolist()))})
        print(f"  {name:>10}  {inc_b:>10,.0f}  {si_av:>+7.1f}%  {sc_av:>+9.1f}%")

    # Range across combos
    si_arr = np.array([p["sick_av"] for p in pol_results])
    sc_arr = np.array([p["school_av"] for p in pol_results])
    print(f"\n  sick_leave averted range:    {si_arr.min():+.1f}% ~ {si_arr.max():+.1f}%  "
          f"(span {si_arr.max() - si_arr.min():.1f}%)")
    print(f"  school_closure averted range: {sc_arr.min():.1f}% ~ {sc_arr.max():.1f}%  "
          f"(span {sc_arr.max() - sc_arr.min():.1f}%)")

    out = "outputs/metapop/channel_prior_4combos.json"
    json.dump({
        "seasons": SEASONS, "amp": AMP, "gamma": "CDC absolute",
        "target_A_pi": [float(x) for x in pi_A],
        "target_B_pi": [float(x) for x in pi_B],
        "unit_R0": UNIT_R0.tolist(),
        "fits": results, "policy": pol_results,
    }, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
