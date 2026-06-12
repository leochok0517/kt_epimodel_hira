"""Diagnose why home π = 0 under winter-break model.

Two competing causes:
  (a) Winter break absorption — home 0.27 (under no-holiday) was the
      "students at home → household transmission" signal that the
      holiday model now explains directly.
  (b) Home-other non-identifiability — home and other have similar
      all-age contact patterns; data lets the optimizer route adult
      transmission to whichever channel is "more efficient."

(a) → home is genuinely small; fill via household-attack-rate literature.
(b) → fix home:other ratio externally, like work:other was fixed.

D1. Contact pattern overlap (home vs other age signatures).
D2. NLL when home is forced on vs current optimum.
D3. Single-channel forward — does home produce a distinct age curve?
D4. Comparison of home π under (holiday OFF, γ CDC) vs (holiday ON, γ CDC)
    at free 4-channel — did holiday introduction itself drop home to 0?
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
from kt_data.data.load_contact import load_contact_matrices
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
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
AMP = 0.7
R0_TARGET = 1.85
HOLIDAY_PARAMS_ON = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
)
HOLIDAY_PARAMS_OFF = dict(school_holiday_amp=0.0)


def setup():
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
    return dict(
        pop_15=pop_15, matrices=matrices, mobility=mobility, rho_emp=rho_emp,
        disease=disease, vax=vax, policy=policy,
        state0=state0, obs=obs, w_obs=w_obs, nw=nw,
    )


def shared_with(s, *, amp, holiday: bool):
    d = dict(
        C_home=jnp.asarray(s["matrices"]["C_home"]),
        C_school=jnp.asarray(s["matrices"]["C_school"]),
        C_work=jnp.asarray(s["matrices"]["C_work"]),
        C_other=jnp.asarray(s["matrices"]["C_other"]),
        M_home=jnp.asarray(s["mobility"]["home"]),
        M_school=jnp.asarray(s["mobility"]["school"]),
        M_work=jnp.asarray(s["mobility"]["work"]),
        M_other=jnp.asarray(s["mobility"]["other"]),
        pop_15=jnp.asarray(s["pop_15"]), rho=jnp.asarray(s["rho_emp"]),
        kappa=jnp.asarray(s["disease"].kappa_array),
        sigma=s["disease"].sigma, gamma=s["disease"].gamma,
        p_school=s["policy"].p_school, p_work=s["policy"].p_work,
        VE=s["vax"].VE,
        annual_coverage=jnp.asarray(s["vax"].annual_coverage),
        vax_peak_iso_week=s["vax"].peak_iso_week,
        vax_spread_weeks=s["vax"].spread_weeks,
        seasonality_amp=float(amp),
        seasonality_base=s["disease"].seasonality_base,
        seasonality_peak_day=s["disease"].seasonality_peak_day,
        seasonality_period=s["disease"].seasonality_period,
    )
    d.update(HOLIDAY_PARAMS_ON if holiday else HOLIDAY_PARAMS_OFF)
    return d


def loss_at(s, gamma_15, shared):
    loss_fn = make_multi_season_loss_fn(
        initial_states=[s["state0"]],
        obs_hira_list=[jnp.asarray(s["obs"])],
        weights_hira_list=[jnp.asarray(s["w_obs"])],
        shared_static=shared, n_weeks=s["nw"],
        min_rate=0.01, discretize_time=False,
    )
    gamma_3 = jnp.array([float(gamma_15[0]), float(gamma_15[4]), float(gamma_15[13])])
    return loss_fn, gamma_3


def ngm_at(s, amp):
    return make_ngm_eigvalue_fn(
        pop_15=s["pop_15"], rho=s["rho_emp"],
        C_home=s["matrices"]["C_home"], C_work=s["matrices"]["C_work"],
        C_school=s["matrices"]["C_school"], C_other=s["matrices"]["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=s["disease"].gamma, seasonal_factor=1.0 + float(amp),
    )


def fit_free_4ch(s, gamma_15, *, amp, holiday: bool):
    shared = shared_with(s, amp=amp, holiday=holiday)
    loss_fn, gamma_3 = loss_at(s, gamma_15, shared)
    ngm = ngm_at(s, amp)
    phi_14 = jnp.ones(14); phi_full = jnp.ones(15)

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
    res = minimize(fg, np.array([np.log(1.85), 0., 0., 0., 0.]),
                    jac=True, method="L-BFGS-B",
                    bounds=[(np.log(0.7), np.log(3.5))] + [(-4.0, 4.0)] * 4,
                    options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-6})
    return {"nll": float(res.fun), "R0": float(np.exp(res.x[0])),
            "pi": [float(x) for x in jax.nn.softmax(jnp.asarray(res.x[1:5]))]}


def main():
    print("=" * 70)
    print(f"home=0 root-cause diagnostic ({SEASON})")
    print("=" * 70)
    s = setup()

    # ─── D1: contact age structure side-by-side ───
    print("\n[D1] contact rowsum by age — home vs other (all-age) "
          "and school/work")
    C = load_contact_matrices()
    rs_home = C["C_home"].sum(axis=1)
    rs_other = C["C_other"].sum(axis=1)
    rs_work = C["C_work"].sum(axis=1)
    rs_school = C["C_school"].sum(axis=1)
    print(f"  {'age':>4}  {'home':>7}  {'other':>7}  {'home/other':>10}  "
          f"{'school':>7}  {'work':>7}")
    for a in range(15):
        ratio = rs_home[a] / max(rs_other[a], 1e-3)
        print(f"  {a:>4}  {rs_home[a]:>7.2f}  {rs_other[a]:>7.2f}  "
              f"{ratio:>10.2f}  {rs_school[a]:>7.2f}  {rs_work[a]:>7.2f}")
    corr = float(np.corrcoef(rs_home, rs_other)[0, 1])
    print(f"  Pearson corr(home_rowsum, other_rowsum) = {corr:.3f}")

    # ─── D2: NLL with home forced on ───
    print("\n[D2] NLL at fixed π — home forced (holiday ON, γ CDC, amp=0.7)")
    shared = shared_with(s, amp=AMP, holiday=True)
    loss_fn, gamma_3 = loss_at(s, GAMMA_CDC, shared)
    ngm = ngm_at(s, AMP)
    phi_14 = jnp.ones(14); phi_full = jnp.ones(15)

    def nll_at_pi(pi_raw, R0):
        pi = jnp.asarray(pi_raw); pi = pi / pi.sum()
        beta_4 = derive_beta_from_R0_simplex(ngm, R0, pi, phi_full)
        beta_16 = jnp.concatenate([beta_4, jnp.zeros(12)])
        return float(loss_fn(jnp.concatenate([phi_14, gamma_3, beta_16])))

    # current optimum (under holiday ON + CDC + free): re-fit
    opt = fit_free_4ch(s, GAMMA_CDC, amp=AMP, holiday=True)
    print(f"  free optimum: π={[round(x, 3) for x in opt['pi']]}  R0={opt['R0']:.3f}  "
          f"NLL={opt['nll']:,.0f}")
    base_nll = opt["nll"]
    configs = [
        ("home_0.10", [0.10, 0.0, opt["pi"][2], 1.0 - 0.10 - opt["pi"][2]]),
        ("home_0.20", [0.20, 0.0, opt["pi"][2], 1.0 - 0.20 - opt["pi"][2]]),
        ("home_0.30", [0.30, 0.0, opt["pi"][2], 1.0 - 0.30 - opt["pi"][2]]),
        ("home_0.40", [0.40, 0.0, opt["pi"][2], 1.0 - 0.40 - opt["pi"][2]]),
    ]
    print(f"  {'config':>12}  {'pi':>32}  {'NLL':>14}  {'ΔNLL':>10}")
    d2_table = []
    for name, pi_raw in configs:
        # cap negative other
        pi = np.array(pi_raw, dtype=float)
        if (pi < 0).any():
            print(f"  {name:>12}  skipped (other would be negative)")
            continue
        nll = nll_at_pi(pi, opt["R0"])
        d = nll - base_nll
        print(f"  {name:>12}  {str([round(x, 3) for x in pi.tolist()]):>32}  "
              f"{nll:>14,.0f}  {d:>+10,.0f}")
        d2_table.append({"config": name, "pi": pi.tolist(), "nll": nll, "delta": d})

    # ─── D3: single-channel forward — home vs other age curve ───
    print("\n[D3] single-channel forward — home vs other age signatures "
          "(holiday ON, R0=1.85)")
    def forward(pi_raw):
        pi = jnp.asarray(pi_raw); pi = pi / pi.sum()
        beta_4 = derive_beta_from_R0_simplex(ngm, R0_TARGET, pi, phi_full)
        kw = dict(shared)
        kw["beta_h"] = float(beta_4[0]); kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2]); kw["beta_o"] = float(beta_4[3])
        kw["phi_susc"] = phi_full
        states = simulate_jax(s["state0"], **kw, discretize_time=False)
        return np.asarray(daily_new_infection_by_age_jax(states))

    inc_home_only = forward([1.0, 0.0, 0.0, 0.0])
    inc_other_only = forward([0.0, 0.0, 0.0, 1.0])
    inc_school_only = forward([0.0, 0.0, 1.0, 0.0])

    print(f"  {'channel':>14}  {'total':>10}  "
          + "".join(f"a{a:02d}".rjust(7) for a in (0, 2, 4, 6, 8, 10, 13)))
    for name, inc in [("home_only", inc_home_only),
                       ("other_only", inc_other_only),
                       ("school_only", inc_school_only)]:
        peak = inc.max(axis=0)
        print(f"  {name:>14}  {inc.sum():>10,.0f}  "
              + "".join(f"{peak[a]:>6,.0f}" for a in (0, 2, 4, 6, 8, 10, 13)))

    # Compute Pearson correlation of age-peak vectors
    corr_age = float(np.corrcoef(inc_home_only.max(axis=0),
                                  inc_other_only.max(axis=0))[0, 1])
    print(f"  corr(age-peak home vs other): {corr_age:.3f}  "
          f"(high → similar age signature → harder to separate)")

    # ─── D4: home π under holiday OFF vs ON at same γ (CDC) ───
    print("\n[D4] home π — (holiday OFF, γ CDC) vs (holiday ON, γ CDC), "
          "free 4-channel, amp=0.7")
    off = fit_free_4ch(s, GAMMA_CDC, amp=AMP, holiday=False)
    on  = fit_free_4ch(s, GAMMA_CDC, amp=AMP, holiday=True)
    print(f"  {'holiday':>8}  {'home π':>7}  {'R0':>6}  {'π (h, w, s, o)':>32}  {'NLL':>14}")
    print(f"  {'OFF':>8}  {off['pi'][0]:>7.3f}  {off['R0']:>6.3f}  "
          f"{str([round(x, 3) for x in off['pi']]):>32}  {off['nll']:>14,.0f}")
    print(f"  {'ON':>8}  {on['pi'][0]:>7.3f}  {on['R0']:>6.3f}  "
          f"{str([round(x, 3) for x in on['pi']]):>32}  {on['nll']:>14,.0f}")
    delta_home = on['pi'][0] - off['pi'][0]
    print(f"  Δhome π (ON − OFF) = {delta_home:+.3f}")

    # save
    out = "outputs/metapop/home_zero_diagnostic.json"
    json.dump({
        "season": SEASON, "amp": AMP, "R0_target": R0_TARGET,
        "D1_contact_corr_home_other": corr,
        "D1_contact_rowsum": {"home": rs_home.tolist(), "other": rs_other.tolist(),
                                "school": rs_school.tolist(), "work": rs_work.tolist()},
        "D2_home_forced": {"base": opt, "configs": d2_table},
        "D3_corr_age_peak": corr_age,
        "D3_peak_per_age": {
            "home_only": inc_home_only.max(axis=0).tolist(),
            "other_only": inc_other_only.max(axis=0).tolist(),
            "school_only": inc_school_only.max(axis=0).tolist(),
        },
        "D4_holiday_effect": {"off": off, "on": on, "delta_home": delta_home},
    }, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
