"""Trace why work π ≈ 0: contact scale vs FOI vs normalization (not non-id).

D1. Contact matrix per-age rowsum and work/school overlap.
D2. Unit-β NGM R0 per channel (school dominance test).
D3. R0 contribution decomposition at the current simplex (π vs actual R0 share).
D4. NLL with work forced on.
D5. Single-channel age-curve separation.
D6. Attack rate by age at current optimum.

All at γ_level=0.6, CDC-standard relative shape, φ=1.0, season 2019-20.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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
GAMMA_LEVEL = 0.6
CHILD_REL, ADULT_REL, ELDER_REL = 1.45, 0.65, 0.90
R0_TARGET = 1.83
CURRENT_OPT_PI = [0.270, 0.009, 0.619, 0.103]


def build_gamma_15():
    K = 3.0 / (CHILD_REL + ADULT_REL + ELDER_REL)
    g_c = GAMMA_LEVEL * K * CHILD_REL
    g_a = GAMMA_LEVEL * K * ADULT_REL
    g_e = GAMMA_LEVEL * K * ELDER_REL
    return np.concatenate([np.full(4, g_c), np.full(9, g_a), np.full(2, g_e)])


def main():
    print("=" * 70)
    print(f"work-channel root-cause trace  (γ_level={GAMMA_LEVEL}, CDC ratio, {SEASON})")
    print("=" * 70)

    # ─── D1: contact age structure ───
    C_raw = load_contact_matrices()
    print("\n[D1] contact rowsum per age (NIMS 15)  — does work age slice live?")
    print(f"  {'age':>4} {'home':>7} {'work':>7} {'school':>7} {'other':>7}")
    for a in range(15):
        print(f"  {a:>4} {C_raw['C_home'].sum(axis=1)[a]:>7.2f} "
              f"{C_raw['C_work'].sum(axis=1)[a]:>7.2f} "
              f"{C_raw['C_school'].sum(axis=1)[a]:>7.2f} "
              f"{C_raw['C_other'].sum(axis=1)[a]:>7.2f}")
    print(f"  totals : home {C_raw['C_home'].sum():.1f}  work {C_raw['C_work'].sum():.1f}  "
          f"school {C_raw['C_school'].sum():.1f}  other {C_raw['C_other'].sum():.1f}")

    # ─── setup shared infra (aggregated) ───
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
        pop_15=jnp.asarray(pop_15), rho=jnp.asarray(rho_emp),
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

    phi_full = jnp.ones(15); phi_14 = jnp.ones(14)
    gamma_15 = build_gamma_15()
    gamma_3 = jnp.array([float(gamma_15[0]), float(gamma_15[4]), float(gamma_15[13])])

    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.7,
    )

    loss_fn = make_multi_season_loss_fn(
        initial_states=[state0], obs_hira_list=[jnp.asarray(obs)],
        weights_hira_list=[jnp.asarray(w)], shared_static=shared,
        n_weeks=nw, min_rate=0.01, discretize_time=False,
    )

    # ─── D2: unit R0 per channel ───
    print("\n[D2] unit-β NGM R0 per channel (β=1 only in that channel, others 0)")
    unit_R0 = {}
    for ch, beta_4 in [("home",   [1.0, 0.0, 0.0, 0.0]),
                       ("work",   [0.0, 1.0, 0.0, 0.0]),
                       ("school", [0.0, 0.0, 1.0, 0.0]),
                       ("other",  [0.0, 0.0, 0.0, 1.0])]:
        r = float(ngm(*beta_4, phi_full))
        unit_R0[ch] = r
        print(f"  {ch:>7}: unit R0 = {r:7.4f}  (β=1 → R0 contribution)")
    ratio = {k: v / unit_R0["work"] for k, v in unit_R0.items()}
    print(f"  → school is {unit_R0['school']/unit_R0['work']:.2f}× the work unit-R0")

    # ─── D3: at current optimum π, what is each channel's actual R0 contribution? ───
    print("\n[D3] At current best π = {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(*CURRENT_OPT_PI))
    pi_cur = jnp.asarray(CURRENT_OPT_PI); pi_cur = pi_cur / pi_cur.sum()
    beta_cur = derive_beta_from_R0_simplex(ngm, R0_TARGET, pi_cur, phi_full)
    print(f"  β derived (h, w, s, o) = {[round(float(x), 5) for x in beta_cur]}")
    # NGM is 1-homogeneous in each β; channel R0 "contribution" defined as
    # β_ch · (unit R0 of channel ch). Sum is upper bound (subadditive eigenvalue).
    contributions = {ch: float(beta_cur[i]) * unit_R0[ch]
                     for i, ch in enumerate(["home", "work", "school", "other"])}
    print(f"  R0 contribution (β·unit_R0) per channel:")
    for ch, r in contributions.items():
        share = r / sum(contributions.values()) * 100
        print(f"    {ch:>7}: {r:>7.4f}  ({share:>5.1f}% of additive total)")
    print(f"  full eigenvalue (= R0_actual): {float(ngm(*beta_cur, phi_full)):.4f}")

    # ─── D4: NLL — force work on ───
    print("\n[D4] NLL — current vs work-forced-on configurations")
    configs = {
        "current_opt":                CURRENT_OPT_PI,
        "work=0.20 ↓school":          [0.270, 0.200, 0.430, 0.100],
        "work=0.20 ↓other":           [0.270, 0.200, 0.520, 0.010],
        "work=0.30":                  [0.270, 0.300, 0.330, 0.100],
        "literature_typical":         [0.300, 0.250, 0.250, 0.200],
    }
    print(f"  {'config':>22}  {'h':>5} {'w':>5} {'s':>5} {'o':>5}  "
          f"{'R0':>6}  {'NLL':>15}  {'ΔNLL':>10}")
    base_nll = None
    for name, pi_raw in configs.items():
        pi = jnp.asarray(pi_raw); pi = pi / pi.sum()
        beta = derive_beta_from_R0_simplex(ngm, R0_TARGET, pi, phi_full)
        beta_16 = jnp.concatenate([beta, jnp.zeros(12)])
        vec_33 = jnp.concatenate([phi_14, gamma_3, beta_16])
        nll = float(loss_fn(vec_33))
        if base_nll is None:
            base_nll = nll; d = 0.0
        else:
            d = nll - base_nll
        print(f"  {name:>22}  {pi[0]:>5.2f} {pi[1]:>5.2f} {pi[2]:>5.2f} {pi[3]:>5.2f}  "
              f"{R0_TARGET:>6.2f}  {nll:>15,.0f}  {d:>+10,.0f}")

    # ─── D5: single-channel forward (age-curve separation) ───
    print("\n[D5] forward with single-channel π — does age curve separate work vs school?")
    print(f"  {'config':>14}  {'total':>10}  "
          + "".join(f"a{a:02d}".rjust(7) for a in (0, 2, 4, 6, 8, 10, 13)))
    for name, pi_raw in [("home_only",   [1.0, 0.0, 0.0, 0.0]),
                         ("work_only",   [0.0, 1.0, 0.0, 0.0]),
                         ("school_only", [0.0, 0.0, 1.0, 0.0]),
                         ("other_only",  [0.0, 0.0, 0.0, 1.0])]:
        pi = jnp.asarray(pi_raw); pi = pi / pi.sum()
        beta = derive_beta_from_R0_simplex(ngm, R0_TARGET, pi, phi_full)
        kw = dict(shared)
        kw["beta_h"] = float(beta[0]); kw["beta_w"] = float(beta[1])
        kw["beta_s"] = float(beta[2]); kw["beta_o"] = float(beta[3])
        kw["phi_susc"] = phi_full
        states = simulate_jax(state0, **kw, discretize_time=False)
        inc_age = np.asarray(daily_new_infection_by_age_jax(states))
        peak_per_age = inc_age.max(axis=0)
        print(f"  {name:>14}  {inc_age.sum():>10,.0f}  "
              + "".join(f"{peak_per_age[a]:>6,.0f}" for a in (0, 2, 4, 6, 8, 10, 13)))

    # ─── D6: attack rate by age at current optimum ───
    print("\n[D6] attack rate by age at current optimum")
    pi = jnp.asarray(CURRENT_OPT_PI); pi = pi / pi.sum()
    beta = derive_beta_from_R0_simplex(ngm, R0_TARGET, pi, phi_full)
    kw = dict(shared)
    kw["beta_h"] = float(beta[0]); kw["beta_w"] = float(beta[1])
    kw["beta_s"] = float(beta[2]); kw["beta_o"] = float(beta[3])
    kw["phi_susc"] = phi_full
    states = simulate_jax(state0, **kw, discretize_time=False)
    inc_age = np.asarray(daily_new_infection_by_age_jax(states))
    pop_per_age = np.asarray(pop_15).flatten()
    for a in range(15):
        print(f"    a{a:02d} ({a*5}-{a*5+4}): AR = "
              f"{inc_age.sum(axis=0)[a] / pop_per_age[a] * 100:5.2f}%   "
              f"pop {pop_per_age[a]:>10,.0f}")
    print(f"  overall AR: {inc_age.sum() / pop_per_age.sum() * 100:.2f}%")


if __name__ == "__main__":
    main()
