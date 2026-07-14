"""Diagnostic 2 + 3 + 4: does the data inform channel decomposition?

For 2019-2020 season at R0 = 1.93 (NB posterior mean), φ = 1.0, γ from registry:

D2. Channel → per-age epidemic curve sensitivity.
    Run forward with 3 different simplexes (school-dominant, home-dominant,
    balanced). Compare per-age peak heights/timing.

D3. NLL on the actual HIRA data at:
    - π_NB (posterior mean) — home/work ≈ 0
    - π_balanced (contact rowsum proportional, the "epidemiologically obvious"
      mix)
    Same R0, same φ, same γ. If close → non-identifiable; if balanced lower →
    optimizer/prior trapped the NB run; if extreme lower → data really says it.

D4. Per-age fit (model vs HIRA) under the two simplexes.
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
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.calibration.gamma_registry import get_active_gamma
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, make_multi_season_loss_fn,
    simulation_to_hira_by_age_jax, gamma_triple_to_15,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)


SEASON = "2019-2020"


def main():
    print("=" * 70)
    print(f"Diagnostic 2/3/4 — channel decomposition for {SEASON}")
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

    # Initial state (1 admdong)
    seed_15 = estimate_initial_infected_from_hira(
        SEASON, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))

    phi_full = jnp.ones(15)
    g_active = get_active_gamma()
    gamma_3 = jnp.array([float(g_active[0]), float(g_active[5]), float(g_active[13])])

    # NGM eigvalue
    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.7,
    )

    R0_target = 1.94    # NB posterior 2019-20 mean

    # Three simplex configurations
    configs = {
        "NB_posterior": [0.001, 0.001, 0.380, 0.618],
        "balanced":     [0.272, 0.198, 0.165, 0.365],   # contact rowsum-prop (23.3 / 17.0 / 14.2 / 31.8)
        "school_only":  [0.001, 0.001, 0.998, 0.001],
        "home_dominant":[0.500, 0.150, 0.150, 0.200],
    }
    # Normalize and derive β
    forward_results = {}
    for name, pi_raw in configs.items():
        pi = jnp.asarray(pi_raw)
        pi = pi / pi.sum()
        beta_4 = derive_beta_from_R0_simplex(ngm, R0_target, pi, phi_full)
        R0_check = float(ngm(beta_4[0], beta_4[1], beta_4[2], beta_4[3], phi_full))
        kw = dict(shared)
        kw["beta_h"] = float(beta_4[0])
        kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2])
        kw["beta_o"] = float(beta_4[3])
        kw["phi_susc"] = phi_full
        states = simulate_jax(state0, **kw, discretize_time=False)
        inc_age = np.asarray(daily_new_infection_by_age_jax(states))  # (n_days-1, 15)
        forward_results[name] = {
            "pi": [float(x) for x in pi.tolist()],
            "beta_4": [float(x) for x in beta_4.tolist()],
            "R0_check": R0_check,
            "inc_age": inc_age,
        }

    # ─── D2: per-age peak height / timing comparison ───
    print("\n[D2] forward results — per-age peak (NIMS 15)")
    print(f"{'config':>14}  {'R0_check':>10}  {'total':>10}  "
          + "".join(f"a{a:02d}".rjust(7) for a in (0, 2, 4, 6, 8, 10, 13)))
    for name, r in forward_results.items():
        inc = r["inc_age"]
        peak_per_age = inc.max(axis=0)
        total = inc.sum()
        print(f"{name:>14}  {r['R0_check']:>10.4f}  {total:>10,.0f}  "
              + "".join(f"{peak_per_age[a]:>6,.0f}" for a in (0, 2, 4, 6, 8, 10, 13)))

    # ─── D3: NLL on real HIRA ───
    print("\n[D3] NLL at the actual HIRA target")
    tgt = load_hira_target_by_age(
        SEASON, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    nw = tgt["n_weeks"]
    obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]
    loss_fn = make_multi_season_loss_fn(
        initial_states=[state0], obs_hira_list=[jnp.asarray(obs)],
        weights_hira_list=[jnp.asarray(w)], shared_static=shared,
        n_weeks=nw, min_rate=0.01, discretize_time=False,
    )
    # vec_33 layout: [phi(14), gamma(3), beta_2017(4), ...] — single-season loss
    # so we put beta_4 in the first season slot (we built only 1 season here)
    phi_14 = jnp.ones(14)
    for name, r in forward_results.items():
        # 16-vec β; only first 4 used by single-season loss
        beta_16 = jnp.concatenate([jnp.asarray(r["beta_4"]),
                                    jnp.zeros(12)])
        vec_33 = jnp.concatenate([phi_14, gamma_3, beta_16])
        nll = float(loss_fn(vec_33))
        print(f"  {name:>14}: NLL = {nll:.2f}")

    # ─── D4: per-age fit (HIRA mapping) ───
    print("\n[D4] per-age reported (γ-scaled) weekly fit — first 6 weeks vs observed")
    gamma_15 = gamma_triple_to_15(gamma_3[0], gamma_3[1], gamma_3[2])
    print(f"{'config':>14}  " + "  ".join(f"{ag:>7}" for ag in HIRA_AGE_GROUPS))
    print(f"{'OBSERVED':>14}  " + "  ".join(
        f"{int(obs[:6, i].sum()):>7,d}" for i in range(6)))
    for name, r in forward_results.items():
        pred_hira = np.asarray(simulation_to_hira_by_age_jax(
            jnp.asarray(r["inc_age"]), gamma_15, n_weeks=nw,
        ))
        first6 = pred_hira[:6].sum(axis=0)
        print(f"{name:>14}  " + "  ".join(f"{int(x):>7,d}" for x in first6))


if __name__ == "__main__":
    main()
