"""Channel mix grid forward — policy robustness without re-calibration.

Same R0 (1.90), φ=1.0, γ_15 (level=0.6 × CDC ratio), season 2019-20. Sweep π
across a plausible grid and run 3 policy scenarios. Aggregated forward for
speed (Step A confirmed NGM equivalence to spatial within 0.3%).

Key questions:
- Is school_closure 99% robust across channels? → if yes, seasonality is the
  cause (TODO-4 justified).
- Does sick_leave become positive at any plausible channel? → tests whether
  −0.8% is structural (home spillover, TODO-5) or just our channel choice.
"""
from __future__ import annotations
import os, itertools, json, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)


SEASON = "2019-2020"
R0_FIXED = 1.90
GAMMA_LEVEL = 0.6
CHILD_REL, ADULT_REL, ELDER_REL = 1.45, 0.65, 0.90


def build_channel_grid():
    """Plausible 4-channel simplex grid: h ∈ [.15,.35], w ∈ [.05,.25],
    s ∈ [.20,.50], with o = 1−sum constrained to [.15,.35]."""
    home_grid   = [0.15, 0.20, 0.25, 0.30, 0.35]
    work_grid   = [0.05, 0.10, 0.15, 0.20, 0.25]
    school_grid = [0.20, 0.30, 0.40, 0.50]
    points = []
    for h, w, s in itertools.product(home_grid, work_grid, school_grid):
        o = 1.0 - h - w - s
        if 0.15 <= o <= 0.35:
            points.append([round(h, 3), round(w, 3), round(s, 3), round(o, 3)])
    # Add reference points
    points += [
        [0.234, 0.112, 0.444, 0.210],   # NB_WO 2019-20 posterior mean
        [0.300, 0.250, 0.250, 0.200],   # literature balanced
        [0.350, 0.200, 0.300, 0.150],   # home-heavy
        [0.200, 0.100, 0.500, 0.200],   # school-heavy
    ]
    # de-dup
    seen = set(); uniq = []
    for p in points:
        k = tuple(p)
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq


def main():
    print("=" * 70)
    print(f"Channel sensitivity grid forward — R0={R0_FIXED}, season {SEASON}")
    print("=" * 70)

    grid = build_channel_grid()
    print(f"  grid size: {len(grid)} plausible channel mixes")

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    rho_emp = inputs["rho"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination

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
    phi_full = jnp.ones(15)

    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.7,
    )

    def forward(beta_4, p_school, p_work):
        kw = dict(shared)
        kw["beta_h"] = float(beta_4[0]); kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2]); kw["beta_o"] = float(beta_4[3])
        kw["phi_susc"] = phi_full
        kw["p_school"] = float(p_school); kw["p_work"] = float(p_work)
        states = simulate_jax(state0, **kw, discretize_time=False)
        return float(daily_new_infection_by_age_jax(states).sum())

    results = []
    t0 = time.perf_counter()
    print(f"\n{'pi (h, w, s, o)':>32}  {'base':>10}  {'sch_av':>7}  {'sick_av':>7}  "
          f"{'Reff_sch0.5':>11}")
    for i, pi_raw in enumerate(grid):
        pi = jnp.asarray(pi_raw); pi = pi / pi.sum()
        beta_4 = derive_beta_from_R0_simplex(ngm, R0_FIXED, pi, phi_full)
        # R_eff at school 50% closure (β_s × 0.5)
        R_eff_sch05 = float(ngm(beta_4[0], beta_4[1], 0.5 * beta_4[2], beta_4[3], phi_full))
        inc_base = forward(beta_4, 1.0, 1.0)
        inc_sch  = forward(beta_4, 0.5, 1.0)
        inc_sick = forward(beta_4, 1.0, 0.4)
        sch_av = (inc_base - inc_sch) / max(inc_base, 1.0) * 100
        sick_av = (inc_base - inc_sick) / max(inc_base, 1.0) * 100
        results.append({
            "pi": [float(x) for x in pi.tolist()],
            "base_total": inc_base, "sch_av": sch_av,
            "sick_av": sick_av, "R_eff_sch05": R_eff_sch05,
        })
        if i < 4 or i % 8 == 0 or i == len(grid) - 1:
            print(f"  {str([round(x, 2) for x in pi_raw]):>32}  "
                  f"{inc_base:>10,.0f}  {sch_av:>6.1f}%  {sick_av:>+6.1f}%  "
                  f"{R_eff_sch05:>11.3f}")
    print(f"\n  total forwards: {len(grid)*3}  wall: {time.perf_counter()-t0:.1f}s")

    sch_arr = np.array([r["sch_av"] for r in results])
    sick_arr = np.array([r["sick_av"] for r in results])
    base_arr = np.array([r["base_total"] for r in results])
    print()
    print(f"[summary across {len(results)} channel points]")
    print(f"  school_closure averted: {sch_arr.min():.1f}% ~ {sch_arr.max():.1f}%  "
          f"mean {sch_arr.mean():.1f}  std {sch_arr.std():.1f}")
    print(f"  sick_leave averted:     {sick_arr.min():.1f}% ~ {sick_arr.max():.1f}%  "
          f"mean {sick_arr.mean():.1f}  std {sick_arr.std():.1f}")
    print(f"  baseline total infections: {base_arr.min():,.0f} ~ {base_arr.max():,.0f}")

    n_cliff = int((sch_arr > 90).sum())
    n_non_cliff = int((sch_arr < 80).sum())
    print(f"  school_closure >90% averted: {n_cliff}/{len(grid)}  (cliff)")
    print(f"  school_closure <80% averted: {n_non_cliff}/{len(grid)}  (not cliff)")
    n_pos = int((sick_arr > 1).sum())
    n_neg = int((sick_arr < 0).sum())
    print(f"  sick_leave averted >1%: {n_pos}/{len(grid)}")
    print(f"  sick_leave averted <0%: {n_neg}/{len(grid)}  (counterproductive)")

    if n_pos > 0:
        pos_pts = [r for r in results if r["sick_av"] > 1]
        pos_sorted = sorted(pos_pts, key=lambda r: -r["sick_av"])[:5]
        print("  top sick_leave-positive channels:")
        for r in pos_sorted:
            print(f"    π={[round(x, 2) for x in r['pi']]}  sick_av {r['sick_av']:+.1f}%")
    if n_non_cliff > 0:
        ncf_pts = sorted([r for r in results if r["sch_av"] < 80], key=lambda r: r["sch_av"])
        print("  lowest school_closure-averted channels:")
        for r in ncf_pts[:5]:
            print(f"    π={[round(x, 2) for x in r['pi']]}  sch_av {r['sch_av']:.1f}%  "
                  f"R_eff_sch0.5={r['R_eff_sch05']:.3f}")

    # save
    out = "outputs/metapop/channel_sensitivity.json"
    json.dump({
        "R0_fixed": R0_FIXED, "season": SEASON, "n_points": len(grid),
        "results": results,
        "summary": {
            "sch_av_min": float(sch_arr.min()), "sch_av_max": float(sch_arr.max()),
            "sch_av_mean": float(sch_arr.mean()), "sch_av_std": float(sch_arr.std()),
            "sick_av_min": float(sick_arr.min()), "sick_av_max": float(sick_arr.max()),
            "sick_av_mean": float(sick_arr.mean()), "sick_av_std": float(sick_arr.std()),
            "n_sch_cliff": n_cliff, "n_sch_noncliff": n_non_cliff,
            "n_sick_pos": n_pos, "n_sick_neg": n_neg,
        },
    }, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")

    # plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pi_arr = np.array([r["pi"] for r in results])
    axes[0, 0].hist(sch_arr, bins=20, color="#1a5490", alpha=0.85)
    axes[0, 0].set_title(f"school_closure (p=0.5) — averted % across {len(grid)} channels")
    axes[0, 0].set_xlabel("averted %"); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].hist(sick_arr, bins=20, color="#c0392b", alpha=0.85)
    axes[0, 1].axvline(0, color="grey", ls="--")
    axes[0, 1].set_title(f"sick_leave (work=0.4) — averted % across {len(grid)} channels")
    axes[0, 1].set_xlabel("averted % (negative = counterproductive)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].scatter(pi_arr[:, 1], sick_arr, alpha=0.6, c=pi_arr[:, 0], cmap="viridis")
    axes[1, 0].axhline(0, color="grey", ls="--")
    axes[1, 0].set_xlabel("work π"); axes[1, 0].set_ylabel("sick_leave averted %")
    axes[1, 0].set_title("sick_leave effect vs work π (color = home π)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].scatter(pi_arr[:, 2], sch_arr, alpha=0.6, c=pi_arr[:, 3], cmap="viridis")
    axes[1, 1].set_xlabel("school π"); axes[1, 1].set_ylabel("school_closure averted %")
    axes[1, 1].set_title("school_closure effect vs school π (color = other π)")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Channel sensitivity grid — R0={R0_FIXED}, season {SEASON}, "
        f"aggregated forward", fontsize=13)
    fig.tight_layout()
    out_png = "presentations/figures/channel_sensitivity.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
