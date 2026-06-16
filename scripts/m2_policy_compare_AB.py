"""Compare A vs B policy posteriors — sick_leave and school_closure effects.

For each of two posteriors (chprior A NIMS, B literature), draw N samples
of β_2019-20, run baseline + sick_leave + school_closure forward, and
compute averted % per draw. Report CIs and overlap.
"""
from __future__ import annotations
import os, json, time
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
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


SEASON_IDX = 2   # 2019-2020 in production beta vec
AMP = 0.9
HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)
N_DRAWS = 100


def load_posterior(label):
    data = np.load(f"outputs/calibration/m2_prod_{label}_samples.npz")
    return data["beta"].reshape(-1, 16)


def main():
    print("=" * 70)
    print("Policy posterior comparison — A (NIMS) vs B (literature)")
    print("=" * 70)

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
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
        VE=vax.VE,
        annual_coverage=jnp.asarray(vax.annual_coverage),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared.update(HOLIDAY)
    phi_full = jnp.ones(15)

    SEASON_LABEL = "2019-2020"
    seed_15 = estimate_initial_infected_from_hira(
        SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))

    def fwd(beta_4, p_school, p_work):
        kw = dict(shared)
        kw["beta_h"] = float(beta_4[0]); kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2]); kw["beta_o"] = float(beta_4[3])
        kw["phi_susc"] = phi_full
        kw["p_school"] = float(p_school); kw["p_work"] = float(p_work)
        st = simulate_jax(state0, **kw, discretize_time=False)
        return float(daily_new_infection_by_age_jax(st).sum())

    rng = np.random.default_rng(0)
    results = {}
    for label in ("A", "B"):
        beta_post = load_posterior(label)              # (1200, 16)
        idx = rng.choice(beta_post.shape[0], N_DRAWS, replace=False)
        sick_av, sch_av, baseline = [], [], []
        t0 = time.perf_counter()
        for i, j in enumerate(idx):
            beta_4 = beta_post[j, SEASON_IDX*4:(SEASON_IDX+1)*4]
            inc_b  = fwd(beta_4, 1.0, 1.0)
            inc_si = fwd(beta_4, 1.0, 0.4)
            inc_sc = fwd(beta_4, 0.5, 1.0)
            baseline.append(inc_b)
            sick_av.append((inc_b - inc_si) / max(inc_b, 1.0) * 100)
            sch_av.append((inc_b - inc_sc) / max(inc_b, 1.0) * 100)
        sick_arr = np.array(sick_av); sch_arr = np.array(sch_av)
        bl_arr = np.array(baseline)
        print(f"\n{label}: {N_DRAWS} draws  ({time.perf_counter()-t0:.1f}s)")
        print(f"  baseline:        {bl_arr.mean():>10,.0f}  "
              f"[{np.quantile(bl_arr, 0.025):,.0f}, {np.quantile(bl_arr, 0.975):,.0f}]")
        print(f"  sick_leave av:   {sick_arr.mean():>+10.2f}%  "
              f"[{np.quantile(sick_arr, 0.025):+.2f}, {np.quantile(sick_arr, 0.975):+.2f}]")
        print(f"  school_closure:  {sch_arr.mean():>+10.2f}%  "
              f"[{np.quantile(sch_arr, 0.025):+.2f}, {np.quantile(sch_arr, 0.975):+.2f}]")
        results[label] = {
            "baseline_mean": float(bl_arr.mean()),
            "sick_av_mean": float(sick_arr.mean()),
            "sick_av_q025": float(np.quantile(sick_arr, 0.025)),
            "sick_av_q975": float(np.quantile(sick_arr, 0.975)),
            "school_av_mean": float(sch_arr.mean()),
            "school_av_q025": float(np.quantile(sch_arr, 0.025)),
            "school_av_q975": float(np.quantile(sch_arr, 0.975)),
            "sick_samples": sick_arr.tolist(),
            "school_samples": sch_arr.tolist(),
        }

    # Overlap analysis
    print(f"\n[CI overlap]")
    for metric, key_lo, key_hi in [("sick_leave", "sick_av_q025", "sick_av_q975"),
                                     ("school_closure", "school_av_q025", "school_av_q975")]:
        A_lo, A_hi = results["A"][key_lo], results["A"][key_hi]
        B_lo, B_hi = results["B"][key_lo], results["B"][key_hi]
        overlap = max(0, min(A_hi, B_hi) - max(A_lo, B_lo))
        a_width = A_hi - A_lo; b_width = B_hi - B_lo
        avg_width = (a_width + b_width) / 2
        rel = overlap / max(avg_width, 1e-6) * 100
        print(f"  {metric}: A [{A_lo:+.2f}, {A_hi:+.2f}]  B [{B_lo:+.2f}, {B_hi:+.2f}]  "
              f"overlap {overlap:.2f} ({rel:.1f}% of avg width)")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(results["A"]["sick_samples"], bins=20, alpha=0.6,
                  label="A (NIMS)", color="#1a5490")
    axes[0].hist(results["B"]["sick_samples"], bins=20, alpha=0.6,
                  label="B (literature)", color="#c0392b")
    axes[0].axvline(0, color="grey", ls="--")
    axes[0].set_xlabel("sick_leave averted %")
    axes[0].set_title(f"sick_leave (p_work=0.4)\n"
                       f"A {results['A']['sick_av_mean']:+.1f}% [{results['A']['sick_av_q025']:+.1f},{results['A']['sick_av_q975']:+.1f}]\n"
                       f"B {results['B']['sick_av_mean']:+.1f}% [{results['B']['sick_av_q025']:+.1f},{results['B']['sick_av_q975']:+.1f}]")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].hist(results["A"]["school_samples"], bins=20, alpha=0.6,
                  label="A (NIMS)", color="#1a5490")
    axes[1].hist(results["B"]["school_samples"], bins=20, alpha=0.6,
                  label="B (literature)", color="#c0392b")
    axes[1].set_xlabel("school_closure averted %")
    axes[1].set_title(f"school_closure (p_school=0.5)\n"
                       f"A {results['A']['school_av_mean']:.1f}% [{results['A']['school_av_q025']:.1f},{results['A']['school_av_q975']:.1f}]\n"
                       f"B {results['B']['school_av_mean']:.1f}% [{results['B']['school_av_q025']:.1f},{results['B']['school_av_q975']:.1f}]")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Policy posterior comparison — 2019-20 (N={N_DRAWS} draws each)")
    fig.tight_layout()
    out = "presentations/figures/policy_posterior_AB.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")

    with open("outputs/metapop/policy_compare_AB.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if "samples" not in kk}
                    for k, v in results.items()}, f, indent=2)
    print(f"saved outputs/metapop/policy_compare_AB.json")


if __name__ == "__main__":
    main()
