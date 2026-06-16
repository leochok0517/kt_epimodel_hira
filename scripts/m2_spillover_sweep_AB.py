"""Sweep spillover κ × kappa_scale and measure sick_leave averted under
A (NIMS) and B (literature) posteriors. kappa_scale=0 = perfect home
isolation (no spillover); kappa_scale=1 = current values (slide 15:
students 0.42, adults 0.60, 70+ 0). The sign-flip point identifies the
home-isolation level required for sick-leave policy to be beneficial.
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
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


SEASON_LABEL = "2019-2020"
SEASON_IDX = 2
AMP = 0.9
HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)
KAPPA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_DRAWS = 60


def load_posterior(label):
    data = np.load(f"outputs/calibration/m2_prod_{label}_samples.npz")
    return data["beta"].reshape(-1, 16)


def main():
    print("=" * 70)
    print("Spillover κ sweep: sick_leave (p_work=0.4) vs home-isolation level")
    print("=" * 70)

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy
    kappa_base = disease.kappa_array

    seed_15 = estimate_initial_infected_from_hira(
        SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    phi_full = jnp.ones(15)

    def shared_with_kappa(kappa_scale):
        d = dict(
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
            kappa=jnp.asarray(kappa_base * float(kappa_scale)),
            sigma=disease.sigma, gamma=disease.gamma,
            VE=vax.VE,
            annual_coverage=jnp.asarray(vax.annual_coverage),
            vax_peak_iso_week=vax.peak_iso_week,
            vax_spread_weeks=vax.spread_weeks,
            seasonality_amp=AMP,
            seasonality_base=disease.seasonality_base,
            seasonality_peak_day=disease.seasonality_peak_day,
            seasonality_period=disease.seasonality_period,
        )
        d.update(HOLIDAY)
        return d

    def fwd(shared, beta_4, p_school, p_work):
        kw = dict(shared)
        kw["beta_h"] = float(beta_4[0]); kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2]); kw["beta_o"] = float(beta_4[3])
        kw["phi_susc"] = phi_full
        kw["p_school"] = float(p_school); kw["p_work"] = float(p_work)
        st = simulate_jax(state0, **kw, discretize_time=False)
        return float(daily_new_infection_by_age_jax(st).sum())

    rng = np.random.default_rng(0)
    results = {}     # results[label][policy][κ]
    for label in ("A", "B"):
        beta_post = load_posterior(label)
        idx = rng.choice(beta_post.shape[0], N_DRAWS, replace=False)
        per_kappa = {"sick": {}, "school": {}}
        for ks in KAPPA_GRID:
            shared = shared_with_kappa(ks)
            sick_arr, sch_arr = [], []
            for j in idx:
                beta_4 = beta_post[j, SEASON_IDX*4:(SEASON_IDX+1)*4]
                inc_b  = fwd(shared, beta_4, 1.0, 1.0)
                inc_si = fwd(shared, beta_4, 1.0, 0.4)
                inc_sc = fwd(shared, beta_4, 0.5, 1.0)
                sick_arr.append((inc_b - inc_si) / max(inc_b, 1.0) * 100)
                sch_arr.append((inc_b - inc_sc) / max(inc_b, 1.0) * 100)
            sick_arr = np.array(sick_arr); sch_arr = np.array(sch_arr)
            for policy, arr in (("sick", sick_arr), ("school", sch_arr)):
                per_kappa[policy][ks] = {
                    "mean": float(arr.mean()),
                    "q025": float(np.quantile(arr, 0.025)),
                    "q975": float(np.quantile(arr, 0.975)),
                    "samples": arr.tolist(),
                }
            print(f"  {label} κ×{ks:>4.1f}: "
                  f"sick {sick_arr.mean():+6.2f}% [{np.quantile(sick_arr, 0.025):+6.2f}, {np.quantile(sick_arr, 0.975):+6.2f}]   "
                  f"school {sch_arr.mean():+6.2f}% [{np.quantile(sch_arr, 0.025):+6.2f}, {np.quantile(sch_arr, 0.975):+6.2f}]")
        results[label] = per_kappa
        print()

    # sign-flip estimate per label × policy
    print("[sign-flip κ × kappa_scale (mean crossing zero)]")
    for policy in ("sick", "school"):
        for label in ("A", "B"):
            means = [results[label][policy][ks]["mean"] for ks in KAPPA_GRID]
            flip_at = None
            for k in range(1, len(KAPPA_GRID)):
                if (means[k] > 0) != (means[k-1] > 0):
                    m0, m1 = means[k-1], means[k]
                    k0, k1 = KAPPA_GRID[k-1], KAPPA_GRID[k]
                    flip_at = k0 + (k1 - k0) * (0 - m0) / (m1 - m0)
                    break
            if flip_at is None:
                state = "all + " if all(m > 0 for m in means) else (
                    "all − " if all(m < 0 for m in means) else "mixed")
                print(f"  {policy:>6} {label}: no flip in [0, 1] ({state})")
            else:
                print(f"  {policy:>6} {label}: flip at κ_scale ≈ {flip_at:.3f}")

    # plot — sick vs school side by side, both with A & B
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    colors = {"A": "#1a5490", "B": "#c0392b"}
    for ax, policy, ptitle in [
        (axes[0], "sick", "sick_leave (p_work=0.4)"),
        (axes[1], "school", "school_closure (p_school=0.5)"),
    ]:
        for label in ("A", "B"):
            means = np.array([results[label][policy][ks]["mean"] for ks in KAPPA_GRID])
            lo = np.array([results[label][policy][ks]["q025"] for ks in KAPPA_GRID])
            hi = np.array([results[label][policy][ks]["q975"] for ks in KAPPA_GRID])
            ax.plot(KAPPA_GRID, means, "o-", color=colors[label], lw=2,
                    label=f"{label} (mean)")
            ax.fill_between(KAPPA_GRID, lo, hi, color=colors[label], alpha=0.20,
                             label=f"{label} (95% CI)")
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xlabel("κ_scale  (0 = perfect home isolation, 1 = slide-15 default)")
        ax.set_ylabel("averted %")
        ax.set_title(ptitle)
        ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Spillover sweep — sick_leave vs school_closure under A (NIMS) vs B (literature)\n"
                  f"2019-20, posterior draws N={N_DRAWS}, holiday ON+realloc, amp={AMP}",
                  y=1.02)
    fig.tight_layout()
    out = "presentations/figures/spillover_sweep_both_policies.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")

    out_json = "outputs/metapop/spillover_sweep_AB.json"
    with open(out_json, "w") as f:
        json.dump({
            "kappa_base": kappa_base.tolist(),
            "kappa_grid": KAPPA_GRID,
            "results": {
                k: {
                    policy: {str(ks): {kk: vv for kk, vv in v.items() if kk != "samples"}
                              for ks, v in vd.items()}
                    for policy, vd in d.items()
                }
                for k, d in results.items()
            },
        }, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
