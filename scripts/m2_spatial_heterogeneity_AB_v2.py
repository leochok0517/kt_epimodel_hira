"""Spatial heterogeneity v2 — reclassify by day/night population ratio.

v1 used net work inflow / population which was dominated by population
denominator and washed out the commercial/residential signal.

build_M_work already uses hours 9-17 (work hours), so it's NOT a daily
sum that cancels. The right metric for commercial vs residential is the
day-population / night-population ratio:
  - day_pop[j]   = sum_a sum_i M_work[a, i, j] * pop_15[a, i] * rho[a, i]
  - night_pop[j] = sum_a pop_15[a, j] * rho[a, j]   (workers sleeping at j)
  - ratio[j]    = day_pop / night_pop
  - >> 1 : commercial / job centre
  - << 1 : bedroom community / residential

Then per-region per-capita averted is recomputed under A (NIMS) and B
(literature) posterior-mean β.
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
from kt_data.data.load_contact import load_contact_matrices
from kt_data.data.load_population import load_population_15groups, get_population_matrix
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters, EmploymentParameters,
)
from kt_epimodel_hira.model.mobility_tensor import (
    build_M_home, build_M_school, build_M_work, build_M_other,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


SEASON = "2019-2020"
YYYYMM = "201909"
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


def load_posterior_mean_beta(label):
    data = np.load(f"outputs/calibration/m2_prod_{label}_samples.npz")
    beta_post = data["beta"].reshape(-1, 16)
    return beta_post.mean(axis=0)[SEASON_IDX*4:(SEASON_IDX+1)*4]


def main():
    print("=" * 70)
    print("Spatial heterogeneity v2 — day/night population ratio classification")
    print("=" * 70)

    # spatial inputs
    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop_15 = N_mat.T.astype(np.float64)            # (15, n)
    n_adm = pop_15.shape[1]
    print(f"  pop shape (15, {n_adm})")

    t0 = time.perf_counter()
    M_work = build_M_work(YYYYMM, daytype="weekday", admdong_codes=codes, pop_15=pop_15).astype(np.float64)
    M_other = build_M_other(YYYYMM, daytype="weekday", admdong_codes=codes, pop_15=pop_15).astype(np.float64)
    M_home = build_M_home(n_adm).astype(np.float64)
    M_school = build_M_school(n_adm).astype(np.float64)
    print(f"  mobility built ({time.perf_counter()-t0:.1f}s); "
          f"M_work shape {M_work.shape}, row-sum range {M_work.sum(axis=2).min():.3f}-{M_work.sum(axis=2).max():.3f}")

    matrices_raw = load_contact_matrices()
    matrices = {k: matrices_raw[k] for k in ("C_home", "C_work", "C_school", "C_other")}
    emp = EmploymentParameters.from_kt_data(admdong_codes=codes)
    rho = emp.rho                                  # (n, 15)

    # --- Step 1: classify by day-population / night-population ratio ---
    # Workers from age a in admdong i: workers_ai = pop_15[a, i] × rho[i, a]
    # During work hours (9-17), they distribute according to M_work[a, i, j].
    # day_workers_at_j[a] = sum_i M_work[a, i, j] × workers_ai
    # night_workers_at_j[a] = workers_aj (they're home at night)
    workers_ai = pop_15 * rho.T                    # (15, n)  age × admdong
    # einsum: M_work has shape (15, n_origin, n_dest)
    day_workers_at_j = np.einsum("aij,ai->aj", M_work, workers_ai)   # (15, n)
    night_workers_at_j = workers_ai
    day_workers_total = day_workers_at_j.sum(axis=0)
    night_workers_total = night_workers_at_j.sum(axis=0)
    # ratio with small floor on denominator
    ratio = day_workers_total / np.maximum(night_workers_total, 1.0)
    print(f"\n[Step 1] day/night worker ratio per admdong")
    print(f"  range:   [{ratio.min():.3f}, {ratio.max():.3f}]")
    print(f"  q33 {np.quantile(ratio, 1/3):.3f}  median {np.quantile(ratio, 0.5):.3f}  "
          f"q66 {np.quantile(ratio, 2/3):.3f}")

    q33 = np.quantile(ratio, 1/3)
    q66 = np.quantile(ratio, 2/3)
    region = np.where(ratio >= q66, "commercial",
              np.where(ratio <= q33, "residential", "mixed"))
    n_com = int((region == "commercial").sum())
    n_res = int((region == "residential").sum())
    n_mix = int((region == "mixed").sum())
    print(f"  commercial (>q66): {n_com}, mixed: {n_mix}, residential (<q33): {n_res}")
    codes_arr = np.asarray(codes)
    top_com = np.argsort(-ratio)[:8]
    print(f"\n  top 8 commercial (highest ratio):")
    for i in top_com:
        print(f"    {codes_arr[i]} ratio={ratio[i]:.2f} (day {day_workers_total[i]:>9,.0f} / night {night_workers_total[i]:>9,.0f})")
    top_res = np.argsort(ratio)[:8]
    print(f"  top 8 residential (lowest ratio):")
    for i in top_res:
        print(f"    {codes_arr[i]} ratio={ratio[i]:.2f} (day {day_workers_total[i]:>9,.0f} / night {night_workers_total[i]:>9,.0f})")

    # --- initial state ---
    seed_15 = estimate_initial_infected_from_hira(
        SEASON, pop_15.sum(axis=1),
        sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    phi_full = jnp.ones(15)

    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    shared_base = dict(
        C_home=jnp.asarray(matrices["C_home"]),
        C_school=jnp.asarray(matrices["C_school"]),
        C_work=jnp.asarray(matrices["C_work"]),
        C_other=jnp.asarray(matrices["C_other"]),
        M_home=jnp.asarray(M_home),
        M_school=jnp.asarray(M_school),
        M_work=jnp.asarray(M_work),
        M_other=jnp.asarray(M_other),
        pop_15=jnp.asarray(pop_15),
        rho=jnp.asarray(rho),
        kappa=jnp.asarray(disease.kappa_array),
        sigma=disease.sigma, gamma=disease.gamma,
        phi_susc=phi_full,
        VE=vax.VE,
        annual_coverage=jnp.asarray(vax.annual_coverage),
        vax_peak_iso_week=vax.peak_iso_week,
        vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared_base.update(HOLIDAY)

    def fwd(beta_4, p_school, p_work):
        kw = dict(shared_base)
        kw["beta_h"] = float(beta_4[0]); kw["beta_w"] = float(beta_4[1])
        kw["beta_s"] = float(beta_4[2]); kw["beta_o"] = float(beta_4[3])
        kw["p_school"] = float(p_school); kw["p_work"] = float(p_work)
        states = simulate_jax(state0, **kw, discretize_time=False)
        states.block_until_ready()
        states_np = np.asarray(states)
        EIR = states_np[:, 2] + states_np[:, 3] + states_np[:, 4]    # (n_t, 15, n)
        per_adm_inc = np.diff(EIR.sum(axis=1), axis=0).sum(axis=0)   # (n,)
        return per_adm_inc, float(daily_new_infection_by_age_jax(states).sum())

    # --- Step 2: spatial forward ---
    print(f"\n[Step 2] spatial forward A & B (posterior mean β)")
    SCENARIOS = {"baseline": (1.0, 1.0),
                  "sick_worker_absence": (1.0, 0.4),
                  "sick_student_absence": (0.5, 1.0)}
    results = {}
    for label in ("A", "B"):
        beta_4 = load_posterior_mean_beta(label)
        results[label] = {}
        for name, (psch, pwk) in SCENARIOS.items():
            t0 = time.perf_counter()
            per_adm, total = fwd(beta_4, psch, pwk)
            results[label][name] = per_adm
            print(f"  {label} [{name:>22}] wall={time.perf_counter()-t0:.1f}s "
                  f"total={total:,.0f}")

    # --- Step 3: per-capita averted by region type ---
    pop_total_per_adm = pop_15.sum(axis=0)         # (n,)
    print(f"\n[Step 3] per-capita averted vs baseline (% of baseline)")
    print(f"  {'label':>6}  {'region':>12}  {'sick_worker_av%':>17}  {'sick_student_av%':>18}")
    pc_summary = {}
    for label in ("A", "B"):
        pc_summary[label] = {}
        base = results[label]["baseline"]
        sick = results[label]["sick_worker_absence"]
        absent = results[label]["sick_student_absence"]
        for rname in ("commercial", "mixed", "residential"):
            mask = region == rname
            pop_r = pop_total_per_adm[mask].sum()
            base_pc = base[mask].sum() / max(pop_r, 1)
            sick_pc = (base[mask] - sick[mask]).sum() / max(pop_r, 1)
            absent_pc = (base[mask] - absent[mask]).sum() / max(pop_r, 1)
            pc_summary[label][rname] = {
                "baseline_pc": float(base_pc),
                "sick_av_pc": float(sick_pc),
                "absence_av_pc": float(absent_pc),
                "sick_av_pct": float(sick_pc / base_pc * 100) if base_pc else 0.0,
                "absence_av_pct": float(absent_pc / base_pc * 100) if base_pc else 0.0,
            }
            print(f"  {label:>6}  {rname:>12}  "
                  f"{sick_pc/base_pc*100:>+16.3f}%  {absent_pc/base_pc*100:>+17.3f}%")

    # --- Step 4: commercial / residential ratio ---
    print(f"\n[Step 4] commercial / residential averted% ratio")
    ratios = {}
    print(f"  {'label':>6}  {'sick_worker':>16}  {'sick_student':>16}")
    for label in ("A", "B"):
        c = pc_summary[label]["commercial"]
        r = pc_summary[label]["residential"]
        rs = c["sick_av_pct"] / r["sick_av_pct"] if abs(r["sick_av_pct"]) > 1e-6 else float("inf")
        ra = c["absence_av_pct"] / r["absence_av_pct"] if abs(r["absence_av_pct"]) > 1e-6 else float("inf")
        ratios[label] = {"sick": rs, "absence": ra}
        print(f"  {label:>6}  {rs:>16.3f}  {ra:>16.3f}")

    # absolute difference (in pp)
    print(f"\n[Step 4b] commercial − residential averted% (percentage-point difference)")
    print(f"  {'label':>6}  {'sick_worker pp':>16}  {'sick_student pp':>17}")
    for label in ("A", "B"):
        c = pc_summary[label]["commercial"]
        r = pc_summary[label]["residential"]
        d_sick = c["sick_av_pct"] - r["sick_av_pct"]
        d_absent = c["absence_av_pct"] - r["absence_av_pct"]
        print(f"  {label:>6}  {d_sick:>+16.3f}   {d_absent:>+16.3f}")

    # --- Step 5: plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    # (a) ratio distribution colored by class
    axes[0].scatter(ratio[region == "residential"],
                    [0.0] * (region == "residential").sum(),
                    s=4, alpha=0.4, c="#1a5490", label=f"residential ({n_res})")
    axes[0].scatter(ratio[region == "mixed"],
                    [0.5] * (region == "mixed").sum(),
                    s=4, alpha=0.4, c="grey", label=f"mixed ({n_mix})")
    axes[0].scatter(ratio[region == "commercial"],
                    [1.0] * (region == "commercial").sum(),
                    s=4, alpha=0.4, c="#c0392b", label=f"commercial ({n_com})")
    axes[0].axvline(q33, color="grey", ls="--", lw=1)
    axes[0].axvline(q66, color="grey", ls="--", lw=1)
    axes[0].set_xlabel("day-worker / night-worker ratio")
    axes[0].set_yticks([0, 0.5, 1.0])
    axes[0].set_yticklabels(["residential", "mixed", "commercial"])
    axes[0].set_xscale("log")
    axes[0].set_title("classification by day/night worker ratio")
    axes[0].legend(loc="best"); axes[0].grid(True, alpha=0.3)

    rnames = ("residential", "mixed", "commercial")
    x = np.arange(len(rnames)); width = 0.35
    for ax, polkey, title in [
        (axes[1], "sick_av_pct", "sick worker absence (p_work=0.4) — averted %"),
        (axes[2], "absence_av_pct", "sick student absence (p_school=0.5) — averted %"),
    ]:
        for i, label in enumerate(("A", "B")):
            vals = [pc_summary[label][r][polkey] for r in rnames]
            ax.bar(x + (i - 0.5) * width, vals, width, label=label,
                    color=["#1a5490", "#c0392b"][i], alpha=0.85)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(rnames)
        ax.set_ylabel("averted % of baseline (per capita)")
        ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Spatial heterogeneity v2 — day/night-worker ratio classification\n"
                  f"posterior mean β, holiday ON, amp={AMP}, 2019-20", y=1.02)
    fig.tight_layout()
    out = "presentations/figures/spatial_heterogeneity_AB_v2.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")

    out_json = "outputs/metapop/spatial_heterogeneity_AB_v2.json"
    with open(out_json, "w") as f:
        json.dump({
            "season": SEASON, "amp": AMP,
            "ratio_quantiles": {"q33": float(q33), "q66": float(q66)},
            "region_counts": {"commercial": n_com, "mixed": n_mix, "residential": n_res},
            "top_commercial_codes": [str(codes[i]) for i in top_com],
            "top_residential_codes": [str(codes[i]) for i in top_res],
            "per_capita": pc_summary,
            "commercial_over_residential_ratio": ratios,
        }, f, indent=2, default=float)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
