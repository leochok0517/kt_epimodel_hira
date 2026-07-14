"""Spatial heterogeneity — commercial vs residential effect of policies (A & B).

Step 1: classify 1,154 admdong by work net inflow (KT mobility).
Step 2: spatial forward at posterior-mean β for A and B, 2019-20, holiday ON.
        Scenarios: baseline, sick_leave (p_work=0.4), sick_student_absence
        (p_school=0.5).
Step 3: per-capita averted by region type (commercial, mixed, residential).
Step 4: ★ commercial/residential ratio per (A, B) × policy — robustness.
Step 5: maps + scatter + bar plot.
"""
from __future__ import annotations
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

from pathlib import Path
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


def load_spatial_inputs(yyyymm):
    print(f"  loading population (1154 admdong) …")
    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)
    n_adm = pop.shape[1]
    print(f"  pop (15, {n_adm})")

    t0 = time.perf_counter()
    print(f"  loading mobility {yyyymm} …")
    M_work = build_M_work(yyyymm, daytype="weekday", admdong_codes=codes, pop_15=pop).astype(np.float64)
    M_other = build_M_other(yyyymm, daytype="weekday", admdong_codes=codes, pop_15=pop).astype(np.float64)
    M_home = build_M_home(n_adm).astype(np.float64)
    M_school = build_M_school(n_adm).astype(np.float64)
    print(f"  mobility built ({time.perf_counter()-t0:.1f}s)")

    matrices_raw = load_contact_matrices()
    matrices = {k: matrices_raw[k] for k in ("C_home", "C_work", "C_school", "C_other")}

    emp = EmploymentParameters.from_kt_data(admdong_codes=codes)
    rho = emp.rho

    return dict(pop_15=pop, codes=codes, n_adm=n_adm, rho=rho,
                C=matrices, M_home=M_home, M_school=M_school,
                M_work=M_work, M_other=M_other)


def build_kwargs(spatial, *, beta_4, phi_full, p_school, p_work):
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    d = dict(
        C_home=jnp.asarray(spatial["C"]["C_home"]),
        C_school=jnp.asarray(spatial["C"]["C_school"]),
        C_work=jnp.asarray(spatial["C"]["C_work"]),
        C_other=jnp.asarray(spatial["C"]["C_other"]),
        M_home=jnp.asarray(spatial["M_home"]),
        M_school=jnp.asarray(spatial["M_school"]),
        M_work=jnp.asarray(spatial["M_work"]),
        M_other=jnp.asarray(spatial["M_other"]),
        pop_15=jnp.asarray(spatial["pop_15"]),
        rho=jnp.asarray(spatial["rho"]),
        kappa=jnp.asarray(disease.kappa_array),
        sigma=disease.sigma, gamma=disease.gamma,
        beta_h=float(beta_4[0]), beta_w=float(beta_4[1]),
        beta_s=float(beta_4[2]), beta_o=float(beta_4[3]),
        phi_susc=jnp.asarray(phi_full),
        p_school=float(p_school), p_work=float(p_work),
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


def main():
    print("=" * 70)
    print(f"Spatial heterogeneity — commercial vs residential (A & B)")
    print("=" * 70)
    spatial = load_spatial_inputs(YYYYMM)

    # --- Step 1: classify by work net inflow ---
    # M_work[a, i, j] = age-a, from admdong i, to admdong j weight (probability-like)
    # work inflow to j: sum_i (i != j) M_work[a, i, j] × something. M is normalized
    # by row-sum or population, but for tercile classification we just need a
    # measure of "commute attractor" — use age-summed sum of column entries
    M_work_age = spatial["M_work"].sum(axis=0)         # (n, n)
    eye = np.eye(spatial["n_adm"])
    inflow = (M_work_age * (1 - eye)).sum(axis=0)      # commuters arriving
    outflow = (M_work_age * (1 - eye)).sum(axis=1)     # commuters leaving
    pop_total = spatial["pop_15"].sum(axis=0)          # (n,)
    net_inflow_pc = (inflow - outflow) / np.maximum(pop_total, 1.0)

    q33 = np.quantile(net_inflow_pc, 1/3)
    q66 = np.quantile(net_inflow_pc, 2/3)
    region = np.where(net_inflow_pc >= q66, "commercial",
              np.where(net_inflow_pc <= q33, "residential", "mixed"))
    n_com = int((region == "commercial").sum())
    n_res = int((region == "residential").sum())
    n_mix = int((region == "mixed").sum())
    print(f"\n[Step 1] region classification (net work inflow per capita)")
    print(f"  upper tercile (commercial):  {n_com}")
    print(f"  lower tercile (residential): {n_res}")
    print(f"  middle (mixed):              {n_mix}")
    print(f"  net_inflow_pc range: [{net_inflow_pc.min():+.4f}, {net_inflow_pc.max():+.4f}]")
    print(f"  upper tercile q66:   {q66:+.4f}")
    print(f"  lower tercile q33:   {q33:+.4f}")
    # Sanity: top 5 commercial admdong (highest net inflow)
    codes_arr = np.asarray(spatial["codes"])
    top_com = np.argsort(-net_inflow_pc)[:5]
    print(f"  top 5 commercial admdong codes: {[codes_arr[i] for i in top_com]}")
    top_res = np.argsort(net_inflow_pc)[:5]
    print(f"  top 5 residential admdong codes: {[codes_arr[i] for i in top_res]}")

    # --- initial state (spatial) ---
    seed_15 = estimate_initial_infected_from_hira(
        SEASON, spatial["pop_15"].sum(axis=1),
        sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        spatial["pop_15"], seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    phi_full = jnp.ones(15)

    # --- Step 2: spatial forward A & B × 3 scenarios ---
    print(f"\n[Step 2] spatial forward — posterior mean β for A and B")
    SCENARIOS = {
        "baseline":               (1.0, 1.0),
        "sick_worker_absence":    (1.0, 0.4),    # sick_leave
        "sick_student_absence":   (0.5, 1.0),    # ex-school_closure
    }
    results = {}
    for label in ("A", "B"):
        beta_4 = load_posterior_mean_beta(label)
        print(f"  {label}: β = {[round(float(x), 4) for x in beta_4]}")
        results[label] = {}
        for name, (psch, pwk) in SCENARIOS.items():
            kw = build_kwargs(spatial, beta_4=beta_4, phi_full=phi_full,
                              p_school=psch, p_work=pwk)
            t0 = time.perf_counter()
            states = simulate_jax(state0, **kw, discretize_time=False)
            states.block_until_ready()
            wall = time.perf_counter() - t0
            states_np = np.asarray(states)
            EIR = states_np[:, 2] + states_np[:, 3] + states_np[:, 4]       # (n_t, 15, n)
            per_adm_total_inc = np.diff(EIR.sum(axis=1), axis=0).sum(axis=0)  # (n,)
            total = float(daily_new_infection_by_age_jax(states).sum())
            results[label][name] = {
                "per_adm_total_inc": per_adm_total_inc,
                "total": total,
                "wall": wall,
            }
            print(f"    [{name:>22}] wall={wall:.1f}s total={total:,.0f}")

    # --- Step 3: per-capita averted by region type ---
    print(f"\n[Step 3] per-capita averted (vs baseline)")
    print(f"  {'label':>6} {'region':>12}  "
          f"{'sick_worker_absence':>22}  {'sick_student_absence':>22}")
    pc_summary = {"A": {}, "B": {}}
    for label in ("A", "B"):
        base = results[label]["baseline"]["per_adm_total_inc"]
        sick = results[label]["sick_worker_absence"]["per_adm_total_inc"]
        absent = results[label]["sick_student_absence"]["per_adm_total_inc"]
        for rname in ("commercial", "mixed", "residential"):
            mask = region == rname
            pop_r = pop_total[mask].sum()
            sick_pc = (base[mask] - sick[mask]).sum() / max(pop_r, 1)
            absent_pc = (base[mask] - absent[mask]).sum() / max(pop_r, 1)
            base_pc = base[mask].sum() / max(pop_r, 1)
            pc_summary[label][rname] = {
                "baseline_pc": float(base_pc),
                "sick_pc": float(sick_pc),
                "absence_pc": float(absent_pc),
                "sick_av_pct": float(sick_pc / base_pc * 100),
                "absence_av_pct": float(absent_pc / base_pc * 100),
            }
            print(f"  {label:>6} {rname:>12}  "
                  f"per-cap {sick_pc*100:>6.3f}% ({sick_pc/base_pc*100:>5.2f}% of base)  "
                  f"per-cap {absent_pc*100:>6.3f}% ({absent_pc/base_pc*100:>5.2f}% of base)")
        print()

    # --- Step 4: commercial / residential ratio (robust direction) ---
    print(f"[Step 4] commercial/residential ratio per (A, B) × policy")
    print(f"  {'label':>6}  {'sick_worker_absence':>22}  {'sick_student_absence':>22}")
    ratios = {}
    for label in ("A", "B"):
        c = pc_summary[label]["commercial"]
        r = pc_summary[label]["residential"]
        # Use averted-as-fraction-of-baseline to compare like-for-like (so we
        # are not just measuring "commercial has more infections per capita")
        ratio_sick = c["sick_av_pct"] / max(r["sick_av_pct"], 1e-6)
        ratio_absent = c["absence_av_pct"] / max(r["absence_av_pct"], 1e-6)
        ratios[label] = {"sick": ratio_sick, "absence": ratio_absent}
        print(f"  {label:>6}  ratio commercial/residential "
              f"{ratio_sick:>6.3f}  {ratio_absent:>6.3f}")
    # robust direction
    def direction(x):
        return "commercial > residential" if x > 1.05 else (
            "residential > commercial" if x < 0.95 else "~ even")
    print(f"\n  sick_worker_absence:    A {direction(ratios['A']['sick'])}, "
          f"B {direction(ratios['B']['sick'])}")
    print(f"  sick_student_absence:   A {direction(ratios['A']['absence'])}, "
          f"B {direction(ratios['B']['absence'])}")

    # --- Step 5: visualization ---
    print(f"\n[Step 5] plots")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) net_inflow distribution colored by class
    axes[0].scatter(net_inflow_pc[region == "residential"],
                    [0.0] * (region == "residential").sum(),
                    s=4, alpha=0.4, c="#1a5490", label=f"residential ({n_res})")
    axes[0].scatter(net_inflow_pc[region == "mixed"],
                    [0.5] * (region == "mixed").sum(),
                    s=4, alpha=0.4, c="grey", label=f"mixed ({n_mix})")
    axes[0].scatter(net_inflow_pc[region == "commercial"],
                    [1.0] * (region == "commercial").sum(),
                    s=4, alpha=0.4, c="#c0392b", label=f"commercial ({n_com})")
    axes[0].axvline(q33, color="grey", ls="--", lw=1)
    axes[0].axvline(q66, color="grey", ls="--", lw=1)
    axes[0].set_xlabel("net work inflow / population")
    axes[0].set_yticks([0, 0.5, 1.0])
    axes[0].set_yticklabels(["residential", "mixed", "commercial"])
    axes[0].set_title("region classification by work net inflow")
    axes[0].legend(loc="best"); axes[0].grid(True, alpha=0.3)

    # (b) per-capita averted % by region — sick_worker_absence
    rnames = ("residential", "mixed", "commercial")
    x = np.arange(len(rnames)); width = 0.35
    for i, label in enumerate(("A", "B")):
        vals = [pc_summary[label][r]["sick_av_pct"] for r in rnames]
        axes[1].bar(x + (i - 0.5) * width, vals, width, label=label,
                    color=["#1a5490", "#c0392b"][i], alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(rnames)
    axes[1].set_ylabel("averted % of baseline (per capita)")
    axes[1].set_title("sick worker absence (병가) — per-capita averted")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    # (c) per-capita averted % by region — sick_student_absence
    for i, label in enumerate(("A", "B")):
        vals = [pc_summary[label][r]["absence_av_pct"] for r in rnames]
        axes[2].bar(x + (i - 0.5) * width, vals, width, label=label,
                    color=["#1a5490", "#c0392b"][i], alpha=0.85)
    axes[2].set_xticks(x); axes[2].set_xticklabels(rnames)
    axes[2].set_ylabel("averted % of baseline (per capita)")
    axes[2].set_title("sick student absence — per-capita averted")
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Spatial heterogeneity — region-type effect under A (NIMS) vs B (literature)\n"
                  f"2019-20, posterior mean β, holiday ON, amp={AMP}", y=1.02)
    fig.tight_layout()
    out = "presentations/figures/spatial_heterogeneity_AB.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  saved {out}")

    # JSON dump
    summary = {
        "season": SEASON, "amp": AMP,
        "region_counts": {"commercial": n_com, "mixed": n_mix, "residential": n_res},
        "net_inflow_quantiles": {"q33": float(q33), "q66": float(q66)},
        "per_capita": pc_summary,
        "commercial_over_residential_ratio": ratios,
    }
    out_json = "outputs/metapop/spatial_heterogeneity_AB.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  saved {out_json}")


if __name__ == "__main__":
    main()
