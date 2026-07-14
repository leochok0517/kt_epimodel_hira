"""Step C: policy scenario forward at 1,154 admdong with NB posterior mean β.

C-1: baseline forward (also serves as A-2 infra sanity)
C-2: 4 scenarios (baseline, school_closure, sick_leave, comprehensive)
C-3: averted infections (overall, by age, by admdong)
C-4: spatial heterogeneity (work-dense vs residential-dense)
C-5: plots saved to presentations/figures/

Uses 2019-2020 season (mid-data, full HIRA coverage). Other seasons later.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import json
import time
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "metapop"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR = REPO_ROOT / "presentations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_data.data.load_population import load_population_15groups, get_population_matrix
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.calibration.gamma_registry import get_active_gamma
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
SEASON_IDX = 2          # 0:2017-18, 1:2018-19, 2:2019-20, 3:2022-23 in NB posterior


def load_spatial_inputs(yyyymm: str):
    print(f"  loading population (1154 admdong) …")
    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)                # (15, 1154)
    n_adm = pop.shape[1]
    print(f"  pop matrix (15, {n_adm})  total={pop.sum():,.0f}")

    t0 = time.perf_counter()
    print(f"  loading mobility {yyyymm} (work/other) …")
    M_work = build_M_work(yyyymm, daytype="weekday", admdong_codes=codes, pop_15=pop).astype(np.float64)
    M_other = build_M_other(yyyymm, daytype="weekday", admdong_codes=codes, pop_15=pop).astype(np.float64)
    M_home = build_M_home(n_adm).astype(np.float64)
    M_school = build_M_school(n_adm).astype(np.float64)
    print(f"  mobility built ({time.perf_counter()-t0:.1f}s)  M_work {M_work.shape}")

    matrices_raw = load_contact_matrices()
    matrices = {k: matrices_raw[k] for k in ("C_home", "C_work", "C_school", "C_other")}

    emp = EmploymentParameters.from_kt_data(admdong_codes=codes)
    rho = emp.rho           # (n_adm, 15)

    return dict(
        pop_15=pop, codes=codes, n_adm=n_adm, rho=rho,
        C=matrices, M_home=M_home, M_school=M_school, M_work=M_work, M_other=M_other,
    )


def build_kwargs(spatial, *, beta_4, phi_full, p_school, p_work):
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    return dict(
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
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=disease.seasonality_amp,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )


def main():
    print("=" * 70)
    print(f"Step C — scenario forward at 1,154 admdong (season {SEASON})")
    print("=" * 70)

    spatial = load_spatial_inputs(YYYYMM)

    # NB posterior mean β for 2019-20 season
    npz_path = REPO_ROOT / "outputs" / "calibration" / "m2_prod_nb_samples.npz"
    data = np.load(npz_path)
    beta_post = data["beta"].reshape(-1, 16)
    beta_mean_16 = beta_post.mean(axis=0)
    beta_4 = beta_mean_16[SEASON_IDX*4:(SEASON_IDX+1)*4]
    print(f"  β (h, w, s, o) = {[round(float(x), 5) for x in beta_4]}")

    phi_full = jnp.ones(15)

    # Initial state (1154 admdong) — distribute seed by population
    seed_15 = estimate_initial_infected_from_hira(
        SEASON, spatial["pop_15"].sum(axis=1),
        sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    print(f"  initial I (total per age) = {[int(x) for x in seed_15]}")
    state0_np = _build_initial_state_with_age_seed(
        spatial["pop_15"], seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0,
    )
    state0 = jnp.asarray(state0_np)
    print(f"  state0 shape: {state0.shape} (5, 15, 1154)")

    # Scenarios — vary (p_school, p_work). Lower = more isolation/attendance reduction.
    SCENARIOS = {
        "baseline":       (1.0, 1.0),
        "school_closure": (0.5, 1.0),    # 50% school attendance
        "sick_leave":     (1.0, 0.4),    # 40% work attendance (sick-leave enhanced)
        "comprehensive":  (0.5, 0.4),
    }

    results = {}
    for name, (psch, pwk) in SCENARIOS.items():
        print(f"\n[{name}] p_school={psch}, p_work={pwk}  forward …")
        kw = build_kwargs(spatial, beta_4=beta_4, phi_full=phi_full,
                          p_school=psch, p_work=pwk)
        t0 = time.perf_counter()
        states = simulate_jax(state0, **kw, discretize_time=False)
        states.block_until_ready()
        wall = time.perf_counter() - t0
        # Daily new infections by age (summed over admdong inside the function)
        daily_inc_age = np.asarray(daily_new_infection_by_age_jax(states))  # (n_days-1, 15)
        # Per-admdong daily infections (need explicit reduction)
        states_np = np.asarray(states)
        E = states_np[:, 2].sum(axis=1)    # (n_t, n_adm) — summed over age
        I = states_np[:, 3].sum(axis=1)
        R = states_np[:, 4].sum(axis=1)
        per_admdong_daily_inc = np.diff(E + I + R, axis=0)   # (n_t-1, n_adm)
        # Per-age × per-admdong daily inc (needs full E+I+R sum diff)
        EIR_age_adm = states_np[:, 2:5].sum(axis=1)          # wrong — let me redo
        results[name] = {
            "daily_inc_age": daily_inc_age,
            "per_admdong_daily_inc": per_admdong_daily_inc,
            "states_np": states_np,
            "wall": wall,
        }
        print(f"  wall: {wall:.1f}s   total new infections (season): "
              f"{daily_inc_age.sum():,.0f}")

    # Per-age × per-admdong inc (recompute properly)
    for name, r in results.items():
        states_np = r["states_np"]
        # daily new = diff over (E+I+R) per (age, admdong)
        EIR = states_np[:, 2] + states_np[:, 3] + states_np[:, 4]   # (n_t, 15, n_adm)
        r["daily_inc_age_adm"] = np.diff(EIR, axis=0)               # (n_t-1, 15, n_adm)

    # --- C-1 sanity: baseline aggregated total vs HIRA target ---
    print("\n[C-1 sanity] baseline weekly aggregated vs HIRA")
    tgt = load_hira_target_by_age(
        SEASON, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    # baseline raw weekly (sum over age × admdong)
    inc_total_daily = results["baseline"]["daily_inc_age"].sum(axis=1)   # (n_days-1,)
    # convert to weekly (52 first weeks)
    nweeks_total = len(inc_total_daily) // 7
    inc_total_weekly = inc_total_daily[:nweeks_total*7].reshape(nweeks_total, 7).sum(axis=1)
    # HIRA observed total weekly (sum over age groups)
    obs_total_weekly = sum(tgt["hira_counts"][ag] for ag in tgt["hira_counts"])
    print(f"  model raw weekly inc total (first 6 wk): {inc_total_weekly[:6].astype(int).tolist()}")
    print(f"  HIRA obs weekly (first 6 wk):           {obs_total_weekly[:6].astype(int).tolist()}")
    # γ scaling for fair comparison
    g_active = np.asarray(get_active_gamma())
    inc_weekly_age = np.zeros((nweeks_total, 15))
    inc_age = results["baseline"]["daily_inc_age"]
    for a in range(15):
        inc_weekly_age[:, a] = inc_age[:nweeks_total*7, a].reshape(nweeks_total, 7).sum(axis=1)
    # apply γ per age → HIRA-like reported
    reported_weekly_age = inc_weekly_age * g_active[None, :]
    reported_weekly_total = reported_weekly_age.sum(axis=1)
    print(f"  model reported weekly (γ-scaled, first 6 wk): "
          f"{reported_weekly_total[:6].astype(int).tolist()}")
    print(f"  → ratio model/obs at peak: "
          f"{reported_weekly_total.max() / obs_total_weekly.max():.3f}")

    # --- C-3 averted ---
    print("\n[C-3] averted infections (baseline - scenario, raw incidence)")
    baseline_total = results["baseline"]["daily_inc_age"].sum()
    averted_table = []
    for name in SCENARIOS:
        if name == "baseline":
            continue
        scenario_total = results[name]["daily_inc_age"].sum()
        averted = baseline_total - scenario_total
        averted_table.append({
            "scenario": name,
            "total_baseline": float(baseline_total),
            "total_scenario": float(scenario_total),
            "averted_total": float(averted),
            "averted_pct": float(averted / baseline_total * 100),
            "averted_by_age": (results["baseline"]["daily_inc_age"].sum(axis=0)
                                - results[name]["daily_inc_age"].sum(axis=0)).tolist(),
        })
        print(f"  {name:15}: averted = {averted:>12,.0f}  ({averted/baseline_total*100:5.1f}%)")
    print(f"  baseline total infected: {baseline_total:,.0f}")

    # --- C-4 spatial heterogeneity ---
    print("\n[C-4] spatial heterogeneity")
    # work_inflow_ratio per admdong: total work travelers in from elsewhere / population
    # Use the destination column sums of M_work (workers' destination), excluding self-loops
    M_work_summed_age = spatial["M_work"].sum(axis=0)   # (n_adm, n_adm) — age-summed
    # entries: from origin i to dest j → mobility weight
    # inflow to j = sum_i M[i, j] where i != j
    eye = np.eye(spatial["n_adm"])
    inflow = (M_work_summed_age * (1 - eye)).sum(axis=0)   # (n_adm,)
    pop_total = spatial["pop_15"].sum(axis=0)               # (n_adm,)
    inflow_ratio = inflow / np.maximum(pop_total, 1.0)
    is_work_dense = inflow_ratio > np.median(inflow_ratio)
    print(f"  median work inflow ratio: {np.median(inflow_ratio):.4f}")
    print(f"  work-dense admdong: {int(is_work_dense.sum())} / {spatial['n_adm']}")

    averted_by_class = []
    for name in ("school_closure", "sick_leave", "comprehensive"):
        diff_adm = (results["baseline"]["per_admdong_daily_inc"].sum(axis=0)
                    - results[name]["per_admdong_daily_inc"].sum(axis=0))    # (n_adm,)
        a_dense = diff_adm[is_work_dense].sum()
        a_resid = diff_adm[~is_work_dense].sum()
        averted_by_class.append({
            "scenario": name,
            "averted_work_dense": float(a_dense),
            "averted_residential": float(a_resid),
            "ratio_dense_per_resid": float(a_dense / max(a_resid, 1)),
        })
        print(f"  {name:15}: dense {a_dense:>11,.0f}  residential {a_resid:>11,.0f}  "
              f"ratio {a_dense/max(a_resid,1):.2f}")

    # --- save results ---
    out_json = OUTDIR / "step_C_results.json"
    json.dump({
        "season": SEASON, "yyyymm": YYYYMM,
        "beta_4": [float(x) for x in beta_4],
        "scenarios": list(SCENARIOS),
        "averted": averted_table,
        "averted_by_admdong_class": averted_by_class,
        "calibration_sanity": {
            "model_peak_reported_weekly": float(reported_weekly_total.max()),
            "obs_peak_weekly": float(obs_total_weekly.max()),
            "ratio_peak": float(reported_weekly_total.max() / obs_total_weekly.max()),
        },
    }, open(out_json, "w"), indent=2, default=float)
    print(f"\n  saved {out_json}")

    # Persist arrays for downstream plots / PSA
    np.savez(
        OUTDIR / "step_C_arrays.npz",
        **{f"{name}_daily_inc_age": r["daily_inc_age"] for name, r in results.items()},
        **{f"{name}_inc_age_adm_total": r["daily_inc_age_adm"].sum(axis=0)  # (15, n_adm)
           for name, r in results.items()},
        admdong_codes=np.array(spatial["codes"]),
        is_work_dense=is_work_dense,
        inflow_ratio=inflow_ratio,
        pop_total=pop_total,
    )
    print(f"  saved {OUTDIR / 'step_C_arrays.npz'}")

    # --- C-5 plots ---
    n_days = results["baseline"]["daily_inc_age"].shape[0]
    days = np.arange(n_days)

    # (a) baseline vs scenarios — total daily incidence curve
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    for name in SCENARIOS:
        ax.plot(days, results[name]["daily_inc_age"].sum(axis=1),
                label=name, lw=2)
    ax.set_xlabel("day of season")
    ax.set_ylabel("new infections / day (total)")
    ax.set_title(f"Scenario forward — {SEASON} (1,154 admdong, NB posterior mean β)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "metapop_scenarios_total.png", dpi=130, bbox_inches="tight")
    print(f"  saved {FIGDIR / 'metapop_scenarios_total.png'}")

    # (b) per-age averted bar chart
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    width = 0.25
    age_ticks = np.arange(15)
    for i, name in enumerate(("school_closure", "sick_leave", "comprehensive")):
        diff = (results["baseline"]["daily_inc_age"].sum(axis=0)
                - results[name]["daily_inc_age"].sum(axis=0))
        ax.bar(age_ticks + (i-1)*width, diff, width, label=name)
    ax.set_xticks(age_ticks)
    ax.set_xticklabels([f"{a*5}–{a*5+4}" for a in range(14)] + ["70+"], rotation=45)
    ax.set_xlabel("age group (NIMS 15)")
    ax.set_ylabel("averted infections (vs baseline)")
    ax.set_title(f"Per-age averted — {SEASON}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "metapop_averted_by_age.png", dpi=130, bbox_inches="tight")
    print(f"  saved {FIGDIR / 'metapop_averted_by_age.png'}")

    # (c) per-admdong averted heat-like scatter (work_inflow vs averted)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, name in zip(axes, ("school_closure", "sick_leave", "comprehensive")):
        diff_adm = (results["baseline"]["per_admdong_daily_inc"].sum(axis=0)
                    - results[name]["per_admdong_daily_inc"].sum(axis=0))
        ax.scatter(inflow_ratio, diff_adm / np.maximum(pop_total, 1) * 100,
                   s=6, alpha=0.45, c="#1a5490")
        ax.axvline(np.median(inflow_ratio), color="grey", ls="--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("work inflow ratio")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("averted per 100 residents")
    fig.suptitle(f"Spatial heterogeneity of policy effect — {SEASON}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "metapop_spatial_heterogeneity.png", dpi=130, bbox_inches="tight")
    print(f"  saved {FIGDIR / 'metapop_spatial_heterogeneity.png'}")


if __name__ == "__main__":
    main()
