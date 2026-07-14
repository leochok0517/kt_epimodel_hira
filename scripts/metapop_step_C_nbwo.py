"""Step C re-run with normalized NB_WO channels.

Key comparisons vs the original (distorted-channel) Step C:
- sick_leave averted: was 2.4% → should rise (work π 0.001 → 0.11)
- school_closure averted: was 99.4% → should drop (school R0 share 82% → ~44%)
- p_school sweep [0.3, 0.5, 0.7] → graded vs cliff
- per-capita spatial heterogeneity (work-dense vs residential)

Forward at 1,154 admdong with NB_WO posterior mean β, season 2019-20.
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
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters, EmploymentParameters,
)
from kt_epimodel_hira.model.mobility_tensor import (
    build_M_home, build_M_school, build_M_work, build_M_other,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn


SEASON = "2019-2020"
YYYYMM = "201909"
SEASON_IDX = 2

# γ used in NB_WO calibration (level 0.6 × CDC relative)
GAMMA_LEVEL = 0.6
CHILD_REL, ADULT_REL, ELDER_REL = 1.45, 0.65, 0.90


def build_gamma_15():
    K = 3.0 / (CHILD_REL + ADULT_REL + ELDER_REL)
    g_c = GAMMA_LEVEL * K * CHILD_REL
    g_a = GAMMA_LEVEL * K * ADULT_REL
    g_e = GAMMA_LEVEL * K * ELDER_REL
    return np.concatenate([np.full(4, g_c), np.full(9, g_a), np.full(2, g_e)])


def load_spatial_inputs(yyyymm: str):
    df_pop = load_population_15groups()
    N_mat, codes, _ = get_population_matrix(df_pop)
    pop = N_mat.T.astype(np.float64)
    n_adm = pop.shape[1]
    print(f"  pop (15, {n_adm}), total {pop.sum():,.0f}")

    t0 = time.perf_counter()
    M_work = build_M_work(yyyymm, daytype="weekday", admdong_codes=codes, pop_15=pop).astype(np.float64)
    M_other = build_M_other(yyyymm, daytype="weekday", admdong_codes=codes, pop_15=pop).astype(np.float64)
    M_home = build_M_home(n_adm).astype(np.float64)
    M_school = build_M_school(n_adm).astype(np.float64)
    print(f"  mobility loaded ({time.perf_counter()-t0:.1f}s)")

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
    print(f"Step C re-run — NB_WO normalized channels (season {SEASON})")
    print("=" * 70)
    spatial = load_spatial_inputs(YYYYMM)

    # NB_WO posterior mean β for season 2019-20
    nbwo = np.load(REPO_ROOT / "outputs/calibration/m2_prod_nb_wo_samples.npz")
    beta_post = nbwo["beta"].reshape(-1, 16)
    beta_mean_16 = beta_post.mean(axis=0)
    beta_4 = beta_mean_16[SEASON_IDX*4:(SEASON_IDX+1)*4]
    pi_post = nbwo["pi"].reshape(-1, 4, 4).mean(axis=0)[SEASON_IDX]
    R0_post = nbwo["R0"].reshape(-1, 4).mean(axis=0)[SEASON_IDX]
    print(f"  NB_WO β (h, w, s, o) = {[round(float(x), 5) for x in beta_4]}")
    print(f"  NB_WO π (h, w, s, o) = {[round(float(x), 3) for x in pi_post]}")
    print(f"  NB_WO R0 = {float(R0_post):.3f}")

    phi_full = jnp.ones(15)

    # Initial state
    seed_15 = estimate_initial_infected_from_hira(
        SEASON, spatial["pop_15"].sum(axis=1),
        sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        spatial["pop_15"], seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0,
    ))

    # ── C-1+C-2 4 main scenarios ──
    SCENARIOS = {
        "baseline":       (1.0, 1.0),
        "school_closure": (0.5, 1.0),
        "sick_leave":     (1.0, 0.4),
        "comprehensive":  (0.5, 0.4),
    }

    results = {}
    for name, (psch, pwk) in SCENARIOS.items():
        kw = build_kwargs(spatial, beta_4=beta_4, phi_full=phi_full,
                          p_school=psch, p_work=pwk)
        t0 = time.perf_counter()
        states = simulate_jax(state0, **kw, discretize_time=False)
        states.block_until_ready()
        wall = time.perf_counter() - t0
        states_np = np.asarray(states)
        daily_inc_age = np.asarray(daily_new_infection_by_age_jax(states))
        EIR = states_np[:, 2] + states_np[:, 3] + states_np[:, 4]
        per_adm_daily_inc = np.diff(EIR.sum(axis=1), axis=0)
        results[name] = {
            "psch": psch, "pwk": pwk,
            "daily_inc_age": daily_inc_age,
            "per_adm_total": per_adm_daily_inc.sum(axis=0),
            "wall": wall,
        }
        print(f"  [{name:>15}] p_school={psch} p_work={pwk}  "
              f"wall={wall:.1f}s  total={daily_inc_age.sum():,.0f}")

    # ── C-3 averted comparison (this run vs prior distorted-channel run) ──
    base_total = results["baseline"]["daily_inc_age"].sum()
    print(f"\n[C-3] averted % vs baseline ({base_total:,.0f} infections)")
    PRIOR_C = {"school_closure": 99.4, "sick_leave": 2.4, "comprehensive": 99.3}
    print(f"  {'scenario':>15}  {'NEW averted':>13}  {'OLD averted':>12}  {'change':>10}")
    averted_table = []
    for name in ("school_closure", "sick_leave", "comprehensive"):
        scen_total = results[name]["daily_inc_age"].sum()
        averted = base_total - scen_total
        pct = averted / base_total * 100
        prior = PRIOR_C[name]
        diff = pct - prior
        print(f"  {name:>15}  {pct:>12.1f}%  {prior:>11.1f}%  {diff:>+9.1f}%")
        averted_table.append({"scenario": name, "averted_new_pct": float(pct),
                               "averted_old_pct": float(prior),
                               "averted_new_total": float(averted),
                               "scenario_total": float(scen_total)})

    # ── C-4 p_school sweep (graded vs cliff) ──
    print(f"\n[C-4] p_school sweep — check graded response vs cliff")
    sweep = {}
    for p in (0.3, 0.5, 0.7, 0.9):
        kw = build_kwargs(spatial, beta_4=beta_4, phi_full=phi_full,
                          p_school=p, p_work=1.0)
        states = simulate_jax(state0, **kw, discretize_time=False)
        daily_inc = np.asarray(daily_new_infection_by_age_jax(states))
        total = float(daily_inc.sum())
        averted_pct = (base_total - total) / base_total * 100
        sweep[p] = {"total": total, "averted_pct": float(averted_pct)}
        print(f"  p_school = {p}: total = {total:>12,.0f}  "
              f"averted = {averted_pct:>5.1f}%")

    # ── R_eff at school-closure (NGM with modified C_school) ──
    # NGM uses aggregated pop (15,) — the spatial pop (15, 1154) doesn't fit
    # the closure shape; the eigenvalue is dominated by the 15-age structure.
    print(f"\n[C-4b] R_eff at DFE with p_school=0.5 modification "
          f"(school transmission halved)")
    from kt_epimodel_hira.calibration.simple_model import build_aggregated_inputs
    agg = build_aggregated_inputs()
    ngm = make_ngm_eigvalue_fn(
        pop_15=agg["pop_15"], rho=agg["rho"],
        C_home=agg["matrices"]["C_home"], C_work=agg["matrices"]["C_work"],
        C_school=agg["matrices"]["C_school"], C_other=agg["matrices"]["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=ModelParameters().disease.gamma, seasonal_factor=1.7,
    )
    R0_base = float(ngm(beta_4[0], beta_4[1], beta_4[2], beta_4[3], phi_full))
    R0_psch05 = float(ngm(beta_4[0], beta_4[1], 0.5 * beta_4[2], beta_4[3], phi_full))
    R0_pwk04 = float(ngm(beta_4[0], 0.4 * beta_4[1], beta_4[2], beta_4[3], phi_full))
    R0_both = float(ngm(beta_4[0], 0.4 * beta_4[1], 0.5 * beta_4[2], beta_4[3], phi_full))
    print(f"  baseline R0     = {R0_base:.3f}")
    print(f"  R0 with p_school=0.5  → β_s ×0.5 = {R0_psch05:.3f}  "
          f"({'still above' if R0_psch05 > 1 else 'BELOW'} threshold)")
    print(f"  R0 with p_work=0.4   → β_w ×0.4 = {R0_pwk04:.3f}")
    print(f"  R0 with both         → both     = {R0_both:.3f}")

    # ── C-6 per-capita spatial heterogeneity ──
    print(f"\n[C-6] spatial heterogeneity (per-capita)")
    M_work_age = spatial["M_work"].sum(axis=0)
    eye = np.eye(spatial["n_adm"])
    inflow = (M_work_age * (1 - eye)).sum(axis=0)
    pop_total = spatial["pop_15"].sum(axis=0)
    inflow_ratio = inflow / np.maximum(pop_total, 1.0)
    is_work_dense = inflow_ratio > np.median(inflow_ratio)
    print(f"  median work inflow ratio: {np.median(inflow_ratio):.4f}")
    print(f"  work-dense admdong: {int(is_work_dense.sum())} / {spatial['n_adm']}")

    per_capita_summary = []
    for name in ("school_closure", "sick_leave", "comprehensive"):
        diff_adm = (results["baseline"]["per_adm_total"]
                    - results[name]["per_adm_total"])     # (n_adm,)
        averted_pc = diff_adm / np.maximum(pop_total, 1.0)
        pc_dense = averted_pc[is_work_dense].mean()
        pc_resid = averted_pc[~is_work_dense].mean()
        print(f"  {name:>15}: dense {pc_dense:.4f}  residential {pc_resid:.4f}  "
              f"ratio {pc_dense/max(pc_resid, 1e-9):.2f}")
        per_capita_summary.append({
            "scenario": name,
            "per_capita_work_dense": float(pc_dense),
            "per_capita_residential": float(pc_resid),
        })

    # ── save ──
    summary_json = OUTDIR / "step_C_nbwo_results.json"
    json.dump({
        "season": SEASON,
        "beta_nb_wo": [float(x) for x in beta_4],
        "pi_nb_wo": [float(x) for x in pi_post],
        "R0_nb_wo": float(R0_post),
        "averted_comparison": averted_table,
        "p_school_sweep": {str(k): v for k, v in sweep.items()},
        "R_eff_at_DFE": {
            "baseline": R0_base, "p_school_0.5": R0_psch05,
            "p_work_0.4": R0_pwk04, "both": R0_both,
        },
        "per_capita_heterogeneity": per_capita_summary,
    }, open(summary_json, "w"), indent=2, default=float)
    print(f"\nsaved {summary_json}")

    # ── plots ──
    n_days = results["baseline"]["daily_inc_age"].shape[0]
    days = np.arange(n_days)
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    for name in SCENARIOS:
        ax.plot(days, results[name]["daily_inc_age"].sum(axis=1), label=name, lw=2)
    ax.set_xlabel("day of season")
    ax.set_ylabel("new infections / day (total)")
    ax.set_title(f"Scenario forward — {SEASON} (NB_WO normalized channels)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "metapop_scenarios_nbwo.png", dpi=130, bbox_inches="tight")
    print(f"  saved {FIGDIR / 'metapop_scenarios_nbwo.png'}")

    # comparison bar: old vs new averted
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    names = ["school_closure", "sick_leave", "comprehensive"]
    new_vals = [PRIOR_C[n] for n in names]   # old
    new_vals_nbwo = [a["averted_new_pct"] for a in averted_table]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w/2, new_vals,        w, label="OLD (distorted channels)", color="#c0392b", alpha=0.85)
    ax.bar(x + w/2, new_vals_nbwo,   w, label="NEW (NB_WO normalized)",   color="#1a5490", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("averted (%)")
    ax.set_title("Policy effect — old vs new (channel normalization)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "metapop_averted_old_vs_new.png", dpi=130, bbox_inches="tight")
    print(f"  saved {FIGDIR / 'metapop_averted_old_vs_new.png'}")


if __name__ == "__main__":
    main()
