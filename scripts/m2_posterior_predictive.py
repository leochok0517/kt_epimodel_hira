"""Posterior predictive check for M2 production NUTS posterior.

Loads m2_prod_posterior.nc (or m2_prod_samples.npz fallback), draws N
posterior samples, forward-simulates each season, aggregates to HIRA
weekly counts (6 age groups), and plots obs vs pred 95% credible band.

Output: presentations/figures/posterior_predictive.png
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import json
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
FIGDIR = REPO_ROOT / "presentations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.calibration.gamma_registry import get_active_gamma
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, gamma_triple_to_15,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


def load_posterior_samples():
    """Return dict with 'beta' shape (n_samples, 16). Prefers npz raw samples."""
    npz_path = OUTDIR / "m2_prod_samples.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        # samples come grouped_by_chain: shape (4, 300, ...)
        beta = data["beta"].reshape(-1, 16)  # (1200, 16)
        return {"beta": beta}
    nc_path = OUTDIR / "m2_prod_posterior.nc"
    if nc_path.exists():
        import arviz as az
        idata = az.from_netcdf(str(nc_path))
        beta = idata.posterior["beta"].values  # (chain, draw, 16)
        beta_flat = beta.reshape(-1, 16)
        return {"beta": beta_flat}
    raise FileNotFoundError("No posterior file found (m2_prod_samples.npz or .nc)")


def main():
    print("=" * 60)
    print("Posterior predictive check")
    print("=" * 60)

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]
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
        rho=jnp.asarray(inputs["rho"]),
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
    phi_full = jnp.ones(15)  # φ FIXED at 1.0 (production setting)

    # γ from active registry (cdc_reed2015)
    g_active = get_active_gamma()
    gamma_15 = jnp.asarray(g_active)  # (15,)

    # Load posterior β
    post = load_posterior_samples()
    beta_all = post["beta"]   # (n_samples_total, 16)
    n_total = beta_all.shape[0]
    print(f"  posterior β samples: {n_total}")

    # Subsample for speed
    N_DRAWS = 100
    rng = np.random.default_rng(0)
    idx = rng.choice(n_total, size=N_DRAWS, replace=False)
    beta_draws = beta_all[idx]                # (N_DRAWS, 16)
    print(f"  drawing {N_DRAWS} posterior samples for predictive simulation")

    # Per season: initial state + obs + weights (weights=0 outside fit window)
    season_data = []
    for s in SEASONS:
        tgt = load_hira_target_by_age(
            s, sido_codes=list(SUDOGWON_SIDO_CODES),
            first_peak_only=True, first_peak_end_week=26,
        )
        seed = estimate_initial_infected_from_hira(
            s, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
            gamma_15_assumed=CalibrationParameters().gamma_15,
        )
        state0 = _build_initial_state_with_age_seed(
            pop_15, seed, seed_e_factor=0.5,
            initial_immunity=R0_IMMUNITY_PROFILE,
            initial_vaccinated_fraction=0.0,
        )
        nw = tgt["n_weeks"]
        obs = np.zeros((nw, 6))
        weights = np.zeros((nw, 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            obs[:, i] = tgt["hira_counts"][ag]
            weights[:, i] = tgt["weights"][ag]
        # Determine fit window: rows where weights > 0 anywhere
        fit_mask = (weights.sum(axis=1) > 0)
        n_fit = int(fit_mask.sum())
        season_data.append({
            "state0": jnp.asarray(state0),
            "obs": obs,
            "weights": weights,
            "fit_mask": fit_mask,
            "n_fit": n_fit,
            "n_weeks": nw,
            "name": s,
        })
        print(f"  {s}: n_weeks={nw}, fit window={n_fit} weeks (weights>0)")

    # JIT a single-season simulate-to-hira closure
    def sim_one(beta_4, state0, n_weeks):
        kw = dict(shared)
        kw["beta_h"] = beta_4[0]
        kw["beta_w"] = beta_4[1]
        kw["beta_s"] = beta_4[2]
        kw["beta_o"] = beta_4[3]
        kw["phi_susc"] = phi_full
        states = simulate_jax(state0, **kw, discretize_time=False)
        inc_15 = daily_new_infection_by_age_jax(states)
        pred_hira = simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)
        return pred_hira  # (n_weeks, 6)

    sim_jit = jax.jit(sim_one, static_argnums=(2,))

    # Run posterior predictive: per draw × per season
    print("  simulating ...")
    preds_per_season = []   # list of (N_DRAWS, n_weeks, 6)
    import time
    for si, sd in enumerate(season_data):
        t0 = time.perf_counter()
        per_draw = []
        for di in range(N_DRAWS):
            beta_4 = jnp.asarray(beta_draws[di, si*4:(si+1)*4])
            pred = sim_jit(beta_4, sd["state0"], sd["n_weeks"])
            per_draw.append(np.asarray(pred))
        per_draw = np.stack(per_draw, axis=0)  # (N_DRAWS, n_weeks, 6)
        preds_per_season.append(per_draw)
        print(f"    {sd['name']}: {time.perf_counter()-t0:.1f}s")

    # Build proper posterior predictive: y_rep ~ Poisson(pred_mean). Without
    # the Poisson observation noise layer, the band only reflects parameter
    # uncertainty (~very narrow under tight posterior) and misses the data's
    # natural variability — that gives misleadingly low coverage.
    rng2 = np.random.default_rng(1)
    preds_with_noise = []
    for si in range(len(season_data)):
        pred = preds_per_season[si]                         # (N, n_weeks, 6)
        # one Poisson draw per (sample, week, age) for posterior predictive
        rep = rng2.poisson(np.maximum(pred, 1e-6))          # (N, n_weeks, 6)
        preds_with_noise.append(rep)

    # Compute coverage on fit-window only (weeks the model was actually fit to)
    in_band_total = 0
    total_points = 0
    per_age_coverage = np.zeros(6)
    per_age_count = np.zeros(6)
    for si, sd in enumerate(season_data):
        rep = preds_with_noise[si]                   # (N, n_weeks, 6)
        q025 = np.quantile(rep, 0.025, axis=0)
        q975 = np.quantile(rep, 0.975, axis=0)
        obs = sd["obs"]
        mask = sd["fit_mask"]
        in_band = (obs >= q025) & (obs <= q975)
        in_band_total += int(in_band[mask].sum())
        total_points += int(mask.sum() * 6)
        for ai in range(6):
            per_age_coverage[ai] += int(in_band[mask, ai].sum())
            per_age_count[ai] += int(mask.sum())
    overall_cov = in_band_total / max(total_points, 1) * 100
    print(f"  coverage (obs in 95% band): {overall_cov:.1f}% ({in_band_total}/{total_points})")
    print(f"  per-age coverage:")
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        cov = per_age_coverage[ai] / max(per_age_count[ai], 1) * 100
        print(f"    {ag}: {cov:.1f}%")

    # ── Plot: 4 seasons × 6 age groups (rows × cols). Restrict to fit window. ──
    fig, axes = plt.subplots(4, 6, figsize=(20, 14), sharex=False)
    for si, sd in enumerate(season_data):
        rep = preds_with_noise[si]    # posterior predictive (with Poisson noise)
        q025 = np.quantile(rep, 0.025, axis=0)
        q50 = np.quantile(rep, 0.5, axis=0)
        q975 = np.quantile(rep, 0.975, axis=0)
        obs = sd["obs"]
        mask = sd["fit_mask"]
        weeks = np.arange(obs.shape[0])[mask]
        q025_f = q025[mask]; q50_f = q50[mask]; q975_f = q975[mask]; obs_f = obs[mask]
        for ai, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[si, ai]
            ax.fill_between(weeks, q025_f[:, ai], q975_f[:, ai],
                             color="#1a5490", alpha=0.25, label="95% CI" if (si == 0 and ai == 0) else None)
            ax.plot(weeks, q50_f[:, ai], color="#1a5490", lw=1.5,
                     label="posterior median" if (si == 0 and ai == 0) else None)
            ax.scatter(weeks, obs_f[:, ai], s=8, color="#c0392b",
                        label="obs (HIRA)" if (si == 0 and ai == 0) else None, zorder=3)
            if si == 0:
                ax.set_title(ag, fontsize=11)
            if ai == 0:
                ax.set_ylabel(f"{sd['name']}\nweekly claims", fontsize=10)
            if si == 3:
                ax.set_xlabel("week", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle("Posterior predictive (4 seasons × 6 HIRA age groups) — production M2",
                  fontsize=13, y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGDIR / "posterior_predictive.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    print(f"  saved {out}")

    # Save summary JSON
    summary = {
        "n_draws": int(N_DRAWS),
        "n_total_posterior": int(n_total),
        "overall_coverage_95pct": float(overall_cov),
        "per_age_coverage_95pct": {
            ag: float(per_age_coverage[ai] / max(per_age_count[ai], 1) * 100)
            for ai, ag in enumerate(HIRA_AGE_GROUPS)
        },
    }
    (OUTDIR / "posterior_predictive_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"  saved {OUTDIR / 'posterior_predictive_summary.json'}")


if __name__ == "__main__":
    main()
