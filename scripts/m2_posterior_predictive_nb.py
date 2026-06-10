"""Posterior predictive for NB production. Replaces Poisson noise layer with NB.

Output: presentations/figures/posterior_predictive_nb.png
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
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


def nb_sample(mu, k, rng):
    """Draw Negative-Binomial samples with mean μ and concentration k.

    var = μ + μ²/k. Use Gamma-Poisson hierarchy:
        λ ~ Gamma(k, μ/k);  y ~ Poisson(λ).
    """
    mu_safe = np.maximum(mu, 1e-6)
    shape = k                # numpy parameterizes Gamma(shape, scale)
    scale = mu_safe / k
    lam = rng.gamma(shape, scale, size=mu_safe.shape)
    return rng.poisson(np.maximum(lam, 1e-6))


def main():
    print("=" * 60)
    print("Posterior predictive (NB observation model)")
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
    phi_full = jnp.ones(15)
    g_active = get_active_gamma()
    gamma_15 = jnp.asarray(g_active)

    # Load NB posterior: β (4, 200, 16) + phi_nb (4, 200)
    data = np.load(OUTDIR / "m2_prod_nb_samples.npz")
    beta_grouped = data["beta"]            # (4, 200, 16)
    phi_nb_grouped = data["phi_nb"]        # (4, 200)
    beta_flat = beta_grouped.reshape(-1, 16)
    phi_nb_flat = phi_nb_grouped.reshape(-1)
    n_total = beta_flat.shape[0]
    print(f"  posterior NB samples: {n_total}")

    N_DRAWS = 100
    rng = np.random.default_rng(0)
    idx = rng.choice(n_total, size=N_DRAWS, replace=False)
    beta_draws = beta_flat[idx]
    phi_nb_draws = phi_nb_flat[idx]
    print(f"  drawing {N_DRAWS} posterior samples")
    print(f"  phi_nb posterior median: {float(np.median(phi_nb_draws)):.3f}")

    # Per-season setup
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
        obs = np.zeros((nw, 6)); weights = np.zeros((nw, 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            obs[:, i] = tgt["hira_counts"][ag]
            weights[:, i] = tgt["weights"][ag]
        fit_mask = (weights.sum(axis=1) > 0)
        season_data.append({
            "state0": jnp.asarray(state0), "obs": obs,
            "fit_mask": fit_mask, "n_weeks": nw, "name": s,
        })

    def sim_one(beta_4, state0, n_weeks):
        kw = dict(shared)
        kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
        kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
        kw["phi_susc"] = phi_full
        states = simulate_jax(state0, **kw, discretize_time=False)
        inc_15 = daily_new_infection_by_age_jax(states)
        return simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)

    sim_jit = jax.jit(sim_one, static_argnums=(2,))

    print("  simulating ...")
    import time
    preds_per_season = []
    for si, sd in enumerate(season_data):
        t0 = time.perf_counter()
        per_draw = []
        for di in range(N_DRAWS):
            beta_4 = jnp.asarray(beta_draws[di, si*4:(si+1)*4])
            per_draw.append(np.asarray(sim_jit(beta_4, sd["state0"], sd["n_weeks"])))
        preds_per_season.append(np.stack(per_draw, axis=0))
        print(f"    {sd['name']}: {time.perf_counter()-t0:.1f}s")

    # NB posterior predictive: y_rep ~ NB(μ=pred, k=phi_nb_draw) per draw
    rng2 = np.random.default_rng(1)
    preds_with_nb = []
    for si in range(len(season_data)):
        pred = preds_per_season[si]                 # (N, n_weeks, 6)
        rep = np.empty_like(pred, dtype=np.int64)
        for di in range(N_DRAWS):
            rep[di] = nb_sample(pred[di], phi_nb_draws[di], rng2)
        preds_with_nb.append(rep)

    # Coverage on fit window
    in_band_total = 0
    total_points = 0
    per_age_cov = np.zeros(6); per_age_n = np.zeros(6)
    for si, sd in enumerate(season_data):
        rep = preds_with_nb[si]
        q025 = np.quantile(rep, 0.025, axis=0)
        q975 = np.quantile(rep, 0.975, axis=0)
        obs = sd["obs"]; mask = sd["fit_mask"]
        in_band = (obs >= q025) & (obs <= q975)
        in_band_total += int(in_band[mask].sum())
        total_points += int(mask.sum() * 6)
        for ai in range(6):
            per_age_cov[ai] += int(in_band[mask, ai].sum())
            per_age_n[ai] += int(mask.sum())
    overall_cov = in_band_total / max(total_points, 1) * 100
    print(f"  coverage (obs in 95% band): {overall_cov:.1f}% ({in_band_total}/{total_points})")
    print(f"  per-age coverage:")
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        cov = per_age_cov[ai] / max(per_age_n[ai], 1) * 100
        print(f"    {ag}: {cov:.1f}%")

    # Plot
    fig, axes = plt.subplots(4, 6, figsize=(20, 14), sharex=False)
    for si, sd in enumerate(season_data):
        rep = preds_with_nb[si]
        q025 = np.quantile(rep, 0.025, axis=0)
        q50 = np.quantile(rep, 0.5, axis=0)
        q975 = np.quantile(rep, 0.975, axis=0)
        obs = sd["obs"]; mask = sd["fit_mask"]
        weeks = np.arange(obs.shape[0])[mask]
        q025_f = q025[mask]; q50_f = q50[mask]; q975_f = q975[mask]; obs_f = obs[mask]
        for ai, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[si, ai]
            ax.fill_between(weeks, q025_f[:, ai], q975_f[:, ai],
                             color="#1a5490", alpha=0.25,
                             label="95% CI (NB)" if (si == 0 and ai == 0) else None)
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
    fig.suptitle(
        f"Posterior predictive (NB obs, φ_nb ≈ {float(np.median(phi_nb_draws)):.2f}) "
        f"— overall 95% coverage {overall_cov:.0f}%", fontsize=13, y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGDIR / "posterior_predictive_nb.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    print(f"  saved {out}")

    summary = {
        "n_draws": int(N_DRAWS),
        "n_total_posterior": int(n_total),
        "phi_nb_median": float(np.median(phi_nb_draws)),
        "overall_coverage_95pct": float(overall_cov),
        "per_age_coverage_95pct": {
            ag: float(per_age_cov[ai] / max(per_age_n[ai], 1) * 100)
            for ai, ag in enumerate(HIRA_AGE_GROUPS)
        },
    }
    (OUTDIR / "posterior_predictive_nb_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"  saved {OUTDIR / 'posterior_predictive_nb_summary.json'}")


if __name__ == "__main__":
    main()
