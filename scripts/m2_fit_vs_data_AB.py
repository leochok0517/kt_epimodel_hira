"""Fit vs observed (HIRA) per age × season, with A (NIMS) and B (literature)
posterior bands overlaid. Also emit an age-resolution rationale figure
showing why home/work signals are buried in the 18-44 HIRA bin.
"""
from __future__ import annotations
import os, time
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
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target_by_age, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
AMP = 0.9
HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
N_DRAWS = 60


def load_posterior_beta(label):
    """Return (n_draws, 4_seasons, 4_channels) β posterior."""
    data = np.load(f"outputs/calibration/m2_prod_{label}_samples.npz")
    return data["beta"].reshape(-1, 4, 4)   # (samples, season, channel)


def main():
    print("=" * 70)
    print("Fit vs observed (A NIMS vs B literature) + age-resolution figure")
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
        p_school=policy.p_school, p_work=policy.p_work,
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
    gamma_15 = jnp.asarray(GAMMA_CDC)

    # initial states per season
    init_states = []
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
        st0 = _build_initial_state_with_age_seed(
            pop_15, seed, seed_e_factor=0.5,
            initial_immunity=R0_IMMUNITY_PROFILE,
            initial_vaccinated_fraction=0.0,
        )
        init_states.append(jnp.asarray(st0))
        nw = tgt["n_weeks"]
        obs = np.zeros((nw, 6)); weights = np.zeros((nw, 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            obs[:, i] = tgt["hira_counts"][ag]
            weights[:, i] = tgt["weights"][ag]
        season_data.append({
            "name": s, "obs": obs, "fit_mask": weights.sum(axis=1) > 0,
            "n_weeks": nw,
        })

    def sim_one(beta_4, state0, n_weeks):
        kw = dict(shared)
        kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
        kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
        kw["phi_susc"] = phi_full
        st = simulate_jax(state0, **kw, discretize_time=False)
        inc_15 = daily_new_infection_by_age_jax(st)
        return simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)

    sim_jit = jax.jit(sim_one, static_argnums=(2,))

    # forward N draws per posterior
    rng = np.random.default_rng(0)
    per_post = {}
    for label in ("A", "B"):
        beta_post = load_posterior_beta(label)               # (n_total, 4, 4)
        n_total = beta_post.shape[0]
        idx = rng.choice(n_total, size=N_DRAWS, replace=False)
        beta_draws = beta_post[idx]                          # (N, 4, 4)
        per_season = []
        for si, sd in enumerate(season_data):
            t0 = time.perf_counter()
            preds = []
            for di in range(N_DRAWS):
                pred = sim_jit(jnp.asarray(beta_draws[di, si]),
                                init_states[si], sd["n_weeks"])
                preds.append(np.asarray(pred))               # (n_weeks, 6)
            preds = np.stack(preds, axis=0)                  # (N, n_weeks, 6)
            per_season.append(preds)
            print(f"  {label} {sd['name']}: {time.perf_counter()-t0:.1f}s")
        per_post[label] = per_season

    # ─── Figure 1: fit vs data ─────────────────────────
    fig, axes = plt.subplots(len(SEASONS), 6, figsize=(22, 14), sharex=False)
    colors = {"A": "#1a5490", "B": "#c0392b"}
    label_map = {"A": "A (NIMS)", "B": "B (literature)"}
    for si, sd in enumerate(season_data):
        obs = sd["obs"]; mask = sd["fit_mask"]
        weeks = np.arange(obs.shape[0])[mask]
        for ai, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[si, ai]
            for label in ("A", "B"):
                p = per_post[label][si]                       # (N, n_weeks, 6)
                q025 = np.quantile(p[:, mask, ai], 0.025, axis=0)
                q50  = np.quantile(p[:, mask, ai], 0.5, axis=0)
                q975 = np.quantile(p[:, mask, ai], 0.975, axis=0)
                ax.fill_between(weeks, q025, q975, color=colors[label], alpha=0.20)
                ax.plot(weeks, q50, color=colors[label], lw=1.5,
                        label=label_map[label] if (si == 0 and ai == 0) else None)
            ax.scatter(weeks, obs[mask, ai], s=10, color="black", zorder=3,
                        label="observed (HIRA)" if (si == 0 and ai == 0) else None)
            if si == 0:
                ax.set_title(ag, fontsize=11)
            if ai == 0:
                ax.set_ylabel(f"{sd['name']}\nweekly claims", fontsize=10)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=11,
                bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Fit vs observed — 4 seasons × 6 HIRA age groups, "
                  "A (NIMS) vs B (literature) posterior median + 95% CI",
                  fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out1 = "presentations/figures/fit_vs_data_AB.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    print(f"saved {out1}")

    # ─── Figure 2: age-resolution rationale ─────────────────────────
    C = load_contact_matrices()
    age_centers = np.arange(15) * 5 + 2     # 0-4 center 2, 5-9 center 7, etc.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # left: per-age rowsum per channel + HIRA boundaries shaded
    ch_colors = {"C_home": "#1a5490", "C_work": "#27ae60",
                 "C_school": "#c0392b", "C_other": "#7f4cb0"}
    ch_labels = {"C_home": "home", "C_work": "work",
                 "C_school": "school", "C_other": "other"}
    for ch, color in ch_colors.items():
        axes[0].plot(age_centers, C[ch].sum(axis=1), "o-",
                      color=color, label=ch_labels[ch], lw=2)
    # HIRA bin boundaries: 0-5, 6-11, 12-17, 18-44, 45-64, 65+
    hira_edges = [0, 6, 12, 18, 45, 65, 75]
    for e in hira_edges:
        axes[0].axvline(e, ls=":", color="grey", alpha=0.7)
    # shade 18-44 bin to emphasise home/work merge
    axes[0].axvspan(18, 45, alpha=0.10, color="#c0392b")
    axes[0].annotate("HIRA bin 18-44 absorbs\nhome (30-44 parent peak)\nand work (25-64) peaks",
                      xy=(36, max(C["C_home"].sum(axis=1)) * 0.9),
                      ha="center", color="#c0392b", fontsize=11,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                 ec="#c0392b", alpha=0.85))
    axes[0].set_xlabel("age (5-year centres)")
    axes[0].set_ylabel("contact rowsum")
    axes[0].set_title("Channel age profile vs HIRA bin boundaries")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # right: HIRA 6 bins as bars, with width = age span. Highlight 18-44.
    hira_names = HIRA_AGE_GROUPS
    bin_starts = [0, 6, 12, 18, 45, 65]
    bin_widths = [6, 6, 6, 27, 20, 10]
    hira_colors = ["#7f7f7f"] * 6
    hira_colors[3] = "#c0392b"            # 18-44 in red
    axes[1].barh(np.arange(6), bin_widths, left=bin_starts,
                  color=hira_colors, alpha=0.75, edgecolor="black")
    for i, name in enumerate(hira_names):
        axes[1].text(bin_starts[i] + bin_widths[i] / 2, i, name,
                     ha="center", va="center", color="white", fontsize=10)
    axes[1].axvline(30, ls="--", color="#1a5490")
    axes[1].text(30, -0.5, "home peak (30-44)", color="#1a5490", fontsize=10)
    axes[1].axvline(25, ls="--", color="#27ae60")
    axes[1].text(25, 5.5, "work peak (25-64)", color="#27ae60", fontsize=10)
    axes[1].set_yticks(np.arange(6))
    axes[1].set_yticklabels(hira_names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("age")
    axes[1].set_xlim(0, 80)
    axes[1].set_title("HIRA 6 bins — width-proportional (18-44 = 27 years)")
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.suptitle("Age-resolution limit — why home/work are non-identifiable from HIRA",
                  fontsize=13)
    fig.tight_layout()
    out2 = "presentations/figures/age_resolution_limit.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
