"""Diagnose age × season obs/model ratio at peak window.

Quantifies the "young under-, adult over-prediction" pattern seen in
fit_vs_data figures. For each (season, HIRA age group) compute:
    obs_peak_sum   = obs   counts summed over the obs peak week ±2 weeks
    model_peak_sum = model counts summed over the model peak week ±2 weeks
    r = obs_peak_sum / max(model_peak_sum, 1.0)
- r > 1  → model UNDER-predicts that age/season
- r < 1  → model OVER-predicts that age/season

Decision guide (in comments only — script does not interpret):
- column (age) of r near-constant across 4 seasons → age-constant distortion
  (γ_report mis-scale per age, or φ=1 fixed when truth ≠ 1)
- row 2022-2023 differs sharply from other rows → post-pandemic initial
  immunity issue (TODO-2)
- large phase offset (obs vs model peak week) → dynamics signal
  (σ/γ/seasonality), separate from amplitude ratio

Setup mirrors m2_fit_vs_data_AB.py forward path. Uses posterior MEDIAN β
(single forward per season per {A,B}) for speed — no full draw bands needed
for ratios.
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
import matplotlib.colors as mcolors

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "age_season_ratio.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "age_season_ratio_heatmap.png"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

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
PEAK_HALF_WIN = 2   # ± weeks around each peak


def load_posterior_beta_median(label: str) -> np.ndarray:
    """Return per-season median β: (4 seasons, 4 channels)."""
    data = np.load(f"outputs/calibration/m2_prod_{label}_samples.npz")
    beta = data["beta"].reshape(-1, 4, 4)   # (draws, seasons, channels)
    return np.median(beta, axis=0)          # (4, 4)


def peak_window_sum(arr_1d: np.ndarray, peak_w: int, half: int) -> float:
    """Sum arr_1d[peak_w-half : peak_w+half+1] with bounds clipping."""
    lo = max(0, peak_w - half)
    hi = min(arr_1d.shape[0], peak_w + half + 1)
    return float(arr_1d[lo:hi].sum())


def main():
    print("=" * 70)
    print("DIAGNOSE: age × season obs/model ratio at peak window (±2w)")
    print(f"  4 seasons × 6 HIRA ages × {{A,B}} posterior median β")
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
            "name": s, "obs": obs, "n_weeks": nw,
            "fit_mask": weights.sum(axis=1) > 0,
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

    # ─── Compute ratios per {A,B} ─────────────────────────────
    results = {}
    for label in ("A", "B"):
        beta_med = load_posterior_beta_median(label)   # (4 seasons, 4 channels)
        print(f"\n── label [{label}] — posterior median β per season ──")
        for si, s in enumerate(SEASONS):
            b = beta_med[si]
            print(f"    {s}: β_h/w/s/o = "
                  f"{b[0]:.4f} / {b[1]:.4f} / {b[2]:.4f} / {b[3]:.4f}")

        per_season = []
        t0 = time.perf_counter()
        for si, sd in enumerate(season_data):
            pred = np.asarray(sim_jit(
                jnp.asarray(beta_med[si]), init_states[si], sd["n_weeks"],
            ))   # (n_weeks, 6)
            obs = sd["obs"]
            mask = sd["fit_mask"]                    # (n_weeks,)
            # Restrict peak search to fit window (mask True)
            obs_masked = np.where(mask[:, None], obs, -1e18)
            pred_masked = np.where(mask[:, None], pred, -1e18)

            season_row = {"season": sd["name"], "ages": {}}
            for ai, ag in enumerate(HIRA_AGE_GROUPS):
                obs_pw = int(np.argmax(obs_masked[:, ai]))
                mdl_pw = int(np.argmax(pred_masked[:, ai]))
                obs_sum = peak_window_sum(obs[:, ai], obs_pw, PEAK_HALF_WIN)
                mdl_sum = peak_window_sum(pred[:, ai], mdl_pw, PEAK_HALF_WIN)
                r = obs_sum / max(mdl_sum, 1.0)
                season_row["ages"][ag] = {
                    "obs_peak_week": obs_pw,
                    "model_peak_week": mdl_pw,
                    "phase_offset_weeks": mdl_pw - obs_pw,
                    "obs_peak_sum_pm2w": obs_sum,
                    "model_peak_sum_pm2w": mdl_sum,
                    "ratio": r,
                }
            per_season.append(season_row)
        wall = time.perf_counter() - t0
        print(f"    forward wall: {wall:.1f}s")
        results[label] = {
            "beta_median_per_season": beta_med.tolist(),
            "seasons": per_season,
        }

    # ─── Console tables: r (rows=seasons, cols=ages) ─────────
    print("\n" + "=" * 78)
    print(f"  Ratio r = obs_peak_sum(±{PEAK_HALF_WIN}w) / model_peak_sum(±{PEAK_HALF_WIN}w)")
    print(f"  r > 1: model UNDER-predicts.  r < 1: model OVER-predicts.")
    print("=" * 78)
    header = "  season       " + "  ".join(f"{ag:>8s}" for ag in HIRA_AGE_GROUPS)
    for label in ("A", "B"):
        print(f"\n  [{label}] ratio table")
        print(header)
        for sr in results[label]["seasons"]:
            row = f"  {sr['season']:12s} "
            for ag in HIRA_AGE_GROUPS:
                r = sr["ages"][ag]["ratio"]
                row += f"  {r:>8.2f}"
            print(row)

    # ─── Phase offsets ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("  Phase offset (model peak − obs peak, weeks)")
    print("=" * 78)
    for label in ("A", "B"):
        print(f"\n  [{label}] phase offsets")
        print(header)
        for sr in results[label]["seasons"]:
            row = f"  {sr['season']:12s} "
            for ag in HIRA_AGE_GROUPS:
                po = sr["ages"][ag]["phase_offset_weeks"]
                row += f"  {po:>+8d}"
            print(row)

    # ─── Save JSON ───────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump({
            "setup": {
                "seasons": SEASONS, "ages": HIRA_AGE_GROUPS,
                "peak_half_window_weeks": PEAK_HALF_WIN,
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3_used": [float(GAMMA_CDC[0]), float(GAMMA_CDC[4]),
                                  float(GAMMA_CDC[13])],
                "beta_source": "posterior median per season (m2_prod_{A,B}_samples.npz)",
            },
            "results": results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Heatmap figure: A and B side-by-side ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    # Use log-scale color so symmetric around r=1
    log_norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdBu_r")    # red=positive=UNDER (r>1), blue=OVER (r<1)
    for ci, label in enumerate(("A", "B")):
        ax = axes[ci]
        mat = np.zeros((len(SEASONS), len(HIRA_AGE_GROUPS)))
        for si, sr in enumerate(results[label]["seasons"]):
            for ai, ag in enumerate(HIRA_AGE_GROUPS):
                mat[si, ai] = np.log10(max(sr["ages"][ag]["ratio"], 1e-3))
        im = ax.imshow(mat, cmap=cmap, norm=log_norm, aspect="auto")
        ax.set_xticks(range(len(HIRA_AGE_GROUPS)))
        ax.set_xticklabels(HIRA_AGE_GROUPS, rotation=30, ha="right")
        ax.set_yticks(range(len(SEASONS)))
        ax.set_yticklabels(SEASONS)
        ax.set_title(f"{label}  (log10 r;  red = under, blue = over)")
        for si in range(len(SEASONS)):
            for ai in range(len(HIRA_AGE_GROUPS)):
                r = results[label]["seasons"][si]["ages"][HIRA_AGE_GROUPS[ai]]["ratio"]
                ax.text(ai, si, f"{r:.2f}", ha="center", va="center",
                         fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.04, label="log10 r")
    fig.suptitle(f"obs/model peak-window (±{PEAK_HALF_WIN}w) ratio  "
                 f"— A (NIMS) vs B (literature)  posterior median β")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
