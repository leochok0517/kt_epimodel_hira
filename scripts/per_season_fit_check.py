"""Per-season INDEPENDENT π+R0 fits → clean epicurve fit (main-text quality).

Contrast to the shared-π multi-season claim: here each season gets its own π_4
and R0 (best per-season fit). Purpose: demonstrate the model reproduces each
season's first wave, and diagnose whether school-age over-estimation persists
under per-season π (→ φ/γ_report structural) or vanishes (→ common-π artifact).

Setup: Step A+B, κ 3-way, φ U-shape, γ CDC Reed. Pin: work σ0.10 (loosened),
school 0.05, home/other 0.15. first_peak_only=True (first wave). 12-start L-BFGS.
Reuses the multiseason_joint_sharedpi independent-fit machinery.

Output: outputs/eda/per_season_fit.json + figures viz_fit_perseason_{total,byage}.png
"""
from __future__ import annotations
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.devices()
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.unicode_minus": False,
    "font.family": "AppleGothic", "font.size": 9,
})

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import simulate_jax, daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
import sens_workshare_kappa_v2 as S
import multiseason_joint_sharedpi as MJ

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "per_season_fit.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
FIGDIR = REPO_ROOT / "presentations" / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
FA = FIGDIR / "viz_fit_perseason_total.png"
FB = FIGDIR / "viz_fit_perseason_byage.png"

SEASONS = MJ.SEASONS
COMMON_PI = np.array([0.357, 0.255, 0.067, 0.321])   # multi-season joint (for comparison)
GAMMA15 = jnp.asarray(np.array([0.40]*4 + [0.18]*9 + [0.25]*2))
PHI = np.array(S.PHI_USHAPE)
AGE_COLORS = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#7b3294"]
DATA_GRAY = "#666666"; MODEL_RED = "#B23A48"


def predict(C, i, R0, pi, nw=52):
    beta = np.asarray(derive_beta_from_R0_simplex(
        C["ngm"], jnp.asarray(R0), jnp.asarray(pi), jnp.asarray(PHI)))
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = [float(x) for x in beta]
    kw["phi_susc"] = jnp.asarray(PHI)
    st = simulate_jax(C["states"][i], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    pred = np.asarray(simulation_to_hira_by_age_jax(jnp.asarray(inc), GAMMA15, n_weeks=nw))
    return pred, beta


def load_full_obs(season):
    t = load_hira_target_by_age(season, sido_codes=list(SUDOGWON_SIDO_CODES), first_peak_only=False)
    obs = np.zeros((t["n_weeks"], 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = t["hira_counts"][ag]
    return obs


def main():
    print("=" * 92)
    print("PER-SEASON INDEPENDENT FIT — π_4 + R0 per season (first wave)")
    print(f"  pin: work σ0.10 school σ0.05 home/other σ0.15 | {MJ.N_STARTS} starts | first_peak_only")
    print("=" * 92)
    t0 = time.perf_counter(); C = MJ.build_common()
    print(f"[common setup] {time.perf_counter()-t0:.1f}s\n")

    fits = {}; preds = {}; full_obs = {}
    for i, s in enumerate(SEASONS):
        ti = time.perf_counter()
        f = MJ.fit_independent(i, C)           # {R0, pi, pi_work, obj, data_nll}
        pred52, beta = predict(C, i, f["R0"], f["pi"])
        # fit-window obs/model per age (weights>0)
        obs_fit = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i])
        mask = w.sum(axis=1) > 0                # weeks in fit window
        om_age = {}
        for a, ag in enumerate(HIRA_AGE_GROUPS):
            om_age[ag] = float(obs_fit[mask, a].sum() / max(pred52[mask, a].sum(), 1.0))
        om_total = float(obs_fit[mask].sum() / max(pred52[mask].sum(), 1.0))
        fits[s] = dict(R0=f["R0"], pi=f["pi"], pi_work=f["pi_work"],
                       beta_4=[float(x) for x in beta], data_nll=f["data_nll"],
                       obs_model_total=om_total, obs_model_by_age=om_age)
        preds[s] = pred52; full_obs[s] = load_full_obs(s)
        print(f"  {s}: R0={f['R0']:.3f} π=[{f['pi'][0]:.3f},{f['pi'][1]:.3f},{f['pi'][2]:.3f},{f['pi'][3]:.3f}] "
              f"obs/model={om_total:.2f} nll={f['data_nll']:.1f} ({time.perf_counter()-ti:.1f}s)")

    # ── console (1) π + R0 + total obs/model ──
    print("\n(1) SEASON  R0    π[home  work  school  other]   obs/model(total)")
    for s in SEASONS:
        f = fits[s]; p = f["pi"]
        print(f"  {s} {f['R0']:.3f}  [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f}]   {f['obs_model_total']:.2f}")

    # ── console (2) season × age obs/model grid ──
    print("\n(2) obs/model by age  (>1 model under-predicts, <1 over-predicts)")
    print(f"  {'season':>11} " + " ".join(f"{ag:>7}" for ag in HIRA_AGE_GROUPS))
    for s in SEASONS:
        om = fits[s]["obs_model_by_age"]
        print(f"  {s:>11} " + " ".join(f"{om[ag]:>7.2f}" for ag in HIRA_AGE_GROUPS))
    # school-age systematic?
    sa = [np.mean([fits[s]["obs_model_by_age"][ag] for ag in ("6-11", "12-17")]) for s in SEASONS]
    ad = [np.mean([fits[s]["obs_model_by_age"][ag] for ag in ("18-44", "45-64")]) for s in SEASONS]
    print(f"  school-age(6-17) mean obs/model = {np.mean(sa):.2f}  (adult 18-64 = {np.mean(ad):.2f})")
    print(f"  → school-age {'OVER-estimated (obs/model<1) persists' if np.mean(sa)<0.9 else 'roughly balanced'} under per-season π")

    # ── console (3) vs common π ──
    print("\n(3) per-season π vs common π [0.357,0.255,0.067,0.321]  (Δ = season − common)")
    print(f"  {'season':>11}  {'π_work':>7} {'Δwork':>7} | {'π_home':>7} {'π_school':>8} {'π_other':>7}")
    for s in SEASONS:
        p = fits[s]["pi"]
        print(f"  {s:>11}  {p[1]:>7.3f} {p[1]-COMMON_PI[1]:>+7.3f} | {p[0]:>7.3f} {p[2]:>8.3f} {p[3]:>7.3f}")
    pw = [fits[s]["pi_work"] for s in SEASONS]
    print(f"  π_work range [{min(pw):.3f}, {max(pw):.3f}] mean {np.mean(pw):.3f}±{np.std(pw):.3f}  (common 0.255)")

    # ── FIG A: total ──
    weeks = np.arange(52)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for k, s in enumerate(SEASONS):
        ax = axes[k // 3, k % 3]
        o = full_obs[s].sum(axis=1); p = preds[s].sum(axis=1)
        ax.plot(weeks, o, "o", color=DATA_GRAY, ms=3.5, alpha=0.75, label="HIRA 데이터")
        ax.plot(weeks, p, "-", color=MODEL_RED, lw=2, label="모델 (시즌별 π)")
        ax.set_title(f"{s}   R0={fits[s]['R0']:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("주차"); ax.set_ylabel("주간 진료에피소드"); ax.grid(alpha=0.25)
        ax.text(0.03, 0.90, f"obs/model={fits[s]['obs_model_total']:.2f}", transform=ax.transAxes,
                ha="left", fontsize=8, color="#555")
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("6시즌 유행곡선 — 데이터 vs 모델 (시즌별 독립 π, 첫 유행파 fit)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FA, bbox_inches="tight"); plt.close(fig)
    print(f"\n[figA] {FA}")

    # ── FIG B: by-age ──
    fig, axes = plt.subplots(len(SEASONS), 6, figsize=(16, 13), sharex=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            ax.plot(weeks, full_obs[s][:, c], "o", color=DATA_GRAY, ms=2.5, alpha=0.7)
            ax.plot(weeks, preds[s][:, c], "-", color=AGE_COLORS[c], lw=1.6)
            ax.grid(alpha=0.2)
            om = fits[s]["obs_model_by_age"][ag]
            ax.text(0.04, 0.86, f"{om:.2f}", transform=ax.transAxes, fontsize=7, color="#777")
            if r == 0: ax.set_title(f"{ag}세", fontsize=10, fontweight="bold")
            if c == 0: ax.set_ylabel(f"{s}\n주간 건수", fontsize=9, fontweight="bold")
            if r == len(SEASONS)-1: ax.set_xlabel("주차", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("6시즌 연령별 fitting — 시즌별 독립 π (셀 숫자 = obs/model, 첫 유행파 fit)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(FB, bbox_inches="tight"); plt.close(fig)
    print(f"[figB] {FB}")

    out = dict(
        meta=dict(seasons=SEASONS, pin_sigma=MJ.SIGMA_PIN.tolist(), common_pi=COMMON_PI.tolist(),
                  n_starts=MJ.N_STARTS, first_peak_only=True, note="per-season independent π+R0"),
        fits=fits,
        school_age_mean_obs_model=float(np.mean(sa)), adult_mean_obs_model=float(np.mean(ad)),
        pi_work_by_season={s: fits[s]["pi_work"] for s in SEASONS},
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}")
    print("=" * 92)


if __name__ == "__main__":
    main()
