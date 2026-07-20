"""6-season overview figures — fit (total + by-age) and policy effect by age.

Setup: Step A+B, shared π FIXED [0.357,0.255,0.067,0.321], per-season R0 (joint),
κ 3-way, φ U-shape, γ CDC Reed. β_{4,s}=derive(R0_s,π,φ), NO refit. Model pred =
infections × γ_report[0.40,0.18,0.25] → HIRA-6 weekly. Data = HIRA weekly (gray).

Fig1: 2×3 total-fit (all-age). Fig2: 6×6 by-age fit grid. Fig3: 6×6 policy
(p_work ∈ {1.0,0.6,0.2}) whole-season sick-leave.

Style: AppleGothic, unicode_minus False, dpi150. figures/.
"""
from __future__ import annotations
import os, time
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
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax
import sens_workshare_kappa_v2 as S
import policy_compare_school_vs_sickleave as PC
import holiday_reversal_multiseason as HR

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGDIR = REPO_ROOT / "presentations" / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)
F1 = FIGDIR / "viz_fit_6seasons_total.png"
F2 = FIGDIR / "viz_fit_6seasons_byage.png"
F3 = FIGDIR / "viz_policy_6seasons_byage.png"

SEASONS = HR.SEASONS
R0_BY = HR.R0_BY_SEASON
GAMMA15 = jnp.asarray(np.array([0.40]*4 + [0.18]*9 + [0.25]*2))
AGE_COLORS = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#7b3294"]
DATA_GRAY = "#666666"


def load_obs(season):
    t = load_hira_target_by_age(season, sido_codes=list(SUDOGWON_SIDO_CODES), first_peak_only=False)
    nw = t["n_weeks"]
    obs = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = t["hira_counts"][ag]
    return obs   # (52, 6)


def model_pred(setup, kappa, p_school, p_work, work_win=PC.WHOLE_WIN, nw=52):
    inc = PC.run(setup, kappa, p_school, p_work, PC.WHOLE_WIN, work_win)   # (T-1,15)
    return np.asarray(simulation_to_hira_by_age_jax(jnp.asarray(inc), GAMMA15, n_weeks=nw))  # (nw,6)


def main():
    print("=" * 78); print("VIZ multiseason overview (fig1/2/3)"); print("=" * 78)
    t0 = time.perf_counter(); common = HR.build_common()
    setups = {}; obs = {}; pred_base = {}
    for s in SEASONS:
        st, _, _ = HR.season_setup(s, common); setups[s] = st
        obs[s] = load_obs(s)
        pred_base[s] = model_pred(st, 1.0*HR.KAPPA_BASE, 1.0, 1.0)
    print(f"[setup+base sims] {time.perf_counter()-t0:.1f}s")
    weeks = np.arange(52)

    # ═══ FIG 1: total-fit 2×3 ═══
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for k, s in enumerate(SEASONS):
        ax = axes[k // 3, k % 3]
        o = obs[s].sum(axis=1); p = pred_base[s].sum(axis=1)
        ax.plot(weeks, o, "o", color=DATA_GRAY, ms=3.5, alpha=0.75, label="HIRA 데이터")
        ax.plot(weeks, p, "-", color="#B23A48", lw=2, label="모델 baseline")
        ax.set_title(f"{s}   R0={R0_BY[s]:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("주차"); ax.set_ylabel("주간 진료에피소드")
        ax.grid(alpha=0.25)
        rat = o.sum() / max(p.sum(), 1)
        ax.text(0.03, 0.90, f"obs/model={rat:.2f}", transform=ax.transAxes, ha="left",
                fontsize=8, color="#555")
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("6시즌 유행곡선 — 데이터 vs 모델 (전 연령 합)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(F1, bbox_inches="tight"); plt.close(fig)
    print(f"[fig1] {F1}")

    # ═══ FIG 2: by-age 6×6 ═══
    fig, axes = plt.subplots(len(SEASONS), 6, figsize=(16, 13), sharex=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            ax.plot(weeks, obs[s][:, c], "o", color=DATA_GRAY, ms=2.5, alpha=0.7)
            ax.plot(weeks, pred_base[s][:, c], "-", color=AGE_COLORS[c], lw=1.6)
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(f"{ag}세", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{s}\n주간 건수", fontsize=9, fontweight="bold")
            if r == len(SEASONS) - 1:
                ax.set_xlabel("주차", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("6시즌 연령별 fitting — HIRA 데이터(회색 점) vs 모델(선), y축 패널별 독립",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(F2, bbox_inches="tight"); plt.close(fig)
    print(f"[fig2] {F2}")

    # ═══ FIG 3: policy p_work sweep 6×6 ═══
    P_LEVELS = [1.0, 0.6, 0.2]
    P_COLORS = {1.0: "#f4a582", 0.6: "#d6604d", 0.2: "#8b0000"}
    t1 = time.perf_counter()
    pol = {s: {p: model_pred(setups[s], 1.0*HR.KAPPA_BASE, 1.0, p) for p in P_LEVELS} for s in SEASONS}
    print(f"[fig3 policy sims] {time.perf_counter()-t1:.1f}s")
    fig, axes = plt.subplots(len(SEASONS), 6, figsize=(16, 13), sharex=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            for p in P_LEVELS:
                ax.plot(weeks, pol[s][p][:, c], "-", color=P_COLORS[p], lw=1.5,
                        label=f"p_work={p}" if (r == 0 and c == 5) else None)
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(f"{ag}세", fontsize=10, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{s}\n주간 건수", fontsize=9, fontweight="bold")
            if r == len(SEASONS) - 1:
                ax.set_xlabel("주차", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 5].legend(fontsize=7, loc="upper right")
    fig.suptitle("6시즌 병가 정책효과 — p_work별 연령별 유행곡선 (전 기간 상시 병가, κ μ=1.0)\n"
                 "연함=p_work 1.0(무정책) → 진함=0.2(강한 병가). ※ 실험의 시간창 정책과 다름(전 기간)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(F3, bbox_inches="tight"); plt.close(fig)
    print(f"[fig3] {F3}")

    # fit-quality report
    print("\n  obs/model total ratio (full 52wk):")
    for s in SEASONS:
        print(f"   {s}: {obs[s].sum()/max(pred_base[s].sum(),1):.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
