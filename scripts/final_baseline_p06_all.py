"""FINAL recalc + figures at presenteeism baseline p_work=0.6 (presentation).

Tasks (point estimate, 6 normal seasons, first_peak, Step A+B, κ3-way, φ U-shape,
γ CDC Reed):
  1. per-season independent π_4 + R0 (pin work σ0.10) → per_season_fit_p06.json
  2. joint shared-π + per-season R0 → joint_fit_p06.json
  3. holiday sign-reversal (windowed sick-leave 0.6→0.4, term vs vacation)
     → holiday_reversal_p06.json
  4. figures: total fit, by-age fit, policy sweep, holiday reversal (per-season π)

Baseline p_work=0.6 (sick workers 40% absent) applied in every forward sim.
Reuses multiseason_joint_sharedpi machinery; only shared["p_work"]=0.6 differs.
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
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target_by_age, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import simulate_jax, daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
import sens_workshare_kappa_v2 as S
import multiseason_joint_sharedpi as MJ

REPO_ROOT = Path(__file__).resolve().parent.parent
ED = REPO_ROOT / "outputs" / "eda"; ED.mkdir(parents=True, exist_ok=True)
FIGDIR = REPO_ROOT / "presentations" / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)

SEASONS = MJ.SEASONS
BASE_PWORK = 0.6
PHI = np.array(S.PHI_USHAPE)
GAMMA15 = jnp.asarray(np.array([0.40]*4 + [0.18]*9 + [0.25]*2))
COMMON_PI = np.array([0.357, 0.255, 0.067, 0.321])
TERM_WIN = (70.0, 113.0); VAC_WIN = (113.0, 183.0); WHOLE = (-1.0e9, 1.0e9)
CHILD = ["0-5", "6-11", "12-17"]; SCHOOL = ["6-11", "12-17"]
AGE_COLORS = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#7b3294"]
DATA_GRAY = "#666666"; MODEL_RED = "#B23A48"; BLUE = "#2166AC"; RED = "#B23A48"


def build_common_p06():
    C = MJ.build_common()
    C["shared"]["p_work"] = BASE_PWORK           # presenteeism baseline
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for idx, wt in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, idx] = wt
    pflat = np.asarray(C["shared"]["pop_15"])
    pflat = pflat.sum(axis=1) if pflat.ndim == 2 else pflat
    C["H"] = H; C["pop6"] = H @ pflat
    C["full_obs"] = {}
    for s in SEASONS:
        t = load_hira_target_by_age(s, sido_codes=list(SUDOGWON_SIDO_CODES), first_peak_only=False)
        o = np.zeros((t["n_weeks"], 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            o[:, i] = t["hira_counts"][ag]
        C["full_obs"][s] = o
    return C


def run_inc(C, i, beta_4, p_work=BASE_PWORK, p_school=1.0, work_win=WHOLE, work_base=1.0):
    kw = dict(C["shared"]); beta_4 = jnp.asarray(beta_4)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]; kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_work_baseline"] = work_base
    st = simulate_jax(C["states"][i], **kw, discretize_time=False)
    return daily_new_infection_by_age_jax(st)


def pred_hira(C, inc, nw=52):
    return np.asarray(simulation_to_hira_by_age_jax(jnp.asarray(inc), GAMMA15, n_weeks=nw))


def attack6(C, inc):
    return C["H"] @ np.asarray(inc).sum(axis=0)


def beta_of(C, R0, pi):
    return np.asarray(derive_beta_from_R0_simplex(C["ngm"], jnp.asarray(R0), jnp.asarray(pi), jnp.asarray(PHI)))


def main():
    print("=" * 96)
    print("FINAL baseline p_work=0.6 — recalc + figures (6 seasons, point estimate)")
    print("=" * 96)
    t0 = time.perf_counter(); C = build_common_p06()
    print(f"[common setup, baseline p_work=0.6] {time.perf_counter()-t0:.1f}s\n")

    # ═══ TASK 1: per-season independent fits ═══
    print("── TASK 1: per-season independent π+R0 (baseline 0.6) ──")
    t1 = {}; beta_by_season = {}; pred_by_season = {}
    for i, s in enumerate(SEASONS):
        ti = time.perf_counter()
        f = MJ.fit_independent(i, C)                 # π+R0 at p_work=0.6 (from C shared)
        b = beta_of(C, f["R0"], f["pi"]); beta_by_season[s] = b
        p52 = pred_hira(C, run_inc(C, i, b, p_work=BASE_PWORK)); pred_by_season[s] = p52
        obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i]); mask = w.sum(1) > 0
        om_age = {ag: float(obs[mask, a].sum()/max(p52[mask, a].sum(), 1.0))
                  for a, ag in enumerate(HIRA_AGE_GROUPS)}
        om_tot = float(obs[mask].sum()/max(p52[mask].sum(), 1.0))
        t1[s] = dict(R0=f["R0"], pi=f["pi"], pi_work=f["pi_work"], beta_4=[float(x) for x in b],
                     data_nll=f["data_nll"], obs_model_total=om_tot, obs_model_by_age=om_age)
        print(f"  {s}: R0={f['R0']:.3f} π=[{f['pi'][0]:.3f},{f['pi'][1]:.3f},{f['pi'][2]:.3f},{f['pi'][3]:.3f}] "
              f"obs/model={om_tot:.2f} ({time.perf_counter()-ti:.1f}s)")
    print("\n  season × age obs/model:")
    print(f"  {'season':>11} " + " ".join(f"{ag:>7}" for ag in HIRA_AGE_GROUPS))
    for s in SEASONS:
        print(f"  {s:>11} " + " ".join(f"{t1[s]['obs_model_by_age'][ag]:>7.2f}" for ag in HIRA_AGE_GROUPS))
    sa = np.mean([np.mean([t1[s]['obs_model_by_age'][ag] for ag in ('6-11', '12-17')]) for s in SEASONS])
    ad = np.mean([np.mean([t1[s]['obs_model_by_age'][ag] for ag in ('18-44', '45-64')]) for s in SEASONS])
    print(f"  school-age(6-17) obs/model mean={sa:.2f}  adult(18-64)={ad:.2f}  "
          f"→ school-age {'과대(obs/model<1) 지속' if sa < 0.9 else '균형'}")
    (ED/"per_season_fit_p06.json").write_text(json.dumps(dict(
        meta=dict(baseline_pwork=BASE_PWORK, seasons=SEASONS, first_peak=True), fits=t1,
        school_age_obs_model=float(sa), adult_obs_model=float(ad)), indent=2, default=float))

    # ═══ TASK 2: joint shared-π ═══
    print("\n── TASK 2: joint shared-π + per-season R0 (baseline 0.6) ──")
    tj = time.perf_counter(); J = MJ.fit_joint(C)
    indep_nll = {s: t1[s]["data_nll"] for s in SEASONS}
    dpct = [100.0*(J["data_nll_by_season"][i]-indep_nll[s])/max(indep_nll[s], 1.0)
            for i, s in enumerate(SEASONS)]
    print(f"  shared π = [home {J['pi'][0]:.3f}, work {J['pi'][1]:.3f}, school {J['pi'][2]:.3f}, other {J['pi'][3]:.3f}]")
    print(f"  π_work={J['pi_work']:.3f} (multistart std {J['pi_work_std']:.3f})  ({time.perf_counter()-tj:.1f}s)")
    print(f"  {'season':>11} {'R0':>6} {'nll_joint':>10} {'nll_indep':>10} {'Δ%':>6}")
    for i, s in enumerate(SEASONS):
        print(f"  {s:>11} {J['R0'][i]:>6.3f} {J['data_nll_by_season'][i]:>10.1f} {indep_nll[s]:>10.1f} {dpct[i]:>+5.1f}%")
    print(f"  mean NLL 열화 {np.mean(dpct):+.1f}% (max {max(dpct):+.1f}%)")
    (ED/"joint_fit_p06.json").write_text(json.dumps(dict(
        meta=dict(baseline_pwork=BASE_PWORK, seasons=SEASONS),
        shared_pi=J["pi"], pi_work=J["pi_work"], pi_work_std=J["pi_work_std"],
        R0_by_season=dict(zip(SEASONS, J["R0"])),
        nll_joint=dict(zip(SEASONS, J["data_nll_by_season"])), nll_independent=indep_nll,
        nll_delta_pct=dict(zip(SEASONS, dpct))), indent=2, default=float))

    # ═══ TASK 3: holiday reversal (windowed 0.6→0.4) ═══
    print("\n── TASK 3: holiday sign-reversal (windowed sick-leave 0.6→0.4) ──")
    t3 = {}
    for i, s in enumerate(SEASONS):
        b = beta_by_season[s]
        base6 = attack6(C, run_inc(C, i, b, p_work=BASE_PWORK))
        rec = {}
        for wname, wwin in (("term", TERM_WIN), ("vacation", VAC_WIN)):
            inc = run_inc(C, i, b, p_work=0.4, work_win=wwin, work_base=BASE_PWORK)
            d = (attack6(C, inc) - base6) / C["pop6"]
            da = {ag: float(100.0*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)}
            rec[wname] = dict(child={c: da[c] for c in CHILD}, child_sum=sum(da[c] for c in CHILD), all=da)
        rev = (rec["term"]["child_sum"] < 0) and (rec["vacation"]["child_sum"] > 0)
        rec["reversal"] = bool(rev); t3[s] = rec
        print(f"  {s}: term_child={rec['term']['child_sum']:+.3f} vac_child={rec['vacation']['child_sum']:+.3f} reversal={'Y' if rev else 'N'}")
    n_rev = sum(1 for s in SEASONS if t3[s]["reversal"])
    print(f"  ★ 부호반전 재현: {n_rev}/{len(SEASONS)}")
    (ED/"holiday_reversal_p06.json").write_text(json.dumps(dict(
        meta=dict(baseline_pwork=BASE_PWORK, term_window=TERM_WIN, vacation_window=VAC_WIN,
                  intervention_pwork=0.4), results=t3, reversal_count=n_rev), indent=2, default=float))

    # ═══ TASK 4: figures ═══
    print("\n── TASK 4: figures ──")
    weeks = np.arange(52)
    # FIG1 total
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for k, s in enumerate(SEASONS):
        ax = axes[k//3, k % 3]; o = C["full_obs"][s].sum(1); p = pred_by_season[s].sum(1)
        ax.plot(weeks, o, "o", color=DATA_GRAY, ms=3.5, alpha=0.75, label="HIRA 데이터")
        ax.plot(weeks, p, "-", color=MODEL_RED, lw=2, label="모델 (시즌별 π)")
        ax.set_title(f"{s}   R0={t1[s]['R0']:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("주차"); ax.set_ylabel("주간 진료에피소드"); ax.grid(alpha=0.25)
        ax.text(0.03, 0.90, f"obs/model={t1[s]['obs_model_total']:.2f}", transform=ax.transAxes, fontsize=8, color="#555")
        if k == 0: ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("6시즌 유행곡선 — 데이터 vs 모델 (시즌별 π, baseline p_work=0.6)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGDIR/"viz_fit_6seasons_total.png", bbox_inches="tight"); plt.close(fig)
    print(f"  [fig1] viz_fit_6seasons_total.png")
    # FIG2 by-age
    fig, axes = plt.subplots(6, 6, figsize=(16, 13), sharex=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            ax.plot(weeks, C["full_obs"][s][:, c], "o", color=DATA_GRAY, ms=2.5, alpha=0.7)
            ax.plot(weeks, pred_by_season[s][:, c], "-", color=AGE_COLORS[c], lw=1.6)
            ax.grid(alpha=0.2); ax.text(0.04, 0.86, f"{t1[s]['obs_model_by_age'][ag]:.2f}", transform=ax.transAxes, fontsize=7, color="#777")
            if r == 0: ax.set_title(f"{ag}세", fontsize=10, fontweight="bold")
            if c == 0: ax.set_ylabel(f"{s}\n주간 건수", fontsize=9, fontweight="bold")
            if r == 5: ax.set_xlabel("주차", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("6시즌 연령별 fitting — 시즌별 π (baseline p_work=0.6, 셀=obs/model)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(FIGDIR/"viz_fit_6seasons_byage.png", bbox_inches="tight"); plt.close(fig)
    print(f"  [fig2] viz_fit_6seasons_byage.png")
    # FIG3 policy p_work 0.6/0.4/0.2
    P_LEV = [0.6, 0.4, 0.2]; P_COL = {0.6: "#f4a582", 0.4: "#d6604d", 0.2: "#8b0000"}
    polc = {s: {p: pred_hira(C, run_inc(C, i, beta_by_season[s], p_work=p)) for p in P_LEV}
            for i, s in enumerate(SEASONS)}
    fig, axes = plt.subplots(6, 6, figsize=(16, 13), sharex=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            for p in P_LEV:
                ax.plot(weeks, polc[s][p][:, c], "-", color=P_COL[p], lw=1.5,
                        label=f"p_work={p}" if (r == 0 and c == 5) else None)
            ax.grid(alpha=0.2)
            if r == 0: ax.set_title(f"{ag}세", fontsize=10, fontweight="bold")
            if c == 0: ax.set_ylabel(f"{s}\n주간 건수", fontsize=9, fontweight="bold")
            if r == 5: ax.set_xlabel("주차", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 5].legend(fontsize=7, loc="upper right")
    fig.suptitle("6시즌 병가 정책효과 — p_work별 (baseline 0.6 → 0.4 → 0.2, 전 기간)\n"
                 "연함=0.6(무정책 baseline) → 진함=0.2(강한 병가)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(FIGDIR/"viz_policy_6seasons_byage.png", bbox_inches="tight"); plt.close(fig)
    print(f"  [fig3] viz_policy_6seasons_byage.png")
    # FIG4 holiday reversal (2-panel)
    labels = [f"{s[2:4]}–{s[7:9]}" for s in SEASONS]
    term = np.array([t3[s]["term"]["child_sum"] for s in SEASONS])
    vac = np.array([t3[s]["vacation"]["child_sum"] for s in SEASONS])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.2))
    x = np.arange(len(SEASONS)); bw = 0.38
    ax1.bar(x-bw/2, term, bw, color=BLUE, label="학기중 창", edgecolor="k", lw=0.5)
    ax1.bar(x+bw/2, vac, bw, color=RED, label="방학중 창", edgecolor="k", lw=0.5)
    ax1.axhline(0, color="k", lw=1.8)
    for xi, (tv, vv) in enumerate(zip(term, vac)):
        ax1.text(xi-bw/2, tv-0.01*np.sign(tv)-0.005, f"{tv:+.2f}", ha="center", va="top" if tv < 0 else "bottom", fontsize=8, color=BLUE)
        ax1.text(xi+bw/2, vv+0.005, f"{vv:+.2f}", ha="center", va="bottom", fontsize=8, color=RED)
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylabel("아동(0–17) Δattack 합 (%pt)")
    ax1.set_title(f"병가 아동영향: 학기중 이득(파랑) → 방학중 역효과(빨강)  ({n_rev}/6 시즌, baseline 0.6)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right"); ax1.grid(axis="y", alpha=0.3); ax1.margins(y=0.18)
    age_c = {"0-5": "#9ecae1", "6-11": "#ef6548", "12-17": "#b30000"}; bw2 = 0.26
    for j, ag in enumerate(CHILD):
        vals = np.array([t3[s]["vacation"]["child"][ag] for s in SEASONS])
        ax2.bar(x+(j-1)*bw2, vals, bw2, color=age_c[ag], label=f"{ag}세", edgecolor="k", lw=0.4)
    ax2.axhline(0, color="k", lw=1.5); ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("방학중 Δattack (%pt)")
    ax2.set_title("방학중 병가: 학령기(6–17세) 집중 악화", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", ncol=3); ax2.grid(axis="y", alpha=0.3); ax2.margins(y=0.15)
    fig.suptitle("방학 부호반전 (baseline p_work=0.6, 창내 0.6→0.4 개입)", fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIGDIR/"viz_holiday_reversal_6seasons.png", bbox_inches="tight"); plt.close(fig)
    print(f"  [fig4] viz_holiday_reversal_6seasons.png")
    print("=" * 96)


if __name__ == "__main__":
    main()
