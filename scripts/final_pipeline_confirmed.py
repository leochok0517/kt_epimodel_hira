"""FINAL confirmed-parameter pipeline: fit + policy + holiday + figures.

★ CONFIRMED FIXED PARAMETERS (from here on):
  baseline p_work=0.6 (presenteeism)
  φ linear U-shape [2.0,1.75,1.5,1.25,1,1,1,1,1,1.083,1.167,1.25,1.333,1.417,1.5]
  γ_15 [0.40,0.40,0.25,0.18, 0.18×9, 0.25,0.25]  (12-17=0.18 adult-level)
  κ 3-way [0.29×4, 0.30×10, 0.0];  Step A+B (C(t), v(t))
  6 normal seasons, first_peak_only=True.

Tasks: (1) fit (per-season π + joint shared-π, obs/model), (2) policy p_work
{0.6,0.4,0.2,0.0} per-age Δattack, (3) holiday reversal by intensity, (4) figures.
Point estimate. Common π = joint p06 (supplement); per-season π = main text.

Outputs: outputs/eda/final_{fit,policy,holiday_reversal}.json + 4 figures.
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
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.unicode_minus": False,
                     "font.family": "AppleGothic", "font.size": 9})

from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.hira_target import HIRA_GROUP_TO_NIMS_WEIGHTED, load_hira_target_by_age
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax
from kt_epimodel_hira.jax_model.solver_jax import simulate_jax, daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
import sens_workshare_kappa_v2 as S
import multiseason_joint_sharedpi as MJ

REPO_ROOT = Path(__file__).resolve().parent.parent
ED = REPO_ROOT / "outputs" / "eda"; ED.mkdir(parents=True, exist_ok=True)
FIGDIR = REPO_ROOT / "presentations" / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)

SEASONS = MJ.SEASONS
BASE_PWORK = 0.6
PHI = np.array([2.0, 1.75, 1.5, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0+0.5*1/6, 1.0+0.5*2/6, 1.0+0.5*3/6, 1.0+0.5*4/6, 1.0+0.5*5/6, 1.5])
GAMMA15 = jnp.asarray(np.array([0.40, 0.40, 0.25, 0.18] + [0.18]*9 + [0.25, 0.25]))
COMMON_PI = np.array([0.322, 0.282, 0.069, 0.327])
LOG_R0_B = (float(np.log(0.8)), float(np.log(3.5))); PHI_NB_B = (1e-3, 1e6)
N_STARTS = 10; SEED = 51
TERM_WIN = (70.0, 113.0); VAC_WIN = (113.0, 183.0); WHOLE = (-1.0e9, 1.0e9)
CHILD = ["0-5", "6-11", "12-17"]; SCHOOL = ["6-11", "12-17"]; ADULT = ["18-44", "45-64"]
AGE_COLORS = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#7b3294"]
DATA_GRAY = "#666666"; MODEL_RED = "#B23A48"; BLUE = "#2166AC"; RED = "#B23A48"
SIGMA_PIN = np.array([0.15, 0.10, 0.05, 0.15]); PI_REF = np.array(S.build_pi_target(0.29))
LOGIT_REF = S.logit_centered_target(PI_REF)


def build():
    C = MJ.build_common(); C["shared"]["p_work"] = BASE_PWORK; C["nw"] = C["nweeks"]
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for idx, wt in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, idx] = wt
    pflat = np.asarray(C["shared"]["pop_15"]); pflat = pflat.sum(1) if pflat.ndim == 2 else pflat
    C["H"] = H; C["pop6"] = H @ pflat
    C["full_obs"] = {}
    for s in SEASONS:
        t = load_hira_target_by_age(s, sido_codes=list(SUDOGWON_SIDO_CODES), first_peak_only=False)
        o = np.zeros((t["n_weeks"], 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            o[:, i] = t["hira_counts"][ag]
        C["full_obs"][s] = o
    return C


def run_inc(C, i, R0, pi, p_work=BASE_PWORK, work_win=WHOLE, work_base=1.0):
    phi = jnp.asarray(PHI)
    beta = derive_beta_from_R0_simplex(C["ngm"], jnp.asarray(R0), jnp.asarray(pi), phi)
    kw = dict(C["shared"])
    kw["beta_h"] = beta[0]; kw["beta_w"] = beta[1]; kw["beta_s"] = beta[2]; kw["beta_o"] = beta[3]
    kw["phi_susc"] = phi; kw["p_school"] = 1.0; kw["p_work"] = p_work
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_work_baseline"] = work_base
    st = simulate_jax(C["states"][i], **kw, discretize_time=False)
    return daily_new_infection_by_age_jax(st)


def pred_h(C, inc, nw=52):
    return np.asarray(simulation_to_hira_by_age_jax(jnp.asarray(inc), GAMMA15, n_weeks=nw))


def attack6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


def fit_season_pi(C, i):
    """Per-season π+R0+phi_nb (pin work σ0.10), φ/γ fixed, baseline 0.6."""
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i]); phi = jnp.asarray(PHI)
    def loss(x):
        R0 = jnp.exp(x[0]); pi = jax.nn.softmax(x[1:5])
        beta = derive_beta_from_R0_simplex(C["ngm"], R0, pi, phi)
        kw = dict(C["shared"]); kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = phi
        st = simulate_jax(C["states"][i], **kw, discretize_time=False)
        pred = simulation_to_hira_by_age_jax(daily_new_infection_by_age_jax(st), GAMMA15, n_weeks=C["nw"])
        nll = nb_nll_jax(obsj, pred, wj, concentration=x[5], min_rate=0.01)
        centered = x[1:5] - jnp.mean(x[1:5])
        return nll + 0.5*jnp.sum((centered - jnp.asarray(LOGIT_REF))**2 / jnp.asarray(SIGMA_PIN)**2)
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(SEED+i)
    bounds = [LOG_R0_B] + [(-10, 10)]*4 + [PHI_NB_B]
    best = None
    for k in range(N_STARTS):
        x0 = np.concatenate([[np.log(rng.uniform(1.8, 2.4))], LOGIT_REF + rng.normal(0, 0.5, 4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun: best = r
    x = best.x; R0 = float(np.exp(x[0])); pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    return dict(R0=R0, pi=[float(p) for p in pi], nll=float(best.fun))


def obs_model(C, i, R0, pi):
    pred = pred_h(C, run_inc(C, i, R0, pi))
    obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i]); mask = w.sum(1) > 0
    om = {ag: float(obs[mask, a].sum()/max(pred[mask, a].sum(), 1.0)) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return om, float(obs[mask].sum()/max(pred[mask].sum(), 1.0))


def main():
    print("=" * 96); print("FINAL confirmed pipeline — baseline 0.6, φ linear, γ 12-17=0.18"); print("=" * 96)
    t0 = time.perf_counter(); C = build(); print(f"[setup] {time.perf_counter()-t0:.1f}s\n")

    # ═══ TASK 1: fitting ═══
    print("── TASK 1: per-season π+R0 fits ──")
    t1 = {}; beta_pi = {}
    for i, s in enumerate(SEASONS):
        f = fit_season_pi(C, i); om, omt = obs_model(C, i, f["R0"], f["pi"])
        t1[s] = dict(**f, obs_model=om, obs_model_total=omt); beta_pi[s] = (f["R0"], f["pi"])
        print(f"  {s}: R0={f['R0']:.3f} π=[{f['pi'][0]:.3f},{f['pi'][1]:.3f},{f['pi'][2]:.3f},{f['pi'][3]:.3f}] obs/model={omt:.2f} 12-17={om['12-17']:.2f}")
    print("  season×age obs/model:")
    print(f"  {'season':>11} " + " ".join(f"{ag:>6}" for ag in HIRA_AGE_GROUPS))
    for s in SEASONS:
        print(f"  {s:>11} " + " ".join(f"{t1[s]['obs_model'][ag]:>6.2f}" for ag in HIRA_AGE_GROUPS))
    m1217 = np.mean([t1[s]["obs_model"]["12-17"] for s in SEASONS])
    print(f"  ★ 12-17 obs/model mean = {m1217:.2f}")
    # supplement: per-season π summary (shared-π evidence, consistent φ/γ)
    pi_arr = np.array([t1[s]["pi"] for s in SEASONS])
    pi_mean = pi_arr.mean(0); pi_std = pi_arr.std(0)
    print(f"  [supplement] per-season π mean=[{pi_mean[0]:.3f},{pi_mean[1]:.3f},{pi_mean[2]:.3f},{pi_mean[3]:.3f}] "
          f"std=[{pi_std[0]:.3f},{pi_std[1]:.3f},{pi_std[2]:.3f},{pi_std[3]:.3f}]  π_work range [{pi_arr[:,1].min():.3f},{pi_arr[:,1].max():.3f}]")
    (ED/"final_fit.json").write_text(json.dumps(dict(
        meta=dict(baseline=BASE_PWORK, phi=PHI.tolist(), gamma15=np.asarray(GAMMA15).tolist()),
        per_season=t1, pi_mean=pi_mean.tolist(), pi_std=pi_std.tolist(),
        obs_model_12_17_mean=float(m1217)), indent=2, default=float))

    # ═══ TASK 2: policy ═══
    print("\n── TASK 2: policy p_work {0.6,0.4,0.2,0.0} per-age Δattack ──")
    P_LEV = [0.6, 0.4, 0.2, 0.0]; pol = {}
    for i, s in enumerate(SEASONS):
        R0, pi = beta_pi[s]
        base6 = attack6(C, run_inc(C, i, R0, pi, p_work=0.6))
        pr = {}
        for p in P_LEV:
            inc = run_inc(C, i, R0, pi, p_work=p)
            inf6 = attack6(C, inc)
            av = 100.0*(base6.sum()-inf6.sum())/max(base6.sum(), 1.0)
            d = (inf6 - base6)/C["pop6"]
            pr[p] = dict(averted=float(av), d_attack={ag: float(100.0*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)})
        pol[s] = pr
    print("  averted% by intensity (rows=season):")
    print(f"  {'season':>11} " + " ".join(f"p={p}" for p in P_LEV))
    for s in SEASONS:
        print(f"  {s:>11} " + " ".join(f"{pol[s][p]['averted']:>+6.2f}" for p in P_LEV))
    print("  2019-2020 per-age Δattack (p=0.2):")
    da = pol["2019-2020"][0.2]["d_attack"]
    print("    " + " ".join(f"{ag}:{da[ag]:+.3f}" for ag in HIRA_AGE_GROUPS))
    (ED/"final_policy.json").write_text(json.dumps(dict(meta=dict(baseline=BASE_PWORK, levels=P_LEV), policy=pol), indent=2, default=float))

    # ═══ TASK 3: holiday reversal by intensity ═══
    print("\n── TASK 3: holiday reversal by intensity (term vs vacation) ──")
    hol = {}; INTENS = [0.4, 0.2, 0.0]
    for i, s in enumerate(SEASONS):
        R0, pi = beta_pi[s]; base6 = attack6(C, run_inc(C, i, R0, pi, p_work=0.6))
        rec = {}
        for p in INTENS:
            wr = {}
            for wn, ww in (("term", TERM_WIN), ("vacation", VAC_WIN)):
                inc = run_inc(C, i, R0, pi, p_work=p, work_win=ww, work_base=0.6)
                d = (attack6(C, inc) - base6)/C["pop6"]
                da = {ag: float(100.0*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)}
                wr[wn] = dict(all=da, child_sum=sum(da[c] for c in CHILD))
            wr["reversal"] = bool(wr["term"]["child_sum"] < 0 and wr["vacation"]["child_sum"] > 0)
            rec[p] = wr
        hol[s] = rec
    for p in INTENS:
        nrev = sum(1 for s in SEASONS if hol[s][p]["reversal"])
        print(f"  개입 p_work={p} (창내): 부호반전 {nrev}/6"
              + "  vac_child=[" + ",".join(f"{hol[s][p]['vacation']['child_sum']:+.2f}" for s in SEASONS) + "]")
    (ED/"final_holiday_reversal.json").write_text(json.dumps(dict(
        meta=dict(baseline=BASE_PWORK, intensities=INTENS, term=TERM_WIN, vacation=VAC_WIN), results=hol), indent=2, default=float))

    # ═══ TASK 4: figures ═══
    print("\n── TASK 4: figures ──")
    weeks = np.arange(52)
    preds = {s: pred_h(C, run_inc(C, i, *beta_pi[s])) for i, s in enumerate(SEASONS)}
    # FIG1
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for k, s in enumerate(SEASONS):
        ax = axes[k//3, k % 3]; o = C["full_obs"][s].sum(1); p = preds[s].sum(1)
        ax.plot(weeks, o, "o", color=DATA_GRAY, ms=3.5, alpha=0.75, label="HIRA 데이터")
        ax.plot(weeks, p, "-", color=MODEL_RED, lw=2, label="모델 (시즌별 π)")
        ax.set_title(f"{s}   R0={t1[s]['R0']:.2f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("주차"); ax.set_ylabel("주간 진료에피소드"); ax.grid(alpha=0.25)
        ax.text(0.03, 0.90, f"obs/model={t1[s]['obs_model_total']:.2f}", transform=ax.transAxes, fontsize=8, color="#555")
        if k == 0: ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("6시즌 유행곡선 — 데이터 vs 모델 (확정: baseline 0.6, φ선형, γ 12-17=0.18)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGDIR/"viz_fit_6seasons_total.png", bbox_inches="tight"); plt.close(fig)
    print("  [fig1] viz_fit_6seasons_total.png")
    # FIG2
    fig, axes = plt.subplots(6, 6, figsize=(16, 13), sharex=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            ax.plot(weeks, C["full_obs"][s][:, c], "o", color=DATA_GRAY, ms=2.5, alpha=0.7)
            ax.plot(weeks, preds[s][:, c], "-", color=AGE_COLORS[c], lw=1.6); ax.grid(alpha=0.2)
            ax.text(0.04, 0.86, f"{t1[s]['obs_model'][ag]:.2f}", transform=ax.transAxes, fontsize=7, color="#777")
            if r == 0: ax.set_title(f"{ag}세", fontsize=10, fontweight="bold")
            if c == 0: ax.set_ylabel(f"{s}\n주간 건수", fontsize=9, fontweight="bold")
            if r == 5: ax.set_xlabel("주차", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("6시즌 연령별 fitting — 확정 파라미터 (셀=obs/model, 12-17 개선)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(FIGDIR/"viz_fit_6seasons_byage.png", bbox_inches="tight"); plt.close(fig)
    print("  [fig2] viz_fit_6seasons_byage.png")
    # FIG3 policy
    P3 = [0.6, 0.4, 0.2]; PC = {0.6: "#f4a582", 0.4: "#d6604d", 0.2: "#8b0000"}
    polc = {s: {p: pred_h(C, run_inc(C, i, *beta_pi[s], p_work=p)) for p in P3} for i, s in enumerate(SEASONS)}
    fig, axes = plt.subplots(6, 6, figsize=(16, 13), sharex=True)
    for r, s in enumerate(SEASONS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            for p in P3:
                ax.plot(weeks, polc[s][p][:, c], "-", color=PC[p], lw=1.5, label=f"p_work={p}" if (r == 0 and c == 5) else None)
            ax.grid(alpha=0.2)
            if r == 0: ax.set_title(f"{ag}세", fontsize=10, fontweight="bold")
            if c == 0: ax.set_ylabel(f"{s}\n주간 건수", fontsize=9, fontweight="bold")
            if r == 5: ax.set_xlabel("주차", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, 5].legend(fontsize=7, loc="upper right")
    fig.suptitle("6시즌 병가 정책효과 — p_work 0.6→0.4→0.2 (확정 파라미터, 전 기간)\n연함=0.6(baseline) → 진함=0.2", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(FIGDIR/"viz_policy_6seasons_byage.png", bbox_inches="tight"); plt.close(fig)
    print("  [fig3] viz_policy_6seasons_byage.png")
    # FIG4 holiday reversal (intensity p=0.2, strongest)
    PSHOW = 0.2
    labels = [f"{s[2:4]}–{s[7:9]}" for s in SEASONS]
    term = np.array([hol[s][PSHOW]["term"]["child_sum"] for s in SEASONS])
    vac = np.array([hol[s][PSHOW]["vacation"]["child_sum"] for s in SEASONS])
    nrev = sum(1 for s in SEASONS if hol[s][PSHOW]["reversal"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.2)); x = np.arange(6); bw = 0.38
    ax1.bar(x-bw/2, term, bw, color=BLUE, label="학기중 창", edgecolor="k", lw=0.5)
    ax1.bar(x+bw/2, vac, bw, color=RED, label="방학중 창", edgecolor="k", lw=0.5)
    ax1.axhline(0, color="k", lw=1.8)
    for xi, (tv, vv) in enumerate(zip(term, vac)):
        ax1.text(xi-bw/2, tv-0.006, f"{tv:+.2f}", ha="center", va="top", fontsize=7.5, color=BLUE)
        ax1.text(xi+bw/2, vv+0.006, f"{vv:+.2f}", ha="center", va="bottom", fontsize=7.5, color=RED)
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylabel("아동(0–17) Δattack 합 (%pt)")
    ax1.set_title(f"병가 아동영향: 학기중(파랑) vs 방학중(빨강)  개입 p_work={PSHOW}  ({nrev}/6 반전, baseline 0.6)", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="upper right"); ax1.grid(axis="y", alpha=0.3); ax1.margins(y=0.18)
    age_c = {"0-5": "#9ecae1", "6-11": "#ef6548", "12-17": "#b30000"}; bw2 = 0.26
    for j, ag in enumerate(CHILD):
        vals = np.array([hol[s][PSHOW]["vacation"]["all"][ag] for s in SEASONS])
        ax2.bar(x+(j-1)*bw2, vals, bw2, color=age_c[ag], label=f"{ag}세", edgecolor="k", lw=0.4)
    ax2.axhline(0, color="k", lw=1.5); ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylabel("방학중 Δattack (%pt)")
    ax2.set_title("방학중 병가: 연령별 (확정 γ 반영)", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper left", ncol=3); ax2.grid(axis="y", alpha=0.3); ax2.margins(y=0.15)
    fig.suptitle(f"방학 부호반전 (확정 파라미터, 개입 강도 p_work={PSHOW})", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIGDIR/"viz_holiday_reversal_6seasons.png", bbox_inches="tight"); plt.close(fig)
    print("  [fig4] viz_holiday_reversal_6seasons.png")
    print("=" * 96)


if __name__ == "__main__":
    main()
