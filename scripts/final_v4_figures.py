"""v4 확정 파라미터 그림 8종 재생성. presymp + κ수정 반영. point est.

파라미터: baseline p=0.6, φ 선형, γ CDC Reed, Erlang I₃, R(0)신규,
  κ={0.34/0.40/0} (η제거), presymp w=0.22 (NGM factor 0.7407),
  관측=발현시점(I₁→I₂ 유입), C(t) term↔vacation.
시즌: 16-17, 17-18, 19-20.
출력: presentations/figures/v4/*.png
기존 그림은 archive_v3/ 로 이동 완료.
"""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({"figure.dpi":110, "savefig.dpi":150,
                      "axes.unicode_minus":False,
                      "font.family":"AppleGothic", "font.size":9})

from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
import final_pipeline_confirmed as F

FIG = Path(__file__).resolve().parent.parent / "presentations" / "figures" / "v4"
FIG.mkdir(parents=True, exist_ok=True)
ED = Path(__file__).resolve().parent.parent / "outputs" / "eda"
D_CM = Path("../kt_data/data/external/contact_matrices")

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEAS]
# ★ v4 대칭 baseline (p_work = p_school = 0.6). φ 선형.
PHI = np.array(F.PHI); BASE = 0.6
GAMMA = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP = np.array([0.34]*4+[0.40]*10+[0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)
TERM = (70.0, 113.0); VAC = (113.0, 183.0); WH = (-1e9, 1e9)

CHILD = ["0-5","6-11","12-17"]; ADULT = ["18-44","45-64"]
AGE_C = ["#4575b4","#74add1","#fdae61","#f46d43","#d73027","#7b3294"]
COL_SICK = "#2166AC"; COL_SCHOOL = "#B2182B"; GRAY = "#666"


def build():
    C = F.build()
    pf = np.asarray(C["shared"]["pop_15"])
    C["pf"] = pf.sum(1) if pf.ndim == 2 else pf
    M = C["shared"]
    C["ngm3"] = make_ngm_eigvalue_fn(
        pop_15=np.asarray(M["pop_15"]), rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
        C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]),
        R0_immunity=IMM, gamma=float(M["gamma"]), seasonal_factor=1.0+F.S.AMP)
    C["st"] = {}
    for s in SEAS:
        sd = estimate_initial_infected_from_hira(s, C["pf"],
            sido_codes=list(SUDOGWON_SIDO_CODES), gamma_15_assumed=GAMMA)
        C["st"][s] = jnp.asarray(_build_initial_state_with_age_seed(
            C["pf"], sd, seed_e_factor=0.5, initial_immunity=IMM,
            initial_vaccinated_fraction=0.0))
    return C


def beta_from(C, R0, pi):
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    return b0 / NGM_F


def sim(C, s, R0, pi, p_school=BASE, p_work=BASE, sch_win=WH, work_win=WH):
    beta = beta_from(C, R0, pi)
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def pred_h(C, inc):
    return np.asarray(simulation_to_hira_by_age_jax(
        jnp.asarray(inc), jnp.asarray(GAMMA), n_weeks=52))


def att6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


def fit(C, s, i):
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i])
    def loss(x):
        R0 = jnp.exp(x[0]); pi = jax.nn.softmax(x[1:5])
        beta = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, jnp.asarray(PHI)) / NGM_F
        kw = dict(C["shared"])
        kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
        kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st"][s]),
                                          w_presymp=W, **kw, discretize_time=False)
        pred = simulation_to_hira_by_age_jax(
            daily_new_onset_by_age_erlang_presymp(st),
            jnp.asarray(GAMMA), n_weeks=C["nw"])
        c = x[1:5] - jnp.mean(x[1:5])
        return nb_nll_jax(obsj, pred, wj, concentration=x[5], min_rate=0.01) \
            + 0.5 * jnp.sum((c - jnp.asarray(F.LOGIT_REF))**2
                             / jnp.asarray(F.SIGMA_PIN)**2)
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v):
            v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(121 + i)
    bounds = [F.LOG_R0_B] + [(-10, 10)]*4 + [F.PHI_NB_B]
    best = None
    for k in range(10):
        x0 = np.concatenate([[np.log(rng.uniform(1.8, 2.5))],
                              F.LOGIT_REF + rng.normal(0, 0.5, 4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    R0 = float(np.exp(x[0])); pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    return dict(R0=R0, pi=[float(p) for p in pi])


def omw(C, i, pred):
    obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i])
    mask = w.sum(1) > 0; o = obs[mask].sum(1); m = pred[mask].sum(1)
    om = {ag: float(obs[mask,a].sum() / max(pred[mask,a].sum(), 1.0))
          for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return om, float(o.sum() / max(m.sum(), 1))


def dattack(C, s, i, R0, pi, base6, **pol):
    inc = sim(C, s, R0, pi, **pol); inf6 = att6(C, inc)
    d = (inf6 - base6) / C["pop6"]
    return (float(100*(base6.sum()-inf6.sum())/max(base6.sum(),1)),
            {ag: float(100*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)})


def main():
    print("="*90)
    print(f"v4 figures — presymp w={W}, κ={KAP.tolist()[0]}/{KAP[4]:.2f}/0, "
          f"baseline={BASE}, Erlang I₃")
    print("="*90)
    t0 = time.perf_counter(); C = build()
    print(f"[setup] {time.perf_counter()-t0:.1f}s")

    # ── fit ──
    print("\n[fit]")
    t1 = {}; bp = {}; preds = {}
    for s, i in zip(SEAS, IDX):
        f = fit(C, s, i); bp[s] = (f["R0"], f["pi"])
        pr = pred_h(C, sim(C, s, f["R0"], f["pi"])); preds[s] = pr
        om, omt = omw(C, i, pr)
        t1[s] = dict(**f, obs_model=om, om_total=omt)
        print(f"  {s}: R0={f['R0']:.3f}  om={omt:.2f}  |18-44={om['18-44']:.2f} "
              f"45-64={om['45-64']:.2f} 12-17={om['12-17']:.2f}")

    # 정책 계산 (school vs sick, rate/number, intensity, holiday)
    print("\n[정책 계산]")
    P = [0.6, 0.4, 0.2, 0.0]
    t2 = {}
    for s, i in zip(SEAS, IDX):
        R0, pi = bp[s]; base = att6(C, sim(C, s, R0, pi))
        rec = {"sick":{}, "school":{}}
        for p in P:
            av, d = dattack(C, s, i, R0, pi, base, p_work=p, work_win=TERM)
            rec["sick"][str(p)] = dict(av=av, da=d)
            av2, d2 = dattack(C, s, i, R0, pi, base, p_school=p, sch_win=TERM)
            rec["school"][str(p)] = dict(av=av2, da=d2)
        t2[s] = rec
    da_sk = np.mean([[t2[s]["sick"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS]
                     for s in SEAS], 0)
    da_sc = np.mean([[t2[s]["school"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS]
                     for s in SEAS], 0)
    pop6 = np.asarray(C["pop6"])
    num_sk = -da_sk / 100 * pop6; num_sc = -da_sc / 100 * pop6
    sk04 = np.mean([t2[s]["sick"]["0.4"]["av"] for s in SEAS])
    sc04 = np.mean([t2[s]["school"]["0.4"]["av"] for s in SEAS])
    print(f"  averted@p=0.4: sick={sk04:.2f}% school={sc04:.2f}%")

    PL = [0.6, 0.4, 0.2, 0.0]; t4 = {}
    for s, i in zip(SEAS, IDX):
        R0, pi = bp[s]; base = att6(C, sim(C, s, R0, pi)); pr = {}
        for p in PL:
            av, d = dattack(C, s, i, R0, pi, base, p_work=p)
            pr[str(p)] = dict(av=av, da=d)
        t4[s] = pr

    t5 = {}; INT = [0.4, 0.2, 0.0]
    for s, i in zip(SEAS, IDX):
        R0, pi = bp[s]; base = att6(C, sim(C, s, R0, pi)); rec = {}
        for p in INT:
            _, dt = dattack(C, s, i, R0, pi, base, p_work=p, work_win=TERM)
            _, dv = dattack(C, s, i, R0, pi, base, p_work=p, work_win=VAC)
            cst = sum(dt[c] for c in CHILD); csv = sum(dv[c] for c in CHILD)
            rec[str(p)] = dict(term=dt, vac=dv, tc=cst, vc=csv)
        t5[s] = rec

    # ═══════════ 그림 ═══════════
    weeks = np.arange(52)
    ax_ages = [a+"세" for a in HIRA_AGE_GROUPS]

    # 1. fit_total
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, s in enumerate(SEAS):
        a = ax[k]
        a.plot(weeks, C["full_obs"][s].sum(1), "o", color=GRAY, ms=3.5,
               alpha=0.7, label="데이터")
        a.plot(weeks, preds[s].sum(1), "-", color=COL_SCHOOL, lw=2, label="모델")
        a.set_title(f"{s}   R0={t1[s]['R0']:.2f}", fontsize=11, fontweight="bold")
        a.set_xlabel("주차"); a.grid(alpha=0.25)
        a.text(0.03, 0.86, f"obs/model = {t1[s]['om_total']:.2f}",
               transform=a.transAxes, fontsize=8, color="#333")
        if k == 0:
            a.legend(fontsize=9); a.set_ylabel("주간 진료에피소드")
    fig.suptitle("3시즌 유행곡선 — 데이터 vs 모델", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(FIG/"fit_total.png", bbox_inches="tight"); plt.close(fig)
    print("  [1/8] fit_total.png")

    # 2. fit_byage  (y 통일)
    fig, ax = plt.subplots(3, 6, figsize=(16, 7.5), sharex=True, sharey=True)
    for r, s in enumerate(SEAS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            a = ax[r, c]
            a.plot(weeks, C["full_obs"][s][:, c], "o", color=GRAY, ms=2, alpha=0.6)
            a.plot(weeks, preds[s][:, c], "-", color=AGE_C[c], lw=1.5)
            a.set_ylim(0, 100000); a.grid(alpha=0.2)
            a.text(0.04, 0.82, f"{t1[s]['obs_model'][ag]:.2f}",
                    transform=a.transAxes, fontsize=7, color="#333")
            if r == 0: a.set_title(f"{ag}세", fontsize=9, fontweight="bold")
            if c == 0: a.set_ylabel(s, fontsize=8, fontweight="bold")
            a.tick_params(labelsize=6)
    fig.suptitle("연령별 유행곡선", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(FIG/"fit_byage.png", bbox_inches="tight"); plt.close(fig)
    print("  [2/8] fit_byage.png")

    # 2b. fit_byage per 1000 & per 100k (인구 정규화, y 통일)
    pop6_arr = np.asarray(C["pop6"])   # (6,)
    for per, tag, ylabel in [(1000, "per1000", "주간 발생 (1000명당)"),
                              (100000, "per100k", "주간 발생 (10만명당)")]:
        divisor = pop6_arr / per       # (6,)
        # y 통일 위해 데이터 최대값 계산 (obs + model)
        ymax = 0.0
        for s in SEAS:
            for c in range(6):
                d_obs = C["full_obs"][s][:, c] / divisor[c]
                d_mod = preds[s][:, c] / divisor[c]
                ymax = max(ymax, float(d_obs.max()), float(d_mod.max()))
        ymax = ymax * 1.05
        fig, ax = plt.subplots(3, 6, figsize=(16, 7.5), sharex=True, sharey=True)
        for r, s in enumerate(SEAS):
            for c, ag in enumerate(HIRA_AGE_GROUPS):
                a = ax[r, c]
                a.plot(weeks, C["full_obs"][s][:, c] / divisor[c], "o",
                       color=GRAY, ms=2, alpha=0.6)
                a.plot(weeks, preds[s][:, c] / divisor[c], "-",
                       color=AGE_C[c], lw=1.5)
                a.set_ylim(0, ymax); a.grid(alpha=0.2)
                a.text(0.04, 0.82, f"{t1[s]['obs_model'][ag]:.2f}",
                       transform=a.transAxes, fontsize=7, color="#333")
                if r == 0:
                    a.set_title(f"{ag}세", fontsize=9, fontweight="bold")
                if c == 0:
                    a.set_ylabel(f"{s}\n{ylabel}", fontsize=8, fontweight="bold")
                a.tick_params(labelsize=6)
        title = "연령별 유행곡선 (인구 1000명당)" if per == 1000 \
            else "연령별 유행곡선 (인구 10만명당)"
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0,0,1,0.96])
        fname = f"fit_byage_{tag}.png"
        fig.savefig(FIG/fname, bbox_inches="tight"); plt.close(fig)
        # peak-per-age 최대 연령 (2019-2020 대표)
        peak_by_age = [(preds["2019-2020"][:, c] / divisor[c]).max() for c in range(6)]
        max_age = HIRA_AGE_GROUPS[int(np.argmax(peak_by_age))]
        print(f"  [+2] {fname}  ymax={ymax:.1f}  max연령(19-20)={max_age}")

    # 3. school_vs_sick  (좌 averted 곡선, 우 연령별 averted %pt 양수방향)
    xs = [BASE - p for p in P]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    sk_v = [np.mean([t2[s]["sick"][str(p)]["av"] for s in SEAS]) for p in P]
    sc_v = [np.mean([t2[s]["school"][str(p)]["av"] for s in SEAS]) for p in P]
    a1.plot(xs, sk_v, "o-", color=COL_SICK, lw=2, ms=7, label="병가")
    a1.plot(xs, sc_v, "s-", color=COL_SCHOOL, lw=2, ms=7, label="학교결석")
    a1.axhline(0, color="k", lw=0.7, alpha=0.4)
    a1.set_xlabel("p 감소량 (공통 baseline 0.6 기준)")
    a1.set_ylabel("averted %")
    a1.set_title("averted", fontsize=11, fontweight="bold")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    x = np.arange(6); bw = 0.38
    a2.bar(x-bw/2, -da_sk, bw, color=COL_SICK, label="병가",
           edgecolor="k", lw=0.4)
    a2.bar(x+bw/2, -da_sc, bw, color=COL_SCHOOL, label="학교결석",
           edgecolor="k", lw=0.4)
    a2.axhline(0, color="k", lw=1)
    a2.set_xticks(x); a2.set_xticklabels(ax_ages, rotation=0, fontsize=9)
    a2.set_ylabel("averted (%pt)")
    a2.set_title("연령 영향: 병가→성인, 학교→학생",
                  fontsize=11, fontweight="bold")
    a2.legend(fontsize=9); a2.grid(axis="y", alpha=0.3)
    fig.suptitle("학교결석 vs 병가", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(FIG/"school_vs_sick.png", bbox_inches="tight"); plt.close(fig)
    print("  [3/8] school_vs_sick.png")

    # 4. school_vs_sick_number  (우 number)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(xs, sk_v, "o-", color=COL_SICK, lw=2, ms=7, label="병가")
    a1.plot(xs, sc_v, "s-", color=COL_SCHOOL, lw=2, ms=7, label="학교결석")
    a1.axhline(0, color="k", lw=0.7, alpha=0.4)
    a1.set_xlabel("p 감소량 (공통 baseline 0.6 기준)")
    a1.set_ylabel("averted %")
    a1.set_title("averted", fontsize=11, fontweight="bold")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    a2.bar(x-bw/2, num_sk, bw, color=COL_SICK, label="병가",
           edgecolor="k", lw=0.4)
    a2.bar(x+bw/2, num_sc, bw, color=COL_SCHOOL, label="학교결석",
           edgecolor="k", lw=0.4)
    a2.axhline(0, color="k", lw=1)
    a2.set_xticks(x); a2.set_xticklabels(ax_ages, rotation=0, fontsize=9)
    a2.set_ylabel("averted (감염 수, 명)")
    a2.set_title("연령 영향 (감염 수): 병가→성인, 학교→성인",
                  fontsize=11, fontweight="bold")
    for xi in range(6):
        a2.text(xi-bw/2, num_sk[xi], f"{num_sk[xi]/1000:.0f}k",
                ha="center", va="bottom", fontsize=7, color=COL_SICK)
        a2.text(xi+bw/2, num_sc[xi], f"{num_sc[xi]/1000:.0f}k",
                ha="center", va="bottom", fontsize=7, color=COL_SCHOOL)
    a2.legend(fontsize=9); a2.grid(axis="y", alpha=0.3)
    fig.suptitle("학교결석 vs 병가 (감염 수)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(FIG/"school_vs_sick_number.png", bbox_inches="tight"); plt.close(fig)
    print("  [4/8] school_vs_sick_number.png")

    # 5. rate_vs_number
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.bar(x-bw/2, -da_sk, bw, color=COL_SICK, label="병가",
           edgecolor="k", lw=0.4)
    a1.bar(x+bw/2, -da_sc, bw, color=COL_SCHOOL, label="학교결석",
           edgecolor="k", lw=0.4)
    a1.axhline(0, color="k", lw=1)
    a1.set_xticks(x); a1.set_xticklabels(ax_ages, fontsize=9)
    a1.set_ylabel("averted (%pt)")
    a1.set_title("비율 (rate) — 학령기 최대", fontsize=11, fontweight="bold")
    a1.legend(fontsize=9); a1.grid(axis="y", alpha=0.3)

    a2.bar(x-bw/2, num_sk, bw, color=COL_SICK, label="병가",
           edgecolor="k", lw=0.4)
    a2.bar(x+bw/2, num_sc, bw, color=COL_SCHOOL, label="학교결석",
           edgecolor="k", lw=0.4)
    a2.axhline(0, color="k", lw=1)
    a2.set_xticks(x); a2.set_xticklabels(ax_ages, fontsize=9)
    a2.set_ylabel("averted (감염 수, 명)")
    a2.set_title("감염 수 (number) — 성인 최대", fontsize=11, fontweight="bold")
    for xi in range(6):
        a2.text(xi-bw/2, num_sk[xi], f"{num_sk[xi]/1000:.0f}k",
                ha="center", va="bottom", fontsize=7, color=COL_SICK)
        a2.text(xi+bw/2, num_sc[xi], f"{num_sc[xi]/1000:.0f}k",
                ha="center", va="bottom", fontsize=7, color=COL_SCHOOL)
    a2.legend(fontsize=9); a2.grid(axis="y", alpha=0.3)
    fig.suptitle("지표에 따른 연령별 효과 — 비율 vs 감염 수",
                  fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(FIG/"rate_vs_number.png", bbox_inches="tight"); plt.close(fig)
    print("  [5/8] rate_vs_number.png")

    # 6. policy_intensity (양수방향 = averted)
    fig, axx = plt.subplots(figsize=(10, 5.5))
    P2 = [0.4, 0.2, 0.0]
    cols = {0.4:"#fdae61", 0.2:"#f46d43", 0.0:"#a50026"}
    bw2 = 0.26
    for j, p in enumerate(P2):
        vals = np.mean([[t4[s][str(p)]["da"][ag] for ag in HIRA_AGE_GROUPS]
                        for s in SEAS], 0)
        axx.bar(x + (j-1)*bw2, -vals, bw2, color=cols[p],
                label=f"p_work={p}", edgecolor="k", lw=0.4)
    axx.axhline(0, color="k", lw=1.2)
    axx.set_xticks(x); axx.set_xticklabels(ax_ages, fontsize=10)
    axx.set_ylabel("averted (%pt) — 양수=감염 감소")
    axx.set_title("병가 강도별 연령 영향", fontsize=12, fontweight="bold")
    axx.legend(); axx.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG/"policy_intensity.png", bbox_inches="tight"); plt.close(fig)
    print("  [6/8] policy_intensity.png")

    # 7. holiday  (지표=Δattack, 양수=감염 증가/전가, 음수=감소/이득)
    P_HOL = 0.2   # 대표 강도
    labels = [f"{s[2:4]}-{s[7:9]}" for s in SEAS]
    # tc/vc 는 이미 Δattack (병가 - baseline)  아동(0-17) 합, %pt
    term_child = np.array([t5[s][str(P_HOL)]["tc"] for s in SEAS])
    vac_child = np.array([t5[s][str(P_HOL)]["vc"] for s in SEAS])
    fig, (au, ad) = plt.subplots(2, 1, figsize=(9.5, 8),
                                    gridspec_kw={"height_ratios":[1,1]})
    xh = np.arange(3); bwh = 0.38
    au.bar(xh-bwh/2, term_child, bwh, color=COL_SICK, label="학기",
           edgecolor="k")
    au.bar(xh+bwh/2, vac_child, bwh, color=COL_SCHOOL, label="방학",
           edgecolor="k")
    au.axhline(0, color="k", lw=1.6)    # 0 기준선 굵게
    au.set_xticks(xh); au.set_xticklabels(labels)
    au.set_ylabel("Δ attack rate (%pt)  — 아동(0-17) 합\n"
                    "양수 = 감염 증가(전가) · 음수 = 감염 감소(이득)",
                    fontsize=9)
    au.set_title(f"학기중과 방학 비교 (p_work={P_HOL})",
                  fontsize=12, fontweight="bold")
    au.legend(fontsize=10); au.grid(axis="y", alpha=0.3)

    # 하단: 방학중 연령별 (아동 3그룹, 시즌별 색), 지표=Δattack
    vals_by_age = np.array([
        [t5[s][str(P_HOL)]["vac"][c] for c in CHILD] for s in SEAS
    ])   # (3 시즌, 3 아동), Δattack %pt
    xc = np.arange(3); bwc = 0.28
    for j, s in enumerate(SEAS):
        ad.bar(xc + (j-1)*bwc, vals_by_age[j], bwc,
               color=AGE_C[j+1], label=s, edgecolor="k", lw=0.4)
    ad.axhline(0, color="k", lw=1.6)    # 0 기준선 굵게
    ad.set_xticks(xc); ad.set_xticklabels([c+"세" for c in CHILD], fontsize=10)
    ad.set_ylabel("Δ attack rate (%pt)\n"
                    "양수 = 감염 증가 · 음수 = 감염 감소",
                    fontsize=9)
    ad.set_title("방학중 연령별 변화", fontsize=12, fontweight="bold")
    ad.legend(fontsize=9); ad.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG/"holiday.png", bbox_inches="tight"); plt.close(fig)
    print("  [7/8] holiday.png")

    # 8. contact_matrix (학기 / 방학 / 차분)
    CH = ("C_home","C_school","C_work","C_other")
    LAB = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
            "40-44","45-49","50-54","55-59","60-64","65-69","70+"]
    def total(path):
        m = load_contact_matrices(path=path)  # [contact, participant]
        return sum(np.asarray(m[c]) for c in CH)
    term_m = total(D_CM/"empirical_matrices_15.npz")
    vac_m = total(D_CM/"empirical_matrices_15_vacation.npz")
    diff_m = term_m - vac_m
    vmax = max(term_m.max(), vac_m.max())
    dmax = np.abs(diff_m).max()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))
    def draw(ax, M, title, cmap, vmin=None, vmax_=None, norm=None):
        if norm is not None:
            im = ax.imshow(M, origin="lower", cmap=cmap, norm=norm, aspect="equal")
        else:
            im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_,
                            aspect="equal")
        ax.set_xticks(range(15)); ax.set_xticklabels(LAB, rotation=90, fontsize=7)
        ax.set_yticks(range(15)); ax.set_yticklabels(LAB, fontsize=7)
        ax.set_xlabel("응답자 연령 (participant)", fontsize=9)
        ax.set_ylabel("접촉 상대 연령 (contacted)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        return im
    im1 = draw(axes[0], term_m, "학기", "YlOrRd", vmin=0, vmax_=vmax)
    im2 = draw(axes[1], vac_m, "방학", "YlOrRd", vmin=0, vmax_=vmax)
    norm = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
    im3 = draw(axes[2], diff_m, "학기 − 방학", "RdBu_r", norm=norm)
    fig.colorbar(im2, ax=axes[:2], fraction=0.023, pad=0.02,
                  label="일 평균 접촉 수")
    fig.colorbar(im3, ax=axes[2], fraction=0.045, pad=0.02,
                  label="접촉 수 차분")
    fig.suptitle("학기 vs 방학 접촉행렬", fontsize=13, fontweight="bold")
    fig.savefig(FIG/"contact_matrix.png", bbox_inches="tight"); plt.close(fig)
    print("  [8/8] contact_matrix.png")

    print("\n"+"="*90)
    print(f"v4 그림 8종 완료 → {FIG}")
    print("="*90)


if __name__ == "__main__":
    main()
