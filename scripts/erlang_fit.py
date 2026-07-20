"""Erlang(3) infectious compartment fit — 3 seasons, baseline 0.6.

Compares single-I vs Erlang-I₃: width_ratio, obs/model, peak timing, redistribution.
Confirmed params (baseline 0.6, φ linear, γ 12-17=0.18, κ 3-way, Step A+B, peak_day
105). Seasons 16-17,17-18,19-20. Point estimate, per-season π+R0 (pin work σ0.10).

Output: outputs/eda/erlang_fit.json + viz_fit_erlang_{total,byage}.png
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

from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang import (
    simulate_jax_erlang, daily_new_infection_by_age_erlang, split_seed_to_erlang,
)
import final_pipeline_confirmed as F

REPO_ROOT = Path(__file__).resolve().parent.parent
ED = REPO_ROOT / "outputs" / "eda"; FIGDIR = REPO_ROOT / "presentations" / "figures"
OUT = ED / "erlang_fit.json"
FT = FIGDIR / "viz_fit_erlang_total.png"; FB = FIGDIR / "viz_fit_erlang_byage.png"

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEAS]
AGE_COLORS = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#7b3294"]
GRAY = "#666666"; ERL = "#1a9850"; SNG = "#B23A48"
ADULT = ["18-44", "45-64"]; SCHOOL = ["6-11", "12-17"]


def run_erlang(C, i, R0, pi, p_work=0.6, work_win=(-1e9, 1e9), work_base=1.0):
    phi = jnp.asarray(F.PHI)
    beta = derive_beta_from_R0_simplex(C["ngm"], jnp.asarray(R0), jnp.asarray(pi), phi)
    kw = dict(C["shared"])
    kw["beta_h"] = beta[0]; kw["beta_w"] = beta[1]; kw["beta_s"] = beta[2]; kw["beta_o"] = beta[3]
    kw["phi_susc"] = phi; kw["p_school"] = 1.0; kw["p_work"] = p_work
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_work_baseline"] = work_base
    st = simulate_jax_erlang(split_seed_to_erlang(C["states"][i]), **kw, discretize_time=False)
    return daily_new_infection_by_age_erlang(st)


def pred_erlang(C, i, R0, pi, nw=52):
    return np.asarray(simulation_to_hira_by_age_jax(run_erlang(C, i, R0, pi), F.GAMMA15, n_weeks=nw))


def fit_erlang(C, i):
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i]); phi = jnp.asarray(F.PHI)
    def loss(x):
        R0 = jnp.exp(x[0]); pi = jax.nn.softmax(x[1:5])
        beta = derive_beta_from_R0_simplex(C["ngm"], R0, pi, phi)
        kw = dict(C["shared"]); kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = phi
        st = simulate_jax_erlang(split_seed_to_erlang(C["states"][i]), **kw, discretize_time=False)
        pred = simulation_to_hira_by_age_jax(daily_new_infection_by_age_erlang(st), F.GAMMA15, n_weeks=C["nw"])
        nll = nb_nll_jax(obsj, pred, wj, concentration=x[5], min_rate=0.01)
        centered = x[1:5] - jnp.mean(x[1:5])
        return nll + 0.5*jnp.sum((centered - jnp.asarray(F.LOGIT_REF))**2 / jnp.asarray(F.SIGMA_PIN)**2)
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(61+i); bounds = [F.LOG_R0_B] + [(-10, 10)]*4 + [F.PHI_NB_B]; best = None
    for k in range(10):
        x0 = np.concatenate([[np.log(rng.uniform(1.8, 2.4))], F.LOGIT_REF + rng.normal(0, 0.5, 4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun: best = r
    x = best.x
    return dict(R0=float(np.exp(x[0])), pi=[float(p) for p in np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))], nll=float(best.fun))


def metrics(C, i, pred):
    obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i]); mask = w.sum(1) > 0
    wk = np.where(mask)[0]; o = obs[mask].sum(1); m = pred[mask].sum(1)
    opk = int(wk[np.argmax(o)]); mpk = int(wk[np.argmax(m)])
    owid = o.sum()/max(o.max(), 1); mwid = m.sum()/max(m.max(), 1)
    om = {ag: float(obs[mask, a].sum()/max(pred[mask, a].sum(), 1.0)) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return dict(obs_pk=opk, mdl_pk=mpk, width_ratio=float(owid/mwid), om_total=float(o.sum()/max(m.sum(), 1.0)), om_age=om)


def redistribution(C, i, R0, pi):
    base = C["H"] @ np.asarray(run_erlang(C, i, R0, pi, p_work=0.6)).sum(0)
    inc = run_erlang(C, i, R0, pi, p_work=0.4)
    d = (C["H"] @ np.asarray(inc).sum(0) - base) / C["pop6"]
    da = {ag: float(100.0*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return da, all(da[a] < 0 for a in ADULT)


def main():
    print("=" * 92); print("ERLANG(3) I-compartment fit — 3 seasons, baseline 0.6, peak_day 105"); print("=" * 92)
    t0 = time.perf_counter(); C = F.build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
    old = json.load(open(ED/"final_fit.json"))["per_season"]   # single-I confirmed fit

    res = {}; preds_e = {}; preds_s = {}
    for s, i in zip(SEAS, IDX):
        f = fit_erlang(C, i)
        pe = pred_erlang(C, i, f["R0"], f["pi"]); preds_e[s] = pe
        # single-I pred from confirmed fit
        ps = F.pred_h(C, F.run_inc(C, i, old[s]["R0"], old[s]["pi"])); preds_s[s] = ps
        me = metrics(C, i, pe); ms = metrics(C, i, ps)
        da, adown = redistribution(C, i, f["R0"], f["pi"])
        res[s] = dict(erlang=dict(R0=f["R0"], pi=f["pi"], nll=f["nll"], **me),
                      single=dict(R0=old[s]["R0"], **ms), redist_adult_down=bool(adown), d_attack=da)
        print(f"\n  [{s}]")
        print(f"    R0: single={old[s]['R0']:.3f} → erlang={f['R0']:.3f}")
        print(f"    width_ratio: single={ms['width_ratio']:.2f} → erlang={me['width_ratio']:.2f}")
        print(f"    peak주(mdl/obs): single={ms['mdl_pk']}/{ms['obs_pk']} → erlang={me['mdl_pk']}/{me['obs_pk']}")
        print(f"    obs/model total: single={ms['om_total']:.2f} → erlang={me['om_total']:.2f}")
        print(f"    obs/model 45-64: single={ms['om_age']['45-64']:.2f} → erlang={me['om_age']['45-64']:.2f}"
              f"   65+: single={ms['om_age']['65+']:.2f} → erlang={me['om_age']['65+']:.2f}")
        print(f"    재분배 성인↓={adown}")

    wr_s = np.mean([res[s]["single"]["width_ratio"] for s in SEAS]); wr_e = np.mean([res[s]["erlang"]["width_ratio"] for s in SEAS])
    print("\n" + "=" * 92)
    print(f"★ width_ratio 평균: single={wr_s:.2f} → erlang={wr_e:.2f}  (1근접={'개선' if abs(wr_e-1)<abs(wr_s-1) else '미개선'})")
    n_ad = sum(1 for s in SEAS if res[s]["redist_adult_down"])
    print(f"★ 재분배 성인↓ {n_ad}/3 유지")
    print("=" * 92)
    OUT.write_text(json.dumps(dict(meta=dict(baseline=0.6, stages=3, peak_day=105, seasons=SEAS),
                                   results=res, width_ratio_single=float(wr_s), width_ratio_erlang=float(wr_e)),
                              indent=2, default=float))
    print(f"[json] {OUT}")

    # figures (3 seasons)
    weeks = np.arange(52)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, s in enumerate(SEAS):
        ax = axes[k]; o = C["full_obs"][s].sum(1)
        ax.plot(weeks, o, "o", color=GRAY, ms=3.5, alpha=0.7, label="데이터")
        ax.plot(weeks, preds_s[s].sum(1), "-", color=SNG, lw=1.8, alpha=0.8, label="단일 I")
        ax.plot(weeks, preds_e[s].sum(1), "-", color=ERL, lw=2, label="Erlang(3)")
        ax.set_title(f"{s}", fontsize=11, fontweight="bold"); ax.set_xlabel("주차"); ax.grid(alpha=0.25)
        ax.text(0.03, 0.85, f"wid_ratio\n단일{res[s]['single']['width_ratio']:.2f}→Erl{res[s]['erlang']['width_ratio']:.2f}", transform=ax.transAxes, fontsize=8, color="#333")
        if k == 0: ax.legend(fontsize=8); ax.set_ylabel("주간 진료에피소드")
    fig.suptitle("Erlang(3) vs 단일 I — 유행 폭 (3시즌 전연령, baseline 0.6)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FT, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {FT}")

    fig, axes = plt.subplots(3, 6, figsize=(16, 7.5), sharex=True)
    for r, s in enumerate(SEAS):
        for c, ag in enumerate(HIRA_AGE_GROUPS):
            ax = axes[r, c]
            ax.plot(weeks, C["full_obs"][s][:, c], "o", color=GRAY, ms=2, alpha=0.6)
            ax.plot(weeks, preds_s[s][:, c], "-", color=SNG, lw=1.2, alpha=0.7)
            ax.plot(weeks, preds_e[s][:, c], "-", color=ERL, lw=1.5)
            ax.grid(alpha=0.2); ax.text(0.04, 0.8, f"{res[s]['erlang']['om_age'][ag]:.2f}", transform=ax.transAxes, fontsize=7, color="#333")
            if r == 0: ax.set_title(f"{ag}세", fontsize=9, fontweight="bold")
            if c == 0: ax.set_ylabel(f"{s}", fontsize=8, fontweight="bold")
            ax.tick_params(labelsize=6)
    fig.suptitle("Erlang(3, 녹색) vs 단일 I(빨강) vs 데이터 — 연령별 (셀=Erlang obs/model)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FB, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {FB}")


if __name__ == "__main__":
    main()
