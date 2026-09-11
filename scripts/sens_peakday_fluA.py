"""08 -- is the 2015-16 misfit a fixed-calendar artifact?

Sweeps ``seasonality_peak_day`` (and, for 2015-16 only, the winter-break
window) with pi and phi_nb pinned at the 5-season joint posterior mean and
re-fits only the scalar log_R0 per grid point. Everything else - contact
matrices, mobility, employment, vaccination, gamma/kappa/phi_susc, initial
seed, population - comes from ``fluA_setup.build_fluA_setup`` unchanged.

Does not touch ``foi_jax.py``, ``erlang_presymp.py``, ``parameters.py``, or
any v5/v7 output. New file per ``prompts/99_conventions.md``.

Every loss evaluation goes through ``jax.jit(jax.value_and_grad(...))`` -
never bare Python calls to the simulator - per the 2026-08-28 lesson in
CLAUDE.md (Nelder-Mead without jit hung for 182 minutes at 54GB).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fluA_setup as FS
import nuts_seasonpop as NS
from kt_epimodel_hira.jax_model.erlang_presymp import (
    daily_new_onset_by_age_erlang_presymp,
    simulate_jax_erlang_presymp,
    split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    nb_nll_jax,
    simulation_to_hira_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex

REPO = SCRIPTS_DIR.parent
ED = REPO / "outputs" / "fluA"
ED.mkdir(parents=True, exist_ok=True)
POSTERIOR_NPZ = REPO / "outputs" / "eda" / "nuts_fluA_raw.npz"

# Default calendar (unchanged elsewhere in the pipeline; these are the
# ``simulate_jax_erlang_presymp`` kwarg defaults, made explicit here so the
# sweep and the shifted arms share one source of truth).
DEFAULT_PEAK_DAY = 105.0
DEFAULT_HOLIDAY = dict(start_day=113.0, min_start_day=127.0, min_end_day=162.0, end_day=183.0)
SHIFT = 49.0
SHIFTED_HOLIDAY = {k: v + SHIFT for k, v in DEFAULT_HOLIDAY.items()}

PEAK_DAY_GRID = list(range(105, 176, 5))   # 105..175, 15 points
LOG_R0_BOUNDS = (float(np.log(0.8)), float(np.log(3.5)))
LOG_R0_X0 = float(np.log(2.0))


def load_posterior_fixed():
    d = np.load(POSTERIOR_NPZ, allow_pickle=True)
    pi_mean = d["pi"].mean(0)
    phi_nb_mean = float(d["phi_nb"].mean())
    seasons_fitted = [str(s) for s in d["seasons"]]
    return pi_mean, phi_nb_mean, seasons_fitted


def make_season_nll(C, season_idx, pi_fixed, phi_nb_fixed):
    """One jit compile per season. peak_day / holiday knobs stay traced, so
    the whole sweep for a season reuses this single compiled program."""
    s = C["seasons"][season_idx]
    phi_j = jnp.asarray(NS.PHI)
    gamma_j = jnp.asarray(NS.GAMMA)
    kap_j = jnp.asarray(NS.KAP)
    nw = C["nw"]
    obs = jnp.asarray(C["obs"][season_idx])
    w = jnp.asarray(C["w"][season_idx])
    init_state = split_seed_to_erlang(C["st_by_s"][s])
    ngm = C["ngm3_by_s"][s]
    sh = dict(C["shared_by_s"][s])
    pi_j = jnp.asarray(pi_fixed)
    phi_nb_j = jnp.asarray(phi_nb_fixed)

    def nll_and_pred(log_R0, peak_day, hol_start, hol_min_start, hol_min_end, hol_end):
        R0 = jnp.exp(log_R0)
        beta = derive_beta_from_R0_simplex(ngm, R0, pi_j, phi_j) / NS.NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = phi_j
        kw["kappa"] = kap_j
        kw["p_school"] = NS.BASE
        kw["p_work"] = NS.BASE
        kw["seasonality_peak_day"] = peak_day
        kw["school_holiday_start_day"] = hol_start
        kw["school_holiday_min_start_day"] = hol_min_start
        kw["school_holiday_min_end_day"] = hol_min_end
        kw["school_holiday_end_day"] = hol_end
        st = simulate_jax_erlang_presymp(init_state, w_presymp=NS.W, **kw, discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        pred = simulation_to_hira_by_age_jax(inc, gamma_j, n_weeks=nw)
        nll = nb_nll_jax(obs, pred, w, concentration=phi_nb_j, min_rate=0.01)
        return nll, pred

    def nll_only(log_R0, peak_day, hol_start, hol_min_start, hol_min_end, hol_end):
        return nll_and_pred(log_R0, peak_day, hol_start, hol_min_start, hol_min_end, hol_end)[0]

    val_grad = jax.jit(jax.value_and_grad(nll_only, argnums=0))
    pred_fn = jax.jit(lambda *a: nll_and_pred(*a)[1])
    obs_np = np.asarray(obs)
    return val_grad, pred_fn, obs_np


def fit_R0(val_grad, peak_day, hol):
    peak_day = jnp.asarray(float(peak_day))
    hol_j = {k: jnp.asarray(float(v)) for k, v in hol.items()}

    def fg(x):
        v, g = val_grad(jnp.asarray(float(x[0])), peak_day,
                         hol_j["start_day"], hol_j["min_start_day"],
                         hol_j["min_end_day"], hol_j["end_day"])
        return float(v), np.array([float(g)])

    res = minimize(fg, x0=[LOG_R0_X0], jac=True, method="L-BFGS-B",
                    bounds=[LOG_R0_BOUNDS], options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    return res


def peak_stats(pred_fn, log_R0, peak_day, hol, obs_np):
    peak_day_j = jnp.asarray(float(peak_day))
    hol_j = {k: jnp.asarray(float(v)) for k, v in hol.items()}
    pred = np.asarray(pred_fn(jnp.asarray(float(log_R0)), peak_day_j,
                                hol_j["start_day"], hol_j["min_start_day"],
                                hol_j["min_end_day"], hol_j["end_day"]))
    o1 = obs_np.sum(1)
    p1 = pred.sum(1)
    return dict(
        peak_obs=float(o1.max()), peak_model=float(p1.max()),
        peak_ratio=float(o1.max() / max(p1.max(), 1e-9)),
        week_obs=int(o1.argmax()), week_model=int(p1.argmax()),
        pred_weekly=p1.tolist(), obs_weekly=o1.tolist(),
    )


def main():
    t0 = time.time()
    print("[setup] loading posterior mean (pi, phi_nb), building 5-season fluA setup ...")
    pi_mean, phi_nb_mean, seasons_fitted = load_posterior_fixed()
    print(f"  pi_mean = {pi_mean.round(4).tolist()}  phi_nb_mean = {phi_nb_mean:.4f}")
    print(f"  posterior was fit on seasons: {seasons_fitted}")

    C = FS.build_fluA_setup(FS.SEASONS_DEFAULT, verbose=True)
    seasons = C["seasons"]
    assert seasons == seasons_fitted, (
        f"season set mismatch: setup={seasons} vs posterior={seasons_fitted}"
    )

    val_grads = {}
    pred_fns = {}
    obs_nps = {}
    for i, s in enumerate(seasons):
        t_c = time.time()
        vg, pf, onp = make_season_nll(C, i, pi_mean, phi_nb_mean)
        # trigger + time the jit compile once with the default calendar
        r0 = fit_R0(vg, DEFAULT_PEAK_DAY, DEFAULT_HOLIDAY)
        print(f"  [{s}] compile+first fit: {time.time()-t_c:.1f}s  "
              f"R0={np.exp(r0.x[0]):.3f}  nll={r0.fun:.1f}")
        val_grads[s] = vg
        pred_fns[s] = pf
        obs_nps[s] = onp

    # ---- Stage 1: peak_day sweep, all seasons ----
    print("\n[stage 1] peak_day sweep (105..175 by 5), all seasons ...", flush=True)
    stage1 = {s: [] for s in seasons}
    t_s1 = time.time()
    for s in seasons:
        for pd in PEAK_DAY_GRID:
            res = fit_R0(val_grads[s], pd, DEFAULT_HOLIDAY)
            log_R0_opt = float(res.x[0])
            pk = peak_stats(pred_fns[s], log_R0_opt, pd, DEFAULT_HOLIDAY, obs_nps[s])
            row = dict(peak_day=pd, R0=float(np.exp(log_R0_opt)), log_R0=log_R0_opt,
                        nll=float(res.fun), converged=bool(res.success), **pk)
            del row["pred_weekly"], row["obs_weekly"]  # keep json small; not needed per grid point
            stage1[s].append(row)
        best = min(stage1[s], key=lambda r: r["nll"])
        print(f"  [{s}] best peak_day={best['peak_day']}  R0={best['R0']:.3f}  "
              f"nll={best['nll']:.1f}  peak_ratio={best['peak_ratio']:.3f}", flush=True)
    print(f"[stage 1] wall={time.time()-t_s1:.1f}s")

    # ---- Stage 2: 2-axis split, 2015-16 only ----
    print("\n[stage 2] seasonality vs winter-break window, 2015-16 only ...", flush=True)
    s2015 = "2015-2016"
    arms = {
        "A0_default_default": (DEFAULT_PEAK_DAY, DEFAULT_HOLIDAY),
        "A1_shift_default": (DEFAULT_PEAK_DAY + SHIFT, DEFAULT_HOLIDAY),
        "A2_default_shift": (DEFAULT_PEAK_DAY, SHIFTED_HOLIDAY),
        "A3_shift_shift": (DEFAULT_PEAK_DAY + SHIFT, SHIFTED_HOLIDAY),
    }
    stage2 = {}
    stage2_curves = {}
    t_s2 = time.time()
    for name, (pd, hol) in arms.items():
        res = fit_R0(val_grads[s2015], pd, hol)
        log_R0_opt = float(res.x[0])
        pk = peak_stats(pred_fns[s2015], log_R0_opt, pd, hol, obs_nps[s2015])
        stage2[name] = dict(peak_day=pd, holiday=hol, R0=float(np.exp(log_R0_opt)),
                              nll=float(res.fun), converged=bool(res.success),
                              peak_ratio=pk["peak_ratio"], week_obs=pk["week_obs"],
                              week_model=pk["week_model"],
                              peak_obs=pk["peak_obs"], peak_model=pk["peak_model"])
        stage2_curves[name] = dict(obs_weekly=pk["obs_weekly"], pred_weekly=pk["pred_weekly"])
        print(f"  [{name}] peak_day={pd} holiday_start={hol['start_day']}  "
              f"R0={stage2[name]['R0']:.3f}  nll={stage2[name]['nll']:.1f}  "
              f"peak_ratio={pk['peak_ratio']:.3f}", flush=True)
    print(f"[stage 2] wall={time.time()-t_s2:.1f}s")

    # ---- Stage 3: identifiability read-off from stage 1's 2015-16 profile ----
    # Stage 1, applied to a single season, already optimizes R0 at each fixed
    # peak_day - that *is* the profile likelihood of peak_day with R0
    # profiled out. No separate 2D grid is run (would cost ~15x more compute
    # for a question stage 1 already answers).
    rows = stage1[s2015]
    peak_days = np.array([r["peak_day"] for r in rows], dtype=float)
    r0s = np.array([r["R0"] for r in rows])
    nlls = np.array([r["nll"] for r in rows])
    nll_min = nlls.min()
    within2 = peak_days[nlls <= nll_min + 2.0]   # Wilks-style delta-NLL=2 band
    stage3 = dict(
        peak_day_at_min=float(peak_days[nlls.argmin()]),
        R0_at_min=float(r0s[nlls.argmin()]),
        nll_min=float(nll_min),
        delta_nll_2_band_days=[float(within2.min()), float(within2.max())] if len(within2) else None,
        delta_nll_2_band_width=float(within2.max() - within2.min()) if len(within2) else None,
        corr_peak_day_R0=float(np.corrcoef(peak_days, r0s)[0, 1]),
    )
    print(f"\n[stage 3] 2015-16 profile: min at peak_day={stage3['peak_day_at_min']}  "
          f"R0={stage3['R0_at_min']:.3f}  delta-NLL=2 band={stage3['delta_nll_2_band_days']}  "
          f"corr(peak_day,R0)={stage3['corr_peak_day_R0']:.3f}")

    total_wall = time.time() - t0
    print(f"\n[total wall] {total_wall:.1f}s ({total_wall/60:.1f} min)")
    if total_wall > 600:
        print("  !!! exceeded the 10-minute budget in prompts/08 - report this, something is off")

    out = dict(
        pi_mean=pi_mean.tolist(), phi_nb_mean=phi_nb_mean, seasons=seasons,
        default_peak_day=DEFAULT_PEAK_DAY, default_holiday=DEFAULT_HOLIDAY,
        shift_days=SHIFT, shifted_holiday=SHIFTED_HOLIDAY,
        peak_day_grid=PEAK_DAY_GRID,
        stage1=stage1, stage2=stage2, stage2_curves=stage2_curves, stage3=stage3,
        wall_sec_total=total_wall, generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    out_json = ED / "sens_peakday_fluA.json"
    out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {out_json}")

    # ---- plot 1: NLL and peak_ratio vs peak_day, all seasons ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seasons)))
    for s, c in zip(seasons, colors):
        rows = stage1[s]
        pd = [r["peak_day"] for r in rows]
        nll = [r["nll"] for r in rows]
        pr = [r["peak_ratio"] for r in rows]
        axes[0].plot(pd, nll, "o-", color=c, label=s, ms=4)
        best = min(rows, key=lambda r: r["nll"])
        axes[0].plot(best["peak_day"], best["nll"], "*", color=c, ms=16, mec="black")
        axes[1].plot(pd, pr, "o-", color=c, label=s, ms=4)
    axes[0].axvline(DEFAULT_PEAK_DAY, color="grey", ls="--", lw=1, label="default (105)")
    axes[0].set_xlabel("seasonality_peak_day"); axes[0].set_ylabel("NLL (R0 profiled out)")
    axes[0].set_title("Season-wise NLL vs peak day"); axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("seasonality_peak_day"); axes[1].set_ylabel("obs peak / model peak")
    axes[1].set_title("Peak-ratio recovery vs peak day"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig_path = ED / "sens_peakday_fluA.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"[fig] {fig_path}")

    # ---- plot 2: stage 2 arms, 2015-16 obs vs model ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    weeks = np.arange(len(stage2_curves["A0_default_default"]["obs_weekly"]))
    ax.plot(weeks, stage2_curves["A0_default_default"]["obs_weekly"], "k-", lw=2, label="observed (A)")
    style = {"A0_default_default": ("C0", "-"), "A1_shift_default": ("C1", "-"),
             "A2_default_shift": ("C2", "--"), "A3_shift_shift": ("C3", "--")}
    for name, curve in stage2_curves.items():
        col, ls = style[name]
        r0 = stage2[name]["R0"]
        ax.plot(weeks, curve["pred_weekly"], color=col, ls=ls, lw=1.6,
                 label=f"{name}  (R0={r0:.2f})")
    ax.set_xlabel("fit-window week"); ax.set_ylabel("weekly influenza-A claims (6 age groups summed)")
    ax.set_title("2015-2016: observed vs model, four calendar arms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path2 = ED / "fit_2015_16_arms.png"
    fig.savefig(fig_path2, dpi=140)
    plt.close(fig)
    print(f"[fig] {fig_path2}")


if __name__ == "__main__":
    main()
