"""Figure 3 (fit_total) with credible intervals and an ISO-week axis.

Co-author #14: the panel has no credible interval, and an "Epidemic week 0-50"
axis hides that week 0 is ISO week 36. ``fig_fit_total_fluA4.py`` stored only
the posterior-mean curve (N=60), so no band could be drawn; this recomputes at
N=500 and adds quantiles.

Two bands are produced, because "credible interval" can mean either and the
question "does the observation fall inside?" only makes sense for the second:

``q05/q95``, ``q25/q75``
    spread of the modelled weekly total across posterior draws — parameter
    uncertainty alone. phi_NB does not enter.
``pred_q05/pred_q95``, ``pred_q25/pred_q75``
    posterior predictive, i.e. each draw's mean curve passed through the
    negative-binomial observation model (mean mu, variance mu + mu^2/phi_NB,
    matching ``nb_nll_jax``) per age band and summed over bands. This is the
    interval an observed weekly count should fall in.

The existing ``mean`` and ``obs`` keys are preserved: ``_fig_Re_readable_fluA4.py``
reads ``mean``.

Writes:
  outputs/fluA/fit_total_curves_fluA4.json   (existing keys kept, quantiles added)
  figures/paper_fluA4/{pdf,png}/fit_total.{pdf,png}   (overwrites)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import fluA_setup as FS  # noqa: E402
import nuts_seasonpop as NS  # noqa: E402
from figstyle import savefig, COL_ZERO, W_DOUBLE  # noqa: E402
from kt_epimodel_hira.jax_model.erlang_presymp import (  # noqa: E402
    daily_new_onset_by_age_erlang_presymp, simulate_jax_erlang_presymp,
    split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax  # noqa: E402
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex  # noqa: E402

ED = REPO / "outputs" / "eda"
FLUA = REPO / "outputs" / "fluA"
MERGED = ED / "nuts_fluA4_raw.npz"
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
N_SUB = 500
SEED = 1234
ISO_START_WEEK = 36     # hira_target._season_start: ISO week 36 Monday
TICK_EVERY = 4


def iso_week_labels(season: str, n_weeks: int, every: int = TICK_EVERY):
    """(tick indices, ISO week labels). Computed from real dates, so the 52/53
    week years and the 52->1 wrap are handled rather than assumed."""
    start_year = int(season.split("-")[0])
    d0 = _dt.date.fromisocalendar(start_year, ISO_START_WEEK, 1)
    idx = list(range(0, n_weeks, every))
    labels = [str((d0 + _dt.timedelta(weeks=i)).isocalendar()[1]) for i in idx]
    return idx, labels


def make_sim_fn(sh, ngm3, st0, nw):
    def one(pi_k, R0):
        beta = derive_beta_from_R0_simplex(
            ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NS.NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = (
            beta[0], beta[1], beta[2], beta[3])
        kw["phi_susc"] = jnp.asarray(NS.PHI)
        kw["kappa"] = jnp.asarray(NS.KAP)
        kw["p_school"] = NS.BASE
        kw["p_work"] = NS.BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(st0),
                                          w_presymp=NS.W, **kw,
                                          discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return simulation_to_hira_by_age_jax(inc, jnp.asarray(NS.GAMMA), n_weeks=nw)
    return jax.jit(jax.vmap(one, in_axes=(0, 0)))


def main():
    t0 = time.time()
    print(f"[fit_total CI] seasons={SEASONS}  N={N_SUB}  seed={SEED}")
    C = FS.build_fluA_setup(SEASONS, verbose=False)

    d = np.load(MERGED, allow_pickle=True)
    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    phi_all = np.asarray(d["phi_nb"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pi_all.shape[0], size=N_SUB, replace=False)
    pi_batch = jnp.asarray(pi_all[idx])
    phi_sub = phi_all[idx]

    curves = json.loads((FLUA / "fit_total_curves_fluA4.json").read_text()) \
        if (FLUA / "fit_total_curves_fluA4.json").exists() else {}

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.0), constrained_layout=True)
    summary = {}
    for k, s in enumerate(SEASONS):
        ax = axes[k // 2][k % 2]
        sh = C["shared_by_s"][s]; ngm3 = C["ngm3_by_s"][s]; st0 = C["st_by_s"][s]
        nw = C["nw"]
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, k]))
        pred = np.asarray(make_sim_fn(sh, ngm3, st0, nw)(pi_batch, R0_batch))  # (N, nw, 6)

        tot = pred.sum(axis=2)                       # (N, nw) modelled weekly total
        mean_pr = tot.mean(axis=0)

        # posterior predictive: NB2(mu, k=phi) per age-week, then sum over ages.
        # numpy's negative_binomial(n, p) with n=k, p=k/(k+mu) has mean mu and
        # variance mu + mu^2/k, matching nb_nll_jax's parametrization.
        mu = np.maximum(pred, 0.01)
        kk = phi_sub[:, None, None]
        p = kk / (kk + mu)
        rng_pred = np.random.default_rng(SEED + 1)
        draws = rng_pred.negative_binomial(np.broadcast_to(kk, mu.shape), p)
        pred_tot = draws.sum(axis=2)                 # (N, nw)

        obs = np.asarray(C["obs"][k])
        o_tot = obs.sum(axis=1)
        fit_mask = np.asarray(C["w"][k])[:, 0] > 0
        n_fit = int(fit_mask.sum())

        q = {f"q{int(v*100):02d}": np.quantile(tot, v, axis=0)
             for v in (0.05, 0.25, 0.75, 0.95)}
        qp = {f"pred_q{int(v*100):02d}": np.quantile(pred_tot, v, axis=0)
              for v in (0.05, 0.25, 0.75, 0.95)}

        entry = dict(curves.get(s, {}))
        entry["mean"] = mean_pr.tolist()
        entry["obs"] = o_tot.tolist()
        entry.update({kk_: vv.tolist() for kk_, vv in q.items()})
        entry.update({kk_: vv.tolist() for kk_, vv in qp.items()})
        entry["n_draws"] = int(N_SUB)
        entry["fitted_weeks"] = n_fit
        curves[s] = entry

        wks = np.arange(nw)
        ax.fill_between(wks, qp["pred_q05"], qp["pred_q95"], color="#BBBBBB",
                        alpha=0.45, lw=0, zorder=1)
        ax.fill_between(wks, q["q05"], q["q95"], color="#777777",
                        alpha=0.55, lw=0, zorder=2)
        ax.plot(wks, mean_pr, color="black", lw=1.4, zorder=3)
        ax.plot(wks, o_tot, "o", color=COL_ZERO, ms=2.6, alpha=0.85, zorder=4)
        ax.axvline(n_fit - 0.5, color="#333333", lw=0.8, ls=":", zorder=2)

        ticks, labels = iso_week_labels(s, nw)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlim(-0.5, nw - 0.5)
        ax.set_title(s, fontsize=9, fontweight="bold")
        if k // 2 == 1:
            ax.set_xlabel("Week (ISO, season starts at week 36)", fontsize=8.5)
        if k % 2 == 0:
            ax.set_ylabel("Weekly incidence (influenza A)", fontsize=8.5)
        ax.tick_params(labelsize=7)

        inside = int(((o_tot >= qp["pred_q05"]) & (o_tot <= qp["pred_q95"]))[fit_mask].sum())
        pk = int(np.argmax(o_tot))
        summary[s] = dict(
            obs_inside_pred_90_fitted=inside, n_fitted=n_fit,
            frac_inside=inside / n_fit,
            peak_week_index=pk,
            peak_obs=float(o_tot[pk]),
            peak_pred_q05=float(qp["pred_q05"][pk]),
            peak_pred_q95=float(qp["pred_q95"][pk]),
            peak_inside_pred=bool(qp["pred_q05"][pk] <= o_tot[pk] <= qp["pred_q95"][pk]),
            peak_param_q05=float(q["q05"][pk]), peak_param_q95=float(q["q95"][pk]),
            peak_inside_param=bool(q["q05"][pk] <= o_tot[pk] <= q["q95"][pk]),
            param_band_rel_width_at_peak=float(
                (q["q95"][pk] - q["q05"][pk]) / max(mean_pr[pk], 1.0)),
            pred_band_rel_width_at_peak=float(
                (qp["pred_q95"][pk] - qp["pred_q05"][pk]) / max(mean_pr[pk], 1.0)),
        )
        print(f"  [{s}] obs inside 90% predictive: {inside}/{n_fit} fitted weeks; "
              f"peak obs={o_tot[pk]:,.0f} vs predictive "
              f"[{qp['pred_q05'][pk]:,.0f}, {qp['pred_q95'][pk]:,.0f}] "
              f"inside={summary[s]['peak_inside_pred']}", flush=True)

    handles = [
        Line2D([0], [0], marker="o", color=COL_ZERO, ls="", markersize=4.5,
               label="Observed (influenza A)"),
        Line2D([0], [0], color="black", lw=1.4, label="Posterior mean"),
        Patch(facecolor="#777777", alpha=0.55, label="90% credible interval (parameter)"),
        Patch(facecolor="#BBBBBB", alpha=0.45,
              label="90% credible interval (posterior predictive)"),
        Line2D([0], [0], color="#333333", lw=0.8, ls=":", label="End of fitted window"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=3, frameon=False, fontsize=7.5)
    savefig(fig, "fit_total")

    (FLUA / "fit_total_curves_fluA4.json").write_text(json.dumps(curves))
    (FLUA / "fit_total_ci_summary_fluA4.json").write_text(
        json.dumps(dict(n_draws=N_SUB, seed=SEED, seasons=SEASONS,
                        summary=summary,
                        note=("parameter band = quantiles of the modelled weekly "
                              "total across draws; predictive band additionally "
                              "passes each draw through NB2(mu, phi_NB) per age "
                              "band and sums"),
                        generated=time.strftime("%Y-%m-%d %H:%M:%S")),
                   indent=2, default=float))
    print(f"[json] {FLUA / 'fit_total_curves_fluA4.json'} (quantiles added)")
    print(f"[json] {FLUA / 'fit_total_ci_summary_fluA4.json'}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
