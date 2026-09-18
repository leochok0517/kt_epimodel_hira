"""T5 -- fit_total figure, 4-season fluA4 (2x2 grid).

Standalone lean re-implementation of v5's ``paper_figures_extra.py`` fig_F7
(observed influenza-A weekly claims vs. posterior-mean model, per season),
scoped to just this one figure rather than importing the full 3-season F7-F12
bundle. Posterior mean forward sim uses jax.vmap + jax.jit over an N=60
subsample of nuts_fluA4_raw.npz, matching v5's subsample size.

Also writes outputs/fluA/fit_total_curves_fluA4.json (mean model curve +
observed points per season), the fluA4 analogue of v5's
outputs/eda/fit_total_curves.json, so _fig_Re_readable_fluA4.py (T1) can
overlay the fitted incidence curve without re-simulating.

Saves:
  figures/paper_fluA4/{pdf,png}/fit_total.{pdf,png}
  outputs/fluA/fit_total_curves_fluA4.json
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["FIG_OUTDIR_SUFFIX"] = "_fluA4"
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import savefig, COL_SEASON, COL_ZERO, W_DOUBLE
import fluA_setup as FS
import nuts_seasonpop as NS
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang,
)
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax

REPO = Path(__file__).resolve().parent.parent
ED = REPO / "outputs" / "eda"
FLUA = REPO / "outputs" / "fluA"
MERGED = ED / "nuts_fluA4_raw.npz"
SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
N_SUB = 60
SEED = 1234

#: figstyle.COL_SEASON has no 2018-2019 entry (v5 only had 3 seasons) -- a
#: local copy with a 4th color added, rather than mutating the shared module.
COL_SEASON_FLUA4 = dict(COL_SEASON, **{"2018-2019": "#E69F00"})


def make_sim_fn(sh, ngm3, st0, nw):
    def one(pi_k, R0):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NS.NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = jnp.asarray(NS.KAP)
        kw["p_school"] = NS.BASE; kw["p_work"] = NS.BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(st0), w_presymp=NS.W, **kw,
                                          discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        pred = simulation_to_hira_by_age_jax(inc, jnp.asarray(NS.GAMMA), n_weeks=nw)
        return pred   # (nw, 6)
    return jax.jit(jax.vmap(one, in_axes=(0, 0)))


def main():
    t0 = time.time()
    print(f"[fit_total_fluA4] seasons={SEASONS}  N_sub={N_SUB}")
    C = FS.build_fluA_setup(SEASONS, verbose=True)

    d = np.load(MERGED, allow_pickle=True)
    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pi_all.shape[0], size=N_SUB, replace=False)
    pi_batch = jnp.asarray(pi_all[idx])

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 6.0), constrained_layout=True)
    curves_cache = {}
    for k, s in enumerate(SEASONS):
        ax = axes[k // 2][k % 2]
        sh = C["shared_by_s"][s]; ngm3 = C["ngm3_by_s"][s]; st0 = C["st_by_s"][s]
        nw = C["nw"]
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, k]))
        fn = make_sim_fn(sh, ngm3, st0, nw)
        pred = np.asarray(fn(pi_batch, R0_batch))       # (N_sub, nw, 6)
        pred_tot = pred.sum(axis=2)                       # (N_sub, nw)
        mean_pr = pred_tot.mean(axis=0)
        obs = np.asarray(C["obs"][k])                     # (nw, 6) influenza-A counts
        o_tot = obs.sum(axis=1)
        wks = np.arange(nw)
        # 모형 곡선은 시즌별 색 대신 검정으로 통일한다 (본문 Figure 3).
        ax.plot(wks, mean_pr, color="black", lw=1.6,
                label="Posterior mean" if k == 0 else None)
        ax.plot(wks, o_tot, "o", color=COL_ZERO, ms=3, alpha=0.75,
                label="Observed (influenza A)" if k == 0 else None)
        ax.set_title(s, color="black", fontsize=9, fontweight="bold")
        ax.set_xlabel("Epidemic week")
        if k % 2 == 0:
            ax.set_ylabel("Weekly incidence (influenza A)")
        curves_cache[s] = dict(mean=mean_pr.tolist(), obs=o_tot.tolist())
        print(f"  [{s}] wall={time.time()-t0:.1f}s cumulative", flush=True)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", lw=1.6, label="Posterior mean"),
        Line2D([0], [0], marker="o", color=COL_ZERO, ls="", markersize=4,
               alpha=0.75, label="Observed (influenza A)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, frameon=False, fontsize=8)
    savefig(fig, "fit_total")

    curves_path = FLUA / "fit_total_curves_fluA4.json"
    curves_path.write_text(json.dumps(curves_cache))
    print(f"[json] {curves_path}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
