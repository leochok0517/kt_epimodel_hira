"""Stage B (prompts/09) -- kappa sweep on the fluA4 posterior, theta=0.

Recounts main_v5 line 415's claim ("winter-break child-ward redistribution
positive in 749 of 750 draws") for fluA4 (4 seasons instead of 3, kappa grid
extended with 0.20 per the prompt). The metric is the population-weighted
child (0-5,6-11,12-17) Delta-attack under the sick-leave scenario, winter
break window - policy_seasonpop.child_agg_weighted's exact definition -
counted positive/negative/zero over every (kappa, season, draw) combination.

No refit. jax.vmap over draws, jax.jit throughout (prompts/99_conventions.md).
theta fixed at 0 -> mathematically identical to the untouched erlang_presymp
pipeline (T1), so this uses erlang_presymp_theta purely for code-path
consistency with sens_theta_fluA4.py, not because theta matters here.
"""
from __future__ import annotations

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
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nuts_seasonpop as NS
import fluA_setup as FS
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang_presymp import ngm_factor, W_PRESYMP, split_seed_to_erlang, daily_new_onset_by_age_erlang_presymp
from kt_epimodel_hira.jax_model.erlang_presymp_theta import simulate_jax_erlang_presymp_theta

ED = REPO / "outputs" / "eda"
OUTD = REPO / "outputs" / "fluA"
OUTD.mkdir(parents=True, exist_ok=True)
FLUA4_NPZ = ED / "nuts_fluA4_raw.npz"

W = W_PRESYMP
NGM_F = ngm_factor(W)
BASE = 0.6
VAC = (113.0, 183.0)
P_POL = 0.4
THETA = 0.0

KAPPA_GRID = [0.20, 0.34, 0.40, 0.50, 0.70, 1.00]   # main_v5's grid + 0.20 per prompt
KAPPA_RATIO = 0.34 / 0.40   # child:adult ratio, matches sens2_kappa_upper.py's kap_vec

N_POST = 50
SEED = 2027

CHILD_AGE = ["0-5", "6-11", "12-17"]


def kap_vec(k_adult):
    return np.array([k_adult * KAPPA_RATIO] * 4 + [k_adult] * 10 + [0.0])


def child_agg_weighted(d_att_N6, pop6, ages):
    idx = [ages.index(a) for a in CHILD_AGE]
    ws = np.array([pop6[i] for i in idx]); ws = ws / ws.sum()
    return (d_att_N6[:, idx] * ws[None, :]).sum(axis=1)


def make_scenario_fns(sh, ngm3, st0):
    def baseline(pi_k, R0, kap):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = kap
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang_presymp_theta(split_seed_to_erlang(st0), w_presymp=W, **kw,
                                                 theta=THETA, discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return inc.sum(0)

    def sick_vac(pi_k, R0, kap):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = kap
        kw["p_school"] = BASE; kw["p_work"] = P_POL
        kw["policy_work_start_day"], kw["policy_work_end_day"] = VAC
        kw["policy_school_start_day"], kw["policy_school_end_day"] = VAC
        kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
        st = simulate_jax_erlang_presymp_theta(split_seed_to_erlang(st0), w_presymp=W, **kw,
                                                 theta=THETA, discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return inc.sum(0)

    base_fn = jax.jit(jax.vmap(baseline, in_axes=(0, 0, None)))
    sick_fn = jax.jit(jax.vmap(sick_vac, in_axes=(0, 0, None)))
    return base_fn, sick_fn


def main():
    t0 = time.time()
    print(f"[Stage B] fluA4 posterior, kappa grid {KAPPA_GRID}, theta=0, N={N_POST}, seed={SEED}")

    d = np.load(FLUA4_NPZ, allow_pickle=True)
    seasons = [str(s) for s in d["seasons"]]
    print(f"  seasons: {seasons}")

    C = FS.build_fluA_setup(seasons, verbose=False)
    H = np.asarray(C["H"])
    ages = list(FS.HIRA_AGE_GROUPS)

    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pi_all.shape[0], size=N_POST, replace=False)
    pi_batch = jnp.asarray(pi_all[idx])

    all_vals = []   # every (kappa, season, draw) weighted-child-d_att value
    per_season_kappa = {}
    for si, s in enumerate(seasons):
        sh = C["shared_by_s"][s]; ngm3 = C["ngm3_by_s"][s]; st0 = C["st_by_s"][s]
        pop6 = np.asarray(C["pop6_by_s"][s])
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, si]))
        base_fn, sick_fn = make_scenario_fns(sh, ngm3, st0)

        per_season_kappa[s] = {}
        for kap_scalar in KAPPA_GRID:
            kap_j = jnp.asarray(kap_vec(kap_scalar))
            base15 = np.asarray(base_fn(pi_batch, R0_batch, kap_j))
            sick15 = np.asarray(sick_fn(pi_batch, R0_batch, kap_j))
            base6 = base15 @ H.T
            sick6 = sick15 @ H.T
            d_att = (sick6 - base6) / pop6[None, :] * 100.0    # (N,6) %p
            child_w = child_agg_weighted(d_att, pop6, ages)     # (N,)
            all_vals.extend(child_w.tolist())
            n_pos = int((child_w > 0).sum())
            n_neg = int((child_w < 0).sum())
            n_zero = N_POST - n_pos - n_neg
            per_season_kappa[s][f"{kap_scalar:.2f}"] = dict(
                n_pos=n_pos, n_neg=n_neg, n_zero=n_zero,
                mean=float(child_w.mean()),
                q05=float(np.quantile(child_w, 0.05)),
                q95=float(np.quantile(child_w, 0.95)),
            )
            print(f"  [{s}][kappa_adult={kap_scalar:.2f}] child-weighted vac Delta-attack: "
                  f"mean={child_w.mean():+.4f}%p  pos={n_pos}/{N_POST}", flush=True)

    all_vals = np.asarray(all_vals)
    n_total = all_vals.shape[0]
    n_pos_total = int((all_vals > 0).sum())
    n_neg_total = int((all_vals < 0).sum())
    n_zero_total = n_total - n_pos_total - n_neg_total

    print(f"\n[Stage B RESULT] {n_pos_total}/{n_total} combinations positive "
          f"({n_neg_total} negative, {n_zero_total} exactly zero)")
    print(f"  (v5's original claim, main_v5 line 415: 749/750, "
          f"kappa grid={{0.34,0.40,0.50,0.70,1.00}} x 3 seasons x N=50)")

    out = dict(seasons=seasons, kappa_grid=KAPPA_GRID, kappa_ratio=KAPPA_RATIO,
                n_post=N_POST, seed=SEED, theta=THETA,
                n_total_combinations=n_total, n_positive=n_pos_total,
                n_negative=n_neg_total, n_zero=n_zero_total,
                positive_fraction=n_pos_total / n_total,
                v5_reference="749/750 positive (3 seasons x 5 kappa x N=50)",
                per_season_kappa=per_season_kappa,
                wall_sec=time.time() - t0, generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_json = OUTD / "sens_kappa_fluA4.json"
    out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {out_json}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in seasons:
        means = [per_season_kappa[s][f"{k:.2f}"]["mean"] for k in KAPPA_GRID]
        q05 = [per_season_kappa[s][f"{k:.2f}"]["q05"] for k in KAPPA_GRID]
        q95 = [per_season_kappa[s][f"{k:.2f}"]["q95"] for k in KAPPA_GRID]
        ax.plot(KAPPA_GRID, means, "o-", label=s)
        ax.fill_between(KAPPA_GRID, q05, q95, alpha=0.12)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlabel("kappa (adult)")
    ax.set_ylabel("population-weighted child Delta-attack, winter break (%p)")
    ax.set_title(f"Stage B: {n_pos_total}/{n_total} combinations positive")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = OUTD / "sens_kappa_fluA4.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"[fig] {fig_path}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
