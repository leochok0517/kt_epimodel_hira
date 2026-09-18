"""Full theta x kappa grid for fluA4 (theta_kappa_heatmap_v2, supplement).

sens_theta_fluA4.json only has the empirical-kappa column and
sens_theta_kappa1_fluA4.json only kappa=1.00. This fills the full grid:
kappa in {0.20, 0.34, 0.40, 0.50, 0.70, 1.00} x theta 21 points (-0.25..0.75)
x 4 seasons, sick leave, school-term window (the figure's scope per its
caption), N_POST=200.

No refit -- forward-only, jax.vmap + jax.jit over the posterior draw axis
per (kappa, theta) combo, so this is cheap despite the larger grid (6x the
single-kappa scripts' cost, ~10-20 min total).
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

import nuts_seasonpop as NS
import fluA_setup as FS
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang_presymp import (
    daily_new_onset_by_age_erlang_presymp, split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
from kt_epimodel_hira.jax_model.erlang_presymp_theta import (
    simulate_jax_erlang_presymp_theta,
)

ED = REPO / "outputs" / "eda"
OUTD = REPO / "outputs" / "fluA"
OUTD.mkdir(parents=True, exist_ok=True)
FLUA4_NPZ = ED / "nuts_fluA4_raw.npz"

W = W_PRESYMP
NGM_F = ngm_factor(W)
BASE = 0.6
TERM = (70.0, 113.0)
P_POL = 0.4

N_POST = 200
SEED = 2027
THETA_GRID = [round(-0.25 + 0.05 * i, 2) for i in range(21)]
KAPPA_GRID = [0.20, 0.34, 0.40, 0.50, 0.70, 1.00]


def kap_vec(k_scalar):
    ratio = 0.34 / 0.40
    return np.array([k_scalar * ratio] * 4 + [k_scalar] * 10 + [0.0])


def sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), q05=float(np.quantile(x, 0.05)),
                q50=float(np.quantile(x, 0.5)), q95=float(np.quantile(x, 0.95)))


def make_sick_fn(sh, ngm3, st0, win, kap):
    def one(pi_k, R0, theta):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = jnp.asarray(kap)
        kw["p_school"] = BASE; kw["p_work"] = P_POL
        kw["policy_work_start_day"], kw["policy_work_end_day"] = win
        kw["policy_school_start_day"], kw["policy_school_end_day"] = win
        kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
        st = simulate_jax_erlang_presymp_theta(split_seed_to_erlang(st0), w_presymp=W, **kw,
                                                 theta=theta, discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return inc.sum(0)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def make_baseline_fn(sh, ngm3, st0, kap):
    from kt_epimodel_hira.jax_model.erlang_presymp import simulate_jax_erlang_presymp

    def one(pi_k, R0):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = jnp.asarray(kap)
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(st0), w_presymp=W, **kw,
                                          discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return inc.sum(0)
    return jax.jit(jax.vmap(one, in_axes=(0, 0)))


def main():
    t0 = time.time()
    print(f"[theta x kappa grid] kappa={KAPPA_GRID}  theta {THETA_GRID[0]}..{THETA_GRID[-1]} "
          f"({len(THETA_GRID)} pts)  N={N_POST}  seed={SEED}")

    if not FLUA4_NPZ.exists():
        raise FileNotFoundError(f"{FLUA4_NPZ} missing - run the fluA4 chain first")
    d = np.load(FLUA4_NPZ, allow_pickle=True)
    seasons = [str(s) for s in d["seasons"]]
    print(f"  seasons: {seasons}")

    C = FS.build_fluA_setup(seasons, verbose=True)
    H = np.asarray(C["H"])

    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pi_all.shape[0], size=N_POST, replace=False)
    pi_batch = jnp.asarray(pi_all[idx])

    results = {}
    for si, s in enumerate(seasons):
        t_s = time.time()
        sh = C["shared_by_s"][s]; ngm3 = C["ngm3_by_s"][s]; st0 = C["st_by_s"][s]
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, si]))

        season_out = {}
        for kap_scalar in KAPPA_GRID:
            kap = kap_vec(kap_scalar)
            base_fn = make_baseline_fn(sh, ngm3, st0, kap)
            base15 = np.asarray(base_fn(pi_batch, R0_batch))
            base_tot = base15.sum(1)

            fn = make_sick_fn(sh, ngm3, st0, TERM, kap)
            grid_total = []
            for theta in THETA_GRID:
                pol15 = np.asarray(fn(pi_batch, R0_batch, jnp.asarray(theta)))
                tot = pol15.sum(1)
                av = 100.0 * (base_tot - tot) / np.maximum(base_tot, 1.0)
                grid_total.append(dict(theta=theta, **sm(av)))
            season_out[f"kappa_{kap_scalar}"] = grid_total
            means = [r["mean"] for r in grid_total]
            print(f"  [{s}][kappa={kap_scalar}] term averted range "
                  f"[{min(means):+.3f}, {max(means):+.3f}]%", flush=True)
        results[s] = season_out
        print(f"  [{s}] wall={time.time()-t_s:.1f}s", flush=True)

    out = dict(seasons=seasons, n_post=N_POST, seed=SEED, p_pol=P_POL,
                window="term", term_window=list(TERM),
                theta_grid=THETA_GRID, kappa_grid=KAPPA_GRID,
                results=results, wall_sec=time.time() - t0,
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_json = OUTD / "sens_theta_kappa_grid_fluA4.json"
    out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {out_json}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
