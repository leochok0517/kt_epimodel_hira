"""Sens 4 ported to fluA4 -- presymptomatic weight w sweep, 4 seasons.

main_v8 sections 4.5 and 5.6 claim the conclusions hold while w is varied, but
no fluA4 output backed that: only the v5 three-season, all-influenza
``outputs/eda/sens_w_sweep_seasonpop.json`` existed. This supplies the missing
one.

Design follows ``sens4_w_sweep.py`` (same 15-point grid, same N=50 draws,
forward-only, no refit) and the fluA4 porting pattern of
``sens_kappa_fluA4.py`` (fluA_setup, nuts_fluA4_raw.npz, jax.vmap over draws,
jax.jit, output to outputs/fluA/).

w enters in two places and both are handled: it weights the presymptomatic
stage in the force of infection, and it sets the NGM factor (w+2)/3 that
converts R_peak to beta. w is passed as a traced argument, so the whole grid
runs on one compilation.
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
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import nuts_seasonpop as NS  # noqa: E402
import fluA_setup as FS  # noqa: E402
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex  # noqa: E402
from kt_epimodel_hira.jax_model.erlang_presymp import (  # noqa: E402
    daily_new_onset_by_age_erlang_presymp, simulate_jax_erlang_presymp,
    split_seed_to_erlang,
)

ED = REPO / "outputs" / "eda"
OUTD = REPO / "outputs" / "fluA"
OUTD.mkdir(parents=True, exist_ok=True)
FLUA4_NPZ = ED / "nuts_fluA4_raw.npz"

BASE = float(NS.BASE)
TERM = (70.0, 113.0)
VAC = (113.0, 183.0)
P_POL = 0.4

#: identical to sens4_w_sweep.py's grid so the two are comparable
W_GRID = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.22, 0.25, 0.28, 0.31,
          0.34, 0.37, 0.40, 0.43, 0.45]
N_POST = 50
SEED = 2027   # fluA4 family convention (sens_kappa_fluA4.py); v5 used 0

CHILD_AGE = ["0-5", "6-11", "12-17"]


def sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), q05=float(np.quantile(x, 0.05)),
                q95=float(np.quantile(x, 0.95)))


def make_fn(sh, ngm3, st0, kind, win):
    """kind: 'baseline' | 'sick' | 'school'. w is traced, so one compile."""
    def one(pi_k, R0, w):
        ngm_f = (w + 2.0) / 3.0          # ngm_factor(w) for the 3-stage chain
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / ngm_f
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = (
            beta[0], beta[1], beta[2], beta[3])
        kw["phi_susc"] = jnp.asarray(NS.PHI)
        kw["kappa"] = jnp.asarray(NS.KAP)
        if kind == "baseline":
            kw["p_school"] = BASE; kw["p_work"] = BASE
        elif kind == "sick":
            kw["p_school"] = BASE; kw["p_work"] = P_POL
            kw["policy_work_start_day"], kw["policy_work_end_day"] = win
            kw["policy_school_start_day"], kw["policy_school_end_day"] = win
        else:
            kw["p_school"] = P_POL; kw["p_work"] = BASE
            kw["policy_school_start_day"], kw["policy_school_end_day"] = win
            kw["policy_work_start_day"], kw["policy_work_end_day"] = win
        kw["policy_school_baseline"] = BASE
        kw["policy_work_baseline"] = BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(st0), w_presymp=w,
                                          **kw, discretize_time=False)
        return daily_new_onset_by_age_erlang_presymp(st).sum(0)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def main():
    t0 = time.time()
    print(f"[sens4 fluA4] w grid {W_GRID[0]}..{W_GRID[-1]} ({len(W_GRID)} points), "
          f"N={N_POST}, seed={SEED}")

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

    results = {}
    for si, s in enumerate(seasons):
        t_s = time.time()
        sh = C["shared_by_s"][s]; ngm3 = C["ngm3_by_s"][s]; st0 = C["st_by_s"][s]
        pop6 = np.asarray(C["pop6_by_s"][s])
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, si]))

        base_fn = make_fn(sh, ngm3, st0, "baseline", None)
        fns = {(k, wn): make_fn(sh, ngm3, st0, k, win)
               for k in ("sick", "school")
               for wn, win in (("term", TERM), ("vac", VAC))}

        season_out = {}
        for w_val in W_GRID:
            wj = jnp.asarray(w_val)
            base15 = np.asarray(base_fn(pi_batch, R0_batch, wj))
            base6 = base15 @ H.T
            base_tot = base6.sum(1)
            row = dict(total_baseline=float(base_tot.mean()))
            for kind in ("sick", "school"):
                for wn in ("term", "vac"):
                    pol15 = np.asarray(fns[(kind, wn)](pi_batch, R0_batch, wj))
                    pol6 = pol15 @ H.T
                    av = 100.0 * (base_tot - pol6.sum(1)) / np.maximum(base_tot, 1.0)
                    d_att = (pol6 - base6) / pop6[None, :] * 100.0
                    row[f"averted_{kind}_{wn}"] = sm(av)
                    row[f"d_attack_{kind}_by_age_{wn}"] = {
                        ag: dict(**sm(d_att[:, a]),
                                 n_pos=int((d_att[:, a] > 0).sum()),
                                 n_neg=int((d_att[:, a] < 0).sum()))
                        for a, ag in enumerate(ages)}
            season_out[f"{w_val:.2f}"] = row
            print(f"  [{s}][w={w_val:.2f}] sick_term={row['averted_sick_term']['mean']:+.3f}%  "
                  f"school_term={row['averted_school_term']['mean']:+.3f}%", flush=True)
        results[s] = season_out
        print(f"  [{s}] wall={time.time()-t_s:.1f}s", flush=True)

    # sign-reversal audit: does the sign of mean sick-leave Delta-attack flip
    # anywhere across the grid, for any (season, age, window)?
    reversals = []
    for s in seasons:
        for wn in ("term", "vac"):
            for ag in ages:
                signs = []
                for w_val in W_GRID:
                    m = results[s][f"{w_val:.2f}"][f"d_attack_sick_by_age_{wn}"][ag]["mean"]
                    signs.append(0 if m == 0 else (1 if m > 0 else -1))
                nz = [x for x in signs if x != 0]
                if nz and len(set(nz)) > 1:
                    flips = [(W_GRID[i], W_GRID[i + 1])
                             for i in range(len(signs) - 1)
                             if signs[i] * signs[i + 1] < 0]
                    reversals.append(dict(season=s, window=wn, age=ag,
                                          signs=signs, flip_between=flips))

    out = dict(
        meta=dict(w_grid=W_GRID, seasons=seasons, n_post=N_POST, seed=SEED,
                  p_policy=P_POL, baseline_p=BASE,
                  term_window=list(TERM), vac_window=list(VAC),
                  main_analysis_w=float(NS.W),
                  note=("forward-only on nuts_fluA4_raw.npz, no refit; w enters "
                        "both the presymptomatic FOI weight and the NGM factor "
                        "(w+2)/3. Grid identical to v5's sens4_w_sweep.py."),
                  v5_counterpart="outputs/eda/sens_w_sweep_seasonpop.json"),
        results=results,
        sign_reversals=reversals,
        n_sign_reversals=len(reversals),
        wall_sec=time.time() - t0,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    outp = OUTD / "sens_w_sweep_fluA4.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[sign reversals] {len(reversals)} (season, window, age) combinations "
          f"flip sign across the w grid")
    for r in reversals:
        print(f"    {r['season']} {r['window']} {r['age']}: flips between {r['flip_between']}")
    print(f"[json] {outp}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
