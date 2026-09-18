"""T3 (prompts/10) -- theta x kappa=1.00 corner check on the fluA4 posterior.

sens_theta_fluA4.json only ever swept theta at the empirical kappa (NS.KAP =
[0.34]*4 + [0.40]*10 + [0]). main_v5 additionally claims "at kappa=1.00, sick
leave's total averted is negative across the whole theta grid" -- that claim
came from the v5 (3-season, A+B) fit and has not been checked against fluA4.
This script reruns exactly the sick-leave axis of sens_theta_fluA4.py at
kappa=1.00 (same ratio convention as sens1_piwork_kappa.py's kap_vec: student
0.85*k / adult k / 70+ 0), theta grid unchanged, N_POST=200 (prompt: "200 이면
충분").

No refit -- forward-only, jax.vmap + jax.jit throughout (prompts/99_conventions.md).
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
VAC = (113.0, 183.0)
P_POL = 0.4
KAPPA_SCALAR = 1.00

N_POST = 200
SEED = 2027
THETA_GRID = [round(-0.25 + 0.05 * i, 2) for i in range(21)]   # -0.25 .. 0.75


def kap_vec(k_scalar):
    """Same ratio convention as sens1_piwork_kappa.py's kap_vec."""
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


def main():
    t0 = time.time()
    kap = kap_vec(KAPPA_SCALAR)
    print(f"[T3] fluA4 posterior, kappa={KAPPA_SCALAR} (vec={kap[[0,4,14]]}), "
          f"theta grid {THETA_GRID[0]}..{THETA_GRID[-1]} ({len(THETA_GRID)} pts), "
          f"N={N_POST}, seed={SEED}")

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

    # baseline uses NS.KAP as usual (kappa only enters the sick-leave home
    # spillover term, not the no-policy baseline, but the theta scaling and
    # crowding kw both key off the same kappa arg for the *policy* fn — the
    # baseline forward sim itself doesn't take a kappa-dependent input beyond
    # the shared kw dict, so we reuse the same baseline function pattern from
    # sens_theta_fluA4.py, evaluated with kap held at kap_vec(1.0) throughout
    # for internal consistency of this kappa=1.00 corner check).
    from kt_epimodel_hira.jax_model.erlang_presymp import simulate_jax_erlang_presymp

    def make_baseline_fn(sh, ngm3, st0, kap):
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

    results = {}
    negative_points = []   # (season, theta) where sick-leave total averted is NOT negative
    for si, s in enumerate(seasons):
        t_s = time.time()
        sh = C["shared_by_s"][s]; ngm3 = C["ngm3_by_s"][s]; st0 = C["st_by_s"][s]
        pop6 = np.asarray(C["pop6_by_s"][s])
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, si]))

        base_fn = make_baseline_fn(sh, ngm3, st0, kap)
        base15 = np.asarray(base_fn(pi_batch, R0_batch))
        base6 = base15 @ H.T
        base_tot = base6.sum(1)

        season_out = {}
        for win_name, win in (("term", TERM), ("vac", VAC)):
            fn = make_sick_fn(sh, ngm3, st0, win, kap)
            grid_total = []
            for theta in THETA_GRID:
                pol15 = np.asarray(fn(pi_batch, R0_batch, jnp.asarray(theta)))
                pol6 = pol15 @ H.T
                tot = pol6.sum(1)
                av = 100.0 * (base_tot - tot) / np.maximum(base_tot, 1.0)
                row = dict(theta=theta, **sm(av))
                grid_total.append(row)
                if row["mean"] >= 0:
                    negative_points.append((s, win_name, theta, row["mean"]))
            season_out[f"sick_{win_name}"] = dict(total_grid=grid_total)
            means = [r["mean"] for r in grid_total]
            print(f"  [{s}][sick/{win_name}] mean averted range: "
                  f"[{min(means):+.3f}, {max(means):+.3f}]%  "
                  f"(all negative: {all(m < 0 for m in means)})", flush=True)
        results[s] = season_out
        print(f"  [{s}] wall={time.time()-t_s:.1f}s", flush=True)

    out = dict(seasons=seasons, n_post=N_POST, seed=SEED, p_pol=P_POL,
                kappa_scalar=KAPPA_SCALAR, kappa_vec_student_adult_elder=[
                    float(kap[0]), float(kap[4]), float(kap[14])],
                theta_grid=THETA_GRID, term_window=list(TERM), vac_window=list(VAC),
                results=results,
                non_negative_points=[dict(season=s, window=w, theta=th, mean=m)
                                       for s, w, th, m in negative_points],
                wall_sec=time.time() - t0,
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_json = OUTD / "sens_theta_kappa1_fluA4.json"
    out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {out_json}")
    if negative_points:
        print(f"\n[NOT-all-negative points] {len(negative_points)}:")
        for s, w, th, m in negative_points:
            print(f"    {s} / {w} / theta={th:+.2f}: mean={m:+.3f}%")
    else:
        print("\n[confirmed] sick-leave total averted is negative at every "
              "(season, window, theta) grid point at kappa=1.00 -- matches "
              "main_v5's claim.")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
