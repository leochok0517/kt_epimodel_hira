"""T3 (prompts/09) -- reproduce main_v5 sec:theta with the new erlang_presymp_theta.

v5 conditions exactly: A+B target, v5 3 seasons (2016-2017, 2017-2018,
2019-2020), v5 posterior (outputs/eda/nuts_seasonpop_merged.npz), empirical
kappa, sick-leave scenario p_work=0.4 / term window, theta on the "other"
channel only (school untouched: p_school == policy_school_baseline).

No refit. Forward-only, jax.vmap over posterior draws, jax.jit throughout
(prompts/99_conventions.md - never call the simulator eagerly in a loop).

Does not touch season_pop_setup.py / nuts_seasonpop.py / any v5 output path.
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
from season_pop_setup import build_seasonwise_setup
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.jax_model.numpyro_model import (
    derive_beta_from_R0_simplex, make_ngm_eigvalue_fn,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
from kt_epimodel_hira.jax_model.erlang_presymp_theta import (
    simulate_jax_erlang_presymp_theta,
)

ED = REPO / "outputs" / "eda"
OUTD = REPO / "outputs" / "fluA"
OUTD.mkdir(parents=True, exist_ok=True)
CONTACT_DIR = REPO.parent / "kt_data" / "data" / "external" / "contact_matrices"
ORIG_TERM_NPZ = CONTACT_DIR / "empirical_matrices_15_raw_asymmetric.npz"
ORIG_VAC_NPZ = CONTACT_DIR / "empirical_matrices_15_vacation_raw_asymmetric.npz"
# outputs/eda/nuts_seasonpop_merged.npz was overwritten by this session's own
# later Phase 2 work (v6, fit under the reciprocity-corrected C) - it is no
# longer the posterior that produced v5's theta table. The genuine v5
# posterior survives only in the pre-Phase-2 backup taken before that refit.
V5_MERGED_NPZ = REPO / "outputs" / "eda_v5_uncorrected" / "nuts_seasonpop_merged.npz"


def inject_original_matrices(C_all, seasons):
    """kt_data's empirical_matrices_15*.npz are the Method-A reciprocity-
    corrected deployment (2026-08-24, this session's own Phase 2 work) - the
    default loader season_pop_setup/sens_common hit above therefore returns
    corrected matrices, not the ones that actually produced v5's posterior
    and its theta table. "v5 conditions" means genuinely uncorrected, so
    every shared dict and NGM function is rebuilt here from the backed-up
    raw_asymmetric files - by dict override, no file on disk is touched.
    Mirrors phase1_reciprocity_partD.build_corrected_C_all, direction
    reversed (undoing the correction for this comparison only).
    """
    if not ORIG_TERM_NPZ.exists() or not ORIG_VAC_NPZ.exists():
        raise FileNotFoundError(
            f"backup matrices missing ({ORIG_TERM_NPZ}, {ORIG_VAC_NPZ}) - "
            "cannot verify v5 conditions without them"
        )
    orig_term = load_contact_matrices(path=ORIG_TERM_NPZ)
    orig_vac = load_contact_matrices(path=ORIG_VAC_NPZ)

    shared_by_s = {}
    ngm3_by_s = {}
    for s in seasons:
        sh = dict(C_all["shared_by_s"][s])
        for c in ("home", "school", "work", "other"):
            sh[f"C_{c}"] = jnp.asarray(orig_term[f"C_{c}"])
            sh[f"C_{c}_vac"] = jnp.asarray(orig_vac[f"C_{c}"])
        shared_by_s[s] = sh

        pop_2d = np.asarray(sh["pop_15"]).reshape(15, 1)
        ngm3_by_s[s] = make_ngm_eigvalue_fn(
            pop_15=pop_2d, rho=np.asarray(sh["rho"]),
            C_home=orig_term["C_home"], C_work=orig_term["C_work"],
            C_school=orig_term["C_school"], C_other=orig_term["C_other"],
            R0_immunity=NS.IMM, gamma=float(sh["gamma"]),
            seasonal_factor=1.0 + NS.F.S.AMP,
        )
    print(f"[matrices] injected ORIGINAL (uncorrected) contact matrices from "
          f"{ORIG_TERM_NPZ.name} / {ORIG_VAC_NPZ.name} - overriding the live "
          f"(reciprocity-corrected) deployment for this v5-reproduction run only")
    return shared_by_s, ngm3_by_s


SEASONS = ["2016-2017", "2017-2018", "2019-2020"]
W = W_PRESYMP
NGM_F = ngm_factor(W)
BASE = 0.6
TERM = (70.0, 113.0)
P_POL = 0.4

N_POST = 500
SEED = 2027

THETA_GRID = [round(-0.5 + 0.05 * i, 2) for i in range(31)]   # -0.50 .. 1.00
V5_REF = {
    "theta_0": {"2016-2017": 0.46, "2017-2018": 0.23, "2019-2020": 0.40},
    "theta_0.75": {"2016-2017": 1.07, "2017-2018": 1.04, "2019-2020": 1.07},
    "theta_star": {"2016-2017": -0.04, "2017-2018": 0.39, "2019-2020": 0.09},
}


def sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), q05=float(np.quantile(x, 0.05)),
                q50=float(np.quantile(x, 0.5)), q95=float(np.quantile(x, 0.95)))


def make_baseline_fn(sh, ngm3, st0):
    def one(pi_k, R0):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = jnp.asarray(NS.KAP)
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(st0), w_presymp=W, **kw,
                                          discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return inc.sum(0)
    return jax.jit(jax.vmap(one, in_axes=(0, 0)))


def make_sick_fn(sh, ngm3, st0):
    def one(pi_k, R0, theta):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = jnp.asarray(NS.KAP)
        kw["p_school"] = BASE; kw["p_work"] = P_POL
        kw["policy_work_start_day"], kw["policy_work_end_day"] = TERM
        kw["policy_school_start_day"], kw["policy_school_end_day"] = TERM
        kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
        st = simulate_jax_erlang_presymp_theta(split_seed_to_erlang(st0), w_presymp=W, **kw,
                                                 theta=theta, discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return inc.sum(0)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def theta_star(theta_grid, q05_by_theta):
    """Linear-interpolated theta where the 90% CI lower bound crosses 0."""
    g = np.asarray(theta_grid); q = np.asarray(q05_by_theta)
    for i in range(len(g) - 1):
        if q[i] < 0 <= q[i + 1]:
            t0, t1 = g[i], g[i + 1]; q0, q1 = q[i], q[i + 1]
            return float(t0 + (0 - q0) * (t1 - t0) / (q1 - q0))
    if (q > 0).all():
        return float(g[0])   # already excludes 0 at the grid's lower edge
    return None   # never crosses within the grid


def main():
    t0 = time.time()
    print("[T3] v5 conditions: A+B target, 3 seasons, v5 posterior, empirical kappa, p=0.4")
    C_all = build_seasonwise_setup(imm=NS.IMM, gamma_15=NS.GAMMA, use_season_pop=True)
    shared_by_s_orig, ngm3_by_s_orig = inject_original_matrices(C_all, SEASONS)

    if not V5_MERGED_NPZ.exists():
        raise FileNotFoundError(
            f"{V5_MERGED_NPZ} missing - the true v5 posterior backup is required; "
            f"outputs/eda/nuts_seasonpop_merged.npz is NOT it (overwritten by this "
            f"session's later Phase 2 refit)"
        )
    d = np.load(V5_MERGED_NPZ)
    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pi_all.shape[0], size=N_POST, replace=False)
    pi_batch = jnp.asarray(pi_all[idx])
    print(f"[posterior] N={N_POST} seed={SEED} from {V5_MERGED_NPZ}")

    results = {}
    for si, s in enumerate(SEASONS):
        t_s = time.time()
        sh = shared_by_s_orig[s]
        ngm3 = ngm3_by_s_orig[s]
        st0 = C_all["st_by_s"][s]
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, si]))

        base_fn = make_baseline_fn(sh, ngm3, st0)
        sick_fn = make_sick_fn(sh, ngm3, st0)

        base_tot = np.asarray(base_fn(pi_batch, R0_batch)).sum(1)   # (N,) total pop

        rows = []
        for theta in THETA_GRID:
            sick_tot = np.asarray(sick_fn(pi_batch, R0_batch, jnp.asarray(theta))).sum(1)
            av = 100.0 * (base_tot - sick_tot) / np.maximum(base_tot, 1.0)
            rows.append(dict(theta=theta, **sm(av)))

        q05s = [r["q05"] for r in rows]
        th_star = theta_star(THETA_GRID, q05s)

        by_theta = {f"{r['theta']:.2f}": r for r in rows}
        at0 = by_theta["0.00"]
        at075 = by_theta["0.75"]

        results[s] = dict(
            grid=rows, theta_star=th_star,
            at_theta_0=at0, at_theta_0_75=at075,
            v5_ref_theta_0=V5_REF["theta_0"][s],
            v5_ref_theta_0_75=V5_REF["theta_0.75"][s],
            v5_ref_theta_star=V5_REF["theta_star"][s],
            match_theta_0=abs(at0["mean"] - V5_REF["theta_0"][s]),
            match_theta_0_75=abs(at075["mean"] - V5_REF["theta_0.75"][s]),
            match_theta_star=(abs(th_star - V5_REF["theta_star"][s])
                                if th_star is not None else None),
        )
        print(f"  [{s}] theta=0: {at0['mean']:+.2f}% (v5 {V5_REF['theta_0'][s]:+.2f}%)  "
              f"theta=0.75: {at075['mean']:+.2f}% (v5 {V5_REF['theta_0.75'][s]:+.2f}%)  "
              f"theta*={th_star} (v5 {V5_REF['theta_star'][s]:+.2f})  "
              f"[{time.time()-t_s:.1f}s]", flush=True)

    tol = 0.05   # percentage points - "reproduces" tolerance
    pass_theta0 = all(results[s]["match_theta_0"] <= tol for s in SEASONS)
    pass_theta075 = all(results[s]["match_theta_0_75"] <= tol for s in SEASONS)
    pass_thetastar = all(
        results[s]["match_theta_star"] is not None and results[s]["match_theta_star"] <= 0.05
        for s in SEASONS
    )
    t3_pass = pass_theta0 and pass_theta075 and pass_thetastar

    print(f"\n[T3 VERDICT] theta=0 match: {pass_theta0}  theta=0.75 match: {pass_theta075}  "
          f"theta* match: {pass_thetastar}  -> {'PASS' if t3_pass else 'FAIL'}")

    out = dict(seasons=SEASONS, n_post=N_POST, seed=SEED, theta_grid=THETA_GRID,
                results=results, tolerance_pp=tol,
                verdict=dict(theta_0=pass_theta0, theta_0_75=pass_theta075,
                              theta_star=pass_thetastar, overall=t3_pass),
                wall_sec=time.time() - t0, generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_path = OUTD / "theta_v5_reproduction.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {out_path}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
