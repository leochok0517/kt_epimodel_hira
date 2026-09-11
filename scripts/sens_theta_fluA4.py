"""Stage A (prompts/09) -- theta sweep on the fluA4 posterior (4 seasons,
2015-16 excluded per prompts/08's finding).

No refit - forward-only reuse of outputs/eda/nuts_fluA4_raw.npz, jax.vmap
over posterior draws, jax.jit throughout (prompts/99_conventions.md).

Mirrors policy_seasonpop.py's sick/student x term/vacation scenario
definitions exactly (same P_POL, TERM/VAC windows, HIRA-6 age Delta-attack
convention) - the only new axis is theta, applied only to the "other"
channel via erlang_presymp_theta (validated by T1/T2/T3, see
tests/test_theta_equivalence.py and outputs/fluA/theta_v5_reproduction.json).

Uses the live (reciprocity-corrected) contact matrices - that is what the
fluA4 posterior was actually fit under, so no matrix injection here (unlike
theta_v5_reproduction.py, which specifically needed genuine v5 conditions).
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
FLUA4_NPZ = ED / "nuts_fluA4_raw.npz"

W = W_PRESYMP
NGM_F = ngm_factor(W)
BASE = 0.6
TERM = (70.0, 113.0)
VAC = (113.0, 183.0)
P_POL = 0.4

N_POST = 500
SEED = 2027
THETA_GRID = [round(-0.25 + 0.05 * i, 2) for i in range(21)]   # -0.25 .. 0.75

ADULT_AGE = ["18-44", "45-64"]
CHILD_AGE = ["0-5", "6-11", "12-17"]


def sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), q05=float(np.quantile(x, 0.05)),
                q50=float(np.quantile(x, 0.5)), q95=float(np.quantile(x, 0.95)))


def theta_star(theta_grid, q05_by_theta):
    g = np.asarray(theta_grid); q = np.asarray(q05_by_theta)
    for i in range(len(g) - 1):
        if q[i] < 0 <= q[i + 1]:
            t0, t1 = g[i], g[i + 1]; q0, q1 = q[i], q[i + 1]
            return float(t0 + (0 - q0) * (t1 - t0) / (q1 - q0))
    if (q > 0).all():
        return float(g[0])
    if (q < 0).all():
        return None
    return None


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
        return inc.sum(0)   # (15,)
    return jax.jit(jax.vmap(one, in_axes=(0, 0)))


def make_policy_fn(sh, ngm3, st0, kind, win):
    """kind: 'sick' (p_work=P_POL) or 'student' (p_school=P_POL)."""
    def one(pi_k, R0, theta):
        beta = derive_beta_from_R0_simplex(ngm3, R0, pi_k, jnp.asarray(NS.PHI)) / NGM_F
        kw = dict(sh)
        kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta[0], beta[1], beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(NS.PHI); kw["kappa"] = jnp.asarray(NS.KAP)
        if kind == "sick":
            kw["p_school"] = BASE; kw["p_work"] = P_POL
            kw["policy_work_start_day"], kw["policy_work_end_day"] = win
            kw["policy_school_start_day"], kw["policy_school_end_day"] = win
        else:
            kw["p_school"] = P_POL; kw["p_work"] = BASE
            kw["policy_school_start_day"], kw["policy_school_end_day"] = win
            kw["policy_work_start_day"], kw["policy_work_end_day"] = win
        kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
        st = simulate_jax_erlang_presymp_theta(split_seed_to_erlang(st0), w_presymp=W, **kw,
                                                 theta=theta, discretize_time=False)
        inc = daily_new_onset_by_age_erlang_presymp(st)
        return inc.sum(0)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def main():
    t0 = time.time()
    print(f"[Stage A] fluA4 posterior, theta grid {THETA_GRID[0]}..{THETA_GRID[-1]} "
          f"({len(THETA_GRID)} points), N={N_POST}, seed={SEED}, p={P_POL}")

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
        pop6 = np.asarray(C["pop6_by_s"][s])
        R0_batch = jnp.asarray(np.exp(log_R0_all[idx, si]))

        base_fn = make_baseline_fn(sh, ngm3, st0)
        base15 = np.asarray(base_fn(pi_batch, R0_batch))            # (N,15)
        base6 = base15 @ H.T                                        # (N,6)
        base_tot = base6.sum(1)

        season_out = {}
        for kind in ("sick", "student"):
            for win_name, win in (("term", TERM), ("vac", VAC)):
                fn = make_policy_fn(sh, ngm3, st0, kind, win)
                grid_total = []
                grid_age = {ag: [] for ag in FS.HIRA_AGE_GROUPS}
                for theta in THETA_GRID:
                    pol15 = np.asarray(fn(pi_batch, R0_batch, jnp.asarray(theta)))
                    pol6 = pol15 @ H.T
                    tot = pol6.sum(1)
                    av = 100.0 * (base_tot - tot) / np.maximum(base_tot, 1.0)
                    d_att = (pol6 - base6) / pop6[None, :] * 100.0    # (N,6) %p
                    grid_total.append(dict(theta=theta, **sm(av)))
                    for a, ag in enumerate(FS.HIRA_AGE_GROUPS):
                        grid_age[ag].append(dict(theta=theta, **sm(d_att[:, a])))
                q05s = [r["q05"] for r in grid_total]
                th_star = theta_star(THETA_GRID, q05s)
                season_out[f"{kind}_{win_name}"] = dict(
                    total_grid=grid_total, theta_star_total=th_star,
                    age_grid=grid_age,
                )
                at0 = next(r for r in grid_total if r["theta"] == 0.0)
                print(f"  [{s}][{kind}/{win_name}] theta=0 total averted: "
                      f"{at0['mean']:+.2f}% [{at0['q05']:+.2f},{at0['q95']:+.2f}]  "
                      f"theta*={th_star}", flush=True)
        results[s] = season_out
        print(f"  [{s}] wall={time.time()-t_s:.1f}s", flush=True)

    out = dict(seasons=seasons, n_post=N_POST, seed=SEED, p_pol=P_POL,
                theta_grid=THETA_GRID, term_window=list(TERM), vac_window=list(VAC),
                hira_age_groups=list(FS.HIRA_AGE_GROUPS),
                results=results, wall_sec=time.time() - t0,
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_json = OUTD / "sens_theta_fluA4.json"
    out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {out_json}")

    # ---- plot: total averted (sick x term) vs theta, all seasons; +
    #      child (0-5,6-11,12-17) term Delta-attack (sick) vs theta ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seasons)))
    for panel, (kind, win_name) in zip(axes[0], [("sick", "term"), ("sick", "vac")]):
        for s, c in zip(seasons, colors):
            g = results[s][f"{kind}_{win_name}"]["total_grid"]
            th = [r["theta"] for r in g]; mean = [r["mean"] for r in g]
            q05 = [r["q05"] for r in g]; q95 = [r["q95"] for r in g]
            panel.plot(th, mean, "-", color=c, label=s)
            panel.fill_between(th, q05, q95, color=c, alpha=0.15)
        panel.axhline(0, color="grey", ls="--", lw=1)
        panel.axvline(0, color="grey", ls=":", lw=1)
        panel.set_xlabel("theta"); panel.set_ylabel("sick-leave total averted (%)")
        panel.set_title(f"sick leave, {win_name}"); panel.legend(fontsize=7)
    for panel, win_name in zip(axes[1], ["term", "vac"]):
        for s, c in zip(seasons, colors):
            child_series = []
            for theta_i in range(len(THETA_GRID)):
                vals = [results[s]["sick_" + win_name]["age_grid"][ag][theta_i]["mean"]
                        * (0 if False else 1) for ag in CHILD_AGE]
                # population-naive average across the three child bands (display only)
                child_series.append(np.mean(vals))
            panel.plot(THETA_GRID, child_series, "-", color=c, label=s)
        panel.axhline(0, color="grey", ls="--", lw=1)
        panel.axvline(0, color="grey", ls=":", lw=1)
        panel.set_xlabel("theta"); panel.set_ylabel("mean child (0-5,6-11,12-17) Delta-attack (%p)")
        panel.set_title(f"sick leave, {win_name} - child bands"); panel.legend(fontsize=7)
    fig.tight_layout()
    fig_path = OUTD / "sens_theta_fluA4.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"[fig] {fig_path}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
