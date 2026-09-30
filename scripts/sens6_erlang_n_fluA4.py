"""Sens 6 ported to fluA4 -- Erlang sub-stage count n, 4 seasons.

Answers co-author #7/#11 ("no justification for Erlang(3)", "how does it relate
to the generation interval") on the fluA4 posterior rather than the v5
three-season all-influenza one.

Design follows ``sens6_erlang_n.py``: n in {1, 2, 3, 5}, with w re-derived at
each n so the presymptomatic *fraction* stays at 10% (w = f(n-1)/(1-f)), N=50
draws, forward-only.

One anchor row is added that v5 did not have. The grid's n=3 uses
w = 0.2222 (fraction exactly 0.10), whereas the main analysis uses
w = 0.22 (fraction 0.0991). They are close but not identical, so n=3 is run at
both and the pair is reported, rather than asserting the grid row reproduces
the main analysis.

The implied mean generation interval is recorded per n: an exponential latent
period of 2.0 d plus, for stage k of n, transmission at mean k*(4/n) d, weighted
(w, 1, ..., 1). n=3 at w=0.22 gives 5.14 d.
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
from kt_epimodel_hira.jax_model.erlang_n import (  # noqa: E402
    daily_new_onset_by_age_erlang_n, ngm_factor, simulate_jax_erlang_n,
    split_seed_to_n, w_for_presymp_fraction,
)

ED = REPO / "outputs" / "eda"
OUTD = REPO / "outputs" / "fluA"
OUTD.mkdir(parents=True, exist_ok=True)
FLUA4_NPZ = ED / "nuts_fluA4_raw.npz"

BASE = float(NS.BASE)
TERM = (70.0, 113.0)
VAC = (113.0, 183.0)
P_POL = 0.4

N_STAGES_LIST = [1, 2, 3, 5]
PRESYMP_FRACTION = 0.10
N_POST = 50
SEED = 2027

LATENT_D = 2.0      # 1/sigma
INFECTIOUS_D = 4.0  # 1/gamma


def implied_generation_interval(n_stages: int, w: float) -> float:
    """Mean GI: latent + weighted mean transmission time over the n stages."""
    weights = [w] + [1.0] * (n_stages - 1)
    times = [(k + 1) * (INFECTIOUS_D / n_stages) for k in range(n_stages)]
    return LATENT_D + sum(a * b for a, b in zip(weights, times)) / sum(weights)


def sm(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), q05=float(np.quantile(x, 0.05)),
                q95=float(np.quantile(x, 0.95)))


def make_fn(sh, ngm3, st0, n_stages, kind, win):
    """n_stages is static (it sets array shapes), so this compiles per n."""
    def one(pi_k, R0, w):
        ngm_f = (w + n_stages - 1.0) / n_stages     # ngm_factor(w, n_stages)
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
        st = simulate_jax_erlang_n(split_seed_to_n(st0, n_stages),
                                    n_stages=n_stages, w_presymp=w, **kw,
                                    discretize_time=False)
        return daily_new_onset_by_age_erlang_n(st, n_stages).sum(0)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def main():
    t0 = time.time()
    w_by_n = {n: w_for_presymp_fraction(PRESYMP_FRACTION, n) for n in N_STAGES_LIST}
    # (label, n_stages, w)
    rows_spec = [(f"n={n}", n, w_by_n[n]) for n in N_STAGES_LIST]
    rows_spec.append(("n=3 (main-analysis w)", 3, float(NS.W)))

    print(f"[sens6 fluA4] n grid {N_STAGES_LIST} (+ n=3 at the main-analysis w), "
          f"N={N_POST}, seed={SEED}")
    for lab, n, w in rows_spec:
        print(f"    {lab:<24} n={n} w={w:.4f}  presymp_fraction="
              f"{(w / (w + n - 1) if n > 1 else 1.0):.4f}  "
              f"GI={implied_generation_interval(n, w):.3f} d")

    d = np.load(FLUA4_NPZ, allow_pickle=True)
    seasons = [str(s) for s in d["seasons"]]
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

        season_out = {}
        for lab, n_stages, w_val in rows_spec:
            base_fn = make_fn(sh, ngm3, st0, n_stages, "baseline", None)
            wj = jnp.asarray(w_val)
            base15 = np.asarray(base_fn(pi_batch, R0_batch, wj))
            base6 = base15 @ H.T
            base_tot = base6.sum(1)
            row = dict(n_stages=n_stages, w=w_val,
                        implied_generation_interval_d=implied_generation_interval(
                            n_stages, w_val),
                        presymp_fraction=(w_val / (w_val + n_stages - 1)
                                           if n_stages > 1 else 1.0),
                        total_baseline=float(base_tot.mean()))
            for kind in ("sick", "school"):
                for wn, win in (("term", TERM), ("vac", VAC)):
                    fn = make_fn(sh, ngm3, st0, n_stages, kind, win)
                    pol15 = np.asarray(fn(pi_batch, R0_batch, wj))
                    pol6 = pol15 @ H.T
                    av = 100.0 * (base_tot - pol6.sum(1)) / np.maximum(base_tot, 1.0)
                    d_att = (pol6 - base6) / pop6[None, :] * 100.0
                    row[f"averted_{kind}_{wn}"] = sm(av)
                    row[f"d_attack_{kind}_by_age_{wn}"] = {
                        ag: dict(**sm(d_att[:, a]),
                                 n_pos=int((d_att[:, a] > 0).sum()),
                                 n_neg=int((d_att[:, a] < 0).sum()))
                        for a, ag in enumerate(ages)}
            season_out[lab] = row
            print(f"  [{s}][{lab}] sick_term={row['averted_sick_term']['mean']:+.3f}%  "
                  f"school_term={row['averted_school_term']['mean']:+.3f}%", flush=True)
        results[s] = season_out
        print(f"  [{s}] wall={time.time()-t_s:.1f}s", flush=True)

    labels = [lab for lab, _, _ in rows_spec]
    reversals = []
    for s in seasons:
        for wn in ("term", "vac"):
            for ag in ages:
                signs = []
                for lab in labels:
                    m = results[s][lab][f"d_attack_sick_by_age_{wn}"][ag]["mean"]
                    signs.append(0 if m == 0 else (1 if m > 0 else -1))
                nz = [x for x in signs if x != 0]
                if nz and len(set(nz)) > 1:
                    reversals.append(dict(season=s, window=wn, age=ag,
                                          labels=labels, signs=signs))

    # how closely does the grid's n=3 row match the main-analysis w?
    anchor = {}
    for s in seasons:
        a = results[s]["n=3"]; b = results[s]["n=3 (main-analysis w)"]
        anchor[s] = dict(
            sick_term_grid_w=a["averted_sick_term"]["mean"],
            sick_term_main_w=b["averted_sick_term"]["mean"],
            abs_diff=abs(a["averted_sick_term"]["mean"] - b["averted_sick_term"]["mean"]))

    out = dict(
        meta=dict(n_stages_list=N_STAGES_LIST, w_by_n=w_by_n,
                  presymp_fraction_target=PRESYMP_FRACTION,
                  main_analysis_w=float(NS.W),
                  main_analysis_presymp_fraction=float(NS.W) / (float(NS.W) + 2),
                  n3_grid_equals_main_analysis=False,
                  n3_note=("the grid's n=3 uses w=0.2222 (presymptomatic fraction "
                           "exactly 0.10); the main analysis uses w=0.22 "
                           "(fraction 0.0991). Both are reported; see "
                           "n3_anchor_check for the size of the difference."),
                  generation_interval=dict(
                      latent_d=LATENT_D, infectious_d=INFECTIOUS_D,
                      formula=("latent + weighted mean of k*(infectious/n) over "
                               "stages k=1..n, weights (w, 1, ..., 1)"),
                      by_row={lab: implied_generation_interval(n, w)
                              for lab, n, w in rows_spec}),
                  seasons=seasons, n_post=N_POST, seed=SEED, p_policy=P_POL,
                  baseline_p=BASE, term_window=list(TERM), vac_window=list(VAC),
                  v5_counterpart="outputs/eda/sens_erlang_n_seasonpop.json"),
        results=results,
        n3_anchor_check=anchor,
        sign_reversals=reversals,
        n_sign_reversals=len(reversals),
        wall_sec=time.time() - t0,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    outp = OUTD / "sens_erlang_n_fluA4.json"
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[n=3 anchor] grid w=0.2222 vs main w=0.22, |diff| in sick_term averted:")
    for s, v in anchor.items():
        print(f"    {s}: {v['sick_term_grid_w']:+.3f}% vs {v['sick_term_main_w']:+.3f}%  "
              f"|diff|={v['abs_diff']:.4f}%p")
    print(f"\n[sign reversals] {len(reversals)}")
    for r in reversals:
        print(f"    {r['season']} {r['window']} {r['age']}: {dict(zip(r['labels'], r['signs']))}")
    print(f"[json] {outp}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
