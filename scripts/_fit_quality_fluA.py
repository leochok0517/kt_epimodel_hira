"""obs/model ratio and peak diagnostics per season at the posterior mean.

Reports, per season, the ratio of observed to modelled total and peak, and the
week each falls in. The timing columns are the ones that matter: a season whose
modelled and observed peak weeks agree but whose heights differ is a magnitude
problem, not a calendar one.

Usage:
    python scripts/_fit_quality_fluA.py [--npz PATH] [--tag fluA4] [--seasons ...]
"""
import argparse, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("scripts").resolve()))
import jax.numpy as jnp
import fluA_setup as FS, nuts_seasonpop as NS
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp, split_seed_to_erlang)
from kt_epimodel_hira.jax_model.loss_jax import simulation_to_hira_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="fluA4")
ap.add_argument("--npz", default=None, help="default outputs/eda/nuts_<tag>_raw.npz")
ap.add_argument("--seasons", nargs="*", default=None)
args = ap.parse_args()
npz = args.npz or f"outputs/eda/nuts_{args.tag}_raw.npz"

d = np.load(npz, allow_pickle=True)
seasons = args.seasons if args.seasons else (
    [str(s) for s in d["seasons"]] if "seasons" in d.files else None)
if seasons is None:
    raise SystemExit(f"{npz} carries no season list — pass --seasons")
print(f"[posterior] {npz}   seasons={seasons}")
pi = d["pi"].mean(0); R0 = np.exp(d["log_R0"]).mean(0)
C = FS.build_fluA_setup(seasons, verbose=False)
phi_j = jnp.asarray(NS.PHI); gamma_j = jnp.asarray(NS.GAMMA)
out = {}
print(f"{'season':>11} {'obs/model':>10} {'peak obs':>10} {'peak mod':>10} {'pk ratio':>9} {'wk obs':>7} {'wk mod':>7}")
for k, s in enumerate(seasons):
    beta = derive_beta_from_R0_simplex(C["ngm3_by_s"][s], jnp.asarray(R0[k]), jnp.asarray(pi), phi_j) / NS.NGM_F
    kw = dict(C["shared_by_s"][s])
    kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"] = beta[0],beta[1],beta[2],beta[3]
    kw["phi_susc"]=phi_j; kw["kappa"]=jnp.asarray(NS.KAP); kw["p_school"]=NS.BASE; kw["p_work"]=NS.BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(C["st_by_s"][s]), w_presymp=NS.W, **kw, discretize_time=False)
    pred = np.asarray(simulation_to_hira_by_age_jax(daily_new_onset_by_age_erlang_presymp(st), gamma_j, n_weeks=C["nw"]))
    obs = np.asarray(C["obs"][k]); w = np.asarray(C["w"][k])[:,0] > 0
    o1, p1 = obs.sum(1), pred.sum(1)
    r = o1[w].sum()/max(p1[w].sum(),1)
    out[s] = dict(obs_over_model=float(r), peak_obs=float(o1.max()), peak_model=float(p1.max()),
                  peak_ratio=float(o1.max()/max(p1.max(),1)), week_obs=int(o1.argmax()), week_model=int(p1.argmax()))
    print(f"{s:>11} {r:>10.3f} {o1.max():>10,.0f} {p1.max():>10,.0f} {o1.max()/p1.max():>9.3f} {o1.argmax():>7d} {p1.argmax():>7d}")
outp = Path(f"outputs/fluA/fit_quality_{args.tag}.json")
outp.write_text(json.dumps(out, indent=2))
print(f"\n[saved] {outp}")
