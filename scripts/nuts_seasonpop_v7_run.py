"""Phase 2 W-2 — 재모수화 좌표계(logit_pi 합0 고정) + 보정 C(방식 A) 본 실행.

스탠드얼론 러너. scripts/nuts_seasonpop.py 원본은 건드리지 않는다(v5 재현용
보존). nuts_seasonpop_v7_reparam.build_model_v7 을 그대로 쓴다.

출력은 전부 새 이름(nuts_seasonpop_v7_*)으로 — 기존 v6a/v6b/merged.npz,
v5 백업 어느 것도 덮어쓰지 않는다.

이어돌리기: warmup pkl 이 이미 있으면 --resume-from 으로 넘겨 warmup을
건너뛰고 sampling만 하거나, sampling draws 를 늘릴 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import jax
jax.config.update("jax_enable_x64", True)
_DEVS = jax.devices()
print(f"[early init] jax.devices() = {_DEVS}", flush=True)
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value
import arviz as az

import phase1_reciprocity_v2 as P1
import phase1_reciprocity_partD as PD
import nuts_seasonpop as NS
import nuts_seasonpop_v7_reparam as V7

ED = REPO / "outputs" / "eda"
MAX_TREE_DEPTH = 8
MAX_NUM_STEPS = 2 ** MAX_TREE_DEPTH - 1


def build_C_all(corrected: bool):
    C_all_orig = NS.build()
    if not corrected:
        return C_all_orig
    G, G_path = P1.load_G("pop_national_2024_01.csv", 51_313_912)
    print(f"[G] source={G_path} sum={G.sum():,}")
    C_all_A, _ = PD.build_corrected_C_all(C_all_orig, G)
    return C_all_A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--tag", type=str, default="v7reparam")
    ap.add_argument("--corrected", type=int, default=1, help="1=방식A 보정 C, 0=미보정 C")
    ap.add_argument("--resume-from", type=str, default=None,
                     help="기존 warmup pkl 경로. 주면 warmup 건너뛰고 sampling만.")
    args = ap.parse_args()

    suffix = f"_{args.tag}"
    WARMUP_PKL = ED / f"nuts_seasonpop{suffix}_warmup_state.pkl"
    RAW_NPZ = ED / f"nuts_seasonpop{suffix}_raw.npz"
    DIAG_JSON = ED / f"nuts_seasonpop{suffix}_diagnostics.json"

    print("=" * 90)
    print(f"NUTS v7 (reparam, {'corrected' if args.corrected else 'UNcorrected'} C)  "
          f"warmup={args.warmup} samples={args.samples} chains={args.chains}  "
          f"target_accept={args.target_accept}  max_tree_depth={MAX_TREE_DEPTH}  "
          f"seed={args.seed}  tag={args.tag}  resume_from={args.resume_from}")
    print("=" * 90, flush=True)

    t0 = time.perf_counter()
    C_all = build_C_all(bool(args.corrected))
    print(f"[setup] {time.perf_counter()-t0:.1f}s", flush=True)

    model = V7.build_model_v7(C_all, reparam=True, with_likelihood=True)
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.2) * jnp.ones(NS.N_SEA),
        logit_pi_free=jnp.asarray(NS.LOGIT_REF[:3]),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=MAX_TREE_DEPTH,
                  target_accept_prob=args.target_accept)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, chain_method="parallel", progress_bar=True)
    numpyro.set_host_device_count(4)

    warmup_extra = None
    if args.resume_from is not None and Path(args.resume_from).exists():
        print(f"[RESUME] loading warmup state from {args.resume_from}", flush=True)
        with open(args.resume_from, "rb") as f:
            mcmc.post_warmup_state = pickle.load(f)
        rng_sample = random.PRNGKey(args.seed + 1)
        wall_warmup = None
    else:
        print(f"\n[WARMUP] {args.chains}ch x {args.warmup} iter (fresh) ...", flush=True)
        t_w = time.perf_counter()
        mcmc.warmup(random.PRNGKey(args.seed), extra_fields=("num_steps", "diverging"),
                    collect_warmup=True)
        warmup_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
        wall_warmup = time.perf_counter() - t_w   # blocked via np.asarray above
        print(f"  warmup wall(blocked)={wall_warmup:.1f}s = {wall_warmup/3600:.2f}h", flush=True)
        with open(WARMUP_PKL, "wb") as f:
            pickle.dump(mcmc.post_warmup_state, f)
        print(f"[warmup state saved] {WARMUP_PKL}", flush=True)
        rng_sample = random.PRNGKey(args.seed + 1)

    print(f"\n[SAMPLING] {args.chains}ch x {args.samples} iter ...", flush=True)
    t_s = time.perf_counter()
    mcmc.run(rng_sample, extra_fields=("num_steps", "diverging"))
    sample_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=False).items()}
    wall_sampling = time.perf_counter() - t_s   # blocked via np.asarray above
    print(f"  sampling wall(blocked)={wall_sampling:.1f}s = {wall_sampling/3600:.2f}h", flush=True)

    np.savez(RAW_NPZ,
             log_R0=samples["log_R0"], logit_pi=samples["logit_pi"],
             log_phi_nb=samples["log_phi_nb"], pi=samples["pi"], phi_nb=samples["phi_nb"])
    print(f"[raw npz saved] {RAW_NPZ}", flush=True)

    def stepstats(extra, label):
        if extra is None:
            return dict(available=False)
        steps = np.asarray(extra["num_steps"])
        sat = float((steps >= MAX_NUM_STEPS).mean() * 100)
        ndiv = int(np.asarray(extra["diverging"]).sum())
        print(f"  [{label}] mean_steps={steps.mean():.1f}  max={steps.max()}  "
              f"saturation={sat:.1f}%  n_divergent={ndiv}/{steps.shape[0]}", flush=True)
        return dict(available=True, mean_num_steps=float(steps.mean()),
                    max_num_steps=int(steps.max()), saturation_pct=sat,
                    n_divergent=ndiv, n_draws=int(steps.shape[0]))

    warmup_stats = stepstats(warmup_extra, "warmup")
    sampling_stats = stepstats(sample_extra, "sampling")

    # diagnostics via arviz (chain axis preserved)
    def reshape(a):
        return a.reshape((args.chains, args.samples) + a.shape[1:])
    post = {k: reshape(v) for k, v in samples.items()}
    idata = az.from_dict({"posterior": post})
    ss = az.summary(idata, round_to=6)
    print("\n=== POSTERIOR DIAGNOSTICS (this run only) ===")
    print(ss[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_string())

    channels = ["home", "work", "school", "other"]
    seas = ["2016-2017", "2017-2018", "2019-2020"]

    def key(base, i):
        return f"{base}[{i}]"

    identified = {}
    for i, c in enumerate(channels):
        identified[f"pi_{c}"] = dict(r_hat=float(ss.loc[key("pi", i), "r_hat"]),
                                       ess_bulk=float(ss.loc[key("pi", i), "ess_bulk"]),
                                       ess_tail=float(ss.loc[key("pi", i), "ess_tail"]))
    for j, s in enumerate(seas):
        identified[f"R0_{s}"] = dict(r_hat=float(ss.loc[key("log_R0", j), "r_hat"]),
                                       ess_bulk=float(ss.loc[key("log_R0", j), "ess_bulk"]),
                                       ess_tail=float(ss.loc[key("log_R0", j), "ess_tail"]))
    identified["phi_nb"] = dict(r_hat=float(ss.loc["phi_nb", "r_hat"]),
                                  ess_bulk=float(ss.loc["phi_nb", "ess_bulk"]),
                                  ess_tail=float(ss.loc["phi_nb", "ess_tail"]))

    rhat_max = max(v["r_hat"] for v in identified.values())
    ess_bulk_min = min(v["ess_bulk"] for v in identified.values())
    ess_tail_min = min(v["ess_tail"] for v in identified.values())
    sampling_div = sampling_stats.get("n_divergent", None)

    goal = dict(rhat_le_1_01=bool(rhat_max <= 1.01),
                ess_bulk_ge_400=bool(ess_bulk_min >= 400),
                ess_tail_ge_400=bool(ess_tail_min >= 400),
                sampling_divergence_zero=bool(sampling_div == 0) if sampling_div is not None else None)
    print(f"\n=== GOAL CHECK ===")
    print(f"  rhat_max={rhat_max:.4f} (<=1.01? {goal['rhat_le_1_01']})")
    print(f"  ess_bulk_min={ess_bulk_min:.1f} (>=400? {goal['ess_bulk_ge_400']})")
    print(f"  ess_tail_min={ess_tail_min:.1f} (>=400? {goal['ess_tail_ge_400']})")
    print(f"  sampling_divergence={sampling_div} (==0? {goal['sampling_divergence_zero']})")

    out = dict(
        config=dict(warmup=args.warmup, samples=args.samples, chains=args.chains,
                     target_accept=args.target_accept, max_tree_depth=MAX_TREE_DEPTH,
                     seed=args.seed, tag=args.tag, corrected=bool(args.corrected),
                     resumed_from=args.resume_from),
        wall_warmup_sec=wall_warmup, wall_sampling_sec=wall_sampling,
        warmup_stats=warmup_stats, sampling_stats=sampling_stats,
        identified_diagnostics=identified, rhat_max=rhat_max,
        ess_bulk_min=ess_bulk_min, ess_tail_min=ess_tail_min,
        goal_check=goal,
        pi_mean=samples["pi"].mean(axis=0).tolist(),
        R0_mean=np.exp(samples["log_R0"]).mean(axis=0).tolist(),
        phi_nb_mean=float(samples["phi_nb"].mean()),
        raw_npz=str(RAW_NPZ), warmup_pkl=str(WARMUP_PKL) if wall_warmup is not None else args.resume_from,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    DIAG_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[json] {DIAG_JSON}")
    print("[DONE]")


if __name__ == "__main__":
    main()
