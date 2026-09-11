"""Influenza-A refit runner — standalone, writes only ``*_fluA_*`` outputs.

Same sampler configuration as ``nuts_seasonpop_v7_run.py`` (the run behind the
v5 manuscript): reparametrized coordinates, max tree depth 8, warm-up state
checkpointed so sampling can be resumed or extended without repeating warm-up.
Nothing here overwrites a v5/v6/v7 artefact.

Two modes:

``--smoke``
    A compiled probe of the joint likelihood, then a very short NUTS. Use it to
    confirm the target, the seeds and the season set behave before committing
    to a long run - it finishes in minutes.

    The probe deliberately runs no optimizer. Calling the joint loss outside a
    compiled function re-executes every season's ODE eagerly, which on the
    five-season setup grew to tens of GB and never returned. Anything that
    needs many loss evaluations must jit first - see ``fit_season_pi`` in
    ``final_pipeline_confirmed`` for the pattern (jit + value_and_grad +
    L-BFGS).

default
    The production run. On a laptop expect hours; run it detached, e.g.::

        nohup caffeinate -i .venv/bin/python scripts/nuts_fluA_run.py \\
            --warmup 1000 --samples 2000 --chains 4 > logs/fluA.log 2>&1 &

Resuming::

    python scripts/nuts_fluA_run.py --resume-from outputs/eda/nuts_fluA_warmup_state.pkl
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

os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
print(f"[early init] jax.devices() = {jax.devices()}", flush=True)
import jax.numpy as jnp  # noqa: E402
import jax.random as random  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from numpyro.infer.util import init_to_value  # noqa: E402
import arviz as az  # noqa: E402

import fluA_setup as FS  # noqa: E402
import nuts_fluA as FA  # noqa: E402
import nuts_seasonpop as NS  # noqa: E402

ED = REPO / "outputs" / "eda"
ED.mkdir(parents=True, exist_ok=True)
FLUA_DIR = REPO / "outputs" / "fluA"
FLUA_DIR.mkdir(parents=True, exist_ok=True)
MAX_TREE_DEPTH = 8
MAX_NUM_STEPS = 2 ** MAX_TREE_DEPTH - 1


def _stepstats(extra, label):
    if extra is None:
        return dict(available=False)
    steps = np.asarray(extra["num_steps"])
    sat = float((steps >= MAX_NUM_STEPS).mean() * 100)
    ndiv = int(np.asarray(extra["diverging"]).sum())
    print(
        f"  [{label}] mean_steps={steps.mean():.1f}  max={steps.max()}  "
        f"saturation={sat:.1f}%  n_divergent={ndiv}/{steps.shape[0]}",
        flush=True,
    )
    return dict(available=True, mean_num_steps=float(steps.mean()),
                max_num_steps=int(steps.max()), saturation_pct=sat,
                n_divergent=ndiv, n_draws=int(steps.shape[0]))


def run_probe(C, seasons):
    """Evaluate the joint NLL at a few parameter points. No optimizer.

    An earlier version ran a Nelder-Mead point fit here. That was a mistake:
    it called the joint loss **without jit**, so every objective evaluation
    re-executed five seasons of diffrax eagerly. Memory grew to tens of GB and
    the smoke never finished. The loss is only cheap inside a compiled
    function, which is how NumPyro uses it and how ``fit_season_pi`` in
    ``final_pipeline_confirmed`` uses it (jit + value_and_grad + L-BFGS).

    So this probe compiles the loss once and evaluates it at three points. That
    is enough to establish what a smoke needs to establish before an
    eight-hour run:

    * the whole graph traces and compiles at all, for this season count;
    * the NLL is finite at the pinned reference;
    * the NLL responds to R0 in the right direction, i.e. the data actually
      constrain the parameter rather than the likelihood being flat.

    It is not a fit and does not claim to be one. Use
    ``final_pipeline_confirmed.fit_season_pi`` if a real point estimate is
    wanted.
    """
    loss = jax.jit(FA.make_joint_loss_fluA(C))
    n_sea = len(seasons)
    logit_ref = jnp.asarray(NS.LOGIT_REF)
    phi_nb = jnp.asarray(10.0)

    def nll_at(r0):
        return float(loss(jnp.full(n_sea, np.log(r0)), logit_ref, phi_nb))

    print(f"\n[probe] compiling the joint loss ({n_sea} seasons) ...", flush=True)
    t0 = time.perf_counter()
    base = nll_at(2.0)
    t_compile = time.perf_counter() - t0
    print(f"  compile + first evaluation: {t_compile:.1f}s", flush=True)

    pts = {}
    t1 = time.perf_counter()
    for r0 in (1.6, 2.0, 2.6):
        pts[r0] = nll_at(r0)
        print(f"  NLL at R0={r0:.1f} (all seasons) = {pts[r0]:,.1f}", flush=True)
    t_eval = (time.perf_counter() - t1) / 3

    finite = all(np.isfinite(v) for v in pts.values())
    # R0 = 2.0 sits between the two flanking values, so a well-posed likelihood
    # should not be monotone across them and should vary by a wide margin.
    spread = max(pts.values()) - min(pts.values())
    responsive = spread > 1.0
    print(f"  finite: {finite}   spread across R0: {spread:,.1f}   "
          f"per-evaluation: {t_eval * 1000:.0f} ms")
    if not finite:
        raise RuntimeError("joint NLL is not finite at the pinned reference — "
                           "stop here, do not start a production run")
    if not responsive:
        print("  WARNING: the NLL barely moves with R0. The likelihood may be "
              "flat or the target may be mis-scaled.", flush=True)

    return dict(nll_by_R0={str(k): v for k, v in pts.items()},
                nll_reference=base, finite=finite, responsive=responsive,
                spread=spread, compile_sec=t_compile, eval_sec=t_eval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=None,
                    help="season labels; default fluA_setup.SEASONS_DEFAULT")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=8001)
    ap.add_argument("--tag", type=str, default="fluA")
    ap.add_argument("--smoke", action="store_true",
                    help="loss probe + 20 warmup / 20 samples x 2 chains")
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.warmup, args.samples, args.chains = 20, 20, 2
        args.tag = f"{args.tag}_smoke"

    suffix = f"_{args.tag}"
    WARMUP_PKL = ED / f"nuts{suffix}_warmup_state.pkl"
    RAW_NPZ = ED / f"nuts{suffix}_raw.npz"
    DIAG_JSON = FLUA_DIR / f"nuts{suffix}_diagnostics.json"

    seasons = args.seasons if args.seasons else list(FS.SEASONS_DEFAULT)

    print("=" * 90)
    print(f"NUTS influenza-A refit  seasons={seasons}")
    print(f"  warmup={args.warmup} samples={args.samples} chains={args.chains} "
          f"target_accept={args.target_accept} max_tree_depth={MAX_TREE_DEPTH} "
          f"seed={args.seed} tag={args.tag} smoke={args.smoke}")
    print("=" * 90, flush=True)

    t0 = time.perf_counter()
    C = FS.build_fluA_setup(seasons)
    print(f"[setup] {time.perf_counter() - t0:.1f}s", flush=True)

    probe = run_probe(C, seasons) if args.smoke else None

    model = FA.build_model_fluA(C, reparam=True, with_likelihood=True)
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.2) * jnp.ones(len(seasons)),
        logit_pi_free=jnp.asarray(NS.LOGIT_REF[:3]),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=MAX_TREE_DEPTH,
                  target_accept_prob=args.target_accept)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, chain_method="parallel", progress_bar=True)
    numpyro.set_host_device_count(max(args.chains, 1))

    warmup_extra = None
    wall_warmup = None
    if args.resume_from and Path(args.resume_from).exists():
        side = Path(args.resume_from).with_suffix(".seasons.json")
        if side.exists():
            prev = json.loads(side.read_text())
            if list(prev) != list(seasons):
                raise ValueError(
                    f"warm-up state at {args.resume_from} was tuned on "
                    f"{prev}, but this run asks for {seasons}. Delete the "
                    f"state and warm up again rather than resuming."
                )
        print(f"[RESUME] warm-up state from {args.resume_from}", flush=True)
        with open(args.resume_from, "rb") as f:
            mcmc.post_warmup_state = pickle.load(f)
    else:
        print(f"\n[WARMUP] {args.chains} chains x {args.warmup} ...", flush=True)
        t_w = time.perf_counter()
        mcmc.warmup(random.PRNGKey(args.seed),
                    extra_fields=("num_steps", "diverging"), collect_warmup=True)
        warmup_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
        wall_warmup = time.perf_counter() - t_w
        print(f"  warmup wall={wall_warmup:.1f}s = {wall_warmup / 3600:.2f}h",
              flush=True)
        with open(WARMUP_PKL, "wb") as f:
            pickle.dump(mcmc.post_warmup_state, f)
        # A warm-up state is only valid for the season set it was tuned on —
        # log_R0 has one component per season. Record it so a resume cannot
        # silently reuse a state of the wrong dimension.
        WARMUP_PKL.with_suffix(".seasons.json").write_text(json.dumps(seasons))
        print(f"[warmup state saved] {WARMUP_PKL}", flush=True)

    print(f"\n[SAMPLING] {args.chains} chains x {args.samples} ...", flush=True)
    t_s = time.perf_counter()
    mcmc.run(random.PRNGKey(args.seed + 1), extra_fields=("num_steps", "diverging"))
    sample_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    samples = {k: np.asarray(v)
               for k, v in mcmc.get_samples(group_by_chain=False).items()}
    wall_sampling = time.perf_counter() - t_s
    print(f"  sampling wall={wall_sampling:.1f}s = {wall_sampling / 3600:.2f}h",
          flush=True)

    np.savez(RAW_NPZ, log_R0=samples["log_R0"], logit_pi=samples["logit_pi"],
             log_phi_nb=samples["log_phi_nb"], pi=samples["pi"],
             phi_nb=samples["phi_nb"], seasons=np.array(seasons))
    print(f"[raw npz saved] {RAW_NPZ}", flush=True)

    warmup_stats = _stepstats(warmup_extra, "warmup")
    sampling_stats = _stepstats(sample_extra, "sampling")

    post = {k: v.reshape((args.chains, args.samples) + v.shape[1:])
            for k, v in samples.items()}
    ss = az.summary(az.from_dict({"posterior": post}), round_to=6)
    print("\n=== POSTERIOR DIAGNOSTICS ===")
    print(ss[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_string())

    identified = {}
    for i, c in enumerate(["home", "work", "school", "other"]):
        identified[f"pi_{c}"] = {
            k: float(ss.loc[f"pi[{i}]", k]) for k in ("r_hat", "ess_bulk", "ess_tail")
        }
    for j, s in enumerate(seasons):
        identified[f"R0_{s}"] = {
            k: float(ss.loc[f"log_R0[{j}]", k])
            for k in ("r_hat", "ess_bulk", "ess_tail")
        }
    identified["phi_nb"] = {
        k: float(ss.loc["phi_nb", k]) for k in ("r_hat", "ess_bulk", "ess_tail")
    }

    rhat_max = max(v["r_hat"] for v in identified.values())
    ess_bulk_min = min(v["ess_bulk"] for v in identified.values())
    ess_tail_min = min(v["ess_tail"] for v in identified.values())
    ndiv = sampling_stats.get("n_divergent")
    goal = dict(rhat_le_1_01=bool(rhat_max <= 1.01),
                ess_bulk_ge_400=bool(ess_bulk_min >= 400),
                ess_tail_ge_400=bool(ess_tail_min >= 400),
                sampling_divergence_zero=(bool(ndiv == 0)
                                          if ndiv is not None else None))
    print("\n=== GOAL CHECK ===")
    print(f"  rhat_max={rhat_max:.4f}  ess_bulk_min={ess_bulk_min:.1f}  "
          f"ess_tail_min={ess_tail_min:.1f}  divergences={ndiv}")
    if args.smoke:
        print("  (smoke run - these thresholds are not meaningful at 20 draws)")

    out = dict(
        config=dict(seasons=seasons, warmup=args.warmup, samples=args.samples,
                    chains=args.chains, target_accept=args.target_accept,
                    max_tree_depth=MAX_TREE_DEPTH, seed=args.seed, tag=args.tag,
                    smoke=args.smoke, resumed_from=args.resume_from),
        target=dict(virus_type="A",
                    a_share_coverage={s: float(C["coverage_by_s"][s])
                                      for s in seasons},
                    obs_A_over_all={
                        s: float(np.asarray(C["obs"][i])[np.asarray(C["w"][i]) > 0].sum()
                                 / max(C["obs_all"][i][np.asarray(C["w"][i]) > 0].sum(), 1))
                        for i, s in enumerate(seasons)},
                    population={s: float(np.sum(C["pop_15_by_s"][s]))
                                for s in seasons}),
        smoke_probe=probe,
        wall_warmup_sec=wall_warmup, wall_sampling_sec=wall_sampling,
        warmup_stats=warmup_stats, sampling_stats=sampling_stats,
        identified_diagnostics=identified, rhat_max=rhat_max,
        ess_bulk_min=ess_bulk_min, ess_tail_min=ess_tail_min, goal_check=goal,
        pi_mean=samples["pi"].mean(axis=0).tolist(),
        R0_mean=np.exp(samples["log_R0"]).mean(axis=0).tolist(),
        phi_nb_mean=float(samples["phi_nb"].mean()),
        raw_npz=str(RAW_NPZ),
        warmup_pkl=str(WARMUP_PKL) if wall_warmup is not None else args.resume_from,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    DIAG_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[json] {DIAG_JSON}")
    print("[DONE]")


if __name__ == "__main__":
    main()
