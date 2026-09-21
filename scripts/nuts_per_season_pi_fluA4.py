"""Per-season independent channel allocation (pi) fits — co-author question #18.

The manuscript claims "a single shared channel allocation reproduced all four
seasons at negligible cost in fit" but carries no number behind it. This fits
each season on its own, so the per-season pi can be compared against the shared
one and the cost in fit can be quantified.

Nothing existing is modified. ``nuts_fluA.build_model_fluA`` takes its season
list from ``C['seasons']``, so a setup dict sliced down to one season is a
single-season fit with no change to the model or the loss. Fixed vectors come
from ``nuts_seasonpop`` via ``nuts_fluA``, never restated here.

Sampler configuration is identical to ``nuts_fluA_run.py`` (the 4-season joint
fit): 1000 warm-up + 2000 samples x 4 chains, target accept 0.95, max tree
depth 8, seed 8001, reparametrized coordinates, same init values.

**Five** seasons are fitted, including 2015-16, which the main fit excludes:
whether a 2015-16-specific pi reproduces that season's peak is the second
question this run answers.

Each season is fitted sequentially and its raw draws are written as soon as it
finishes, so an interrupted run resumes instead of restarting (seasons whose
npz already exists are skipped unless ``--force``).

Usage::

    # smoke: one season, tiny, writes *_smoke_* artefacts only
    python scripts/nuts_per_season_pi_fluA4.py --smoke

    # production (long — run detached)
    nohup caffeinate -i .venv/bin/python scripts/nuts_per_season_pi_fluA4.py \\
        > logs/per_season_pi.log 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Mirrors nuts_fluA_run.py exactly: 4 host devices for 4 parallel chains, CPU.
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

#: All five seasons with a population, including 2015-16 (excluded from the
#: main fit — see the module docstring).
SEASONS_ALL = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020"]

PER_SEASON_KEYS = ("a_share_by_s", "coverage_by_s", "ngm3_by_s", "st_by_s",
                    "shared_by_s", "pop_15_by_s", "pop6_by_s")
POSITIONAL_KEYS = ("obs", "w", "obs_all")


def slice_setup(C: dict, season: str) -> dict:
    """Shallow copy of ``C`` carrying one season only.

    ``C['seasons']`` is the list the model and the loss iterate over, ``obs`` /
    ``w`` / ``obs_all`` are positional lists aligned to it, and the ``*_by_s``
    entries are dicts keyed by the season string itself (verified against
    ``fluA_setup.build_fluA_setup``). ``nw``, ``H`` and ``default_setup`` are
    season-independent and are passed through unchanged.
    """
    if season not in C["seasons"]:
        raise KeyError(f"{season} not in setup seasons {C['seasons']}")
    i = C["seasons"].index(season)
    C_s = dict(C)
    C_s["seasons"] = [season]
    for k in POSITIONAL_KEYS:
        C_s[k] = [C[k][i]]
    for k in PER_SEASON_KEYS:
        C_s[k] = {season: C[k][season]}
    return C_s


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


def fit_one_season(C_s: dict, season: str, args, suffix: str) -> dict:
    """NUTS on a single-season setup. Returns the diagnostics dict."""
    raw_npz = ED / f"nuts_per_season{suffix}_{season}_raw.npz"

    print("\n" + "=" * 90)
    print(f"SEASON {season}   warmup={args.warmup} samples={args.samples} "
          f"chains={args.chains} target_accept={args.target_accept} "
          f"seed={args.seed}")
    print("=" * 90, flush=True)

    model = FA.build_model_fluA(C_s, reparam=True, with_likelihood=True)
    # Same init as nuts_fluA_run.py, with log_R0 of length 1.
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.2) * jnp.ones(1),
        logit_pi_free=jnp.asarray(NS.LOGIT_REF[:3]),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=MAX_TREE_DEPTH,
                  target_accept_prob=args.target_accept)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, chain_method=args.chain_method,
                progress_bar=False)

    t_w = time.perf_counter()
    mcmc.warmup(random.PRNGKey(args.seed),
                extra_fields=("num_steps", "diverging"), collect_warmup=True)
    warmup_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    wall_warmup = time.perf_counter() - t_w
    print(f"  warmup wall={wall_warmup:.1f}s = {wall_warmup / 3600:.2f}h", flush=True)

    t_s = time.perf_counter()
    mcmc.run(random.PRNGKey(args.seed + 1),
             extra_fields=("num_steps", "diverging"))
    sample_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    samples = {k: np.asarray(v)
               for k, v in mcmc.get_samples(group_by_chain=False).items()}
    wall_sampling = time.perf_counter() - t_s
    print(f"  sampling wall={wall_sampling:.1f}s = {wall_sampling / 3600:.2f}h",
          flush=True)

    np.savez(raw_npz, log_R0=samples["log_R0"], logit_pi=samples["logit_pi"],
             log_phi_nb=samples["log_phi_nb"], pi=samples["pi"],
             phi_nb=samples["phi_nb"], seasons=np.array([season]))
    print(f"  [raw npz saved] {raw_npz}", flush=True)

    warmup_stats = _stepstats(warmup_extra, "warmup")
    sampling_stats = _stepstats(sample_extra, "sampling")

    post = {k: v.reshape((args.chains, args.samples) + v.shape[1:])
            for k, v in samples.items()}
    ss = az.summary(az.from_dict({"posterior": post}), round_to=6)
    print(ss[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_string())

    identified = {}
    for i, c in enumerate(["home", "work", "school", "other"]):
        identified[f"pi_{c}"] = {
            k: float(ss.loc[f"pi[{i}]", k]) for k in ("r_hat", "ess_bulk", "ess_tail")
        }
    identified[f"R_peak_{season}"] = {
        k: float(ss.loc["log_R0[0]", k]) for k in ("r_hat", "ess_bulk", "ess_tail")
    }
    identified["phi_nb"] = {
        k: float(ss.loc["phi_nb", k]) for k in ("r_hat", "ess_bulk", "ess_tail")
    }
    rhat_max = max(v["r_hat"] for v in identified.values())
    ess_bulk_min = min(v["ess_bulk"] for v in identified.values())
    ess_tail_min = min(v["ess_tail"] for v in identified.values())
    ndiv = sampling_stats.get("n_divergent")
    print(f"  [goal] rhat_max={rhat_max:.4f} ess_bulk_min={ess_bulk_min:.1f} "
          f"ess_tail_min={ess_tail_min:.1f} divergences={ndiv}", flush=True)

    return dict(
        season=season,
        config=dict(warmup=args.warmup, samples=args.samples, chains=args.chains,
                    target_accept=args.target_accept,
                    max_tree_depth=MAX_TREE_DEPTH, seed=args.seed,
                    chain_method=args.chain_method, smoke=args.smoke),
        raw_npz=str(raw_npz),
        wall_warmup_sec=wall_warmup, wall_sampling_sec=wall_sampling,
        warmup_stats=warmup_stats, sampling_stats=sampling_stats,
        identified_diagnostics=identified,
        rhat_max=rhat_max, ess_bulk_min=ess_bulk_min, ess_tail_min=ess_tail_min,
        goal_check=dict(rhat_le_1_01=bool(rhat_max <= 1.01),
                         ess_bulk_ge_400=bool(ess_bulk_min >= 400),
                         ess_tail_ge_400=bool(ess_tail_min >= 400),
                         sampling_divergence_zero=(bool(ndiv == 0)
                                                   if ndiv is not None else None)),
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=None,
                    help="default: all five (2015-16 .. 2019-20)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=8001)
    ap.add_argument("--chain-method", type=str, default="parallel",
                    help="'sequential' if memory is tight (prompts/13 §5)")
    ap.add_argument("--smoke", action="store_true",
                    help="one season (2017-18), 50 warmup / 50 samples x 1 chain")
    ap.add_argument("--force", action="store_true",
                    help="refit seasons whose raw npz already exists")
    args = ap.parse_args()

    if args.smoke:
        args.warmup, args.samples, args.chains = 50, 50, 1
        if args.seasons is None:
            args.seasons = ["2017-2018"]
    suffix = "_smoke" if args.smoke else ""
    seasons = args.seasons if args.seasons else list(SEASONS_ALL)
    partial_json = FLUA_DIR / f"per_season_pi{suffix}_partial.json"

    print("=" * 90)
    print(f"PER-SEASON pi FITS  seasons={seasons}  smoke={args.smoke}")
    print("=" * 90, flush=True)

    t0 = time.perf_counter()
    # One setup for every season, then sliced per season (prompts/13 §2.2).
    C = FS.build_fluA_setup(seasons)
    print(f"[setup] {time.perf_counter() - t0:.1f}s", flush=True)

    numpyro.set_host_device_count(max(args.chains, 1))

    done = {}
    if partial_json.exists() and not args.force:
        done = json.loads(partial_json.read_text())
        print(f"[resume] {len(done)} season(s) already in {partial_json.name}: "
              f"{sorted(done)}", flush=True)

    for season in seasons:
        raw_npz = ED / f"nuts_per_season{suffix}_{season}_raw.npz"
        if season in done and raw_npz.exists() and not args.force:
            print(f"\n[skip] {season} already fitted ({raw_npz.name})", flush=True)
            continue
        C_s = slice_setup(C, season)
        res = fit_one_season(C_s, season, args, suffix)
        done[season] = res
        # Written after every season so an interruption costs one season at most.
        partial_json.write_text(json.dumps(done, indent=2, default=float))
        print(f"  [partial saved] {partial_json}  "
              f"({len(done)}/{len(seasons)} seasons)", flush=True)
        print(f"  [elapsed total] {(time.perf_counter() - t0) / 3600:.2f}h",
              flush=True)

    print(f"\n[ALL DONE] {len(done)}/{len(seasons)} seasons  "
          f"wall={(time.perf_counter() - t0) / 3600:.2f}h")
    print(f"[partial json] {partial_json}")
    print("Next: python scripts/per_season_pi_summary_fluA4.py")


if __name__ == "__main__":
    main()
