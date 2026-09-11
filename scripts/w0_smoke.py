"""W-0 2단계 — 보정 C, warmup 100/sampling 100/4체인, 기존 샘플러 설정(0.9/8)에서
기존 좌표계 vs 합0-고정 좌표계를 나란히 비교.

본 실행 sentinel(v6a/v6b, merged.npz)과 v5 백업은 건드리지 않는다. 출력은
outputs/eda/timing/ 아래로만 쓴다.

Standalone. 저장소 핵심 코드 미수정.
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

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import jax
jax.config.update("jax_enable_x64", True)
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value
import jax.numpy as jnp

import phase1_reciprocity_v2 as P1
import phase1_reciprocity_partD as PD
import nuts_seasonpop as NS
import nuts_seasonpop_v7_reparam as V7

ED_TIMING = REPO / "outputs" / "eda" / "timing"
ED_TIMING.mkdir(parents=True, exist_ok=True)
OUT_JSON = ED_TIMING / "w0_smoke_report.json"

N_WARMUP = 100
N_SAMPLES = 100
N_CHAINS = 4
TARGET_ACCEPT = 0.9
MAX_TREE_DEPTH = 8
MAX_NUM_STEPS = 2 ** MAX_TREE_DEPTH - 1


def summarize_steps(extra_fields, label):
    steps = np.asarray(extra_fields["num_steps"])
    sat = float((steps >= MAX_NUM_STEPS).mean() * 100)
    ndiv = int(np.asarray(extra_fields.get("diverging", np.zeros_like(steps))).sum())
    print(f"  [{label}] mean_steps={steps.mean():.1f}  max={steps.max()}  "
          f"saturation={sat:.1f}%  n_divergent={ndiv}", flush=True)
    return dict(mean_num_steps=float(steps.mean()), max_num_steps=int(steps.max()),
                saturation_pct=sat, n_divergent=ndiv, n_draws=int(steps.shape[0]))


def run_condition(C_all, label, seed, reparam):
    print(f"\n{'='*70}\n[{label}] reparam={reparam}\n{'='*70}", flush=True)
    model = V7.build_model_v7(C_all, reparam=reparam, with_likelihood=True)

    init_vals = dict(log_R0=jnp.log(2.2) * jnp.ones(NS.N_SEA), log_phi_nb=jnp.log(10.0))
    if reparam:
        init_vals["logit_pi_free"] = jnp.asarray(NS.LOGIT_REF[:3])
    else:
        init_vals["logit_pi"] = jnp.asarray(NS.LOGIT_REF)
    init = init_to_value(values=init_vals)

    kernel = NUTS(model, init_strategy=init, max_tree_depth=MAX_TREE_DEPTH,
                  target_accept_prob=TARGET_ACCEPT)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
                num_chains=N_CHAINS, chain_method="parallel", progress_bar=True)
    numpyro.set_host_device_count(4)

    t0 = time.perf_counter()
    mcmc.warmup(random.PRNGKey(seed), extra_fields=("num_steps", "diverging"),
                collect_warmup=True)
    warmup_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    wall_warmup = time.perf_counter() - t0   # blocked via np.asarray above
    print(f"  [{label}] warmup wall(blocked)={wall_warmup:.1f}s", flush=True)
    warmup_stats = summarize_steps(warmup_extra, f"{label}/warmup")

    t1 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed + 1), extra_fields=("num_steps", "diverging"))
    sample_extra = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=False).items()}
    wall_sampling = time.perf_counter() - t1
    print(f"  [{label}] sampling wall(blocked)={wall_sampling:.1f}s", flush=True)
    sample_stats = summarize_steps(sample_extra, f"{label}/sampling")

    # ESS via arviz on properly chain-shaped samples
    import arviz as az
    def reshape(a):
        return a.reshape((N_CHAINS, N_SAMPLES) + a.shape[1:])
    post = {k: reshape(v) for k, v in samples.items() if k in ("pi", "log_R0", "phi_nb")}
    idata = az.from_dict({"posterior": post})
    ss = az.summary(idata, round_to=6)
    ess_bulk_min = float(ss["ess_bulk"].min())
    ess_tail_min = float(ss["ess_tail"].min())
    rhat_max = float(ss["r_hat"].max())
    print(f"  [{label}] (100-draw diagnostics, informal) rhat_max={rhat_max:.4f}  "
          f"min_ess_bulk={ess_bulk_min:.1f}  min_ess_tail={ess_tail_min:.1f}", flush=True)

    total_wall = wall_warmup + wall_sampling
    ess_per_hour = ess_bulk_min / (total_wall / 3600.0)

    return dict(
        label=label, reparam=reparam, seed=seed,
        wall_warmup=wall_warmup, wall_sampling=wall_sampling, wall_total=total_wall,
        warmup_stats=warmup_stats, sampling_stats=sample_stats,
        rhat_max_100draw=rhat_max, ess_bulk_min_100draw=ess_bulk_min,
        ess_tail_min_100draw=ess_tail_min, ess_bulk_per_hour=ess_per_hour,
        pi_mean=samples["pi"].mean(axis=0).tolist(),
    )


def main():
    t0 = time.time()
    before_files = set(p.name for p in (REPO / "outputs" / "eda").glob("*"))

    G, G_path = P1.load_G("pop_national_2024_01.csv", 51_313_912)
    C_all_orig = NS.build()
    C_all_A, _ = PD.build_corrected_C_all(C_all_orig, G)
    print(f"[setup] corrected(A) C_all ready", flush=True)

    res_orig_coord = run_condition(C_all_A, "correctedC_origcoord", seed=8001, reparam=False)
    res_reparam = run_condition(C_all_A, "correctedC_reparam", seed=8001, reparam=True)

    report = dict(config=dict(n_warmup=N_WARMUP, n_samples=N_SAMPLES, n_chains=N_CHAINS,
                                target_accept=TARGET_ACCEPT, max_tree_depth=MAX_TREE_DEPTH,
                                G_source=G_path),
                   original_coord=res_orig_coord, reparam_sumfixed=res_reparam,
                   wall_sec_total=time.time() - t0,
                   generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float))

    print("\n=== SUMMARY ===")
    for r in (res_orig_coord, res_reparam):
        print(f"{r['label']:24s}  wall={r['wall_total']:.0f}s  "
              f"warmup_sat={r['warmup_stats']['saturation_pct']:.1f}%  "
              f"sampling_sat={r['sampling_stats']['saturation_pct']:.1f}%  "
              f"sampling_div={r['sampling_stats']['n_divergent']}  "
              f"min_ess_bulk={r['ess_bulk_min_100draw']:.1f}  "
              f"ess/hr={r['ess_bulk_per_hour']:.1f}")

    after_files = set(p.name for p in (REPO / "outputs" / "eda").glob("*"))
    new_files = sorted(after_files - before_files)
    print(f"\n[sentinel check] new files under outputs/eda/ (should be empty): {new_files}")

    print(f"\n[json] {OUT_JSON}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
