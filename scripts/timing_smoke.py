"""Phase 2 실행 전 소요시간 추정 — T-2 스모크 (warmup+sampling 각 25).

주의: 본 실행 sentinel(outputs/eda/nuts_seasonpop_raw_extended.npz 등)을
절대 건드리지 않는다. 모든 출력은 outputs/eda/timing/ 아래로만 쓴다.
warmup pkl도 별도 이름(timing_smoke_warmup_state.pkl)을 쓴다.
--resume-from 은 쓰지 않는다 — 본 실행에서 warmup을 새로 돌릴 예정이므로
warmup 비용을 포함해 재야 한다.

Standalone. 저장소 핵심 코드(nuts_seasonpop.py 포함)는 건드리지 않는다 —
이 스크립트는 그 로직을 복제해 C만 주입 지점을 추가한 것이다.
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
_DEVS = jax.devices()
print(f"[early init] jax.devices() = {_DEVS}", flush=True)
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value

import phase1_reciprocity_v2 as P1
import phase1_reciprocity_partD as PD
import nuts_seasonpop as NS

ED_TIMING = REPO / "outputs" / "eda" / "timing"
ED_TIMING.mkdir(parents=True, exist_ok=True)
OUT_JSON = ED_TIMING / "timing_smoke_report.json"

PROBE_N = 2          # tiny probe run to isolate fixed (compile) cost
TARGET_N = 25         # requested smoke size
N_CHAINS = 4
MAX_TREE_DEPTH = 8
MAX_NUM_STEPS = 2 ** MAX_TREE_DEPTH - 1   # 255


def build_model_for(C_all):
    loss_fn = NS.make_joint_loss(C_all)
    logit_ref_j = jnp.asarray(NS.LOGIT_REF); sigma_pin_j = jnp.asarray(NS.SIGMA_PIN)

    def model():
        log_R0 = numpyro.sample("log_R0",
            dist.Normal(jnp.log(2.0) * jnp.ones(NS.N_SEA), 0.5))
        logit_pi = numpyro.sample("logit_pi",
            dist.Uniform(jnp.full(4, -10.0), jnp.full(4, 10.0)))
        centered = logit_pi - jnp.mean(logit_pi)
        numpyro.factor("pin_prior",
            -0.5 * jnp.sum(((centered - logit_ref_j) / sigma_pin_j) ** 2))
        log_phi_nb = numpyro.sample("log_phi_nb", dist.Normal(jnp.log(10.0), 1.5))
        phi_nb = jnp.exp(log_phi_nb)
        numpyro.deterministic("phi_nb", phi_nb)
        numpyro.deterministic("pi", jax.nn.softmax(logit_pi))
        nll = loss_fn(log_R0, logit_pi, phi_nb)
        numpyro.factor("data_loglik", -nll)
    return model


def summarize_steps(extra_fields, label):
    if "num_steps" not in extra_fields:
        return dict(available=False)
    steps = np.asarray(extra_fields["num_steps"])
    sat = float((steps >= MAX_NUM_STEPS).mean() * 100)
    ndiv = int(np.asarray(extra_fields.get("diverging", np.zeros_like(steps))).sum())
    print(f"  [{label}] mean num_steps={steps.mean():.1f}  max={steps.max()}  "
          f"tree_depth_saturation={sat:.1f}%  n_divergent={ndiv}", flush=True)
    return dict(available=True, mean_num_steps=float(steps.mean()),
                max_num_steps=int(steps.max()), saturation_pct=sat,
                n_divergent=ndiv, shape=list(steps.shape))


def run_condition(C_all, label, seed, n_warmup, n_samples):
    print(f"\n{'='*70}\n[{label}] n_warmup={n_warmup} n_samples={n_samples}\n{'='*70}",
          flush=True)
    model = build_model_for(C_all)

    init = init_to_value(values=dict(
        log_R0=jnp.log(2.2) * jnp.ones(NS.N_SEA),
        logit_pi=jnp.asarray(NS.LOGIT_REF),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=MAX_TREE_DEPTH,
                  target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=N_CHAINS, chain_method="parallel", progress_bar=True)

    numpyro.set_host_device_count(4)

    print(f"[{label}] WARMUP {N_CHAINS}ch x {n_warmup} iter (fresh, no resume) …", flush=True)
    t_compile_start = time.perf_counter()
    t_w = time.perf_counter()
    mcmc.warmup(random.PRNGKey(seed), extra_fields=("num_steps", "diverging"),
                collect_warmup=True)
    # NOTE: mcmc.warmup() can return before on-device computation actually
    # finishes (JAX async dispatch) — time.perf_counter() taken immediately
    # after only captures Python-side dispatch/compile time (a near-constant
    # ~40-70s regardless of iteration count, confirmed against nuts_v4 logs
    # showing the identical artifact). Forcing a blocking host-side read via
    # get_extra_fields()+np.asarray() before stopping the clock gives the
    # true wall time; this mirrors what the tqdm bar itself displays.
    warmup_extra = mcmc.get_extra_fields()
    warmup_extra = {k: np.asarray(v) for k, v in warmup_extra.items()}
    wall_warmup = time.perf_counter() - t_w
    print(f"  [{label}] warmup wall (blocked)={wall_warmup:.1f}s = {wall_warmup/60:.2f} min",
          flush=True)
    warmup_stats = summarize_steps(warmup_extra, f"{label}/warmup")

    pkl_path = ED_TIMING / f"timing_smoke_{label}_n{n_warmup}_warmup_state.pkl"
    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(mcmc.post_warmup_state, f)
    print(f"  [{label}] warmup state saved -> {pkl_path}", flush=True)

    print(f"[{label}] SAMPLING {N_CHAINS}ch x {n_samples} iter …", flush=True)
    t_s = time.perf_counter()
    mcmc.run(random.PRNGKey(seed + 1), extra_fields=("num_steps", "diverging"))
    sample_extra = mcmc.get_extra_fields()
    sample_extra = {k: np.asarray(v) for k, v in sample_extra.items()}
    samples = mcmc.get_samples(group_by_chain=False)
    samples = {k: np.asarray(v) for k, v in samples.items()}
    wall_sampling = time.perf_counter() - t_s   # blocked (see warmup note above)
    print(f"  [{label}] sampling wall (blocked)={wall_sampling:.1f}s = "
          f"{wall_sampling/60:.2f} min", flush=True)
    sample_stats = summarize_steps(sample_extra, f"{label}/sampling")

    total_compile_to_sampling_done = time.perf_counter() - t_compile_start

    pi_mean = samples["pi"].mean(axis=0).tolist()
    R0_mean = np.exp(samples["log_R0"]).mean(axis=0).tolist()

    return dict(
        label=label, seed=seed, n_warmup=n_warmup, n_samples=n_samples,
        warmup_wall_sec=wall_warmup, sampling_wall_sec=wall_sampling,
        warmup_stats=warmup_stats, sampling_stats=sample_stats,
        pi_mean_after_smoke=pi_mean, R0_mean_after_smoke=R0_mean,
        total_wall_sec=total_compile_to_sampling_done,
    )


def decompose(probe, target, n_probe, n_target):
    """cost(N) = fixed + rate*N, solved from 2 points (probe, target)."""
    rate = (target - probe) / (n_target - n_probe)
    fixed = probe - rate * n_probe
    return dict(fixed_sec=fixed, rate_sec_per_iter=rate,
                probe_n=n_probe, probe_wall=probe, target_n=n_target, target_wall=target)


def run_two_tier(C_all, label, seed):
    probe = run_condition(C_all, f"{label}_probe", seed, PROBE_N, PROBE_N)
    target = run_condition(C_all, f"{label}_target", seed, TARGET_N, TARGET_N)
    warmup_decomp = decompose(probe["warmup_wall_sec"], target["warmup_wall_sec"],
                                PROBE_N, TARGET_N)
    sampling_decomp = decompose(probe["sampling_wall_sec"], target["sampling_wall_sec"],
                                  PROBE_N, TARGET_N)
    print(f"\n[{label}] decomposition: warmup fixed={warmup_decomp['fixed_sec']:.1f}s "
          f"rate={warmup_decomp['rate_sec_per_iter']:.2f}s/iter | "
          f"sampling fixed={sampling_decomp['fixed_sec']:.1f}s "
          f"rate={sampling_decomp['rate_sec_per_iter']:.2f}s/iter", flush=True)
    return dict(probe=probe, target=target,
                warmup_decomposition=warmup_decomp,
                sampling_decomposition=sampling_decomp)


def main():
    t0 = time.time()
    print(f"[T-2] probe_n={PROBE_N} target_n={TARGET_N} chains={N_CHAINS} "
          f"max_tree_depth={MAX_TREE_DEPTH} (max_num_steps={MAX_NUM_STEPS})")

    before_files = set(p.name for p in (REPO / "outputs" / "eda").glob("*"))

    t_setup = time.perf_counter()
    C_all_orig = NS.build()
    print(f"[setup] uncorrected C_all built {time.perf_counter()-t_setup:.1f}s", flush=True)
    G, G_path = P1.load_G("pop_national_2024_01.csv", 51_313_912)
    t_setup2 = time.perf_counter()
    C_all_A, _ = PD.build_corrected_C_all(C_all_orig, G)
    print(f"[setup] corrected(A) C_all built {time.perf_counter()-t_setup2:.1f}s", flush=True)

    result_orig = run_two_tier(C_all_orig, "uncorrected", seed=9001)
    result_A = run_two_tier(C_all_A, "correctedA", seed=9001)

    report = dict(
        config=dict(probe_n=PROBE_N, target_n=TARGET_N, chains=N_CHAINS,
                     max_tree_depth=MAX_TREE_DEPTH, max_num_steps=MAX_NUM_STEPS,
                     resume_from=None, G_source=G_path),
        uncorrected=result_orig,
        corrected_A=result_A,
        wall_sec_total=time.time() - t0,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[json] {OUT_JSON}")

    after_files = set(p.name for p in (REPO / "outputs" / "eda").glob("*"))
    new_files = sorted(after_files - before_files)
    print(f"\n[sentinel check] new files directly under outputs/eda/ (should be EMPTY): "
          f"{new_files}")
    report["sentinel_check_new_files_in_outputs_eda"] = new_files
    OUT_JSON.write_text(json.dumps(report, indent=2, default=float))

    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
