"""Shared-π joint NUTS — SMOKE ONLY (wall-time measurement).

Parameter block (11-D):
  log_R0_vec (6,)       per-season size axis
  logit_pi   (4,)       shared channel mix (softmax → π = home,work,school,other)
  log_phi_nb (1,)       shared NB concentration

Priors:
  log_R0[i]  ~ Normal(log(2.0), 0.5)
  logit_pi   ~ Normal(logit_ref, σ_pin)  — same weak "pin" that L-BFGS uses,
              needed to suppress the degenerate all-work corner
              (σ_pin = [0.15, 0.10, 0.05, 0.15]).
  log_phi_nb ~ Normal(log(10.0), 1.5)
Likelihood: -make_shared_pi_joint_loss_nb (NB, 6 seasons, weighted per age).

Smoke stages:
  (1) SEQUENTIAL  2 chains × (warmup 50 + samples 50)   → JIT + step/sec
  (2) PARALLEL    4 chains × (warmup 50 + samples 50)   → speedup ratio

Full-run wall estimate: warmup 500 + samples 500 × 4 chains, based on parallel
step/sec.  ★ Full run NOT executed here. Requires separate approval.

Usage:
  XLA_FLAGS="--xla_force_host_platform_device_count=4" OMP_NUM_THREADS=2 \\
    uv run python scripts/multiseason_joint_nuts.py --smoke
"""
from __future__ import annotations
import os, sys, json, time, argparse
# Force 4 host devices BEFORE any jax import (env inheritance unreliable)
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
# ★ Force JAX platform init NOW, before any other module can flip XLA_FLAGS.
_DEVICES = jax.devices()
print(f"[early init] jax.devices() = {_DEVICES}", flush=True)

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import init_to_value
import arviz as az

# Reuse the joint-fit setup wholesale.
# (multiseason_joint_sharedpi resets XLA_FLAGS=1 on import, but JAX is already
# initialised above so the reset is harmless — new env var has no effect on a
# live JAX platform.)
sys.path.insert(0, str(Path(__file__).resolve().parent))
_SAVED_XLA = os.environ["XLA_FLAGS"]
from multiseason_joint_sharedpi import (
    build_common, SEASONS, SIGMA_PIN, PI_REF, LOGIT_REF, PHI_USHAPE,
    LOGIT_B,
)
os.environ["XLA_FLAGS"] = _SAVED_XLA  # restore for informational print
from kt_epimodel_hira.jax_model.loss_jax import make_shared_pi_joint_loss_nb


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "multiseason_joint_nuts_smoke.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

N_SEASONS = len(SEASONS)


def build_model(C):
    loss_fn = make_shared_pi_joint_loss_nb(
        initial_states=C["states"], obs_hira_list=C["obs"], weights_hira_list=C["w"],
        shared_static=C["shared"], ngm_eigval_fn=C["ngm"],
        phi_full=jnp.asarray(PHI_USHAPE), gamma_15=C["gamma_15"], n_weeks=C["nweeks"],
    )
    logit_ref_j = jnp.asarray(LOGIT_REF)
    sigma_pin_j = jnp.asarray(SIGMA_PIN)

    def model():
        # log_R0 per season — weak prior centered at ln(2.0)
        log_R0 = numpyro.sample(
            "log_R0",
            dist.Normal(jnp.log(2.0) * jnp.ones(N_SEASONS), 0.5),
        )
        # Shared logit_pi with pin prior around logit_ref (centered → suppress all-work corner)
        # Effective prior: centered_logit_pi ~ Normal(logit_ref, σ_pin)
        # We sample logit_pi and add centered-logit prior factor.
        logit_pi = numpyro.sample(
            "logit_pi",
            dist.Uniform(jnp.full(4, float(LOGIT_B[0])), jnp.full(4, float(LOGIT_B[1]))),
        )
        centered = logit_pi - jnp.mean(logit_pi)
        pin_logp = -0.5 * jnp.sum(((centered - logit_ref_j) / sigma_pin_j) ** 2)
        numpyro.factor("pin_prior", pin_logp)

        # phi_nb ~ LogNormal(ln(10), 1.5)  → range roughly 0.5–200
        log_phi_nb = numpyro.sample("log_phi_nb", dist.Normal(jnp.log(10.0), 1.5))
        phi_nb = jnp.exp(log_phi_nb)
        numpyro.deterministic("phi_nb", phi_nb)
        numpyro.deterministic("pi", jax.nn.softmax(logit_pi))

        nll = loss_fn(log_R0, logit_pi, phi_nb)
        numpyro.factor("data_loglik", -nll)

    return model


def run_smoke(model, chain_method, n_chains, warmup, samples, seed=0):
    numpyro.set_host_device_count(4)   # ensures parallel actually parallel
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.0) * jnp.ones(N_SEASONS),
        logit_pi=jnp.asarray(LOGIT_REF),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=n_chains, chain_method=chain_method, progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed))
    wall = time.perf_counter() - t0
    ss = az.summary(az.from_numpyro(mcmc), round_to=4)
    rhat_max = float(ss["r_hat"].max())
    ess_min = float(ss["ess_bulk"].min())
    ndiv = int(mcmc.get_extra_fields().get("diverging",
                                            jnp.array([0])).sum()) if hasattr(mcmc, "get_extra_fields") else 0
    try:
        ndiv = int(mcmc.get_extra_fields()["diverging"].sum())
    except Exception:
        ndiv = -1  # unknown
    samples_dict = mcmc.get_samples()
    pi_samples = np.asarray(samples_dict.get("pi"))
    log_R0_samples = np.asarray(samples_dict.get("log_R0"))
    total_iter = (warmup + samples) * n_chains
    step_per_sec = total_iter / wall if wall > 0 else float("nan")
    return dict(
        chain_method=chain_method, n_chains=n_chains,
        warmup=warmup, samples=samples,
        wall_sec=wall, rhat_max=rhat_max, ess_min=ess_min, n_div=ndiv,
        total_iter=total_iter, step_per_sec=step_per_sec,
        pi_work_mean=float(pi_samples[:, 1].mean()) if pi_samples.ndim else None,
        pi_work_q05=float(np.quantile(pi_samples[:, 1], 0.05)) if pi_samples.ndim else None,
        pi_work_q95=float(np.quantile(pi_samples[:, 1], 0.95)) if pi_samples.ndim else None,
        pi_mean=[float(pi_samples[:, k].mean()) for k in range(4)] if pi_samples.ndim else None,
        log_R0_mean=[float(log_R0_samples[:, i].mean()) for i in range(N_SEASONS)]
                        if log_R0_samples.ndim else None,
    )


def run_full(model, warmup, samples, n_chains, seed=42):
    """Full parallel NUTS with immediate raw-posterior save."""
    numpyro.set_host_device_count(4)
    init = init_to_value(values=dict(
        log_R0=jnp.log(2.0) * jnp.ones(N_SEASONS),
        logit_pi=jnp.asarray(LOGIT_REF),
        log_phi_nb=jnp.log(10.0),
    ))
    kernel = NUTS(model, init_strategy=init, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples,
                num_chains=n_chains, chain_method="parallel", progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed))
    wall = time.perf_counter() - t0

    # ★ Immediate raw save — before any post-processing.
    samples_dict = mcmc.get_samples(group_by_chain=False)
    raw_path = OUT_JSON.parent / "multiseason_joint_nuts_full_raw.json"
    raw_payload = dict(
        seasons=SEASONS, sigma_pin=SIGMA_PIN.tolist(), pi_ref=PI_REF.tolist(),
        wall_sec=wall, warmup=warmup, samples=samples, n_chains=n_chains,
        log_R0_samples=np.asarray(samples_dict["log_R0"]).tolist(),
        logit_pi_samples=np.asarray(samples_dict["logit_pi"]).tolist(),
        log_phi_nb_samples=np.asarray(samples_dict["log_phi_nb"]).tolist(),
        pi_samples=np.asarray(samples_dict["pi"]).tolist(),
        phi_nb_samples=np.asarray(samples_dict["phi_nb"]).tolist(),
    )
    raw_path.write_text(json.dumps(raw_payload))
    print(f"[raw saved] {raw_path}  wall={wall:.1f}s", flush=True)

    # Diagnostics
    try:
        ss = az.summary(az.from_numpyro(mcmc), round_to=4)
        rhat_max = float(ss["r_hat"].max())
        ess_min = float(ss["ess_bulk"].min())
        rhat_by_param = ss["r_hat"].to_dict()
        ess_by_param = ss["ess_bulk"].to_dict()
    except Exception as e:
        print(f"[diag warn] {e}", flush=True)
        rhat_max = float("nan"); ess_min = float("nan")
        rhat_by_param = {}; ess_by_param = {}
    try:
        ndiv = int(mcmc.get_extra_fields()["diverging"].sum())
    except Exception:
        ndiv = -1
    return dict(
        wall_sec=wall, warmup=warmup, samples_per_chain=samples, n_chains=n_chains,
        rhat_max=rhat_max, ess_min=ess_min, n_div=ndiv,
        rhat_by_param=rhat_by_param, ess_by_param=ess_by_param,
        posterior=samples_dict,
    )


def summarise_full(full_result):
    s = full_result["posterior"]
    pi = np.asarray(s["pi"])                                   # (N, 4)
    log_R0 = np.asarray(s["log_R0"])                            # (N, 6)
    phi_nb = np.asarray(s["phi_nb"])                            # (N,)
    def q(x, p): return float(np.quantile(x, p))
    def sm(x): return dict(mean=float(x.mean()), q025=q(x, 0.025), q05=q(x, 0.05),
                            q50=q(x, 0.5), q95=q(x, 0.95), q975=q(x, 0.975))
    return dict(
        wall_sec=full_result["wall_sec"], warmup=full_result["warmup"],
        samples_per_chain=full_result["samples_per_chain"],
        n_chains=full_result["n_chains"], n_div=full_result["n_div"],
        rhat_max=full_result["rhat_max"], ess_min=full_result["ess_min"],
        pi=[sm(pi[:, k]) for k in range(4)],
        pi_channels=["home", "work", "school", "other"],
        R0_by_season={SEASONS[i]: sm(np.exp(log_R0[:, i])) for i in range(N_SEASONS)},
        phi_nb=sm(phi_nb),
        rhat_by_param=full_result["rhat_by_param"],
        ess_by_param=full_result["ess_by_param"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run smoke.")
    ap.add_argument("--full", action="store_true", help="Run FULL parallel NUTS.")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()
    if not (args.smoke or args.full):
        print("★ neither --smoke nor --full specified.")
        sys.exit(1)

    if args.full:
        print("=" * 78, flush=True)
        print(f"MULTISEASON JOINT NUTS — FULL   warmup={args.warmup} samples={args.samples} "
              f"chains={args.chains}", flush=True)
        print(f"  seasons ({N_SEASONS}): {SEASONS}", flush=True)
        print(f"  jax devices: {jax.device_count()}   XLA_FLAGS={os.environ.get('XLA_FLAGS','')}",
              flush=True)
        print(f"  pin prior σ={SIGMA_PIN.tolist()}  ref π={PI_REF.tolist()}", flush=True)
        print("=" * 78, flush=True)
        t_build = time.perf_counter()
        C = build_common()
        print(f"[common setup] {time.perf_counter()-t_build:.1f}s", flush=True)
        model = build_model(C)
        full_result = run_full(model, args.warmup, args.samples, args.chains)
        summary = summarise_full(full_result)
        FULL_JSON = OUT_JSON.parent / "multiseason_joint_nuts_full.json"
        FULL_JSON.write_text(json.dumps(summary, indent=2, default=float))
        print(f"[summary saved] {FULL_JSON}", flush=True)
        # console short
        pi_s = summary["pi"]
        print("\n=== FULL POSTERIOR SUMMARY ===", flush=True)
        for k, ch in enumerate(summary["pi_channels"]):
            print(f"  π_{ch:6s}: mean={pi_s[k]['mean']:.4f}  "
                  f"95%CI=[{pi_s[k]['q025']:.4f},{pi_s[k]['q975']:.4f}]",
                  flush=True)
        for s, r in summary["R0_by_season"].items():
            print(f"  R0[{s}]: mean={r['mean']:.3f}  "
                  f"95%CI=[{r['q025']:.3f},{r['q975']:.3f}]", flush=True)
        print(f"  phi_nb: mean={summary['phi_nb']['mean']:.2f}  "
              f"95%CI=[{summary['phi_nb']['q025']:.2f},{summary['phi_nb']['q975']:.2f}]",
              flush=True)
        print(f"  r_hat max={summary['rhat_max']:.4f}  ess_min={summary['ess_min']:.1f}  "
              f"div={summary['n_div']}  wall={summary['wall_sec']:.1f}s",
              flush=True)
        print("[DONE FULL]", flush=True)
        return

    print("=" * 78, flush=True)
    print("MULTISEASON JOINT NUTS — SMOKE", flush=True)
    print(f"  seasons ({N_SEASONS}): {SEASONS}", flush=True)
    print(f"  params: log_R0[{N_SEASONS}] + logit_pi[4] + log_phi_nb[1] = "
          f"{N_SEASONS+5}-D", flush=True)
    print(f"  pin prior on centered logit_pi ~ N(logit_ref, σ={SIGMA_PIN.tolist()})",
          flush=True)
    print(f"  jax devices: {jax.device_count()}   XLA_FLAGS={os.environ.get('XLA_FLAGS','')}",
          flush=True)
    print("=" * 78, flush=True)

    t_build = time.perf_counter()
    C = build_common()
    print(f"[common setup] {time.perf_counter()-t_build:.1f}s\n", flush=True)
    model = build_model(C)

    results = {}
    # STAGE 1 — SEQUENTIAL 2 chains
    print(f"[STAGE 1] SEQUENTIAL  2 chains × ({args.warmup}+{args.samples})...", flush=True)
    seq = run_smoke(model, chain_method="sequential", n_chains=2,
                    warmup=args.warmup, samples=args.samples, seed=0)
    results["sequential"] = seq
    print(f"  wall={seq['wall_sec']:.1f}s  step/sec={seq['step_per_sec']:.2f}  "
          f"r_hat_max={seq['rhat_max']:.4f}  ess_min={seq['ess_min']:.1f}  div={seq['n_div']}",
          flush=True)
    print(f"  π_work mean={seq['pi_work_mean']:.4f}  90%CI=[{seq['pi_work_q05']:.4f},{seq['pi_work_q95']:.4f}]",
          flush=True)

    # STAGE 2 — PARALLEL 4 chains
    print(f"\n[STAGE 2] PARALLEL  4 chains × ({args.warmup}+{args.samples})...", flush=True)
    par = run_smoke(model, chain_method="parallel", n_chains=4,
                    warmup=args.warmup, samples=args.samples, seed=1)
    results["parallel"] = par
    print(f"  wall={par['wall_sec']:.1f}s  step/sec={par['step_per_sec']:.2f}  "
          f"r_hat_max={par['rhat_max']:.4f}  ess_min={par['ess_min']:.1f}  div={par['n_div']}",
          flush=True)
    print(f"  π_work mean={par['pi_work_mean']:.4f}  90%CI=[{par['pi_work_q05']:.4f},{par['pi_work_q95']:.4f}]",
          flush=True)

    # Full-run wall estimate
    speedup = seq["wall_sec"] / par["wall_sec"] if par["wall_sec"] > 0 else float("nan")
    full_warmup, full_samples, full_chains = 500, 500, 4
    full_total_iter = (full_warmup + full_samples) * full_chains
    full_wall_est = full_total_iter / par["step_per_sec"] if par["step_per_sec"] > 0 else float("nan")

    print("\n" + "=" * 78, flush=True)
    print(f"SMOKE SUMMARY", flush=True)
    print(f"  sequential wall {seq['wall_sec']:.1f}s   parallel wall {par['wall_sec']:.1f}s   "
          f"speedup {speedup:.2f}×", flush=True)
    print(f"  FULL estimate ({full_warmup} warmup + {full_samples} samples × {full_chains} chains "
          f"= {full_total_iter} iter):", flush=True)
    print(f"    predicted wall @ parallel step/sec {par['step_per_sec']:.2f} → "
          f"{full_wall_est:.1f}s = {full_wall_est/60:.1f} min = {full_wall_est/3600:.2f} h",
          flush=True)
    if full_wall_est < 3600:      band = "<1h : direct background OK"
    elif full_wall_est < 14400:   band = "1-4h : nohup+caffeinate+ntfy+watchdog"
    else:                          band = ">4h  : split seasons or reduce samples"
    print(f"    band: {band}", flush=True)
    print(f"  divergence total: seq={seq['n_div']}  par={par['n_div']}", flush=True)
    print(f"  π_work posterior (parallel smoke): mean={par['pi_work_mean']:.4f}  "
          f"90%CI=[{par['pi_work_q05']:.4f},{par['pi_work_q95']:.4f}]  "
          f"(≥0.05→corner suppressed?  {'yes' if par['pi_work_mean']>0.05 else 'CHECK'})",
          flush=True)
    print("=" * 78, flush=True)

    payload = dict(
        seasons=SEASONS, sigma_pin=SIGMA_PIN.tolist(), pi_ref=PI_REF.tolist(),
        smoke=results,
        full_estimate=dict(
            warmup=full_warmup, samples=full_samples, chains=full_chains,
            total_iter=full_total_iter,
            predicted_wall_sec=full_wall_est,
            band=band, speedup=speedup,
        ),
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[json] {OUT_JSON}", flush=True)
    print("[DONE — smoke only, full run pending approval]", flush=True)


if __name__ == "__main__":
    main()
