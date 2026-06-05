"""M1d short NUTS smoke test on toy budget.

Verifies:
- numpyro model defined correctly (single eval works)
- short NUTS runs without error
- target_accept 0.8 vs 0.9 comparison
- chain init A/B mixing observation
- estimate full-run wall time
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import json
import time
import numpy as np
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density

numpyro.set_host_device_count(4)

# Local
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters, EmploymentParameters,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, make_multi_season_loss_fn,
)
from kt_epimodel_hira.jax_model.numpyro_model import hira_model

OUTDIR = Path("/Users/hwcho/Documents/python/NIMS/kt_epimodel_hira/outputs/calibration")
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]

# ====================================================================
# Setup: build season data + JAX loss closure (continuous mode)
# ====================================================================
print("=" * 70); print("Setup"); print("=" * 70)
t0 = time.perf_counter()

inputs = build_aggregated_inputs()
pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
matrices = inputs["matrices"]; mobility = inputs["mobility"]

disease = ModelParameters().disease
vax = ModelParameters().vaccination
policy = ModelParameters().policy

shared_static = dict(
    C_home=jnp.asarray(matrices["C_home"]),
    C_school=jnp.asarray(matrices["C_school"]),
    C_work=jnp.asarray(matrices["C_work"]),
    C_other=jnp.asarray(matrices["C_other"]),
    M_home=jnp.asarray(mobility["home"]),
    M_school=jnp.asarray(mobility["school"]),
    M_work=jnp.asarray(mobility["work"]),
    M_other=jnp.asarray(mobility["other"]),
    pop_15=jnp.asarray(pop_15),
    rho=jnp.asarray(rho_emp),
    kappa=jnp.asarray(disease.kappa_array),
    sigma=disease.sigma, gamma=disease.gamma,
    p_school=policy.p_school, p_work=policy.p_work,
    VE=vax.VE,
    annual_coverage=jnp.asarray(vax.annual_coverage),
    vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
    seasonality_amp=disease.seasonality_amp,
    seasonality_base=disease.seasonality_base,
    seasonality_peak_day=disease.seasonality_peak_day,
    seasonality_period=disease.seasonality_period,
)

initial_states_jax = []
obs_hira_jax = []
weights_hira_jax = []
for s in SEASONS:
    tgt = load_hira_target_by_age(
        s, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    seed = estimate_initial_infected_from_hira(
        s, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = _build_initial_state_with_age_seed(
        pop_15, seed, seed_e_factor=0.5,
        initial_immunity=0.3, initial_vaccinated_fraction=0.0,
    )
    initial_states_jax.append(jnp.asarray(state0))
    nw = tgt["n_weeks"]
    obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]
    obs_hira_jax.append(jnp.asarray(obs))
    weights_hira_jax.append(jnp.asarray(w))

n_weeks = tgt["n_weeks"]
loss_cont = make_multi_season_loss_fn(
    initial_states=initial_states_jax,
    obs_hira_list=obs_hira_jax,
    weights_hira_list=weights_hira_jax,
    shared_static=shared_static,
    n_weeks=n_weeks, min_rate=0.01,
    discretize_time=False,
)
print(f"  setup wall: {time.perf_counter()-t0:.1f}s")

# ====================================================================
# Phase 1: verify model evaluates
# ====================================================================
print("\n" + "=" * 70)
print("Phase 1: numpyro model single evaluation")
print("=" * 70)

model = hira_model(loss_cont, lambda_phi=0.1)

# Load multistart inits
def load_vec(name):
    return np.array(json.load(open(OUTDIR / f"multistart_{name}.json"))["best_vec"])

corner_A1 = load_vec("warm")        # γ_elder=0.24
corner_A2 = load_vec("bio_prior")   # γ_elder=0.21
corner_B1 = load_vec("neutral")     # γ_elder=0.05
corner_B2 = load_vec("distributed") # γ_elder=0.08

# Per-chain init values (4 chains: A, A, B, B)
def to_init(vec):
    return {
        "phi": jnp.asarray(vec[:14]),
        "gamma_child": float(vec[14]),
        "gamma_adult": float(vec[15]),
        "gamma_elder": float(vec[16]),
        "beta": jnp.asarray(vec[17:33]),
    }

# Stack into (n_chains, *param_shape) for init_params
inits = [to_init(v) for v in [corner_A1, corner_A2, corner_B1, corner_B2]]
init_params = {
    "phi": jnp.stack([d["phi"] for d in inits]),
    "gamma_child": jnp.array([d["gamma_child"] for d in inits]),
    "gamma_adult": jnp.array([d["gamma_adult"] for d in inits]),
    "gamma_elder": jnp.array([d["gamma_elder"] for d in inits]),
    "beta": jnp.stack([d["beta"] for d in inits]),
}

# Single eval — verify model evaluates and log-density is finite at corner A vec
ld, _ = log_density(model, (), {}, inits[0])
print(f"  log_density at corner A1 (warm): {float(ld):.2f}")
print(f"  log_density at corner B1 (neutral): "
      f"{float(log_density(model, (), {}, inits[2])[0]):.2f}")
print(f"  model evaluates: OK")

# ====================================================================
# Phase 3: Short NUTS at target_accept=0.8
# ====================================================================
print("\n" + "=" * 70)
print("Phase 3: NUTS smoke (warmup=50, samples=50, 4 chains)")
print("=" * 70)

# Use smaller budget than spec (200+200×4) for smoke; we want quick verdict
N_WARMUP = 50
N_SAMPLES = 50
N_CHAINS = 4

results = {}
for ta in [0.8, 0.9]:
    print(f"\n--- target_accept = {ta} ---")
    kernel = NUTS(model, target_accept_prob=ta, max_tree_depth=8)
    mcmc = MCMC(
        kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES,
        num_chains=N_CHAINS, chain_method="sequential",
        progress_bar=False,
    )
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(0), init_params=init_params, extra_fields=("diverging",))
    wall = time.perf_counter() - t0
    samples = mcmc.get_samples(group_by_chain=True)
    extras = mcmc.get_extra_fields(group_by_chain=True)

    ge = np.asarray(samples["gamma_elder"])
    n_div = int(np.asarray(extras["diverging"]).sum())
    print(f"  wall: {wall:.1f}s")
    print(f"  divergences (total): {n_div}")
    print(f"  gamma_elder per chain (init A=[0,1], B=[2,3]):")
    for c in range(N_CHAINS):
        tag = "A" if c < 2 else "B"
        print(f"    chain {c} init={tag}: mean={ge[c].mean():.4f}  "
              f"std={ge[c].std():.4f}  range[{ge[c].min():.3f}, {ge[c].max():.3f}]")

    # r_hat-like check (between-chain variance vs within)
    chain_means = ge.mean(axis=1)
    overall_mean = chain_means.mean()
    between = np.var(chain_means)
    within = ge.var(axis=1).mean()
    rhat_approx = np.sqrt((within + between) / max(within, 1e-12)) if within > 0 else float("nan")
    print(f"  chain means: {chain_means.round(4).tolist()}")
    print(f"  chain stds:  {ge.std(axis=1).round(4).tolist()}")
    print(f"  approx r_hat: {rhat_approx:.3f}  (1.0 = perfect mixing)")

    results[ta] = {
        "wall": wall, "n_div": n_div,
        "ge_means": chain_means.tolist(),
        "rhat": rhat_approx,
    }

# ====================================================================
# Phase 4: full-run time estimate
# ====================================================================
print("\n" + "=" * 70)
print("Phase 4: full-run time estimate")
print("=" * 70)
# Smoke: 50 warmup + 50 samples = 100 iters per chain * 4 chains = 400 NUTS iters
# Full: 1000 + 1000 = 2000 per chain * 4 = 8000 NUTS iters (20x)
# Warmup scales differently (more adaptation) — rough 20x estimate
ta_best = 0.8 if results[0.8]["wall"] < results[0.9]["wall"] else 0.9
wall_smoke = results[ta_best]["wall"]
print(f"  smoke wall (best): {wall_smoke:.0f}s @ target_accept={ta_best}")
print(f"  estimate full (1000+1000 x 4 chains, ~20x smoke): {wall_smoke*20/60:.0f} min "
      f"({wall_smoke*20/3600:.1f}h)")

# ====================================================================
# Verdict
# ====================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

# Mixing: chains starting from A vs B should converge to similar γ_elder mean
# if NUTS mixes. Otherwise multimodal trapped.
def assess_mixing(ge_means):
    a_mean = np.mean(ge_means[:2])
    b_mean = np.mean(ge_means[2:])
    return abs(a_mean - b_mean), a_mean, b_mean

for ta, r in results.items():
    gap, a, b = assess_mixing(r["ge_means"])
    mix_status = "mixed" if gap < 0.03 else ("separated" if gap > 0.10 else "partial")
    print(f"  target_accept={ta}: A_mean={a:.3f}  B_mean={b:.3f}  "
          f"gap={gap:.4f}  -> {mix_status}  ({r['n_div']} divergences)")

# Pick the better config
best = min(results.items(), key=lambda kv: (kv[1]["rhat"], kv[1]["n_div"]))
print(f"\n  Best config: target_accept={best[0]} (r_hat={best[1]['rhat']:.3f}, "
      f"divergences={best[1]['n_div']})")
print(f"\n  Model verification: PASS (single eval, NUTS runs without error)")
