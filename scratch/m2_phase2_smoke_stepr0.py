"""Phase 2 smoke: step R(0) + γ_elder Normal(0.25, 0.07) + stepr0 inits.

Verify γ_elder stays in 0.20-0.30 range (not saturate at 0.03) before
launching production sampling.
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import json, time
import numpy as np
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, make_multi_season_loss_fn,
)
from kt_epimodel_hira.jax_model.numpyro_model import hira_model

OUTDIR = Path("/Users/hwcho/Documents/python/NIMS/kt_epimodel_hira/outputs/calibration")
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]

# Build loss closure (continuous time)
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
        initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0,
    )
    initial_states_jax.append(jnp.asarray(state0))
    nw = tgt["n_weeks"]
    obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]
    obs_hira_jax.append(jnp.asarray(obs))
    weights_hira_jax.append(jnp.asarray(w))

loss_fn = make_multi_season_loss_fn(
    initial_states=initial_states_jax,
    obs_hira_list=obs_hira_jax,
    weights_hira_list=weights_hira_jax,
    shared_static=shared_static,
    n_weeks=tgt["n_weeks"], min_rate=0.01,
    discretize_time=False,
)
model = hira_model(loss_fn, lambda_phi=0.1)

# Inits from stepr0 (skip random)
def load_stepr0(name):
    return np.array(json.load(open(OUTDIR / f"stepr0_{name}.json"))["best_vec"])

inits = []
for name in ["warm", "bio_prior", "distributed", "home_dominant"]:
    v = load_stepr0(name)
    inits.append({
        "phi": jnp.asarray(v[:14]),
        "gamma_child": float(v[14]),
        "gamma_adult": float(v[15]),
        "gamma_elder": float(v[16]),
        "beta": jnp.asarray(v[17:33]),
    })
init_params = {
    "phi": jnp.stack([d["phi"] for d in inits]),
    "gamma_child": jnp.array([d["gamma_child"] for d in inits]),
    "gamma_adult": jnp.array([d["gamma_adult"] for d in inits]),
    "gamma_elder": jnp.array([d["gamma_elder"] for d in inits]),
    "beta": jnp.stack([d["beta"] for d in inits]),
}

print("=" * 70)
print("Phase 2 smoke (step R(0) + γ_elder Normal(0.25, 0.07))")
print("=" * 70)
print(f"  init chains: warm, bio_prior, distributed, home_dominant")
print(f"  init γ_elder: {[float(d['gamma_elder']) for d in inits]}")

t0 = time.perf_counter()
kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=8)
mcmc = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=4,
            chain_method="sequential", progress_bar=False)
mcmc.run(random.PRNGKey(1), init_params=init_params,
         extra_fields=("diverging",))
wall = time.perf_counter() - t0

samples = mcmc.get_samples(group_by_chain=True)
extras = mcmc.get_extra_fields(group_by_chain=True)
ge = np.asarray(samples["gamma_elder"])
gc = np.asarray(samples["gamma_child"])
ga = np.asarray(samples["gamma_adult"])
n_div = int(np.asarray(extras["diverging"]).sum())

print(f"\n  wall: {wall:.0f}s ({wall/60:.1f}min)")
print(f"  divergences: {n_div}")
print(f"  per-chain γ_elder:")
for c in range(4):
    print(f"    chain {c} ({['warm','bio_prior','distributed','home_dominant'][c]}): "
          f"γ_elder mean={ge[c].mean():.4f}  std={ge[c].std():.4f}  "
          f"range[{ge[c].min():.3f}, {ge[c].max():.3f}]")

print(f"\n  overall γ posterior:")
print(f"    γ_child: mean={ge.mean():.3f}" if False else "")
for g, vals, name in [(gc, "child", gc.mean()), (ga, "adult", ga.mean()), (ge, "elder", ge.mean())]:
    print(f"    γ_{name if isinstance(name, str) else 'elder':>5}: mean={g.mean():.4f}  std={g.std():.4f}  "
          f"95% [{np.percentile(g, 2.5):.3f}, {np.percentile(g, 97.5):.3f}]")

# Verdict
all_in_cdc = all(0.15 <= ge[c].mean() <= 0.40 for c in range(4))
any_saturate = any(ge[c].mean() < 0.05 for c in range(4))
has_exploration = all(ge[c].std() > 1e-4 for c in range(4))

print(f"\n{'='*70}\nVERDICT\n{'='*70}")
print(f"  all chains in CDC range [0.15, 0.40]: {all_in_cdc}")
print(f"  no saturation at lower bound (>0.05): {not any_saturate}")
print(f"  chains exploring (std > 1e-4): {has_exploration}")
if all_in_cdc and not any_saturate and has_exploration:
    print(f"\n  -> SMOKE PASS: ready for production")
else:
    print(f"\n  -> SMOKE FAIL: investigate before production")
