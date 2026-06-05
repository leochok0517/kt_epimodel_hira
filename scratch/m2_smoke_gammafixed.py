"""M2 smoke with γ fixed (registry) + step R(0). Verify identifiability recovered.

γ is no longer sampled — taken from gamma_registry.get_active_gamma().
Only phi(14) + beta(16) sampled. 30-dim NUTS (vs old 33-dim).

Pass criteria:
- Few divergences (vs previous 37)
- φ converges, range plausible (~0.4-2.5)
- β does NOT bound-saturate (no compensation attempt vs γ being free)
- r_hat reasonable (< 1.5 for short smoke)
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import json
import time
import numpy as np
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
import arviz as az

import mlflow

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.calibration.gamma_registry import (
    get_active_gamma, get_active_source,
)
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, make_multi_season_loss_fn,
)
from kt_epimodel_hira.jax_model.numpyro_model import hira_model

OUTDIR = Path("/Users/hwcho/Documents/python/NIMS/kt_epimodel_hira/outputs/calibration")
SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]

print("=" * 70)
print("M2 smoke (γ fixed + step R(0))")
print("=" * 70)
src = get_active_source()
print(f"  γ source: {src.key}")
print(f"    child={src.child}, adult={src.adult}, elder={src.elder}")
print(f"  R(0): step {R0_IMMUNITY_PROFILE[0]}/{R0_IMMUNITY_PROFILE[4]}/"
      f"{R0_IMMUNITY_PROFILE[10]}/{R0_IMMUNITY_PROFILE[13]}")

# Build loss closure
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

# Init params: only phi + beta (γ no longer sampled)
def load_stepr0(name):
    return np.array(json.load(open(OUTDIR / f"stepr0_{name}.json"))["best_vec"])

init_names = ["warm", "bio_prior", "distributed", "home_dominant"]
inits = []
for name in init_names:
    v = load_stepr0(name)
    inits.append({
        "phi": jnp.asarray(v[:14]),
        "beta": jnp.asarray(v[17:33]),
    })
init_params = {
    "phi": jnp.stack([d["phi"] for d in inits]),
    "beta": jnp.stack([d["beta"] for d in inits]),
}

print(f"\n  init chains: {init_names}")
print(f"  init phi[0] (children): {[float(d['phi'][0]) for d in inits]}")
print(f"  init beta[0] (β_h_2017): {[float(d['beta'][0]) for d in inits]}")

# MLflow
mlflow.set_tracking_uri("sqlite:///" + str(
    (Path(__file__).resolve().parent.parent
     / "outputs" / "mlruns" / "mlflow.db").resolve()))
mlflow.set_experiment("hira_calibration_m2_smoke_gammafixed")

with mlflow.start_run(run_name="smoke_50w_50s_4chains_target_accept_08"):
    mlflow.log_params({
        "num_warmup": 50, "num_samples": 50, "num_chains": 4,
        "target_accept": 0.8, "max_tree_depth": 8,
        "gamma_source": src.key,
        "phi_prior": "LogNormal(0,0.5) + smooth λ=0.1",
        "beta_prior": "HalfNormal(σ=1.0)",
        "R0_profile": "step [.10/.30/.45/.65]",
        "discretize_time": False,
    })

    t0 = time.perf_counter()
    kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=4,
                chain_method="sequential", progress_bar=False)
    mcmc.run(random.PRNGKey(2), init_params=init_params,
             extra_fields=("diverging",))
    wall = time.perf_counter() - t0

    samples = mcmc.get_samples(group_by_chain=True)
    extras = mcmc.get_extra_fields(group_by_chain=True)
    phi = np.asarray(samples["phi"])          # (4, 50, 14)
    beta = np.asarray(samples["beta"])        # (4, 50, 16)
    n_div = int(np.asarray(extras["diverging"]).sum())

    print(f"\n  wall: {wall:.0f}s ({wall/60:.1f}min)")
    print(f"  divergences: {n_div}/200  (previous γ-sampled smoke: 37)")
    mlflow.log_metric("wall_sec", wall)
    mlflow.log_metric("n_divergences", n_div)

    # phi diagnostics
    phi_mean = phi.mean(axis=(0, 1))
    phi_std = phi.std(axis=(0, 1))
    phi_min = phi.min(); phi_max = phi.max()
    print(f"\n  φ posterior:")
    print(f"    mean (14 ages): {phi_mean.round(3).tolist()}")
    print(f"    std  (14 ages): {phi_std.round(3).tolist()}")
    print(f"    overall range: [{phi_min:.3f}, {phi_max:.3f}]")
    mlflow.log_metric("phi_min", float(phi_min))
    mlflow.log_metric("phi_max", float(phi_max))
    mlflow.log_metric("phi_mean_overall", float(phi_mean.mean()))

    # beta diagnostics — bound 0.30 saturation check
    beta_mean = beta.mean(axis=(0, 1))
    beta_max = float(beta.max())
    n_near_bound = int((beta > 0.28).sum())
    n_total = beta.size
    print(f"\n  β posterior (★ bound saturation check):")
    print(f"    mean (16 channels): {beta_mean.round(4).tolist()}")
    print(f"    overall max: {beta_max:.4f}  (bound = 0.30)")
    print(f"    near bound (>0.28): {n_near_bound}/{n_total} "
          f"({n_near_bound/n_total*100:.1f}%)")
    mlflow.log_metric("beta_max", beta_max)
    mlflow.log_metric("beta_near_bound_frac", n_near_bound/n_total)

    # Per-chain phi/beta means (basic mixing check)
    print(f"\n  per-chain φ mean (first 3 ages):")
    for c in range(4):
        print(f"    chain {c} ({init_names[c]:>14}): "
              f"φ[0..2]={phi[c].mean(0)[:3].round(3).tolist()}  "
              f"β[0]={float(beta[c].mean(0)[0]):.4f}")

    # arviz diagnostics
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata, hdi_prob=0.95, kind="diagnostics")
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"\n  r_hat max: {rhat_max:.3f}  (< 1.5 OK for short smoke)")
    print(f"  ess_bulk min: {ess_min:.0f}")
    mlflow.log_metric("rhat_max", rhat_max)
    mlflow.log_metric("ess_min", ess_min)

# VERDICT
print(f"\n{'='*70}\nVERDICT\n{'='*70}")
checks = {
    "Divergences < 20 (vs old 37)": n_div < 20,
    "β does NOT saturate bound (<5% near 0.30)": n_near_bound / n_total < 0.05,
    "β max under 0.30": beta_max < 0.30,
    "φ range plausible [0.1, 5.0]": (phi_min > 0.1) and (phi_max < 5.0),
    "r_hat < 1.5": rhat_max < 1.5,
}
all_ok = True
for label, ok in checks.items():
    print(f"  [{'OK ' if ok else 'WARN'}] {label}")
    if not ok: all_ok = False

if all_ok:
    print(f"\n  -> PASS: γ-fixing recovered identifiability. Ready for production.")
elif n_div >= 20 or n_near_bound/n_total >= 0.05:
    print(f"\n  -> FAIL: even with γ fixed, β/φ posterior has issues.")
    print(f"           Consider prior tightening (phi σ 0.5→0.3, beta σ 1.0→0.5).")
else:
    print(f"\n  -> PARTIAL: most checks pass, review details before production.")
