"""M2 production NUTS sampling.

Setup:
- gamma_elder prior recalibrated to Normal(0.13, 0.05) (M1d diagnosis)
- 4 chains, mixed init from corner A (warm, bio_prior) and corner B (neutral, distributed)
- target_accept = 0.8 (fewer divergences in smoke than 0.9)
- 1000 warmup + 1000 samples per chain
- chain_method = "sequential" (CPU safe)
- continuous time (discretize_time=False)

Outputs (in outputs/calibration/):
- m2_posterior.nc       : arviz InferenceData (netcdf)
- m2_summary.json       : posterior summary (mean, sd, hdi, r_hat, ess)
- m2_meta.json          : wall_sec, divergences, rhat_max, config
- M2_DONE.flag          : sentinel for automatic completion detection

Run detached:
    nohup caffeinate -i -s uv run python scripts/m2_nuts_production.py \\
        > outputs/calibration/m2_nuts.log 2>&1 & disown
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import MCMC, NUTS
import arviz as az
import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)
SENTINEL = OUTDIR / "M2_DONE.flag"
MLFLOW_URI = "sqlite:///" + str((REPO_ROOT / "outputs" / "mlruns" / "mlflow.db").resolve())

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 4
TARGET_ACCEPT = 0.8
MAX_TREE_DEPTH = 10


def setup_loss_and_inits():
    """Build JAX loss closure + init params for 4 chains."""
    from kt_data import SUDOGWON_SIDO_CODES
    from kt_epimodel_hira.calibration.simple_model import (
        build_aggregated_inputs, estimate_initial_infected_from_hira,
        _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
    )
    from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
    from kt_epimodel_hira.model.parameters import (
        CalibrationParameters, ModelParameters, EmploymentParameters,
    )
    from kt_epimodel_hira.jax_model.loss_jax import (
        HIRA_AGE_GROUPS, make_multi_season_loss_fn,
    )

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
            initial_immunity=R0_IMMUNITY_PROFILE,   # step [.10/.30/.45/.65]
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
        n_weeks=tgt["n_weeks"],
        min_rate=0.01,
        discretize_time=False,
    )

    # Init from step-R(0) refit results (Phase 4):
    # γ_elder all in CDC band [0.18, 0.35], β/φ corner diversity preserved.
    # Drop random (γ_elder=0.09 outlier).
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

    return loss_fn, init_params


def main():
    # Clean stale sentinel
    if SENTINEL.exists():
        SENTINEL.unlink()

    print("=" * 70)
    print("M2 production NUTS sampling")
    print("=" * 70)
    print(f"  config: warmup={NUM_WARMUP}, samples={NUM_SAMPLES}, "
          f"chains={NUM_CHAINS}, target_accept={TARGET_ACCEPT}")
    print(f"  gamma_elder prior: Normal(0.13, 0.05)  [recalibrated M1d]")
    print(f"  chain_method: sequential (CPU safe)")
    print(f"  init: corner A (warm, bio_prior) + B (neutral, distributed)")

    t_setup0 = time.perf_counter()
    loss_fn, init_params = setup_loss_and_inits()
    print(f"  setup: {time.perf_counter()-t_setup0:.1f}s")

    from kt_epimodel_hira.jax_model.numpyro_model import hira_model
    model = hira_model(loss_fn, lambda_phi=0.1)

    kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT,
                  max_tree_depth=MAX_TREE_DEPTH)
    mcmc = MCMC(
        kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS, chain_method="sequential",
        progress_bar=False,
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("hira_calibration_m2_production")

    with mlflow.start_run(run_name="m2_nuts_4chains_1000w_1000s"):
        mlflow.log_params({
            "method": "NUTS",
            "num_warmup": NUM_WARMUP, "num_samples": NUM_SAMPLES,
            "num_chains": NUM_CHAINS, "target_accept": TARGET_ACCEPT,
            "max_tree_depth": MAX_TREE_DEPTH,
            "chain_method": "sequential",
            "gamma_elder_prior": "Normal(0.13, 0.05)",
            "discretize_time": False,
        })

        t0 = time.perf_counter()
        mcmc.run(random.PRNGKey(42), init_params=init_params,
                 extra_fields=("diverging",))
        wall = time.perf_counter() - t0
        print(f"\n  NUTS wall: {wall:.0f}s ({wall/60:.1f}min, {wall/3600:.2f}h)")

        # Diagnostics
        idata = az.from_numpyro(mcmc)
        summary = az.summary(idata, hdi_prob=0.95)
        rhat_max = float(summary["r_hat"].max())
        ess_min = float(summary["ess_bulk"].min())
        n_div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

        mlflow.log_metric("wall_sec", wall)
        mlflow.log_metric("n_divergences", n_div)
        mlflow.log_metric("rhat_max", rhat_max)
        mlflow.log_metric("ess_min", ess_min)

        # gamma_elder posterior (key check)
        ge = idata.posterior["gamma_elder"].values
        ge_mean = float(ge.mean()); ge_lo = float(np.quantile(ge, 0.025))
        ge_hi = float(np.quantile(ge, 0.975))
        mlflow.log_metric("gamma_elder_mean", ge_mean)
        mlflow.log_metric("gamma_elder_lo95", ge_lo)
        mlflow.log_metric("gamma_elder_hi95", ge_hi)
        print(f"  gamma_elder posterior: {ge_mean:.3f} [{ge_lo:.3f}, {ge_hi:.3f}]")
        print(f"  divergences: {n_div}")
        print(f"  r_hat max:   {rhat_max:.4f}")
        print(f"  ess min:     {ess_min:.0f}")

        # Save outputs
        nc_path = OUTDIR / "m2_posterior.nc"
        idata.to_netcdf(str(nc_path))
        mlflow.log_artifact(str(nc_path))

        summary_path = OUTDIR / "m2_summary.json"
        summary.to_json(str(summary_path))
        mlflow.log_artifact(str(summary_path))

        meta = {
            "wall_sec": wall, "wall_hours": wall / 3600,
            "n_divergences": n_div,
            "rhat_max": rhat_max, "ess_min": ess_min,
            "num_warmup": NUM_WARMUP, "num_samples": NUM_SAMPLES,
            "num_chains": NUM_CHAINS, "target_accept": TARGET_ACCEPT,
            "gamma_elder_prior": "Normal(0.13, 0.05)",
            "discretize_time": False,
            "gamma_elder_posterior": {
                "mean": ge_mean, "lo95": ge_lo, "hi95": ge_hi,
            },
            "init_strategy": "A,A,B,B (warm, bio_prior, neutral, distributed)",
            "chain_method": "sequential",
        }
        meta_path = OUTDIR / "m2_meta.json"
        json.dump(meta, open(meta_path, "w"), indent=2)
        mlflow.log_artifact(str(meta_path))

    # Sentinel
    SENTINEL.write_text(
        f"M2 DONE wall_h={wall/3600:.2f} div={n_div} "
        f"rhat={rhat_max:.4f} ess={ess_min:.0f} "
        f"gamma_elder={ge_mean:.3f}[{ge_lo:.3f},{ge_hi:.3f}]\n"
    )
    print(f"\n  Sentinel: {SENTINEL}")
    print("=" * 70)
    print("M2 DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
