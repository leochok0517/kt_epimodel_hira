"""M2 production NUTS with Negative-Binomial observation noise (TODO-3).

Identical to m2_production.py except the likelihood is NB instead of Poisson.
NB adds a single global dispersion parameter ``phi_nb``; obs variance is
``μ + μ²/k`` so the posterior predictive band can widen to cover real
weekly variability (Poisson predictive coverage was only 12.2%).

200 warmup + 200 sample × 4 chains (shortened from production v2's
300+300 since the diagnostic question is coverage, not sub-percent CI).

Outputs:
- outputs/calibration/m2_prod_nb_posterior.nc
- outputs/calibration/m2_prod_nb_samples.npz
- outputs/calibration/m2_prod_nb_result.json
- outputs/calibration/M2_PROD_NB_DONE.flag
"""
from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import json
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import MCMC, NUTS
import arviz as az
import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)
SENTINEL = OUTDIR / "M2_PROD_NB_DONE.flag"
RESULT_JSON = OUTDIR / "m2_prod_nb_result.json"
NC_PATH = OUTDIR / "m2_prod_nb_posterior.nc"
NPZ_PATH = OUTDIR / "m2_prod_nb_samples.npz"
MLFLOW_URI = "sqlite:///" + str((REPO_ROOT / "outputs" / "mlruns" / "mlflow.db").resolve())

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.calibration.gamma_registry import get_active_source
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, make_multi_season_loss_fn_nb
from kt_epimodel_hira.jax_model.numpyro_model import hira_model_nb, make_ngm_eigvalue_fn
from kt_epimodel_hira.utils.safe_save import to_native, safe_json_dump, write_flag


def main():
    if SENTINEL.exists():
        SENTINEL.unlink()

    print("=" * 70)
    print("M2 production (NB obs) — reparam A + φ=1.0 + γ registry + NB phi_nb")
    print("=" * 70)
    src = get_active_source()
    print(f"  γ source:  {src.key}")
    print(f"  φ:         FIXED at 1.0 (sample X)")
    print(f"  log_R0:    TN(log 1.5, 0.3, [log 0.8, log 2.8])")
    print(f"  logit_pi:  Normal(0, 0.3)")
    print(f"  phi_nb:    HalfNormal(10) — NB concentration (overdispersion)")
    print(f"  NUTS:      200 warmup + 200 sample × 4 chain, depth=8, accept=0.8")
    print(f"  ODE max_steps: 200K")
    print(f"  R(0): step {R0_IMMUNITY_PROFILE[0]}/{R0_IMMUNITY_PROFILE[4]}/"
          f"{R0_IMMUNITY_PROFILE[10]}/{R0_IMMUNITY_PROFILE[13]}")

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy

    shared = dict(
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

    initial_states, obs_list, w_list = [], [], []
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
        initial_states.append(jnp.asarray(state0))
        nw = tgt["n_weeks"]
        obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            obs[:, i] = tgt["hira_counts"][ag]
            w[:, i] = tgt["weights"][ag]
        obs_list.append(jnp.asarray(obs))
        w_list.append(jnp.asarray(w))

    loss_fn_nb = make_multi_season_loss_fn_nb(
        initial_states=initial_states, obs_hira_list=obs_list,
        weights_hira_list=w_list, shared_static=shared,
        n_weeks=tgt["n_weeks"], min_rate=0.01, discretize_time=False,
    )

    ngm_eigval_fn = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.7,
    )

    model = hira_model_nb(loss_fn_nb, ngm_eigval_fn=ngm_eigval_fn,
                          n_seasons=len(SEASONS))

    # Init from stepr0 (reverse-derive log_R0/logit_pi using phi=1)
    def load_stepr0(name):
        return np.array(json.load(open(OUTDIR / f"stepr0_{name}.json"))["best_vec"])

    init_names = ["warm", "bio_prior", "distributed", "home_dominant"]
    inits = []
    phi_full_ones = jnp.ones(15)
    for name in init_names:
        v = load_stepr0(name)
        log_R0_init = np.zeros(len(SEASONS))
        logit_pi_init = np.zeros((len(SEASONS), 4))
        for si in range(len(SEASONS)):
            beta_4 = v[17 + si*4 : 21 + si*4]
            R0_s = float(ngm_eigval_fn(beta_4[0], beta_4[1], beta_4[2], beta_4[3], phi_full_ones))
            log_R0_init[si] = float(np.log(max(R0_s, 1e-3)))
            beta_pos = np.clip(beta_4, 1e-6, None)
            pi_s = beta_pos / beta_pos.sum()
            lp = np.log(pi_s)
            logit_pi_init[si] = np.clip(lp - lp.mean(), -2.0, 2.0)
        inits.append({
            "log_R0": jnp.asarray(log_R0_init),
            "logit_pi": jnp.asarray(logit_pi_init),
            "phi_nb": jnp.asarray(10.0),
        })
    init_params = {
        "log_R0": jnp.stack([d["log_R0"] for d in inits]),
        "logit_pi": jnp.stack([d["logit_pi"] for d in inits]),
        "phi_nb": jnp.stack([d["phi_nb"] for d in inits]),
    }
    print(f"  init chains: {init_names}")
    print(f"  init phi_nb (all chains): 10.0")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("hira_calibration_m2_production_nb")
    with mlflow.start_run(run_name="m2_prod_nb_phi_nb_global"):
        mlflow.log_params({
            "num_warmup": 200, "num_samples": 200, "num_chains": 4,
            "target_accept": 0.8, "max_tree_depth": 8,
            "dense_mass": False,
            "reparam": "A (log_R0 + simplex)",
            "obs_model": "NegativeBinomial2",
            "phi_nb_prior": "HalfNormal(10)",
            "phi": "FIXED at 1.0",
            "gamma_source": src.key,
            "ode_max_steps": 200_000,
            "R0_profile": "step [.10/.30/.45/.65]",
        })

        t0 = time.perf_counter()
        kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=8,
                      dense_mass=False)
        mcmc = MCMC(kernel, num_warmup=200, num_samples=200, num_chains=4,
                    chain_method="sequential", progress_bar=True)
        mcmc.run(random.PRNGKey(11), init_params=init_params,
                 extra_fields=("diverging",))
        wall = time.perf_counter() - t0
        print(f"\n  wall: {wall:.0f}s ({wall/60:.1f}min)")

        samples = mcmc.get_samples(group_by_chain=True)
        extras = mcmc.get_extra_fields(group_by_chain=True)
        n_div = int(np.asarray(extras["diverging"]).sum())

        beta = np.asarray(samples["beta"])              # (4, 200, 16)
        R0 = np.asarray(samples["R0"])                  # (4, 200, 4)
        pi = np.asarray(samples["pi"])                  # (4, 200, 4, 4)
        phi_nb = np.asarray(samples["phi_nb"])          # (4, 200)

        # Save (independent try/except)
        idata = az.from_numpyro(mcmc)
        nc_saved = False
        try:
            idata.to_netcdf(str(NC_PATH))
            nc_saved = True
            print(f"  saved {NC_PATH}")
        except Exception as e:
            print(f"  [warn] netcdf save failed: {e}")
        npz_saved = False
        try:
            np.savez(
                str(NPZ_PATH),
                **{k: np.asarray(v) for k, v in samples.items()},
                diverging=np.asarray(extras["diverging"]),
            )
            npz_saved = True
            print(f"  saved {NPZ_PATH}")
        except Exception as e:
            print(f"  [warn] npz save failed: {e}")

        rhat_max = float("nan"); ess_min = float("nan")
        rhat_phi_nb = float("nan")
        try:
            rhat = az.rhat(idata)
            rhat_max = float(np.nanmax([float(rhat[v].values.max()) for v in rhat.data_vars]))
            if "phi_nb" in rhat.data_vars:
                rhat_phi_nb = float(rhat["phi_nb"].values.max())
        except Exception as e:
            print(f"  az.rhat error: {e}")
        try:
            ess = az.ess(idata)
            ess_min = float(np.nanmin([float(ess[v].values.min()) for v in ess.data_vars]))
        except Exception as e:
            print(f"  az.ess error: {e}")

        result = {
            "wall_sec": float(wall),
            "n_div": int(n_div),
            "rhat_max": float(rhat_max),
            "rhat_phi_nb": float(rhat_phi_nb),
            "ess_min": float(ess_min),
            "R0": {
                "mean_per_season": to_native(R0.mean(axis=(0, 1)).tolist()),
                "q025": to_native(np.quantile(R0, 0.025, axis=(0, 1)).tolist()),
                "q975": to_native(np.quantile(R0, 0.975, axis=(0, 1)).tolist()),
                "overall_mean": float(R0.mean()),
                "overall_min": float(R0.min()),
                "overall_max": float(R0.max()),
            },
            "phi_nb": {
                "mean": float(phi_nb.mean()),
                "median": float(np.median(phi_nb)),
                "q025": float(np.quantile(phi_nb, 0.025)),
                "q975": float(np.quantile(phi_nb, 0.975)),
                "per_chain_mean": to_native(phi_nb.mean(axis=1).tolist()),
            },
            "pi_mean_per_season": to_native(pi.mean(axis=(0, 1)).tolist()),
            "beta": {
                "mean": to_native(beta.mean(axis=(0, 1)).tolist()),
                "max": float(beta.max()),
            },
            "nc_saved": nc_saved, "npz_saved": npz_saved,
        }
        try:
            safe_json_dump(result, RESULT_JSON)
            print(f"  saved {RESULT_JSON}")
        except Exception as e:
            print(f"  [warn] json save failed: {e}")

        try:
            write_flag(
                SENTINEL,
                f"M2 PROD NB DONE wall={wall:.0f}s div={n_div} "
                f"r_hat={rhat_max:.3f} ess_min={ess_min:.0f} "
                f"R0_mean={float(R0.mean()):.3f} "
                f"phi_nb_mean={float(phi_nb.mean()):.2f} "
                f"phi_nb_rhat={rhat_phi_nb:.3f} "
                f"nc={nc_saved} npz={npz_saved}\n"
            )
            print(f"  Sentinel: {SENTINEL}")
        except Exception as e:
            print(f"  [warn] flag write failed: {e}")

        print("=" * 60)
        print(f"[VERDICT] R0 mean={float(R0.mean()):.3f} "
              f"range=[{float(R0.min()):.3f}, {float(R0.max()):.3f}]")
        print(f"  phi_nb mean={float(phi_nb.mean()):.2f} median={float(np.median(phi_nb)):.2f} "
              f"q025/q975=[{float(np.quantile(phi_nb,0.025)):.2f}, {float(np.quantile(phi_nb,0.975)):.2f}]")
        print(f"  phi_nb r_hat = {rhat_phi_nb:.3f}")
        print(f"  divergences = {n_div}/1600")
        print(f"  r_hat max = {rhat_max:.3f}    ess_min = {ess_min:.0f}")
        print("=" * 60)

        try:
            mlflow.log_metric("wall_sec", float(wall))
            mlflow.log_metric("n_div", float(n_div))
            mlflow.log_metric("rhat_max", float(rhat_max))
            mlflow.log_metric("ess_min", float(ess_min))
            mlflow.log_metric("R0_mean", float(R0.mean()))
            mlflow.log_metric("phi_nb_mean", float(phi_nb.mean()))
            mlflow.log_metric("phi_nb_rhat", float(rhat_phi_nb))
        except Exception as e:
            print(f"  [warn] mlflow log failed: {e}")


if __name__ == "__main__":
    main()
