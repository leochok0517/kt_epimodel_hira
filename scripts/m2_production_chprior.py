"""M2 production NB — channel-prior {A,B} × 4 seasons + winter break + γ CDC.

Usage:  uv run python scripts/m2_production_chprior.py A
        uv run python scripts/m2_production_chprior.py B

A: NIMS contact share / unit R0  → π_A ≈ [0.29, 0.29, 0.06, 0.36]
B: Italy 2009 R0 contribution    → π_B ≈ [0.47, 0.17, 0.11, 0.25]

Settings: 300 warmup + 300 sample × 4 chains sequential, NB obs, holiday
ON + realloc=1, amp=0.9 fixed (best across diagnostics), φ=1.0, γ CDC
absolute, free 4-channel with weak prior (σ_strong on home/work,
σ_loose on school/other).
"""
from __future__ import annotations
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import MCMC, NUTS
import arviz as az
import mlflow

if len(sys.argv) < 2 or sys.argv[1] not in ("A", "B"):
    print("usage: m2_production_chprior.py {A|B}")
    sys.exit(1)
CHANNEL = sys.argv[1]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)
SENTINEL = OUTDIR / f"PROD_{CHANNEL}_DONE.flag"
RESULT_JSON = OUTDIR / f"m2_prod_{CHANNEL}_result.json"
NC_PATH = OUTDIR / f"m2_prod_{CHANNEL}_posterior.nc"
NPZ_PATH = OUTDIR / f"m2_prod_{CHANNEL}_samples.npz"
MLFLOW_URI = "sqlite:///" + str((REPO_ROOT / "outputs" / "mlruns" / "mlflow.db").resolve())

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
AMP = 0.9
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
UNIT_R0 = np.array([8.70, 6.21, 25.40, 9.33])

HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)

SIGMA_PER_CHANNEL = np.array([0.01, 0.01, 0.30, 0.30])   # weak: pin h/w, free s/o

if CHANNEL == "A":
    contact = np.array([23.26, 17.03, 14.24, 31.77])
    share = contact / contact.sum()
    pi_target = share / UNIT_R0
    pi_target = pi_target / pi_target.sum()
    LABEL = "A_NIMS"
else:
    R0_contrib = np.array([0.40, 0.10, 0.27, 0.23])
    R0_contrib = R0_contrib / R0_contrib.sum()
    pi_target = R0_contrib / UNIT_R0
    pi_target = pi_target / pi_target.sum()
    LABEL = "B_literature"

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, make_multi_season_loss_fn_nb
from kt_epimodel_hira.jax_model.numpyro_model import (
    hira_model_nb_chprior, make_ngm_eigvalue_fn,
)
from kt_epimodel_hira.utils.safe_save import to_native, safe_json_dump, write_flag


def main():
    if SENTINEL.exists():
        SENTINEL.unlink()

    print("=" * 70)
    print(f"M2 production NB+chprior {LABEL}")
    print("=" * 70)
    print(f"  target π = {[round(float(x), 3) for x in pi_target]}")
    print(f"  σ per channel (h, w, s, o) = {SIGMA_PER_CHANNEL.tolist()}")
    print(f"  γ = CDC absolute (0.40/0.18/0.25), amp={AMP}, holiday realloc=1.0")

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
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared.update(HOLIDAY)

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
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )
    model = hira_model_nb_chprior(
        loss_fn_nb, ngm_eigval_fn=ngm_eigval_fn, n_seasons=len(SEASONS),
        target_pi=pi_target, sigma_per_channel=SIGMA_PER_CHANNEL,
        gamma_3_override=jnp.array([GAMMA_CDC[0], GAMMA_CDC[4], GAMMA_CDC[13]]),
    )

    # init: use target as initial logit (centered)
    lp = np.log(np.clip(pi_target, 1e-6, None))
    logit_init = lp - lp.mean()
    init_params = {
        "log_R0": jnp.stack([jnp.full(len(SEASONS), float(np.log(2.0)))] * 4),
        "logit_pi": jnp.stack([jnp.tile(jnp.asarray(logit_init), (len(SEASONS), 1))] * 4),
        "phi_nb": jnp.stack([jnp.asarray(10.0)] * 4),
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(f"hira_calibration_m2_prod_chprior_{CHANNEL}")
    with mlflow.start_run(run_name=f"m2_prod_{LABEL}"):
        mlflow.log_params({
            "channel_prior": LABEL,
            "target_pi": [float(x) for x in pi_target],
            "sigma_per_channel": SIGMA_PER_CHANNEL.tolist(),
            "num_warmup": 300, "num_samples": 300, "num_chains": 4,
            "amp": AMP, "holiday_amp": HOLIDAY["school_holiday_amp"],
            "holiday_realloc": HOLIDAY["school_holiday_realloc"],
            "gamma": "CDC absolute",
        })

        t0 = time.perf_counter()
        kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=8,
                      dense_mass=False)
        mcmc = MCMC(kernel, num_warmup=300, num_samples=300, num_chains=4,
                    chain_method="sequential", progress_bar=True)
        seed_int = 17 if CHANNEL == "A" else 23
        mcmc.run(random.PRNGKey(seed_int), init_params=init_params,
                 extra_fields=("diverging",))
        wall = time.perf_counter() - t0
        print(f"\n  wall: {wall:.0f}s ({wall/60:.1f}min)")

        samples = mcmc.get_samples(group_by_chain=True)
        extras = mcmc.get_extra_fields(group_by_chain=True)
        n_div = int(np.asarray(extras["diverging"]).sum())
        beta = np.asarray(samples["beta"])
        R0 = np.asarray(samples["R0"])
        pi = np.asarray(samples["pi"])
        phi_nb = np.asarray(samples["phi_nb"])

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
        try:
            rhat = az.rhat(idata)
            rhat_max = float(np.nanmax([float(rhat[v].values.max()) for v in rhat.data_vars]))
        except Exception as e:
            print(f"  az.rhat error: {e}")
        try:
            ess = az.ess(idata)
            ess_min = float(np.nanmin([float(ess[v].values.min()) for v in ess.data_vars]))
        except Exception as e:
            print(f"  az.ess error: {e}")

        result = {
            "channel_prior": LABEL,
            "target_pi": [float(x) for x in pi_target],
            "sigma_per_channel": SIGMA_PER_CHANNEL.tolist(),
            "wall_sec": float(wall),
            "n_div": int(n_div),
            "rhat_max": float(rhat_max), "ess_min": float(ess_min),
            "R0_mean_per_season": to_native(R0.mean(axis=(0, 1)).tolist()),
            "R0_q025": to_native(np.quantile(R0, 0.025, axis=(0, 1)).tolist()),
            "R0_q975": to_native(np.quantile(R0, 0.975, axis=(0, 1)).tolist()),
            "pi_mean_per_season": to_native(pi.mean(axis=(0, 1)).tolist()),
            "pi_q025_per_season": to_native(np.quantile(pi, 0.025, axis=(0, 1)).tolist()),
            "pi_q975_per_season": to_native(np.quantile(pi, 0.975, axis=(0, 1)).tolist()),
            "phi_nb_mean": float(phi_nb.mean()),
            "phi_nb_q025": float(np.quantile(phi_nb, 0.025)),
            "phi_nb_q975": float(np.quantile(phi_nb, 0.975)),
            "beta_mean": to_native(beta.mean(axis=(0, 1)).tolist()),
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
                f"PROD {CHANNEL} DONE wall={wall:.0f}s div={n_div} "
                f"rhat={rhat_max:.3f} ess_min={ess_min:.0f} "
                f"R0_mean={float(R0.mean()):.3f} "
                f"phi_nb={float(phi_nb.mean()):.2f} "
                f"nc={nc_saved} npz={npz_saved}\n"
            )
            print(f"  Sentinel: {SENTINEL}")
        except Exception as e:
            print(f"  [warn] flag write failed: {e}")

        print("=" * 60)
        print(f"[VERDICT {LABEL}] R0 mean={float(R0.mean()):.3f}  div={n_div}")
        print(f"  r_hat = {rhat_max:.3f}    ess_min = {ess_min:.0f}")
        for si, s in enumerate(SEASONS):
            ps = pi.mean(axis=(0, 1))[si]
            print(f"  {s}: home={ps[0]:.3f} work={ps[1]:.3f} school={ps[2]:.3f} other={ps[3]:.3f}")
        print("=" * 60)

        try:
            mlflow.log_metric("wall_sec", float(wall))
            mlflow.log_metric("n_div", float(n_div))
            mlflow.log_metric("rhat_max", float(rhat_max))
            mlflow.log_metric("ess_min", float(ess_min))
            mlflow.log_metric("R0_mean", float(R0.mean()))
            mlflow.log_metric("phi_nb_mean", float(phi_nb.mean()))
        except Exception as e:
            print(f"  [warn] mlflow log failed: {e}")


if __name__ == "__main__":
    main()
