"""SEM-2 (seminar feedback) — 4→2 channel collapse sanity check.

Usage:
    uv run python scripts/sem2_2ch_collapse_sanity.py --sanity
    uv run python scripts/sem2_2ch_collapse_sanity.py --full

--sanity: warmup 50 + samples 50 × 2 chains (pipeline check, ~minutes)
--full:   warmup 300 + samples 300 × 4 chains sequential (production scale)

Hypothesis: HIRA data does not support the 4-channel decomposition. Collapsing
home + work + other into a single "home_total" channel while keeping school
(winter break) separate should fit comparably to production 4-channel.

Settings (match m2_production_chprior so comparison is apples-to-apples):
- NB observation, holiday ON (amp 0.7, realloc 1.0), seasonal amp 0.9, γ CDC,
  φ = 1.0, unit_R0 NIMS basis.
- C_home_eff = C_h + C_w + C_o, C_school unchanged, C_work = C_other = 0.
- π is 2-simplex (home_total, school); β_w = β_o = 0.
"""
from __future__ import annotations
import os, sys, json, time, argparse
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

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]
AMP = 0.9
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])

HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, make_multi_season_loss_fn_nb,
)
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn
from kt_epimodel_hira.jax_model.numpyro_model_sem2 import hira_model_nb_2ch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--full", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sanity == args.full:
        print("must pass exactly one of --sanity / --full")
        sys.exit(2)

    mode = "sanity" if args.sanity else "full"
    if mode == "sanity":
        n_warmup, n_samples, n_chains = 50, 50, 2
        tag = "sanity"
    else:
        n_warmup, n_samples, n_chains = 300, 300, 4
        tag = "full"

    print("=" * 70)
    print(f"SEM-2: 4→2 channel collapse — {mode} run")
    print(f"  warmup={n_warmup} samples={n_samples} chains={n_chains} sequential")
    print(f"  NB obs, holiday ON, amp={AMP}, γ CDC, φ=1.0")
    print("=" * 70)

    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy

    # ★ collapse: C_home_eff = C_h + C_w + C_o, C_w = C_o = 0, C_s unchanged
    C_home_eff = (
        np.asarray(matrices["C_home"])
        + np.asarray(matrices["C_work"])
        + np.asarray(matrices["C_other"])
    )
    C_school_full = np.asarray(matrices["C_school"])
    C_zero = np.zeros_like(C_school_full)

    print(f"  C_home_eff total mass: {C_home_eff.sum():.2f} "
          f"(orig home {np.asarray(matrices['C_home']).sum():.2f} + "
          f"work {np.asarray(matrices['C_work']).sum():.2f} + "
          f"other {np.asarray(matrices['C_other']).sum():.2f})")
    print(f"  C_school mass: {C_school_full.sum():.2f} (preserved)")

    shared = dict(
        C_home=jnp.asarray(C_home_eff),
        C_school=jnp.asarray(C_school_full),
        C_work=jnp.asarray(C_zero),
        C_other=jnp.asarray(C_zero),
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

    # NGM uses collapsed matrices too — β derivation must see what simulate_jax sees
    ngm_eigval_fn = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=C_home_eff, C_work=C_zero,
        C_school=C_school_full, C_other=C_zero,
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )
    model = hira_model_nb_2ch(
        loss_fn_nb, ngm_eigval_fn=ngm_eigval_fn, n_seasons=len(SEASONS),
        gamma_3_override=jnp.array([GAMMA_CDC[0], GAMMA_CDC[4], GAMMA_CDC[13]]),
    )

    # Init: log_R0 at log(2.0), logit_pi2 at (0, 0) so softmax → (0.5, 0.5).
    init_params = {
        "log_R0": jnp.stack([jnp.full(len(SEASONS), float(np.log(2.0)))] * n_chains),
        "logit_pi2": jnp.stack(
            [jnp.zeros((len(SEASONS), 2))] * n_chains
        ),
        "phi_nb": jnp.stack([jnp.asarray(10.0)] * n_chains),
    }

    t0 = time.perf_counter()
    kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=8,
                  dense_mass=False)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, chain_method="sequential",
                progress_bar=True)
    mcmc.run(random.PRNGKey(31), extra_fields=("diverging",),
             init_params=init_params)
    wall = time.perf_counter() - t0
    print(f"\n  wall: {wall:.0f}s ({wall/60:.1f}min)")

    samples = mcmc.get_samples(group_by_chain=True)
    extras = mcmc.get_extra_fields(group_by_chain=True)
    n_div = int(np.asarray(extras["diverging"]).sum())
    beta = np.asarray(samples["beta"])
    R0 = np.asarray(samples["R0"])
    pi2 = np.asarray(samples["pi2"])
    phi_nb = np.asarray(samples["phi_nb"])

    rhat_max = float("nan"); ess_min = float("nan")
    try:
        idata = az.from_numpyro(mcmc)
        rhat = az.rhat(idata)
        rhat_max = float(np.nanmax(
            [float(rhat[v].values.max()) for v in rhat.data_vars]
        ))
        ess = az.ess(idata)
        ess_min = float(np.nanmin(
            [float(ess[v].values.min()) for v in ess.data_vars]
        ))
    except Exception as e:
        print(f"  arviz diag error: {e}")

    print("=" * 60)
    print(f"[VERDICT SEM-2 2ch {mode}]")
    print(f"  wall={wall:.0f}s  div={n_div}  r_hat={rhat_max:.3f}  ess_min={ess_min:.0f}")
    print(f"  R0 mean per season: "
          f"{[round(float(x), 3) for x in R0.mean(axis=(0,1))]}")
    print(f"  pi2 (home_total, school) per season:")
    for si, s in enumerate(SEASONS):
        ps = pi2.mean(axis=(0, 1))[si]
        print(f"    {s}: home_total={ps[0]:.3f}  school={ps[1]:.3f}")
    print(f"  phi_nb mean: {float(phi_nb.mean()):.3f}")
    print("=" * 60)

    # Save
    npz_path = OUTDIR / f"sem2_2ch_{tag}_samples.npz"
    np.savez(
        str(npz_path),
        **{k: np.asarray(v) for k, v in samples.items()},
        diverging=np.asarray(extras["diverging"]),
    )
    print(f"  saved {npz_path}")

    result = {
        "mode": mode,
        "n_warmup": n_warmup, "n_samples": n_samples, "n_chains": n_chains,
        "wall_sec": float(wall), "n_div": int(n_div),
        "rhat_max": float(rhat_max), "ess_min": float(ess_min),
        "R0_mean_per_season": [float(x) for x in R0.mean(axis=(0, 1))],
        "pi2_mean_per_season": [
            [float(x) for x in pi2.mean(axis=(0, 1))[si]]
            for si in range(len(SEASONS))
        ],
        "phi_nb_mean": float(phi_nb.mean()),
        "phi_nb_q025": float(np.quantile(phi_nb, 0.025)),
        "phi_nb_q975": float(np.quantile(phi_nb, 0.975)),
        "beta_mean": [float(x) for x in beta.mean(axis=(0, 1))],
    }
    json_path = OUTDIR / f"sem2_2ch_{tag}_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved {json_path}")


if __name__ == "__main__":
    main()
