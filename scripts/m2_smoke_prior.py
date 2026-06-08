"""M2 smoke v2 — tightened priors + faster ODE max_steps + sentinel.

Changes vs previous smoke:
- ODE max_steps 500K -> 200K (speed recovery; previous 1h+ stuck)
- β prior high 0.20 -> 0.15 (extreme-region shrink, still covers R0 up to ~5.5
  single-channel)
- arviz API fix (kind="diagnostics" only, no hdi_prob)

Outputs:
- outputs/calibration/smoke_prior_result.json  (per-chain diag + verdict)
- outputs/calibration/SMOKE_DONE.flag           (sentinel for auto-detection)

Run detached:
    nohup caffeinate -i -s uv run python scripts/m2_smoke_prior.py \\
        > outputs/calibration/smoke_prior.log 2>&1 & disown
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
import numpyro
from numpyro.infer import MCMC, NUTS
import arviz as az
import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = REPO_ROOT / "outputs" / "calibration"
OUTDIR.mkdir(parents=True, exist_ok=True)
SENTINEL = OUTDIR / "SMOKE_DONE.flag"
RESULT_JSON = OUTDIR / "smoke_prior_result.json"
MLFLOW_URI = "sqlite:///" + str((REPO_ROOT / "outputs" / "mlruns" / "mlflow.db").resolve())

SEASONS = ["2017-2018", "2018-2019", "2019-2020", "2022-2023"]

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


def compute_R0_uniform(beta_uniform, phi=1.0, seasonal_factor=1.7):
    """NGM R0 at single-β-per-channel uniform value.

    Same calc as Phase 1 R0 inversion: R0 ≈ 36.9 × β at peak sf=1.7.
    """
    inputs = build_aggregated_inputs()
    pop = inputs["pop_15"].flatten()
    N_safe = np.maximum(pop, 1e-10)
    matrices = inputs["matrices"]; rho = inputs["rho"].flatten()
    from kt_epimodel_hira.model.parameters import DiseaseParameters
    dis = DiseaseParameters()
    C_eff = beta_uniform * matrices["C_home"]
    C_eff[:4, :4] += beta_uniform * matrices["C_school"][:4, :4]
    rho_ok = (rho > 0).astype(float)
    for a in range(4, 14):
        C_eff[a, :] += beta_uniform * matrices["C_work"][a, :] * rho[a] * rho_ok
    C_eff += beta_uniform * matrices["C_other"]
    S_frac = 1.0 - R0_IMMUNITY_PROFILE
    phi_v = np.full(15, phi) if np.isscalar(phi) else phi
    K = (seasonal_factor / dis.gamma) * (
        np.diag(pop) @ np.diag(phi_v) @ np.diag(S_frac) @ C_eff @ np.diag(1.0 / N_safe)
    )
    return float(np.max(np.real(np.linalg.eigvals(K))))


def compute_R0_from_sample(beta_4, phi_14, seasonal_factor=1.7):
    """NGM R0 with per-channel β and per-age φ.

    beta_4: (4,) channel betas [h, w, s, o]
    phi_14: (14,) ages 0..14 except 5 — convert to 15 with ref=1.0
    """
    inputs = build_aggregated_inputs()
    pop = inputs["pop_15"].flatten()
    N_safe = np.maximum(pop, 1e-10)
    matrices = inputs["matrices"]; rho = inputs["rho"].flatten()
    from kt_epimodel_hira.model.parameters import DiseaseParameters
    dis = DiseaseParameters()
    bh, bw, bs, bo = beta_4
    # Expand phi to 15 with reference idx 5 = 1.0
    phi_full = np.ones(15)
    idx = 0
    for a in range(15):
        if a == 5: continue
        phi_full[a] = phi_14[idx]; idx += 1
    C_eff = bh * matrices["C_home"]
    C_eff[:4, :4] += bs * matrices["C_school"][:4, :4]
    rho_ok = (rho > 0).astype(float)
    for a in range(4, 14):
        C_eff[a, :] += bw * matrices["C_work"][a, :] * rho[a] * rho_ok
    C_eff += bo * matrices["C_other"]
    S_frac = 1.0 - R0_IMMUNITY_PROFILE
    K = (seasonal_factor / dis.gamma) * (
        np.diag(pop) @ np.diag(phi_full) @ np.diag(S_frac) @ C_eff @ np.diag(1.0 / N_safe)
    )
    return float(np.max(np.real(np.linalg.eigvals(K))))


def main():
    if SENTINEL.exists():
        SENTINEL.unlink()

    print("=" * 70)
    print("M2 smoke v2 — priors tightened + ODE 200K + sentinel")
    print("=" * 70)
    src = get_active_source()
    print(f"  γ source: {src.key}")
    print(f"  β prior:  TN(0.04, 0.04, [0.001, 0.15])  [v2: high 0.20 -> 0.15]")
    print(f"  φ prior:  TN(1.0, 0.3, [0.1, 3.0])")
    print(f"  ODE max_steps: 200K  [v2: 500K -> 200K]")
    print(f"  R(0): step {R0_IMMUNITY_PROFILE[0]}/{R0_IMMUNITY_PROFILE[4]}/"
          f"{R0_IMMUNITY_PROFILE[10]}/{R0_IMMUNITY_PROFILE[13]}")

    # Build loss closure
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
        shared_static=shared,
        n_weeks=tgt["n_weeks"], min_rate=0.01,
        discretize_time=False,
    )
    model = hira_model(loss_fn, lambda_phi=0.1)

    # Init from stepr0 (skip random outlier)
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
    print(f"  init chains: {init_names}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("hira_calibration_m2_smoke_prior_v2")
    with mlflow.start_run(run_name="smoke_v2_50w_50s_4chains"):
        mlflow.log_params({
            "num_warmup": 50, "num_samples": 50, "num_chains": 4,
            "target_accept": 0.8, "max_tree_depth": 8,
            "gamma_source": src.key,
            "beta_prior": "TN(0.04,0.04,[0.001,0.15])",
            "phi_prior": "TN(1.0,0.3,[0.1,3.0])",
            "ode_max_steps": 200_000,
            "R0_profile": "step [.10/.30/.45/.65]",
        })

        t0 = time.perf_counter()
        kernel = NUTS(model, target_accept_prob=0.8, max_tree_depth=8)
        mcmc = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=4,
                    chain_method="sequential", progress_bar=False)
        mcmc.run(random.PRNGKey(3), init_params=init_params,
                 extra_fields=("diverging",))
        wall = time.perf_counter() - t0
        print(f"\n  wall: {wall:.0f}s ({wall/60:.1f}min)")

        samples = mcmc.get_samples(group_by_chain=True)
        extras = mcmc.get_extra_fields(group_by_chain=True)
        phi = np.asarray(samples["phi"])      # (4, 50, 14)
        beta = np.asarray(samples["beta"])    # (4, 50, 16)
        n_div = int(np.asarray(extras["diverging"]).sum())

        # Diagnostics
        phi_mean = phi.mean(axis=(0, 1))
        phi_std = phi.std(axis=(0, 1))
        beta_mean = beta.mean(axis=(0, 1))
        beta_max = float(beta.max())
        phi_min = float(phi.min()); phi_max = float(phi.max())
        # cap thresholds aligned to actual prior bounds: φ in [0.5, 1.5], β in [0.001, 0.15]
        n_near_phi_cap = int((phi > 1.42).sum())   # >95% of upper bound 1.5
        n_near_beta_cap = int((beta > 0.14).sum())

        idata = az.from_numpyro(mcmc)
        # arviz API: use kind="diagnostics", no hdi_prob (fixed)
        try:
            summary = az.summary(idata, kind="diagnostics")
            rhat_max = float(summary["r_hat"].max())
            ess_min = float(summary["ess_bulk"].min())
        except Exception as e:
            print(f"  az.summary error: {e}")
            rhat_max = float("nan"); ess_min = float("nan")

        # Implied R0 from posterior samples (per chain mean β + φ)
        implied_R0_per_chain_season = []
        for c in range(4):
            phi_c = phi[c].mean(0)
            for si, s_name in enumerate(SEASONS):
                beta_c = beta[c].mean(0)[si*4:(si+1)*4]
                R0 = compute_R0_from_sample(beta_c, phi_c, seasonal_factor=1.7)
                implied_R0_per_chain_season.append({
                    "chain": c, "init": init_names[c],
                    "season": s_name, "R0_peak": R0,
                })

        # Per-chain mixing display
        per_chain = []
        for c in range(4):
            per_chain.append({
                "chain": c, "init": init_names[c],
                "phi_mean_first3": phi[c].mean(0)[:3].round(3).tolist(),
                "phi_mean_overall": float(phi[c].mean()),
                "beta_mean_first4_2017": beta[c].mean(0)[:4].round(4).tolist(),
                "beta_mean_overall": float(beta[c].mean()),
                "beta_max": float(beta[c].max()),
            })

        print(f"\n  divergences: {n_div}/200")
        print(f"  φ overall: mean={phi_mean.mean():.3f}  range=[{phi_min:.3f}, {phi_max:.3f}]")
        print(f"  φ near cap 1.5 (>1.42): {n_near_phi_cap}/{phi.size} ({n_near_phi_cap/phi.size*100:.1f}%)")
        print(f"  β overall: mean={beta_mean.mean():.4f}  max={beta_max:.4f}")
        print(f"  β near cap 0.15 (>0.14): {n_near_beta_cap}/{beta.size} ({n_near_beta_cap/beta.size*100:.1f}%)")
        print(f"  r_hat max: {rhat_max:.3f}  ess_min: {ess_min:.0f}")

        print(f"\n  Per-chain β[0] (β_h_2017) / φ[0]:")
        for c in range(4):
            print(f"    chain {c} ({init_names[c]:>14}): β[0]={float(beta[c].mean(0)[0]):.4f}  "
                  f"φ[0]={float(phi[c].mean(0)[0]):.3f}")

        print(f"\n  Implied R0 (peak sf=1.7) per chain × season:")
        R0_arr = []
        for r in implied_R0_per_chain_season:
            R0_arr.append(r["R0_peak"])
        R0_arr = np.array(R0_arr)
        print(f"    min: {R0_arr.min():.2f}  max: {R0_arr.max():.2f}  mean: {R0_arr.mean():.2f}")
        # Show by chain
        for c in range(4):
            chain_R0s = [r["R0_peak"] for r in implied_R0_per_chain_season if r["chain"] == c]
            print(f"    chain {c} ({init_names[c]:>14}): R0_peak = "
                  f"{[f'{r:.2f}' for r in chain_R0s]}")

        # Verdict
        checks = {
            "0 divergences": n_div == 0,
            "β max < 0.15 (within bound)": beta_max < 0.15,
            "β not saturating cap (<5%)": n_near_beta_cap / beta.size < 0.05,
            "φ not saturating cap 1.5 (<10%)": n_near_phi_cap / phi.size < 0.10,
            "Implied R0 ∈ [1.0, 2.5]": (R0_arr.min() >= 1.0 and R0_arr.max() <= 2.5),
            "r_hat < 1.5": rhat_max < 1.5,
        }
        verdict_passed = sum(checks.values())

        # === Save ordering: netcdf -> json -> flag -> print -> mlflow ===
        # Each stage independent try/except so any failure does not block later stages.
        from kt_epimodel_hira.utils.safe_save import to_native, safe_json_dump, write_flag

        # (1) idata netcdf (optional — for posterior persistence)
        nc_path = RESULT_JSON.parent / "smoke_v3_posterior.nc"
        try:
            idata.to_netcdf(str(nc_path))
            nc_saved = True
        except Exception as e:
            print(f"  [warn] netcdf save failed (ignored): {e}")
            nc_saved = False

        # (2) results JSON — numpy/jax 타입 to_native 강제 변환 (int64 영구 차단)
        result = {
            "wall_sec": float(wall), "n_div": int(n_div),
            "phi": {"mean": to_native(phi_mean), "std": to_native(phi_std),
                    "min": float(phi_min), "max": float(phi_max),
                    "near_cap_frac": float(n_near_phi_cap / phi.size)},
            "beta": {"mean": to_native(beta_mean), "max": float(beta_max),
                     "near_cap_frac": float(n_near_beta_cap / beta.size)},
            "rhat_max": float(rhat_max), "ess_min": float(ess_min),
            "implied_R0": {"min": float(R0_arr.min()), "max": float(R0_arr.max()),
                           "mean": float(R0_arr.mean()),
                           "per_chain_season": to_native(implied_R0_per_chain_season)},
            "per_chain": to_native(per_chain),
            "checks": {k: bool(v) for k, v in checks.items()},
            "checks_passed": int(verdict_passed),
            "checks_total": int(len(checks)),
            "nc_saved": nc_saved,
        }
        try:
            safe_json_dump(result, RESULT_JSON)
            print(f"\n  saved {RESULT_JSON}")
            json_saved = True
        except Exception as e:
            print(f"  [warn] json save failed: {e}")
            json_saved = False

        # (3) flag — 가장 단순, 위가 다 실패해도 이건 됨
        try:
            write_flag(
                SENTINEL,
                f"SMOKE DONE wall={wall:.0f}s div={n_div} "
                f"checks={verdict_passed}/{len(checks)} "
                f"beta_max={beta_max:.4f} phi_max={phi_max:.3f} "
                f"R0_range=[{R0_arr.min():.2f}, {R0_arr.max():.2f}] "
                f"nc={nc_saved} json={json_saved}\n"
            )
            print(f"  Sentinel: {SENTINEL}")
        except Exception as e:
            print(f"  [warn] flag write failed: {e}")

        # (4) verdict print — 디스크 저장 실패와 무관하게 로그에 항상 남도록
        print("=" * 60)
        print(f"[VERDICT] R0 mean={R0_arr.mean():.2f} "
              f"range=[{R0_arr.min():.2f},{R0_arr.max():.2f}]")
        print(f"  divergences={int(n_div)}/200")
        print(f"  beta max={beta_max:.4f} cap_frac={n_near_beta_cap/beta.size:.1%}")
        print(f"  phi range=[{phi_min:.3f},{phi_max:.3f}] "
              f"cap_frac={n_near_phi_cap/phi.size:.1%}")
        print(f"  r_hat max={rhat_max:.3f}")
        print("=" * 60)

        # (5) mlflow — 맨 마지막, flag 이미 찍혔으니 실패해도 무관
        def safe_key(k):
            return "".join(c if c.isalnum() or c in "._-/ " else "_" for c in k)
        try:
            for k, v in checks.items():
                mlflow.log_metric(safe_key(k), float(v))
            mlflow.log_metric("wall_sec", float(wall))
            mlflow.log_metric("n_div", float(n_div))
            mlflow.log_metric("beta_max", float(beta_max))
            mlflow.log_metric("phi_max", float(phi_max))
            mlflow.log_metric("rhat_max", float(rhat_max))
            mlflow.log_metric("R0_min", float(R0_arr.min()))
            mlflow.log_metric("R0_max", float(R0_arr.max()))
            mlflow.log_metric("R0_mean", float(R0_arr.mean()))
        except Exception as e:
            print(f"  [warn] mlflow log failed (ignored — flag/JSON saved): {e}")

    # VERDICT
    print(f"\n{'='*70}\nVERDICT ({verdict_passed}/{len(checks)} checks passed)\n{'='*70}")
    for k, v in checks.items():
        print(f"  [{'OK ' if v else 'FAIL'}] {k}")

    if verdict_passed == len(checks):
        print(f"\n  -> PASS: ready for M2 production (1000+1000 × 4 chains, detached)")
    elif verdict_passed >= 4:
        print(f"\n  -> PARTIAL: review before production")
    else:
        print(f"\n  -> FAIL: prior tightening insufficient. Re-evaluate.")

    # Sentinel already written above (before mlflow). End of main.


if __name__ == "__main__":
    main()
