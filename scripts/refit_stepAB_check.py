"""Multi-start refit under Step A (C(t) term↔vacation + κ 3-way) + Step B (v(t)
finite-season normalization). β stability check across work_share ∈ {0.06,0.15,
0.29}, 12 L-BFGS-B starts each, pin (work σ0.01, school σ0.05, home/other σ0.15).

Reuses the sens_workshare_kappa_v2 harness (reparam A: log_R0 + logit_pi simplex
+ channel pin, φ U-shape fixed, γ CDC Reed). Only build_setup is overridden:
  - RAW annual_coverage (Step B vax_rate_vector_jax now owns the -ln(1-C)/Z A-fix;
    passing correct_coverage() would double-apply it),
  - inject C_*_vac → enables C(t) blending, and drop legacy HOLIDAY β-scaling,
  - κ 3-way [0.29×4, 0.30×10, 0.0] (baseline-fit-irrelevant but set for structure).

Point estimate only. Output: console table + outputs/eda/refit_stepAB_check.json.
"""
from __future__ import annotations
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)

import sens_workshare_kappa_v2 as S   # harness reuse (main-guarded, safe to import)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "refit_stepAB_check.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
VAC_NPZ = REPO_ROOT.parent / "kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz"

WORK_SHARE_GRID = [0.06, 0.15, 0.29]
KAPPA_3WAY = np.array([0.29]*4 + [0.30]*10 + [0.0])
NLL_TOL = 0.5          # starts within this of best NLL count as "converged to best"
RAIL_TOL_LR0 = 1e-3
RAIL_TOL_PI = 0.05
BETA_FLOOR = 1e-3      # β channel below this = effectively zero (structural floor)


def build_setup_AB():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination

    seed_15 = estimate_initial_infected_from_hira(
        S.SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    tgt = load_hira_target_by_age(
        S.SEASON_LABEL, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]; w[:, i] = tgt["weights"][ag]

    vac = load_contact_matrices(path=VAC_NPZ)
    shared_base = dict(
        C_home=jnp.asarray(matrices["C_home"]), C_school=jnp.asarray(matrices["C_school"]),
        C_work=jnp.asarray(matrices["C_work"]), C_other=jnp.asarray(matrices["C_other"]),
        C_home_vac=jnp.asarray(vac["C_home"]), C_school_vac=jnp.asarray(vac["C_school"]),
        C_work_vac=jnp.asarray(vac["C_work"]), C_other_vac=jnp.asarray(vac["C_other"]),
        M_home=jnp.asarray(mobility["home"]), M_school=jnp.asarray(mobility["school"]),
        M_work=jnp.asarray(mobility["work"]), M_other=jnp.asarray(mobility["other"]),
        pop_15=jnp.asarray(pop_15), rho=jnp.asarray(rho_emp),
        sigma=disease.sigma, gamma=disease.gamma, VE=vax.VE,
        # RAW coverage — Step B applies -ln(1-C)/Z inside vax_rate_vector_jax
        annual_coverage=jnp.asarray(np.asarray(vax.annual_coverage, dtype=np.float64)),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=S.AMP, seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )   # NOTE: no HOLIDAY update — C(t) owns the winter break
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    ngm_default = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE, gamma=disease.gamma, seasonal_factor=1.0 + S.AMP,
    )
    return dict(shared_base=shared_base, state0=state0,
               obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
               obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default)


def make_shared_3way(setup):
    kw = dict(setup["shared_base"])
    kw["kappa"] = jnp.asarray(KAPPA_3WAY)
    return kw


def fit_ws(work_share, setup):
    pi_target = S.build_pi_target(work_share)
    logit_target = S.logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(S.PHI_USHAPE)
    gamma_15_j = jnp.asarray(S.build_gamma_15(*S.GAMMA_CENTER))
    shared_fit = make_shared_3way(setup)

    fg = S.build_point_loss(logit_target, S.SIGMA_PER_CHANNEL, phi_full_j,
                            gamma_15_j, shared_fit, setup)
    bounds = [S.LOG_R0_BOUNDS] + [S.LOGIT_PI_BOUNDS] * 4 + [S.PHI_NB_BOUNDS]
    starts = S.make_starts(logit_target, S.N_STARTS, S.POINT_START_SEED)

    per_start = []
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        x = np.asarray(res.x); nll = float(res.fun)
        R0 = float(np.exp(x[0])); pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
        b4 = np.asarray(derive_beta_from_R0_simplex(
            setup["ngm_default"], jnp.asarray(R0), jnp.asarray(pi), phi_full_j))
        per_start.append(dict(idx=i, nll=nll, log_R0=float(x[0]), logit_pi=x[1:5].tolist(),
                              R0=R0, pi=pi.tolist(), beta_4=[float(v) for v in b4]))
    wall = time.perf_counter() - t0

    per_start.sort(key=lambda r: r["nll"])
    best = per_start[0]
    nlls = np.array([r["nll"] for r in per_start])
    conv = [r for r in per_start if r["nll"] <= best["nll"] + NLL_TOL]
    n_conv = len(conv)
    # β spread among converged cluster
    conv_b = np.array([r["beta_4"] for r in conv])
    beta_std = conv_b.std(axis=0).tolist() if n_conv > 1 else [0.0]*4

    # railing
    lo, hi = S.LOG_R0_BOUNDS
    lr0_rail = (abs(best["log_R0"]-lo) < RAIL_TOL_LR0) or (abs(best["log_R0"]-hi) < RAIL_TOL_LR0)
    plo, phi = S.LOGIT_PI_BOUNDS
    pi_rail = any(abs(v-plo) < RAIL_TOL_PI or abs(v-phi) < RAIL_TOL_PI for v in best["logit_pi"])
    b4 = np.array(best["beta_4"])
    beta_floored = [c for c, v in zip("hwso", b4) if v < BETA_FLOOR]

    return dict(
        work_share=work_share, pi_target=pi_target.tolist(),
        R0=best["R0"], pi=best["pi"], beta_4=best["beta_4"],
        beta_home=best["beta_4"][0], nll=best["nll"],
        nll_min=float(nlls.min()), nll_max=float(nlls.max()), nll_std=float(nlls.std()),
        n_starts=len(per_start), n_converged_to_best=n_conv,
        beta_std_converged=beta_std,
        railing=dict(log_R0=bool(lr0_rail), logit_pi=bool(pi_rail),
                     beta_floored_channels=beta_floored),
        best_start_idx=best["idx"], wall_sec=float(wall),
    )


def main():
    print("="*90)
    print("REFIT Step A+B — multi-start β stability  (2019-2020, φ U-shape, γ CDC Reed)")
    print(f"  work_share ∈ {WORK_SHARE_GRID}   κ 3-way={list(KAPPA_3WAY[:1])}/{KAPPA_3WAY[4]}/{KAPPA_3WAY[14]}"
          f"   pin σ(h,w,s,o)={list(S.SIGMA_PER_CHANNEL)}   {S.N_STARTS} starts")
    print(f"  C(t) term↔vacation ON  |  v(t) finite-season norm ON  |  RAW coverage")
    print("="*90)
    t0 = time.perf_counter()
    setup = build_setup_AB()
    print(f"[setup] {time.perf_counter()-t0:.1f}s\n")

    results = []
    for ws in WORK_SHARE_GRID:
        r = fit_ws(ws, setup)
        results.append(r)
        b = r["beta_4"]; rail = r["railing"]
        print(f"── work_share={ws:.2f}  ({r['wall_sec']:.1f}s) ──")
        print(f"   R0={r['R0']:.3f}  NLL={r['nll']:.2f}  (start spread: min={r['nll_min']:.2f} max={r['nll_max']:.2f} std={r['nll_std']:.3f})")
        print(f"   β_4 = [h={b[0]:.4f}, w={b[1]:.4f}, s={b[2]:.4f}, o={b[3]:.4f}]")
        print(f"   π   = [h={r['pi'][0]:.3f}, w={r['pi'][1]:.3f}, s={r['pi'][2]:.3f}, o={r['pi'][3]:.3f}]  (target w={ws:.2f})")
        print(f"   converged_to_best: {r['n_converged_to_best']}/{r['n_starts']}   β_std(conv)={[round(v,4) for v in r['beta_std_converged']]}")
        print(f"   railing: log_R0={rail['log_R0']} logit_pi={rail['logit_pi']} β_floored={rail['beta_floored_channels'] or 'none'}")
        print()

    # verdict
    any_rail = any(r["railing"]["log_R0"] or r["railing"]["logit_pi"] or r["railing"]["beta_floored_channels"] for r in results)
    all_converge = all(r["n_converged_to_best"] >= max(2, r["n_starts"]//2) for r in results)
    bh = [r["beta_home"] for r in results]
    verdict = ("STABLE — no railing, majority start agreement → full refit OK"
               if (not any_rail and all_converge)
               else "REVIEW — " + ("railing present (pin re-tune) " if any_rail else "")
                    + ("" if all_converge else "weak start agreement"))
    print("="*90)
    print(f"β_home across ws: {[round(v,4) for v in bh]}   (smoke ref ≈ 0.057)")
    print(f"VERDICT: {verdict}")
    print("="*90)

    out = dict(
        meta=dict(season=S.SEASON_LABEL, work_share_grid=WORK_SHARE_GRID,
                  kappa_3way=KAPPA_3WAY.tolist(), sigma_pin=S.SIGMA_PER_CHANNEL.tolist(),
                  n_starts=S.N_STARTS, step_A="C(t) term-vacation + kappa 3-way",
                  step_B="v(t) -ln(1-C)/Z finite-season norm", coverage="raw (A-fix in model)",
                  phi="U-shape fixed", gamma_15=list(S.GAMMA_CENTER), AMP=S.AMP),
        results=results,
        verdict=verdict, beta_home_by_ws=bh, any_railing=any_rail, all_converge=all_converge,
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
