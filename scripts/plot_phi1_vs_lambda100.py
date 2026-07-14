"""Compare epidemic curves: φ=1.0 (production) vs φ=λ100-best (physical U shape).

Same season 2019-2020, same shared setup (HOLIDAY realloc=1, amp=0.7, AMP=0.9,
γ=CDC, σ=0.5, NB). Fair comparison: β_4 (+ phi_nb) is REFIT under each φ so
neither curve is penalised by holding β constant while φ changes.

Inputs:
- φ_lambda100_best: loaded from outputs/eda/phi_2ndorder.json (λ=100 result).
- φ_ones: jnp.ones(15) — production choice.

Outputs:
- presentations/figures/phi1_vs_lambda100_fit.png (2×3 grid, 6 HIRA ages).
- Console: NLL per condition, per-age r = obs_peak_sum(±2w)/model_peak_sum(±2w).
"""
from __future__ import annotations
import os, json, time
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
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PREV_JSON = REPO_ROOT / "outputs" / "eda" / "phi_2ndorder.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "phi1_vs_lambda100_fit.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

SEASON_LABEL = "2019-2020"
AMP = 0.9
HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)
GAMMA_CDC = np.concatenate([np.full(4, 0.40), np.full(9, 0.18), np.full(2, 0.25)])
BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)
N_STARTS = 8
START_SEED = 13
PEAK_HALF_WIN = 2


def build_setup():
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
    gamma_15 = jnp.asarray(GAMMA_CDC)

    seed_15 = estimate_initial_infected_from_hira(
        SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    tgt = load_hira_target_by_age(
        SEASON_LABEL, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]

    return dict(
        shared=shared, gamma_15=gamma_15, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks,
    )


def predict_hira(beta_4, phi_full_15, *, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_15
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc_15 = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc_15, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def build_loss_fixed_phi(phi_full_15: jnp.ndarray, setup):
    """x = [β_4, phi_nb]. φ is fixed at phi_full_15."""
    def loss(x):
        beta_4 = x[:4]
        phi_nb = x[4]
        pred = predict_hira(beta_4, phi_full_15, setup=setup)
        return nb_nll_jax(setup["obs_j"], pred, setup["w_j"],
                          concentration=phi_nb, min_rate=0.01)
    loss_j = jax.jit(loss)
    grad_j = jax.jit(jax.grad(loss))

    def fg_np(x_np):
        x = jnp.asarray(x_np)
        v = float(loss_j(x))
        g = np.asarray(grad_j(x))
        if not np.isfinite(v):
            v = 1e15
            g = np.where(np.isfinite(g), g, 0.0)
        return v, g

    return fg_np


def make_starts_beta_only(n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = [
        np.array([0.10, 0.10, 0.10, 0.10, 10.0]),
        np.array([0.05, 0.05, 0.05, 0.15, 10.0]),
        np.array([0.07, 0.07, 0.20, 0.10, 10.0]),
        np.array([0.07, 0.07, 0.05, 0.20, 10.0]),
    ]
    starts = list(base)
    while len(starts) < n_starts:
        b = rng.uniform(0.02, 0.20, 4)
        starts.append(np.concatenate([b, np.array([rng.uniform(2.0, 20.0)])]))
    return starts[:n_starts]


def fit_beta_only(phi_full_15: jnp.ndarray, setup) -> dict:
    fg = build_loss_fixed_phi(phi_full_15, setup)
    bounds = BETA_BOUNDS + [PHI_NB_BOUNDS]
    starts = make_starts_beta_only(N_STARTS, START_SEED)
    best = None
    per_start_nll = []
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=300, ftol=1e-9, gtol=1e-6))
            nll = float(res.fun)
        except Exception as e:
            print(f"    [warn] start {i} failed: {e}")
            continue
        per_start_nll.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0
    best["wall_sec"] = float(wall)
    best["nll_per_start"] = per_start_nll
    return best


def peak_window_sum(arr_1d, peak_w, half):
    lo = max(0, peak_w - half)
    hi = min(arr_1d.shape[0], peak_w + half + 1)
    return float(arr_1d[lo:hi].sum())


def per_age_ratios(pred, setup):
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    obs_masked = np.where(mask[:, None], obs, -1e18)
    pred_masked = np.where(mask[:, None], pred, -1e18)
    rows = {}
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        obs_pw = int(np.argmax(obs_masked[:, ai]))
        mdl_pw = int(np.argmax(pred_masked[:, ai]))
        obs_sum = peak_window_sum(obs[:, ai], obs_pw, PEAK_HALF_WIN)
        mdl_sum = peak_window_sum(pred[:, ai], mdl_pw, PEAK_HALF_WIN)
        rows[ag] = dict(
            obs_peak_week=obs_pw, model_peak_week=mdl_pw,
            phase_offset_weeks=mdl_pw - obs_pw,
            ratio=obs_sum / max(mdl_sum, 1.0),
        )
    return rows


def main():
    print("=" * 78)
    print(f"COMPARE: φ=1.0 (production) vs φ=λ100-best  —  {SEASON_LABEL}")
    print(f"  fair comparison: β_4 (+ phi_nb) REFIT under each φ")
    print(f"  NB obs, HOLIDAY realloc=1 amp=0.7, AMP=0.9, γ=CDC[0.40/0.18/0.25]")
    print("=" * 78)

    setup = build_setup()

    # ─── (A) φ = ones ────────────────────────────────
    phi_ones = jnp.ones(15)
    print("\n[A] φ = ones(15)  (production choice) — refit β_4 + phi_nb")
    best_A = fit_beta_only(phi_ones, setup)
    beta_A = best_A["x"][:4]; phi_nb_A = float(best_A["x"][4])
    pred_A = np.asarray(predict_hira(beta_A, phi_ones, setup=setup))
    r_A = per_age_ratios(pred_A, setup)
    print(f"  best NLL={best_A['nll']:.4e}  β_4={[round(x,4) for x in beta_A]}  "
          f"phi_nb={phi_nb_A:.2f}  start_idx={best_A['start_idx']}  "
          f"wall={best_A['wall_sec']:.1f}s")

    # ─── (B) φ = λ=100 best ──────────────────────────
    if not PREV_JSON.exists():
        raise FileNotFoundError(f"required prior result: {PREV_JSON}")
    with open(PREV_JSON) as f:
        prev = json.load(f)
    lam100 = next(r for r in prev["results"] if r["lambda_phi"] == 100.0)
    phi_lam100 = jnp.asarray(lam100["diagnostic"]["phi_full_15"])
    print(f"\n[B] φ = λ=100 best  (loaded from {PREV_JSON.name})")
    print(f"    φ_full(15) = {[round(float(p),3) for p in phi_lam100]}")
    print("    refit β_4 + phi_nb under this φ")
    best_B = fit_beta_only(phi_lam100, setup)
    beta_B = best_B["x"][:4]; phi_nb_B = float(best_B["x"][4])
    pred_B = np.asarray(predict_hira(beta_B, phi_lam100, setup=setup))
    r_B = per_age_ratios(pred_B, setup)
    print(f"  best NLL={best_B['nll']:.4e}  β_4={[round(x,4) for x in beta_B]}  "
          f"phi_nb={phi_nb_B:.2f}  start_idx={best_B['start_idx']}  "
          f"wall={best_B['wall_sec']:.1f}s")

    # ─── Console summary tables ──────────────────────
    print("\n" + "=" * 78)
    print(f"  NLL comparison   (φ=1 vs φ=λ100-best)")
    print(f"  φ=1        NLL={best_A['nll']:.4e}  phi_nb={phi_nb_A:.2f}")
    print(f"  φ=λ100     NLL={best_B['nll']:.4e}  phi_nb={phi_nb_B:.2f}")
    print(f"  ΔNLL = {best_A['nll'] - best_B['nll']:+.4e}   "
          f"(positive → φ=λ100 better)")
    print("=" * 78)

    print(f"\n  Age ratio r = obs_peak(±2w) / model_peak(±2w)")
    header = "  cond          " + "  ".join(f"{ag:>8s}" for ag in HIRA_AGE_GROUPS) \
             + "    phase(off)"
    print(header)
    for label, rr in [("φ=1", r_A), ("φ=λ100", r_B)]:
        row = f"  {label:12s} "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {rr[ag]['ratio']:>8.2f}"
        offs = [rr[ag]['phase_offset_weeks'] for ag in HIRA_AGE_GROUPS]
        row += f"     [{','.join(f'{o:+d}' for o in offs)}]"
        print(row)

    # ─── Figure ─────────────────────────────────────
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    weeks_fit = np.arange(obs.shape[0])[mask]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        ax = axes[ai // 3, ai % 3]
        ax.plot(weeks_fit, obs[mask, ai], "k-o", ms=4, label="HIRA obs")
        ax.plot(weeks_fit, pred_A[mask, ai], "-", color="#888888", lw=2.0,
                label=f"φ=1 pred")
        ax.plot(weeks_fit, pred_B[mask, ai], "-", color="#1a5490", lw=2.0,
                label=f"φ=λ100 pred")
        rA = r_A[ag]["ratio"]; rB = r_B[ag]["ratio"]
        ax.set_title(f"{ag}   r(φ=1)={rA:.2f}  r(φ=λ100)={rB:.2f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("week")
        ax.set_ylabel("HIRA weekly count")
        if ai == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        f"{SEASON_LABEL} single-season fit  —  "
        f"φ=1 (grey)  vs  φ=λ100 best (blue)  vs  obs (black)\n"
        f"NLL: φ=1 {best_A['nll']:.3e}   φ=λ100 {best_B['nll']:.3e}   "
        f"Δ={best_A['nll']-best_B['nll']:+.3e}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"\nsaved {OUT_FIG}")


if __name__ == "__main__":
    main()
