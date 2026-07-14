"""C1 diagnostic: work channel self-recovery under 15-age vs 6-age binning.

Isolate whether HIRA 6-group binning structurally destroys β_w identifiability.
Generate synthetic 15-age daily incidence with known β_true (large β_w on
purpose), then recover β_4 under two observation modes:
    (A) 15-age weekly aggregation (γ_15 element-wise, no HIRA collapse)
    (B) 6-age HIRA mapping (production path)
φ, γ_report, R(0), κ, HOLIDAY, AMP, σ, γ all fixed and identical between
generation and recovery — only the observation binning changes.

Noise: 0 (deterministic forward). Poisson NLL used for recovery.

Decision guide (comments only — script does not interpret):
- A recovers β_true, B collapses β_w→0 → HIRA 6-bin destroys work identifiability.
- Both recover → work is identifiable; production β_w≈0 reflects real HIRA data.
- Neither recovers → binning-independent non-id (channel structure itself).

Note: φ is fixed at ones(15) (production choice) for both generation and
recovery — a "same φ" fair test. Sensitivity to φ choice not tested here.
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
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, poisson_nll_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "work_recovery_C1.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "work_recovery_C1.png"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
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

BETA_TRUE = np.array([0.06, 0.08, 0.05, 0.06])   # [h, w, s, o] — β_w intentionally large
N_STARTS = 12
START_SEED = 23
BETA_BOUNDS = [(0.001, 1.0)] * 4
N_WEEKS_USE = None  # will be set from HIRA target

# Profile scan config for condition B β_w
PROFILE_N = 41
PROFILE_LO = 0.001
PROFILE_HI = 0.15


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
    phi_full = jnp.ones(15)   # ★ production choice — used for BOTH gen and recover

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

    ngm_fn = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )
    return dict(
        shared=shared, gamma_15=gamma_15, phi_full=phi_full,
        state0=state0, n_weeks=n_weeks, ngm_fn=ngm_fn,
    )


def forward_inc_15(beta_4, setup) -> jnp.ndarray:
    """Run simulate_jax with given β_4 (φ fixed at ones) → daily inc (n_days-1, 15)."""
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = setup["phi_full"]
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    return daily_new_infection_by_age_jax(st)   # (n_days-1, 15)


def inc_to_weekly_15(inc_15: jnp.ndarray, gamma_15: jnp.ndarray,
                      n_weeks: int) -> jnp.ndarray:
    """(A) 15-age weekly obs: γ_15 · sum-per-week, no HIRA collapse.
    Mirrors simulation_to_hira_by_age_jax except skips the 6-mapping."""
    N_AGE = 15
    n_complete = inc_15.shape[0] // 7
    weekly = inc_15[: n_complete * 7].reshape(n_complete, 7, N_AGE).sum(axis=1)
    reported = weekly * gamma_15[None, :]   # (n_complete, 15)
    if n_complete < n_weeks:
        pad = jnp.zeros((n_weeks - n_complete, N_AGE))
        return jnp.concatenate([reported, pad], axis=0)
    return reported[:n_weeks]


def predict_A(beta_4, setup) -> jnp.ndarray:
    """15-age weekly (n_weeks, 15)."""
    inc = forward_inc_15(beta_4, setup)
    return inc_to_weekly_15(inc, setup["gamma_15"], setup["n_weeks"])


def predict_B(beta_4, setup) -> jnp.ndarray:
    """6-age HIRA (n_weeks, 6)."""
    inc = forward_inc_15(beta_4, setup)
    return simulation_to_hira_by_age_jax(inc, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def build_loss(mode: str, obs_j, setup):
    if mode == "A":
        pred_fn = predict_A
        w_j = jnp.ones_like(obs_j)
    elif mode == "B":
        pred_fn = predict_B
        w_j = jnp.ones_like(obs_j)
    else:
        raise ValueError(mode)

    def loss(x):
        pred = pred_fn(x, setup)
        return poisson_nll_jax(obs_j, pred, w_j, min_rate=0.01)

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

    return fg_np, loss_j


def make_starts(n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = [
        np.array([0.10, 0.10, 0.10, 0.10]),
        np.array([0.05, 0.05, 0.05, 0.15]),
        np.array([0.07, 0.07, 0.20, 0.10]),
        np.array([0.07, 0.07, 0.05, 0.20]),
        np.array([0.15, 0.15, 0.05, 0.05]),
        np.array([0.02, 0.20, 0.05, 0.10]),
    ]
    starts = list(base)
    while len(starts) < n_starts:
        starts.append(rng.uniform(0.02, 0.20, 4))
    return starts[:n_starts]


def fit_mode(mode: str, obs_j, setup) -> dict:
    fg, loss_j = build_loss(mode, obs_j, setup)
    starts = make_starts(N_STARTS, START_SEED)
    per_start_nll = []
    best = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                            bounds=BETA_BOUNDS,
                            options=dict(maxiter=400, ftol=1e-12, gtol=1e-8))
            nll = float(res.fun)
        except Exception as e:
            print(f"    [warn] start {i} failed: {e}")
            continue
        per_start_nll.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0
    nll_arr = np.array(per_start_nll)
    return dict(
        mode=mode, best_nll=best["nll"],
        beta_recovered=[float(x) for x in best["x"]],
        start_idx=best["start_idx"], wall_sec=float(wall),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        nll_min=float(np.min(nll_arr)),
        nll_max=float(np.max(nll_arr)),
        nll_per_start=[float(x) for x in nll_arr.tolist()],
    )


def main():
    print("=" * 78)
    print(f"C1: work channel self-recovery under 15-age vs 6-age binning  —  "
          f"{SEASON_LABEL}")
    print(f"  β_true = {BETA_TRUE.tolist()}   (β_w intentionally large: "
          f"{BETA_TRUE[1]})")
    print(f"  φ = ones(15)  (production choice, same for gen & recover)")
    print(f"  noise = 0  (deterministic forward, Poisson NLL for recovery)")
    print(f"  multi-start {N_STARTS}, β bounds {BETA_BOUNDS[0]}")
    print("=" * 78)

    setup = build_setup()
    print(f"  n_weeks = {setup['n_weeks']}  pop shape = "
          f"{np.asarray(setup['shared']['pop_15']).shape}")

    # Ground-truth R0 (informational)
    r0_true = float(setup["ngm_fn"](
        jnp.asarray(BETA_TRUE[0]), jnp.asarray(BETA_TRUE[1]),
        jnp.asarray(BETA_TRUE[2]), jnp.asarray(BETA_TRUE[3]),
        setup["phi_full"],
    ))
    print(f"  R0_true (NGM) = {r0_true:.3f}")

    # ─── Generate synthetic obs ─────────────────────────────
    inc_true = forward_inc_15(jnp.asarray(BETA_TRUE), setup)
    obs_A = np.asarray(inc_to_weekly_15(inc_true, setup["gamma_15"],
                                         setup["n_weeks"]))
    obs_B = np.asarray(simulation_to_hira_by_age_jax(
        inc_true, setup["gamma_15"], n_weeks=setup["n_weeks"],
    ))
    print(f"  synthetic obs_A shape={obs_A.shape}  sum={obs_A.sum():,.0f}")
    print(f"  synthetic obs_B shape={obs_B.shape}  sum={obs_B.sum():,.0f}")

    # ─── Recover ────────────────────────────────────────────
    print(f"\n── (A) 15-age recovery ──")
    fit_A = fit_mode("A", jnp.asarray(obs_A), setup)
    print(f"  best NLL={fit_A['best_nll']:.6e}  "
          f"β_recovered={[round(x,4) for x in fit_A['beta_recovered']]}  "
          f"NLL std={fit_A['nll_std']:.3e}  wall={fit_A['wall_sec']:.1f}s")
    err_A = np.array(fit_A["beta_recovered"]) - BETA_TRUE
    print(f"  channel error (recovered − true): "
          f"{[round(float(x),5) for x in err_A]}")

    print(f"\n── (B) 6-age HIRA recovery ──")
    fit_B = fit_mode("B", jnp.asarray(obs_B), setup)
    print(f"  best NLL={fit_B['best_nll']:.6e}  "
          f"β_recovered={[round(x,4) for x in fit_B['beta_recovered']]}  "
          f"NLL std={fit_B['nll_std']:.3e}  wall={fit_B['wall_sec']:.1f}s")
    err_B = np.array(fit_B["beta_recovered"]) - BETA_TRUE
    print(f"  channel error (recovered − true): "
          f"{[round(float(x),5) for x in err_B]}")

    # ─── β_w likelihood profile in condition B ──────────────
    print(f"\n── β_w profile scan (condition B) ──")
    beta_B_best = np.array(fit_B["beta_recovered"])
    scan_bw = np.linspace(PROFILE_LO, PROFILE_HI, PROFILE_N)
    obs_B_j = jnp.asarray(obs_B)
    w_B_j = jnp.ones_like(obs_B_j)

    def nll_B_profile(bw):
        b4 = np.array(beta_B_best)
        b4[1] = bw
        pred = predict_B(jnp.asarray(b4), setup)
        return float(poisson_nll_jax(obs_B_j, pred, w_B_j, min_rate=0.01))

    profile_B = np.array([nll_B_profile(float(bw)) for bw in scan_bw])
    idx_min = int(np.argmin(profile_B))
    print(f"  scan β_w over {PROFILE_N} points [{PROFILE_LO}, {PROFILE_HI}]")
    print(f"  NLL min at β_w={scan_bw[idx_min]:.4f}  (true β_w={BETA_TRUE[1]:.4f})")
    print(f"  NLL(β_w=true)={nll_B_profile(float(BETA_TRUE[1])):.4e}   "
          f"NLL(β_w=min_scan)={profile_B[idx_min]:.4e}")

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "beta_true": BETA_TRUE.tolist(),
            "R0_true_ngm": r0_true,
            "phi": "ones(15)",
            "n_weeks": int(setup["n_weeks"]),
            "obs_A_sum": float(obs_A.sum()),
            "obs_B_sum": float(obs_B.sum()),
            "fit_A": fit_A,
            "fit_B": fit_B,
            "profile_B_bw_scan": scan_bw.tolist(),
            "profile_B_nll": profile_B.tolist(),
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    channels = ["home", "work", "school", "other"]
    x = np.arange(4)
    w = 0.28
    axes[0].bar(x - w, BETA_TRUE, w, label="β_true", color="#333333")
    axes[0].bar(x, fit_A["beta_recovered"], w,
                 label="A (15-age)", color="#1a5490")
    axes[0].bar(x + w, fit_B["beta_recovered"], w,
                 label="B (6-age HIRA)", color="#c0392b")
    axes[0].set_xticks(x); axes[0].set_xticklabels(channels)
    axes[0].set_ylabel("β")
    axes[0].set_title(
        f"Recovery vs truth  (β_w true={BETA_TRUE[1]:.3f})\n"
        f"A best NLL={fit_A['best_nll']:.2e}, B best NLL={fit_B['best_nll']:.2e}"
    )
    axes[0].grid(True, alpha=0.3, axis="y"); axes[0].legend(fontsize=9)
    axes[0].axhline(BETA_BOUNDS[0][0], color="grey", ls=":", lw=1)

    axes[1].plot(scan_bw, profile_B, "-", color="#c0392b", lw=1.8)
    axes[1].axvline(BETA_TRUE[1], color="#333333", ls="--", lw=1,
                    label=f"β_w true={BETA_TRUE[1]}")
    axes[1].axvline(fit_B["beta_recovered"][1], color="#1a5490", ls=":", lw=1.5,
                    label=f"B best={fit_B['beta_recovered'][1]:.3f}")
    axes[1].set_xlabel("β_w  (others fixed at B best)")
    axes[1].set_ylabel("Poisson NLL")
    axes[1].set_title(f"Condition B: β_w likelihood profile\n"
                        f"scan {PROFILE_N} pts [{PROFILE_LO}, {PROFILE_HI}]")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    fig.suptitle(f"C1 work-recovery — 15-age (A) vs 6-age HIRA (B) — "
                  f"{SEASON_LABEL}, noise=0")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
