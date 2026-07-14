"""φ U-shape fixed vs φ=ones, free-channel β_4 fit (no channel prior).

Setup: single season 2019-2020, γ_report = CDC center [0.40, 0.18, 0.25],
R(0) = default [.10×4, .30×6, .45×3, .65×2], vaccine A-fix on
(cov_eff = −ln(1 − cov)), HOLIDAY on, AMP = 0.9, φ FIXED (no sampling, no
smoothing). β_4 (+ phi_nb) free via L-BFGS multi-start 12. NB observation.
No channel prior / pin.

Two φ conditions compared:
  ones     : φ_full = [1.0] × 15
  ushape   : φ_full = [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
                       1.05, 1.1, 1.2, 1.3, 1.4, 1.5]
             (docs/parameter_justification.md A.2)

Decision guide (comments only):
- If r(0-5) drops from ~2.15 toward ~1 under U-shape → φ was the main lever
  for infant fit.
- Report other-age r shifts (U-shape affects all 15 ages).
- β_h/β_w: without pin, expect collapse to lower bound. Record the extent
  as a reference for future pin experiments.

Production code is NOT modified.
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
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "phifixed_freechannel.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "phifixed_freechannel.png"
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

PHI_ONES = np.ones(15, dtype=np.float64)
PHI_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5],
    dtype=np.float64,
)

N_STARTS = 12
START_SEED = 23
PEAK_HALF_WIN = 2
COVERAGE_CAP = 0.99

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_setup():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy

    seed_15 = estimate_initial_infected_from_hira(
        SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    tgt = load_hira_target_by_age(
        SEASON_LABEL, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]

    cov_eff = correct_coverage(np.asarray(vax.annual_coverage, dtype=np.float64))
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
        annual_coverage=jnp.asarray(cov_eff),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared.update(HOLIDAY)
    gamma_15 = jnp.asarray(GAMMA_CDC)

    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0,
    ))
    ngm_default = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )
    return dict(
        shared=shared, state0=state0, gamma_15=gamma_15,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
    )


def predict_hira(beta_4, phi_full, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def build_loss(phi_full_j, setup):
    def loss(x):
        beta_4 = x[:4]; phi_nb = x[4]
        pred = predict_hira(beta_4, phi_full_j, setup)
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


def make_starts(n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = [
        np.array([0.10, 0.10, 0.10, 0.10, 10.0]),
        np.array([0.05, 0.05, 0.05, 0.15, 10.0]),
        np.array([0.07, 0.07, 0.20, 0.10, 10.0]),
        np.array([0.07, 0.07, 0.05, 0.20, 10.0]),
        np.array([0.15, 0.15, 0.05, 0.05, 10.0]),
        np.array([0.02, 0.20, 0.05, 0.10, 10.0]),
    ]
    starts = list(base)
    while len(starts) < n_starts:
        b = rng.uniform(0.02, 0.20, 4)
        pn = rng.uniform(2.0, 20.0)
        starts.append(np.concatenate([b, np.array([pn])]))
    return starts[:n_starts]


def peak_window_sum(arr_1d, peak_w, half):
    lo = max(0, peak_w - half)
    hi = min(arr_1d.shape[0], peak_w + half + 1)
    return float(arr_1d[lo:hi].sum())


def per_age_ratios(pred, setup):
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    obs_m = np.where(mask[:, None], obs, -1e18)
    pred_m = np.where(mask[:, None], pred, -1e18)
    rows = {}
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        obs_pw = int(np.argmax(obs_m[:, ai]))
        mdl_pw = int(np.argmax(pred_m[:, ai]))
        obs_sum = peak_window_sum(obs[:, ai], obs_pw, PEAK_HALF_WIN)
        mdl_sum = peak_window_sum(pred[:, ai], mdl_pw, PEAK_HALF_WIN)
        rows[ag] = dict(
            obs_peak_week=obs_pw, model_peak_week=mdl_pw,
            phase_offset_weeks=mdl_pw - obs_pw,
            ratio=obs_sum / max(mdl_sum, 1.0),
        )
    return rows


def fit_condition(name: str, phi_full_np: np.ndarray, setup) -> dict:
    phi_full_j = jnp.asarray(phi_full_np)
    fg = build_loss(phi_full_j, setup)
    bounds = BETA_BOUNDS + [PHI_NB_BOUNDS]
    starts = make_starts(N_STARTS, START_SEED)

    best = None
    per_start_nll = []
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=300, ftol=1e-9, gtol=1e-6))
            nll = float(res.fun)
        except Exception as e:
            print(f"      [warn] start {i} failed: {e}", flush=True)
            continue
        per_start_nll.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0

    beta_4 = np.array(best["x"][:4])
    phi_nb = float(best["x"][4])
    pred = np.asarray(predict_hira(beta_4, phi_full_j, setup))
    ratios = per_age_ratios(pred, setup)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        phi_full_j,
    ))

    nll_arr = np.array(per_start_nll)
    return dict(
        condition=name,
        phi_full_15=[float(x) for x in phi_full_np],
        beta_4=[float(x) for x in beta_4], phi_nb=phi_nb,
        nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios, wall_sec=float(wall),
        pred_hira=pred.tolist(),
    )


def main():
    print("=" * 78, flush=True)
    print(f"DIAGNOSE: φ U-shape vs ones, free channel β_4  —  {SEASON_LABEL}",
          flush=True)
    print(f"  γ_report = CDC center [0.40, 0.18, 0.25]", flush=True)
    print(f"  R(0) = default   vaccine A fix on (cov_eff = −ln(1 − cov))",
          flush=True)
    print(f"  free: β_4 + phi_nb   φ FIXED   no channel prior", flush=True)
    print(f"  multi-start {N_STARTS}", flush=True)
    print(f"  PHI_USHAPE = {PHI_USHAPE.tolist()}", flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    conditions = [
        ("ones", PHI_ONES),
        ("ushape", PHI_USHAPE),
    ]
    all_results = []
    for name, phi in conditions:
        print(f"\n── fit φ = {name} ──", flush=True)
        r = fit_condition(name, phi, setup)
        all_results.append(r)
        print(f"    NLL={r['nll']:.4e}  β_4={[round(x,4) for x in r['beta_4']]}"
              f"  R0_ngm={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
              f"wall={r['wall_sec']:.1f}s", flush=True)
        print(f"    NLL 12 starts std={r['nll_std']:.3e}  "
              f"(min={r['nll_min']:.4e}, max={r['nll_max']:.4e})", flush=True)
        print(f"    per-age r: ", flush=True)
        for ag in HIRA_AGE_GROUPS:
            print(f"      {ag:>6s}: {r['per_age'][ag]['ratio']:.2f}",
                  flush=True)

    # Console summary
    print("\n" + "=" * 78, flush=True)
    print("  Summary — β/NLL + per-age r", flush=True)
    print(f"  {'cond':>8s}  {'β_h':>7s} {'β_w':>7s} {'β_s':>7s} {'β_o':>7s}  "
          f"{'NLL':>10s}  {'R0':>5s}  {'std':>8s}", flush=True)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['condition']:>8s}  {b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f}  "
              f"{b[3]:.4f}  {r['nll']:.4e}  {r['R0_ngm']:.3f}  "
              f"{r['nll_std']:.2e}", flush=True)

    print("\n  Per-age r  (obs_peak±2w / model_peak±2w)", flush=True)
    header = f"  {'cond':>8s}  " + "  ".join(f"{ag:>7s}"
                                                for ag in HIRA_AGE_GROUPS)
    print(header, flush=True)
    for r in all_results:
        row = f"  {r['condition']:>8s}  "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>7.2f}"
        print(row, flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3": [0.40, 0.18, 0.25],
                "R0_IMMUNITY_default": [float(x) for x in R0_IMMUNITY_PROFILE],
                "PHI_USHAPE": PHI_USHAPE.tolist(),
                "vaccine": "A_fix cov_eff = -ln(1-cov)",
                "channel_prior": "NONE",
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # Figure: per-age r bars, both conditions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ages = HIRA_AGE_GROUPS
    xa = np.arange(len(ages))
    bw = 0.35

    ax = axes[0]
    for i, r in enumerate(all_results):
        vals = [r["per_age"][ag]["ratio"] for ag in ages]
        ax.bar(xa + (i - 0.5) * bw, vals, bw,
                color="#888888" if r["condition"] == "ones" else "#1a5490",
                label=f"φ={r['condition']}   NLL={r['nll']:.3e}",
                alpha=0.85)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(ages)
    ax.set_ylabel("r = obs / model")
    ax.set_title("Per-age r  (free β_4, no channel prior)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    channels = ["home", "work", "school", "other"]
    xc = np.arange(4)
    for i, r in enumerate(all_results):
        ax.bar(xc + (i - 0.5) * bw, r["beta_4"], bw,
                color="#888888" if r["condition"] == "ones" else "#1a5490",
                label=f"φ={r['condition']}", alpha=0.85)
    ax.axhline(0.001, color="grey", ls=":", lw=1,
                label="lower bound 0.001")
    ax.set_xticks(xc); ax.set_xticklabels(channels)
    ax.set_ylabel("β")
    ax.set_title("β_4  (no channel prior → pin-off)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"φ U-shape vs ones, free-channel β  —  {SEASON_LABEL}")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
