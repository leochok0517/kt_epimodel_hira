"""R(0) natural-immunity sweep on 50-64 × 65+ under single-count vaccine.

Fix vaccine bug A globally (coverage_eff = −ln(1 − cov)) and sweep the elder
band R(0) values to find natural-immunity combinations that recover the
baseline age-fit under φ = ones(15) fixed.

Grid:
  R0_5064 ∈ {0.30, 0.35, 0.40}   idx 10-12
  R0_65p  ∈ {0.45, 0.50, 0.55}   idx 13-14
  fixed: 0-19 = 0.10, 20-49 = 0.30
→ 9 combos.

Reference baseline (buggy production): from diagnose_vaccine_doublecount.py
condition (a): r = 2.15 / 1.05 / 0.72 / 0.86 / 0.86 / 0.97 for
(0-5, 6-11, 12-17, 18-44, 45-64, 65+).

Decision guide (comments only):
- Combo whose r(45-64) ≈ 0.86 AND r(65+) ≈ 0.97 = the R(0) natural-only value
  that matches production fit under single-count vaccine (bug A + B fixed).
- Exclude combos with large multistart NLL std (unstable local minima).

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
    simulate_jax, daily_new_infection_by_age_jax, IDX_S, IDX_V, IDX_R,
)
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "R0_immunity_sweep.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "R0_immunity_sweep.png"
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

N_STARTS = 12
START_SEED = 23
PEAK_HALF_WIN = 2
COVERAGE_CAP = 0.99

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)

R0_5064_GRID = [0.30, 0.35, 0.40]     # idx 10-12
R0_65P_GRID = [0.45, 0.50, 0.55]      # idx 13-14

# Baseline reference (buggy production, from diagnose_vaccine_doublecount.py)
BASELINE_R = {"0-5": 2.15, "6-11": 1.05, "12-17": 0.72,
              "18-44": 0.86, "45-64": 0.86, "65+": 0.97}
BASELINE_BETA = [0.0010, 0.0010, 0.0362, 0.1123]
BASELINE_NLL = 1.2158e+03
BASELINE_R0 = 2.035


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    cov_clip = np.minimum(cov_15, COVERAGE_CAP)
    return -np.log(1.0 - cov_clip)


def build_immunity(r_5064: float, r_65p: float) -> np.ndarray:
    return np.array(
        [0.10] * 4 + [0.30] * 6 + [r_5064] * 3 + [r_65p] * 2,
        dtype=np.float64,
    )


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

    shared_base = dict(
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
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared_base.update(HOLIDAY)
    gamma_15 = jnp.asarray(GAMMA_CDC)

    ngm_default = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )

    return dict(
        shared_base=shared_base, gamma_15=gamma_15,
        pop_15=pop_15, seed_15=seed_15, vax=vax,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
    )


def build_state0(immunity_15: np.ndarray, setup) -> jnp.ndarray:
    return jnp.asarray(_build_initial_state_with_age_seed(
        setup["pop_15"], setup["seed_15"], seed_e_factor=0.5,
        initial_immunity=immunity_15, initial_vaccinated_fraction=0.0,
    ))


def build_shared(cov_eff_15: np.ndarray, setup):
    kw = dict(setup["shared_base"])
    kw["annual_coverage"] = jnp.asarray(cov_eff_15)
    return kw


def predict_hira(beta_4, phi_full, state0, shared, setup):
    kw = dict(shared)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    st = simulate_jax(state0, **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"]), st


def build_loss(state0, shared, setup):
    phi_full = jnp.ones(15)

    def loss(x):
        beta_4 = x[:4]; phi_nb = x[4]
        pred, _ = predict_hira(beta_4, phi_full, state0, shared, setup)
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


def s_end_pct(states: jnp.ndarray, age_idx: int) -> float:
    S = np.asarray(states[:, IDX_S, age_idx, :].sum(axis=-1))
    V = np.asarray(states[:, IDX_V, age_idx, :].sum(axis=-1))
    R = np.asarray(states[:, IDX_R, age_idx, :].sum(axis=-1))
    E = np.asarray(states[:, 2, age_idx, :].sum(axis=-1))
    I = np.asarray(states[:, 3, age_idx, :].sum(axis=-1))
    N = S + V + E + I + R
    return 100.0 * float(S[-1]) / max(float(N[0]), 1e-9)


def fit_combo(r_5064: float, r_65p: float, cov_eff_15, setup) -> dict:
    imm = build_immunity(r_5064, r_65p)
    state0 = build_state0(imm, setup)
    shared = build_shared(cov_eff_15, setup)
    fg = build_loss(state0, shared, setup)
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
    phi_full = jnp.ones(15)
    pred_j, states = predict_hira(beta_4, phi_full, state0, shared, setup)
    pred = np.asarray(pred_j)

    ratios = per_age_ratios(pred, setup)
    s_end_5054 = s_end_pct(states, age_idx=10)
    s_end_5559 = s_end_pct(states, age_idx=11)
    s_end_6064 = s_end_pct(states, age_idx=12)
    s_end_6569 = s_end_pct(states, age_idx=13)
    s_end_70p  = s_end_pct(states, age_idx=14)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        jnp.asarray(phi_full),
    ))

    nll_arr = np.array(per_start_nll)
    return dict(
        R0_5064=r_5064, R0_65p=r_65p,
        immunity_15=[float(x) for x in imm],
        beta_4=[float(x) for x in beta_4],
        phi_nb=phi_nb, nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios,
        S_end_pct_5054=s_end_5054,
        S_end_pct_5559=s_end_5559,
        S_end_pct_6064=s_end_6064,
        S_end_pct_6569=s_end_6569,
        S_end_pct_70p=s_end_70p,
        wall_sec=float(wall),
    )


def main():
    print("=" * 78, flush=True)
    print(f"DIAGNOSE: R(0) natural-immunity sweep (50-64 × 65+)  —  {SEASON_LABEL}",
          flush=True)
    print(f"  free: β_4 + phi_nb   φ = ones(15) FIXED", flush=True)
    print(f"  vaccine A fix on globally (cov_eff = −ln(1 − cov))", flush=True)
    print(f"  grid: R0_5064 ∈ {R0_5064_GRID}   R0_65p ∈ {R0_65P_GRID}",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()
    cov_default = np.asarray(setup["vax"].annual_coverage, dtype=np.float64)
    cov_eff = correct_coverage(cov_default)
    print(f"  cov (14, 70+): {cov_default[14]:.3f}  →  cov_eff: {cov_eff[14]:.4f}",
          flush=True)

    print(f"\n  Baseline (buggy prod, from vaccine_doublecount.py):",
          flush=True)
    print(f"    β_4  = {BASELINE_BETA}   NLL = {BASELINE_NLL:.4e}   "
          f"R0 = {BASELINE_R0:.3f}", flush=True)
    print(f"    r    = {BASELINE_R}", flush=True)

    all_results = []
    for r_65p in R0_65P_GRID:
        for r_5064 in R0_5064_GRID:
            print(f"\n── R0_5064={r_5064}  R0_65p={r_65p} ──", flush=True)
            r = fit_combo(r_5064, r_65p, cov_eff, setup)
            all_results.append(r)
            print(f"    NLL={r['nll']:.4e}  β_4={[round(x,4) for x in r['beta_4']]}"
                  f"  R0_ngm={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
                  f"wall={r['wall_sec']:.1f}s", flush=True)
            print(f"    NLL 12 starts std={r['nll_std']:.3e}  "
                  f"(min={r['nll_min']:.4e}, max={r['nll_max']:.4e})",
                  flush=True)
            print(f"    ★ r(45-64)={r['per_age']['45-64']['ratio']:.2f}  "
                  f"r(65+)={r['per_age']['65+']['ratio']:.2f}", flush=True)
            print(f"    S_end%: 50-54={r['S_end_pct_5054']:.2f}  "
                  f"55-59={r['S_end_pct_5559']:.2f}  "
                  f"60-64={r['S_end_pct_6064']:.2f}  "
                  f"65-69={r['S_end_pct_6569']:.2f}  "
                  f"70+={r['S_end_pct_70p']:.2f}", flush=True)
            print(f"    per-age r all: ", flush=True)
            for ag in HIRA_AGE_GROUPS:
                print(f"      {ag:>6s}: {r['per_age'][ag]['ratio']:.2f}",
                      flush=True)

    # Console summary
    print("\n" + "=" * 78, flush=True)
    print("  9-combo sweep summary", flush=True)
    print(f"  {'R0_5064':>8s} {'R0_65p':>7s}  {'β_h':>7s} {'β_w':>7s} "
          f"{'β_s':>7s} {'β_o':>7s}  {'NLL':>10s}  {'R0':>5s}  {'std':>8s}  "
          f"{'r(45-64)':>9s}  {'r(65+)':>7s}", flush=True)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['R0_5064']:>8.2f} {r['R0_65p']:>7.2f}  "
              f"{b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f}  {b[3]:.4f}  "
              f"{r['nll']:.4e}  {r['R0_ngm']:.3f}  {r['nll_std']:.2e}  "
              f"{r['per_age']['45-64']['ratio']:>9.2f}  "
              f"{r['per_age']['65+']['ratio']:>7.2f}", flush=True)

    # per-age r matrix
    print("\n  Per-age r for all 9 combos (obs_peak±2w / model_peak±2w)",
          flush=True)
    header = f"  {'R0_5064':>8s} {'R0_65p':>7s}  " + "  ".join(
        f"{ag:>7s}" for ag in HIRA_AGE_GROUPS)
    print(header, flush=True)
    print(f"  {'baseline':>8s} {'—':>7s}  " + "  ".join(
        f"{BASELINE_R[ag]:>7.2f}" for ag in HIRA_AGE_GROUPS), flush=True)
    for r in all_results:
        row = f"  {r['R0_5064']:>8.2f} {r['R0_65p']:>7.2f}  "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>7.2f}"
        print(row, flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3": [0.40, 0.18, 0.25],
                "R0_5064_grid": R0_5064_GRID,
                "R0_65p_grid": R0_65P_GRID,
                "R0_fixed_0_19": 0.10, "R0_fixed_20_49": 0.30,
                "annual_coverage_default": cov_default.tolist(),
                "annual_coverage_eff_A_fix": cov_eff.tolist(),
                "coverage_cap": COVERAGE_CAP,
                "phi": "ones(15) FIXED",
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "peak_half_window_weeks": PEAK_HALF_WIN,
                "baseline_reference": {
                    "beta_4": BASELINE_BETA, "NLL": BASELINE_NLL,
                    "R0": BASELINE_R0, "r": BASELINE_R,
                },
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # Heatmap figure: 3×3 grid for r(65+) and r(45-64), plus NLL and std
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n_x = len(R0_5064_GRID); n_y = len(R0_65P_GRID)

    def matrix_by(field):
        m = np.zeros((n_y, n_x))
        for i, y in enumerate(R0_65P_GRID):
            for j, x in enumerate(R0_5064_GRID):
                for r in all_results:
                    if r["R0_5064"] == x and r["R0_65p"] == y:
                        m[i, j] = field(r)
                        break
        return m

    matrices = [
        ("r(65+)", matrix_by(lambda r: r["per_age"]["65+"]["ratio"]),
         "coolwarm", 0.4, 1.4, 0.97),
        ("r(45-64)", matrix_by(lambda r: r["per_age"]["45-64"]["ratio"]),
         "coolwarm", 0.4, 1.4, 0.86),
        ("NLL (best)", matrix_by(lambda r: r["nll"]),
         "viridis", None, None, None),
        ("NLL std across 12 starts", matrix_by(lambda r: r["nll_std"]),
         "magma", None, None, None),
    ]
    for ax, (title, M, cmap, vmin, vmax, target) in zip(axes.flat, matrices):
        if vmin is not None:
            im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_x))
        ax.set_xticklabels([f"{v:.2f}" for v in R0_5064_GRID])
        ax.set_yticks(range(n_y))
        ax.set_yticklabels([f"{v:.2f}" for v in R0_65P_GRID])
        ax.set_xlabel("R0_5064")
        ax.set_ylabel("R0_65p")
        for i in range(n_y):
            for j in range(n_x):
                ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center",
                        color="white" if cmap in ("magma", "viridis")
                        else "black", fontsize=9)
        subtitle = title
        if target is not None:
            subtitle += f"  (baseline target = {target:.2f})"
        ax.set_title(subtitle)
        fig.colorbar(im, ax=ax, fraction=0.045)

    fig.suptitle(f"R(0) natural-immunity sweep, 50-64 × 65+  —  {SEASON_LABEL}"
                  f"  (φ fixed, A fix on)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
