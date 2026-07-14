"""γ_report sensitivity — is elder fit driven by reporting fraction, not R(0)?

Fix R(0) at baseline [.10×4, .30×6, .45×3, .65×2], vaccine A-fix on
(coverage_eff = −ln(1 − cov)), φ = ones(15). Sweep γ_report_elder × γ_report_adult
and refit β_4 (+ phi_nb). If γ_elder alone recovers r(65+) ≈ 1, elder fit is
a reporting issue and prior R(0) sweeps were operating on the wrong axis.

Grid:
  γ_elder ∈ {0.18, 0.25 (current), 0.35, 0.50}  (idx 13-14)
  γ_adult ∈ {0.18 (current), 0.25}              (idx 4-12)
  γ_child = 0.40 (fixed)                         (idx 0-3)
→ 4 × 2 = 8 combos.

Baseline reference (buggy production, γ_elder=0.25 γ_adult=0.18) from
diagnose_vaccine_doublecount.py condition (a): r = 2.15/1.05/0.72/0.86/0.86/0.97.

Decision guide (comments only):
- γ_elder ↑ → r(65+) approaches 1 → elder fit is a reporting problem;
  R(0) sweeps were treating the wrong variable.
- r(65+) invariant to γ_elder → not reporting; return to structural (R(0)) analysis.
- γ_adult ↑ → does it fix r(45-64)?

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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "gamma_report_sens.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "gamma_report_sens.png"
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
GAMMA_CHILD_FIXED = 0.40

GAMMA_ELDER_GRID = [0.18, 0.25, 0.35, 0.50]
GAMMA_ADULT_GRID = [0.18, 0.25]

N_STARTS = 12
START_SEED = 23
PEAK_HALF_WIN = 2
COVERAGE_CAP = 0.99

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)

BASELINE_R = {"0-5": 2.15, "6-11": 1.05, "12-17": 0.72,
              "18-44": 0.86, "45-64": 0.86, "65+": 0.97}
BASELINE_BETA = [0.0010, 0.0010, 0.0362, 0.1123]
BASELINE_NLL = 1.2158e+03
BASELINE_R0 = 2.035


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    cov_clip = np.minimum(cov_15, COVERAGE_CAP)
    return -np.log(1.0 - cov_clip)


def build_gamma_15(gamma_child: float, gamma_adult: float,
                    gamma_elder: float) -> np.ndarray:
    return np.concatenate([
        np.full(4, gamma_child),
        np.full(9, gamma_adult),
        np.full(2, gamma_elder),
    ])


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

    cov_default = np.asarray(vax.annual_coverage, dtype=np.float64)
    cov_eff = correct_coverage(cov_default)

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
        shared=shared, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
    )


def predict_hira(beta_4, phi_full, gamma_15_j, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc, gamma_15_j,
                                          n_weeks=setup["n_weeks"])


def build_loss(gamma_15_j, setup):
    phi_full = jnp.ones(15)

    def loss(x):
        beta_4 = x[:4]; phi_nb = x[4]
        pred = predict_hira(beta_4, phi_full, gamma_15_j, setup)
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


def fit_combo(gamma_adult: float, gamma_elder: float, setup) -> dict:
    gamma_15 = build_gamma_15(GAMMA_CHILD_FIXED, gamma_adult, gamma_elder)
    gamma_15_j = jnp.asarray(gamma_15)
    fg = build_loss(gamma_15_j, setup)
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
    pred_j = predict_hira(beta_4, phi_full, gamma_15_j, setup)
    pred = np.asarray(pred_j)
    ratios = per_age_ratios(pred, setup)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        jnp.asarray(phi_full),
    ))

    nll_arr = np.array(per_start_nll)
    return dict(
        gamma_child=GAMMA_CHILD_FIXED,
        gamma_adult=gamma_adult, gamma_elder=gamma_elder,
        gamma_15=[float(x) for x in gamma_15],
        beta_4=[float(x) for x in beta_4], phi_nb=phi_nb,
        nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios, wall_sec=float(wall),
    )


def main():
    print("=" * 78, flush=True)
    print(f"DIAGNOSE: γ_report sensitivity (elder × adult)  —  {SEASON_LABEL}",
          flush=True)
    print(f"  free: β_4 + phi_nb   φ = ones(15) FIXED   R(0) = default", flush=True)
    print(f"  vaccine A fix on (cov_eff = −ln(1 − cov))", flush=True)
    print(f"  γ_child fixed = {GAMMA_CHILD_FIXED}", flush=True)
    print(f"  γ_adult ∈ {GAMMA_ADULT_GRID}   γ_elder ∈ {GAMMA_ELDER_GRID}",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    print(f"\n  Baseline (buggy prod, γ_adult=0.18 γ_elder=0.25):", flush=True)
    print(f"    β_4  = {BASELINE_BETA}   NLL = {BASELINE_NLL:.4e}   "
          f"R0 = {BASELINE_R0:.3f}", flush=True)
    print(f"    r    = {BASELINE_R}", flush=True)

    all_results = []
    for g_ad in GAMMA_ADULT_GRID:
        for g_el in GAMMA_ELDER_GRID:
            print(f"\n── γ_adult={g_ad}  γ_elder={g_el} ──", flush=True)
            r = fit_combo(g_ad, g_el, setup)
            all_results.append(r)
            print(f"    NLL={r['nll']:.4e}  β_4={[round(x,4) for x in r['beta_4']]}"
                  f"  R0_ngm={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
                  f"wall={r['wall_sec']:.1f}s", flush=True)
            print(f"    NLL 12 starts std={r['nll_std']:.3e}", flush=True)
            print(f"    ★ r(45-64)={r['per_age']['45-64']['ratio']:.2f}  "
                  f"r(65+)={r['per_age']['65+']['ratio']:.2f}", flush=True)
            print(f"    per-age r: ", flush=True)
            for ag in HIRA_AGE_GROUPS:
                print(f"      {ag:>6s}: {r['per_age'][ag]['ratio']:.2f}",
                      flush=True)

    print("\n" + "=" * 78, flush=True)
    print("  8-combo sweep summary", flush=True)
    print(f"  {'γ_ad':>5s} {'γ_el':>5s}  {'β_h':>7s} {'β_w':>7s} "
          f"{'β_s':>7s} {'β_o':>7s}  {'NLL':>10s}  {'R0':>5s}  {'std':>8s}  "
          f"{'r(45-64)':>9s}  {'r(65+)':>7s}", flush=True)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['gamma_adult']:>5.2f} {r['gamma_elder']:>5.2f}  "
              f"{b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f}  {b[3]:.4f}  "
              f"{r['nll']:.4e}  {r['R0_ngm']:.3f}  {r['nll_std']:.2e}  "
              f"{r['per_age']['45-64']['ratio']:>9.2f}  "
              f"{r['per_age']['65+']['ratio']:>7.2f}", flush=True)

    print("\n  Per-age r for all 8 combos  (obs_peak±2w / model_peak±2w)",
          flush=True)
    header = f"  {'γ_ad':>5s} {'γ_el':>5s}  " + "  ".join(
        f"{ag:>7s}" for ag in HIRA_AGE_GROUPS)
    print(header, flush=True)
    print(f"  {'base':>5s} {'—':>5s}  " + "  ".join(
        f"{BASELINE_R[ag]:>7.2f}" for ag in HIRA_AGE_GROUPS), flush=True)
    for r in all_results:
        row = f"  {r['gamma_adult']:>5.2f} {r['gamma_elder']:>5.2f}  "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>7.2f}"
        print(row, flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "gamma_child_fixed": GAMMA_CHILD_FIXED,
                "gamma_adult_grid": GAMMA_ADULT_GRID,
                "gamma_elder_grid": GAMMA_ELDER_GRID,
                "R0_IMMUNITY_default": [float(x) for x in R0_IMMUNITY_PROFILE],
                "vaccine": "A_fix cov_eff = -ln(1-cov)", "phi": "ones(15) FIXED",
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

    # Figure: 4 panels — r(65+), r(45-64), r(18-44), NLL — indexed by γ_elder×γ_adult
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    n_el = len(GAMMA_ELDER_GRID); n_ad = len(GAMMA_ADULT_GRID)

    def matrix_by(field):
        m = np.zeros((n_ad, n_el))
        for i, ad in enumerate(GAMMA_ADULT_GRID):
            for j, el in enumerate(GAMMA_ELDER_GRID):
                for r in all_results:
                    if r["gamma_adult"] == ad and r["gamma_elder"] == el:
                        m[i, j] = field(r)
                        break
        return m

    panels = [
        ("r(65+)", matrix_by(lambda r: r["per_age"]["65+"]["ratio"]),
         "coolwarm", 0.4, 1.6, 0.97),
        ("r(45-64)", matrix_by(lambda r: r["per_age"]["45-64"]["ratio"]),
         "coolwarm", 0.4, 1.6, 0.86),
        ("r(18-44)", matrix_by(lambda r: r["per_age"]["18-44"]["ratio"]),
         "coolwarm", 0.4, 1.6, 0.86),
        ("NLL", matrix_by(lambda r: r["nll"]), "viridis", None, None, None),
    ]
    for ax, (title, M, cmap, vmin, vmax, target) in zip(axes, panels):
        if vmin is not None:
            im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_el))
        ax.set_xticklabels([f"{v:.2f}" for v in GAMMA_ELDER_GRID])
        ax.set_yticks(range(n_ad))
        ax.set_yticklabels([f"{v:.2f}" for v in GAMMA_ADULT_GRID])
        ax.set_xlabel("γ_elder")
        ax.set_ylabel("γ_adult")
        for i in range(n_ad):
            for j in range(n_el):
                ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center",
                        color="white" if cmap == "viridis" else "black",
                        fontsize=9)
        subt = title + (f"  target {target:.2f}" if target is not None else "")
        ax.set_title(subt)
        fig.colorbar(im, ax=ax, fraction=0.045)

    fig.suptitle(f"γ_report sensitivity — γ_elder × γ_adult  —  "
                  f"{SEASON_LABEL}  (φ fixed, R(0) default, A fix on)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
