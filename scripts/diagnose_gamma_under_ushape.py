"""γ_report sweep under φ U-shape fixed — recover r≈1 across ages.

Setup: single season 2019-2020, R(0)=default, vaccine A-fix on, HOLIDAY on,
AMP=0.9, φ=PHI_USHAPE fixed. β_4 (+phi_nb) free L-BFGS×12, no channel prior.

φ U-shape under baseline γ (0.40/0.18/0.25) produces (from prior diagnostic):
    r = 1.97 / 1.10 / 0.97 / 1.36 / 1.29 / 1.42   (adult/elder over-predict).

Step A — 3×3 adult × elder sweep with γ_child=0.40 fixed:
    γ_adult ∈ {0.18, 0.25, 0.35}
    γ_elder ∈ {0.25, 0.35, 0.45}
Objective: bring r(18-44), r(45-64), r(65+) all near 1.

Step B — at the (γ_adult, γ_elder) combo minimising Σ_{adult,elder ages}
|r − 1| over {18-44, 45-64, 65+}, sweep γ_child ∈ {0.30, 0.40, 0.50} to
check whether r(0-5) responds (φ-shape vs γ diagnosis).

Decision guide (comments only):
- The (γ_a, γ_e) that lands r(18-44)/r(45-64)/r(65+) ≈ 1 = new γ_report base
  candidate under φ U-shape.
- Record β_h / β_w behaviour across the sweep — does channel pin-free
  reveal work/home revival at some γ?
- Step B: if r(0-5) barely moves with γ_child, infant under-prediction is a
  φ-shape issue, not γ.

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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "gamma_under_ushape.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "gamma_under_ushape.png"
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

PHI_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64,
)

GAMMA_CHILD_FIXED = 0.40
GAMMA_ADULT_GRID = [0.18, 0.25, 0.35]
GAMMA_ELDER_GRID = [0.25, 0.35, 0.45]
GAMMA_CHILD_GRID_STEPB = [0.30, 0.40, 0.50]

N_STARTS = 12
START_SEED = 23
PEAK_HALF_WIN = 2
COVERAGE_CAP = 0.99

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_gamma_15(child: float, adult: float, elder: float) -> np.ndarray:
    return np.concatenate([np.full(4, child), np.full(9, adult),
                            np.full(2, elder)])


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


def predict_hira(beta_4, phi_full_j, gamma_15_j, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_j
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc, gamma_15_j,
                                          n_weeks=setup["n_weeks"])


def build_loss(phi_full_j, gamma_15_j, setup):
    def loss(x):
        beta_4 = x[:4]; phi_nb = x[4]
        pred = predict_hira(beta_4, phi_full_j, gamma_15_j, setup)
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


def fit_combo(step: str, child: float, adult: float, elder: float, setup) -> dict:
    gamma_15 = build_gamma_15(child, adult, elder)
    gamma_15_j = jnp.asarray(gamma_15)
    phi_j = jnp.asarray(PHI_USHAPE)
    fg = build_loss(phi_j, gamma_15_j, setup)
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
    pred = np.asarray(predict_hira(beta_4, phi_j, gamma_15_j, setup))
    ratios = per_age_ratios(pred, setup)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        phi_j,
    ))
    nll_arr = np.array(per_start_nll)
    return dict(
        step=step,
        gamma_child=child, gamma_adult=adult, gamma_elder=elder,
        beta_4=[float(x) for x in beta_4], phi_nb=phi_nb,
        nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios, wall_sec=float(wall),
    )


def adult_elder_score(rec: dict) -> float:
    """Sum |r − 1| over the 3 adult/elder HIRA groups."""
    return (abs(rec["per_age"]["18-44"]["ratio"] - 1.0)
            + abs(rec["per_age"]["45-64"]["ratio"] - 1.0)
            + abs(rec["per_age"]["65+"]["ratio"] - 1.0))


def main():
    print("=" * 78, flush=True)
    print(f"DIAGNOSE: γ_report sweep under φ U-shape  —  {SEASON_LABEL}",
          flush=True)
    print(f"  φ FIXED = {PHI_USHAPE.tolist()}", flush=True)
    print(f"  A-fix cov, R(0) default, no channel prior", flush=True)
    print(f"  Step A: γ_child={GAMMA_CHILD_FIXED}, γ_adult ∈ {GAMMA_ADULT_GRID},"
          f" γ_elder ∈ {GAMMA_ELDER_GRID}", flush=True)
    print(f"  Step B: γ_child ∈ {GAMMA_CHILD_GRID_STEPB} at best (adult, elder)",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    # Step A
    all_results = []
    for a in GAMMA_ADULT_GRID:
        for e in GAMMA_ELDER_GRID:
            print(f"\n── [A] γ=(child={GAMMA_CHILD_FIXED}, adult={a}, "
                  f"elder={e}) ──", flush=True)
            r = fit_combo("A", GAMMA_CHILD_FIXED, a, e, setup)
            all_results.append(r)
            print(f"    NLL={r['nll']:.4e}  "
                  f"β_4={[round(x,4) for x in r['beta_4']]}  "
                  f"R0={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
                  f"std={r['nll_std']:.2e}", flush=True)
            print(f"    ★ β_w={r['beta_4'][1]:.4f}  β_h={r['beta_4'][0]:.4f}",
                  flush=True)
            print(f"    per-age r: " + "  ".join(
                f"{ag}={r['per_age'][ag]['ratio']:.2f}"
                for ag in HIRA_AGE_GROUPS), flush=True)

    # Find best (adult, elder) by adult_elder_score
    best_A = min(all_results, key=adult_elder_score)
    ba, be = best_A["gamma_adult"], best_A["gamma_elder"]
    print(f"\n[Step A winner] (γ_adult, γ_elder) = ({ba}, {be})   "
          f"score = {adult_elder_score(best_A):.3f}", flush=True)
    print(f"  r(18-44)={best_A['per_age']['18-44']['ratio']:.2f}  "
          f"r(45-64)={best_A['per_age']['45-64']['ratio']:.2f}  "
          f"r(65+)={best_A['per_age']['65+']['ratio']:.2f}", flush=True)

    # Step B — γ_child sweep at best (ba, be); dedupe γ_child=0.40 if applicable
    for c in GAMMA_CHILD_GRID_STEPB:
        # Dedupe: if (c, ba, be) already in results, reuse it
        matched = next((r for r in all_results
                          if r["gamma_child"] == c and r["gamma_adult"] == ba
                          and r["gamma_elder"] == be), None)
        if matched is not None:
            rec = dict(matched); rec["step"] = "B"
            all_results.append(rec)
            print(f"\n── [B] γ=(child={c}, adult={ba}, elder={be})  "
                  f"[dedup A] ──", flush=True)
            print(f"    per-age r: " + "  ".join(
                f"{ag}={rec['per_age'][ag]['ratio']:.2f}"
                for ag in HIRA_AGE_GROUPS), flush=True)
            continue
        print(f"\n── [B] γ=(child={c}, adult={ba}, elder={be}) ──", flush=True)
        r = fit_combo("B", c, ba, be, setup)
        all_results.append(r)
        print(f"    NLL={r['nll']:.4e}  "
              f"β_4={[round(x,4) for x in r['beta_4']]}  "
              f"R0={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
              f"std={r['nll_std']:.2e}", flush=True)
        print(f"    ★ β_w={r['beta_4'][1]:.4f}  β_h={r['beta_4'][0]:.4f}",
              flush=True)
        print(f"    per-age r: " + "  ".join(
            f"{ag}={r['per_age'][ag]['ratio']:.2f}"
            for ag in HIRA_AGE_GROUPS), flush=True)

    # Console summary
    print("\n" + "=" * 78, flush=True)
    print("  All combos — β + NLL + β_w focus", flush=True)
    print(f"  {'st':>3s}  {'γ_c':>5s} {'γ_a':>5s} {'γ_e':>5s}  "
          f"{'β_h':>7s} {'β_w':>7s} {'β_s':>7s} {'β_o':>7s}  "
          f"{'NLL':>10s}  {'R0':>5s}  {'std':>8s}  {'score':>6s}",
          flush=True)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['step']:>3s}  {r['gamma_child']:>5.2f} "
              f"{r['gamma_adult']:>5.2f} {r['gamma_elder']:>5.2f}  "
              f"{b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f}  {b[3]:.4f}  "
              f"{r['nll']:.4e}  {r['R0_ngm']:.3f}  {r['nll_std']:.2e}  "
              f"{adult_elder_score(r):>6.3f}", flush=True)

    print("\n  Per-age r (obs_peak±2w / model_peak±2w)", flush=True)
    header = f"  {'st':>3s}  {'γ_c':>5s} {'γ_a':>5s} {'γ_e':>5s}  " + \
             "  ".join(f"{ag:>7s}" for ag in HIRA_AGE_GROUPS)
    print(header, flush=True)
    for r in all_results:
        row = (f"  {r['step']:>3s}  {r['gamma_child']:>5.2f} "
               f"{r['gamma_adult']:>5.2f} {r['gamma_elder']:>5.2f}  ")
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>7.2f}"
        print(row, flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "PHI_USHAPE": PHI_USHAPE.tolist(),
                "R0_IMMUNITY_default": [float(x) for x in R0_IMMUNITY_PROFILE],
                "gamma_child_step_A": GAMMA_CHILD_FIXED,
                "gamma_adult_grid": GAMMA_ADULT_GRID,
                "gamma_elder_grid": GAMMA_ELDER_GRID,
                "gamma_child_grid_step_B": GAMMA_CHILD_GRID_STEPB,
                "vaccine": "A_fix cov_eff = -ln(1-cov)",
                "channel_prior": "NONE",
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
            "step_A_winner": {
                "gamma_adult": ba, "gamma_elder": be,
                "score": adult_elder_score(best_A),
            },
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # Figure: 4 heatmaps for r(18-44), r(45-64), r(65+), NLL over adult × elder
    step_A = [r for r in all_results if r["step"] == "A"]
    n_a = len(GAMMA_ADULT_GRID); n_e = len(GAMMA_ELDER_GRID)

    def mat_by(field):
        M = np.zeros((n_a, n_e))
        for i, a in enumerate(GAMMA_ADULT_GRID):
            for j, e in enumerate(GAMMA_ELDER_GRID):
                for r in step_A:
                    if r["gamma_adult"] == a and r["gamma_elder"] == e:
                        M[i, j] = field(r)
                        break
        return M

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    panels = [
        ("r(18-44)", mat_by(lambda r: r["per_age"]["18-44"]["ratio"]),
         "coolwarm", 0.4, 1.6, 1.0),
        ("r(45-64)", mat_by(lambda r: r["per_age"]["45-64"]["ratio"]),
         "coolwarm", 0.4, 1.6, 1.0),
        ("r(65+)", mat_by(lambda r: r["per_age"]["65+"]["ratio"]),
         "coolwarm", 0.4, 1.6, 1.0),
        ("NLL", mat_by(lambda r: r["nll"]), "viridis", None, None, None),
    ]
    for ax, (title, M, cmap, vmin, vmax, target) in zip(axes, panels):
        if vmin is not None:
            im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_e))
        ax.set_xticklabels([f"{v:.2f}" for v in GAMMA_ELDER_GRID])
        ax.set_yticks(range(n_a))
        ax.set_yticklabels([f"{v:.2f}" for v in GAMMA_ADULT_GRID])
        ax.set_xlabel("γ_elder"); ax.set_ylabel("γ_adult")
        for i in range(n_a):
            for j in range(n_e):
                ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center",
                        color="white" if cmap == "viridis" else "black",
                        fontsize=9)
        subt = title + (f"  target {target:.2f}" if target is not None else "")
        ax.set_title(subt)
        fig.colorbar(im, ax=ax, fraction=0.045)

    fig.suptitle(f"γ_report sweep under φ U-shape  —  {SEASON_LABEL}  "
                  f"(γ_child = {GAMMA_CHILD_FIXED}, no channel pin)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
