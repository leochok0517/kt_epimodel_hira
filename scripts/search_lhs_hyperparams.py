"""LHS search over (γ_report × φ shape × R(0)) — do work/home revive anywhere?

8-D Latin Hypercube:
  γ_child      ∈ [0.30, 0.50]
  γ_adult      ∈ [0.12, 0.35]
  γ_elder      ∈ [0.25, 0.50]
    constraint : γ_adult < γ_elder ≤ γ_child   (rejection sampling)
  φ child_peak  ∈ [0.75, 1.25]
  φ adult_level ∈ [0.85, 1.15]
  φ elder_rise  ∈ [0.70, 1.30]
  R0_adult     ∈ [0.20, 0.40]   (idx 4-9, 20-49y)
  R0_elder     ∈ [0.45, 0.65]   (idx 13-14, 65+y)
  R0 idx 10-12 (50-64y) = (R0_adult + R0_elder) / 2 linear interp
  R0 idx 0-3 (0-19y)     = 0.10 fixed

150 samples target (with rejection top-up). Per sample: β_4 + phi_nb free
L-BFGS × 8 starts (reduced from 12 for throughput), NB obs, no channel prior.

Success criteria (all three):
  ① work_alive  : β_w > 0.005
  ② home_alive  : β_h > 0.005
  ③ fit_score   : Σ|r − 1| over 6 HIRA groups < 1.5
  ④ NLL         : < 1210
  successful = ① and ② and ③ and ④

Reporting: count successes; count work_alive, home_alive individually;
best sample by fit_score; scatter of β_w vs β_h; scatter NLL vs fit_score.

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
from scipy.stats import qmc

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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "lhs_search.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "lhs_search.png"
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

PHI_BASE_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64,
)

RANGES = dict(
    gamma_child=(0.30, 0.50),
    gamma_adult=(0.12, 0.35),
    gamma_elder=(0.25, 0.50),
    child_peak=(0.75, 1.25),
    adult_level=(0.85, 1.15),
    elder_rise=(0.70, 1.30),
    R0_adult=(0.20, 0.40),
    R0_elder=(0.45, 0.65),
)
DIM_ORDER = list(RANGES.keys())   # 8

N_SAMPLES = 150
N_STARTS = 8
LHS_SEED = 31
START_SEED = 23
PEAK_HALF_WIN = 2
COVERAGE_CAP = 0.99

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)

# Success thresholds
WORK_ALIVE_THR = 0.005
HOME_ALIVE_THR = 0.005
FIT_SCORE_MAX = 1.5
NLL_MAX = 1210.0


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_gamma_15(child: float, adult: float, elder: float) -> np.ndarray:
    return np.concatenate([np.full(4, child), np.full(9, adult),
                            np.full(2, elder)])


def build_phi(child_peak: float, adult_level: float,
               elder_rise: float) -> np.ndarray:
    phi = PHI_BASE_USHAPE.copy()
    phi[0:4] = PHI_BASE_USHAPE[0:4] * child_peak
    for i in [4, 6, 7, 8]:
        phi[i] = PHI_BASE_USHAPE[i] * adult_level
    phi[5] = 1.0
    for i in range(9, 15):
        phi[i] = 1.0 + elder_rise * (PHI_BASE_USHAPE[i] - 1.0)
    return phi


def build_R0_immunity(R0_adult: float, R0_elder: float) -> np.ndarray:
    """0-19 fixed 0.10, 20-49 = R0_adult, 50-64 = linear interp, 65+ = R0_elder."""
    R0_5064 = 0.5 * (R0_adult + R0_elder)
    return np.array(
        [0.10] * 4 + [R0_adult] * 6 + [R0_5064] * 3 + [R0_elder] * 2,
        dtype=np.float64,
    )


def sample_feasible_lhs(n_target: int, seed: int) -> np.ndarray:
    """LHS in [0,1]^8, scale to ranges, reject γ_adult<γ_elder≤γ_child violations."""
    rng = np.random.default_rng(seed)
    acc = []
    batch = 0
    while len(acc) < n_target:
        batch += 1
        # Over-sample: rejection rate expected 40-70%
        n_batch = max(400, (n_target - len(acc)) * 3)
        engine = qmc.LatinHypercube(d=8, seed=int(rng.integers(2**32 - 1)))
        u = engine.random(n=n_batch)
        pts = np.zeros_like(u)
        for i, k in enumerate(DIM_ORDER):
            lo, hi = RANGES[k]
            pts[:, i] = lo + u[:, i] * (hi - lo)
        # Constraint
        gc = pts[:, DIM_ORDER.index("gamma_child")]
        ga = pts[:, DIM_ORDER.index("gamma_adult")]
        ge = pts[:, DIM_ORDER.index("gamma_elder")]
        valid = (ga < ge) & (ge <= gc)
        acc.extend(pts[valid].tolist())
        print(f"    [lhs batch {batch}] {int(valid.sum())}/{n_batch} pass  "
              f"total {len(acc)}/{n_target}", flush=True)
        if batch > 20:
            print(f"    [warn] stopping after 20 batches with {len(acc)} valid",
                  flush=True)
            break
    return np.array(acc[:n_target])


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
        annual_coverage=jnp.asarray(cov_eff),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared_base.update(HOLIDAY)

    return dict(
        shared_base=shared_base,
        pop_15=pop_15, rho_emp=rho_emp, matrices=matrices, seed_15=seed_15,
        disease=disease,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks,
    )


def make_state0(immunity_15: np.ndarray, setup) -> jnp.ndarray:
    return jnp.asarray(_build_initial_state_with_age_seed(
        setup["pop_15"], setup["seed_15"], seed_e_factor=0.5,
        initial_immunity=immunity_15, initial_vaccinated_fraction=0.0,
    ))


def make_ngm(immunity_15, setup):
    return make_ngm_eigvalue_fn(
        pop_15=setup["pop_15"], rho=setup["rho_emp"],
        C_home=setup["matrices"]["C_home"],
        C_work=setup["matrices"]["C_work"],
        C_school=setup["matrices"]["C_school"],
        C_other=setup["matrices"]["C_other"],
        R0_immunity=immunity_15,
        gamma=setup["disease"].gamma, seasonal_factor=1.0 + AMP,
    )


def predict_hira(beta_4, phi_full_j, gamma_15_j, state0, setup):
    kw = dict(setup["shared_base"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_j
    st = simulate_jax(state0, **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc, gamma_15_j,
                                          n_weeks=setup["n_weeks"])


def build_loss(phi_full_j, gamma_15_j, state0, setup):
    def loss(x):
        beta_4 = x[:4]; phi_nb = x[4]
        pred = predict_hira(beta_4, phi_full_j, gamma_15_j, state0, setup)
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
        rows[ag] = dict(ratio=obs_sum / max(mdl_sum, 1.0),
                        phase_offset_weeks=mdl_pw - obs_pw)
    return rows


def evaluate_sample(idx: int, params: dict, setup) -> dict:
    phi = build_phi(params["child_peak"], params["adult_level"],
                     params["elder_rise"])
    gamma_15 = build_gamma_15(params["gamma_child"], params["gamma_adult"],
                                params["gamma_elder"])
    immunity_15 = build_R0_immunity(params["R0_adult"], params["R0_elder"])

    state0 = make_state0(immunity_15, setup)
    ngm_fn = make_ngm(immunity_15, setup)
    phi_j = jnp.asarray(phi)
    gamma_15_j = jnp.asarray(gamma_15)

    fg = build_loss(phi_j, gamma_15_j, state0, setup)
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
        except Exception:
            continue
        per_start_nll.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0

    beta_4 = np.array(best["x"][:4])
    phi_nb = float(best["x"][4])
    pred = np.asarray(predict_hira(beta_4, phi_j, gamma_15_j, state0, setup))
    ratios = per_age_ratios(pred, setup)
    r0_ngm = float(ngm_fn(
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        phi_j,
    ))

    fit_score = sum(abs(ratios[ag]["ratio"] - 1.0) for ag in HIRA_AGE_GROUPS)
    work_alive = beta_4[1] > WORK_ALIVE_THR
    home_alive = beta_4[0] > HOME_ALIVE_THR
    success = (work_alive and home_alive and fit_score < FIT_SCORE_MAX
                and best["nll"] < NLL_MAX)

    nll_arr = np.array(per_start_nll)
    return dict(
        idx=idx, params=params,
        beta_4=[float(x) for x in beta_4], phi_nb=phi_nb,
        nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age_r={ag: ratios[ag]["ratio"] for ag in HIRA_AGE_GROUPS},
        fit_score=float(fit_score),
        work_alive=bool(work_alive),
        home_alive=bool(home_alive),
        success=bool(success),
        wall_sec=float(wall),
    )


def main():
    print("=" * 78, flush=True)
    print(f"LHS SEARCH: (γ × φ × R(0))  —  {SEASON_LABEL}", flush=True)
    print(f"  target samples = {N_SAMPLES}   dims = {DIM_ORDER}", flush=True)
    print(f"  multi-start {N_STARTS}", flush=True)
    print(f"  ranges: {RANGES}", flush=True)
    print(f"  success: work>{WORK_ALIVE_THR} + home>{HOME_ALIVE_THR} + "
          f"fit_score<{FIT_SCORE_MAX} + NLL<{NLL_MAX}", flush=True)
    print("=" * 78, flush=True)

    print("\n[LHS sampling with rejection]", flush=True)
    pts = sample_feasible_lhs(N_SAMPLES, LHS_SEED)
    print(f"  → {len(pts)} feasible samples", flush=True)

    setup = build_setup()

    results = []
    t_start = time.perf_counter()
    for i, row in enumerate(pts):
        params = {k: float(row[j]) for j, k in enumerate(DIM_ORDER)}
        try:
            r = evaluate_sample(i, params, setup)
        except Exception as e:
            print(f"  [warn] sample {i} failed: {e}", flush=True)
            continue
        results.append(r)
        if (i + 1) % 10 == 0 or i < 3:
            n_success = sum(1 for x in results if x["success"])
            n_work = sum(1 for x in results if x["work_alive"])
            n_home = sum(1 for x in results if x["home_alive"])
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(pts) - i - 1)
            print(f"  [{i+1:>3d}/{len(pts)}] success={n_success}  "
                  f"work_alive={n_work}  home_alive={n_home}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    # Summary
    n_success = sum(1 for x in results if x["success"])
    n_work = sum(1 for x in results if x["work_alive"])
    n_home = sum(1 for x in results if x["home_alive"])
    n_both_ch = sum(1 for x in results if x["work_alive"] and x["home_alive"])
    n_fit_only = sum(1 for x in results if x["fit_score"] < FIT_SCORE_MAX)
    n_nll_only = sum(1 for x in results if x["nll"] < NLL_MAX)

    print("\n" + "=" * 78, flush=True)
    print("  LHS search summary", flush=True)
    print(f"  total samples evaluated : {len(results)}", flush=True)
    print(f"  work_alive (β_w>{WORK_ALIVE_THR}) : {n_work}", flush=True)
    print(f"  home_alive (β_h>{HOME_ALIVE_THR}) : {n_home}", flush=True)
    print(f"  BOTH channels alive     : {n_both_ch}", flush=True)
    print(f"  fit_score < {FIT_SCORE_MAX}         : {n_fit_only}", flush=True)
    print(f"  NLL < {NLL_MAX}              : {n_nll_only}", flush=True)
    print(f"  ★ SUCCESS (all four)    : {n_success}", flush=True)

    if n_success > 0:
        print("\n  Successful samples:", flush=True)
        for x in results:
            if x["success"]:
                p = x["params"]
                print(f"    idx {x['idx']:>3d}: "
                      f"γ=({p['gamma_child']:.2f}, {p['gamma_adult']:.2f}, "
                      f"{p['gamma_elder']:.2f})  "
                      f"φ=(cp={p['child_peak']:.2f}, al={p['adult_level']:.2f}, "
                      f"er={p['elder_rise']:.2f})  "
                      f"R0=(a{p['R0_adult']:.2f}, e{p['R0_elder']:.2f})",
                      flush=True)
                print(f"          β=({x['beta_4'][0]:.4f}, {x['beta_4'][1]:.4f},"
                      f" {x['beta_4'][2]:.4f}, {x['beta_4'][3]:.4f})  "
                      f"NLL={x['nll']:.4e}  fit_score={x['fit_score']:.3f}  "
                      f"R0_ngm={x['R0_ngm']:.3f}", flush=True)
                print(f"          r = {x['per_age_r']}", flush=True)

    if n_work > 0 and n_success == 0:
        print("\n  work_alive but not success (top 5 by fit_score):", flush=True)
        work_only = sorted([x for x in results if x["work_alive"]],
                             key=lambda z: z["fit_score"])
        for x in work_only[:5]:
            p = x["params"]
            print(f"    idx {x['idx']:>3d}: "
                  f"β=({x['beta_4'][0]:.4f}, {x['beta_4'][1]:.4f}, "
                  f"{x['beta_4'][2]:.4f}, {x['beta_4'][3]:.4f})  "
                  f"NLL={x['nll']:.4e}  fit_score={x['fit_score']:.3f}  "
                  f"γ=({p['gamma_child']:.2f}, {p['gamma_adult']:.2f}, "
                  f"{p['gamma_elder']:.2f})", flush=True)

    # Best sample overall by fit_score
    if results:
        best_fit = min(results, key=lambda z: z["fit_score"])
        best_nll = min(results, key=lambda z: z["nll"])
        print(f"\n  Best by fit_score: idx={best_fit['idx']}  "
              f"fit_score={best_fit['fit_score']:.3f}  "
              f"NLL={best_fit['nll']:.4e}", flush=True)
        print(f"    params = {best_fit['params']}", flush=True)
        print(f"    β = {best_fit['beta_4']}   r = {best_fit['per_age_r']}",
              flush=True)
        print(f"  Best by NLL: idx={best_nll['idx']}  "
              f"NLL={best_nll['nll']:.4e}  "
              f"fit_score={best_nll['fit_score']:.3f}", flush=True)
        print(f"    params = {best_nll['params']}", flush=True)
        print(f"    β = {best_nll['beta_4']}   r = {best_nll['per_age_r']}",
              flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "n_samples_target": N_SAMPLES, "n_samples_evaluated": len(results),
            "ranges": {k: list(v) for k, v in RANGES.items()},
            "dim_order": DIM_ORDER,
            "n_starts": N_STARTS, "lhs_seed": LHS_SEED,
            "success_thresholds": dict(
                work_alive=WORK_ALIVE_THR, home_alive=HOME_ALIVE_THR,
                fit_score_max=FIT_SCORE_MAX, nll_max=NLL_MAX,
            ),
            "summary": dict(
                n_success=n_success, n_work_alive=n_work, n_home_alive=n_home,
                n_both_channels=n_both_ch,
                n_fit_ok=n_fit_only, n_nll_ok=n_nll_only,
            ),
            "PHI_BASE_USHAPE": PHI_BASE_USHAPE.tolist(),
            "results": results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # Figure
    if len(results) == 0:
        print("  no results to plot", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    nlls = np.array([x["nll"] for x in results])
    fits = np.array([x["fit_score"] for x in results])
    bws = np.array([x["beta_4"][1] for x in results])
    bhs = np.array([x["beta_4"][0] for x in results])
    work_alive = np.array([x["work_alive"] for x in results])
    success = np.array([x["success"] for x in results])

    ax = axes[0]
    ax.scatter(nlls[~work_alive], fits[~work_alive], s=18, color="#888888",
                alpha=0.6, label=f"work dead (n={int((~work_alive).sum())})")
    ax.scatter(nlls[work_alive & ~success], fits[work_alive & ~success],
                s=22, color="#f39c12", alpha=0.8,
                label=f"work_alive only (n={int((work_alive & ~success).sum())})")
    ax.scatter(nlls[success], fits[success], s=60, color="#c0392b",
                marker="*", edgecolor="k", zorder=5,
                label=f"SUCCESS (n={int(success.sum())})")
    ax.axhline(FIT_SCORE_MAX, color="grey", ls=":", lw=1,
                label=f"fit_score={FIT_SCORE_MAX}")
    ax.axvline(NLL_MAX, color="grey", ls="--", lw=1,
                label=f"NLL={NLL_MAX}")
    ax.set_xlabel("NLL")
    ax.set_ylabel("fit_score = Σ|r−1|")
    ax.set_title(f"NLL vs fit_score  (n={len(results)})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(bhs, bws, s=18, color="#888888", alpha=0.6,
                label="samples")
    ax.axhline(WORK_ALIVE_THR, color="grey", ls=":", lw=1,
                label=f"β_w={WORK_ALIVE_THR}")
    ax.axvline(HOME_ALIVE_THR, color="grey", ls=":", lw=1,
                label=f"β_h={HOME_ALIVE_THR}")
    ax.scatter(bhs[success], bws[success], s=60, color="#c0392b",
                marker="*", edgecolor="k", zorder=5, label="SUCCESS")
    ax.set_xlabel("β_h")
    ax.set_ylabel("β_w")
    ax.set_title("β_w vs β_h  (channel exclusivity check)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"LHS search — {SEASON_LABEL}  "
                  f"({len(results)} samples, {N_STARTS} starts each)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
