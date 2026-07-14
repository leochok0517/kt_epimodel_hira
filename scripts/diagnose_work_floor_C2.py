"""C2 diagnostic: identifiability floor of β_w under NB noise.

Step 1: fit HIRA 2019-2020 with free β_4 (+phi_nb), NB likelihood
        → β_fit and R0_fit.
Step 2: build β_test grids by varying work_level ∈ {0.08, 0.05, ...} while
        HOLDING R0 constant. β_h, β_s fixed at β_fit; β_o adjusted via bisection
        so that R0(β_test) = R0_ref where
            R0_ref = R0([β_h_fit, 0.05, β_s_fit, β_o_fit])
Step 3: for each β_test, forward → pred → 20 NB replicate observations with
        phi_nb = 1.3 (production level).
Step 4: refit β_4 (+phi_nb) on each replicate (L-BFGS × 8 starts, NB).

Measure per work_level:
  - median recovered β_w and [2.5, 97.5] over 20 reps
  - fraction of reps where β_w collapsed to lower bound (0.001)
  - true work_level ∈ CI?
  - recovered β_h, β_o mean (does work mass leak to other channels?)

Decision guide (comments only):
- Recovered β_w tracks truth down to some level, then collapses → that's the floor.
- Floor > 20% of β_o_fit → real β_w may be "too small to see, not zero".
- Floor < 0.01 → tiny β_w still identifiable; real work=0 is genuine.
- Recovered β_w=0 while β_o inflates above truth → mass leaks (channel non-id).
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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "work_floor_C2.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "work_floor_C2.png"
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

WORK_LEVELS = [0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
WORK_REF = 0.05          # mid work used to define R0_ref
N_REPLICATES = 20
PHI_NB_TRUE = 1.3        # NB dispersion for synthetic obs (production level)
N_STARTS_STEP1 = 12
N_STARTS_STEP4 = 8
STEP1_SEED = 23
STEP4_BASE_SEED = 100    # replicate seed = base + i
BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)
LOWER_BOUND_TOL = 0.0015   # β_w within this of lower bound → "collapsed"


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
    phi_full = jnp.ones(15)

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
    obs_real = np.zeros((n_weeks, 6)); w_real = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs_real[:, i] = tgt["hira_counts"][ag]
        w_real[:, i] = tgt["weights"][ag]

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
        obs_real_j=jnp.asarray(obs_real), w_real_j=jnp.asarray(w_real),
    )


def forward_pred_hira(beta_4, setup) -> jnp.ndarray:
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = setup["phi_full"]
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def R0_of(beta_4, setup) -> float:
    return float(setup["ngm_fn"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        setup["phi_full"],
    ))


def build_nb_loss(obs_j, w_j, setup):
    """x = [β_4, phi_nb]. NB likelihood."""
    def loss(x):
        beta_4 = x[:4]; phi_nb = x[4]
        pred = forward_pred_hira(beta_4, setup)
        return nb_nll_jax(obs_j, pred, w_j, concentration=phi_nb, min_rate=0.01)
    loss_j = jax.jit(loss)
    grad_j = jax.jit(jax.grad(loss))
    def fg_np(x_np):
        x = jnp.asarray(x_np)
        v = float(loss_j(x))
        g = np.asarray(grad_j(x))
        if not np.isfinite(v):
            v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
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


def fit_free(obs_j, w_j, n_starts: int, seed: int, setup):
    fg = build_nb_loss(obs_j, w_j, setup)
    bounds = BETA_BOUNDS + [PHI_NB_BOUNDS]
    starts = make_starts(n_starts, seed)
    best = None
    per_start_nll = []
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
    nll_arr = np.array(per_start_nll)
    return best, nll_arr


def solve_beta_o_for_R0(beta_h, work_level, beta_s, R0_ref, setup,
                         beta_o_lo=0.001, beta_o_hi=1.0, tol=1e-4, max_iter=60):
    """Bisection: find β_o s.t. R0([β_h, work_level, β_s, β_o]) = R0_ref."""
    def R0_at(bo):
        return R0_of([beta_h, work_level, beta_s, bo], setup)
    r_lo = R0_at(beta_o_lo); r_hi = R0_at(beta_o_hi)
    if not (r_lo <= R0_ref <= r_hi):
        return dict(beta_o=None, R0_achieved=None, warn=(
            f"R0_ref={R0_ref:.4f} outside bracket [{r_lo:.4f}, {r_hi:.4f}] "
            f"at work_level={work_level}"))
    a, b = beta_o_lo, beta_o_hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        r_m = R0_at(m)
        if abs(r_m - R0_ref) < tol:
            return dict(beta_o=float(m), R0_achieved=float(r_m), warn=None)
        if r_m < R0_ref:
            a = m
        else:
            b = m
    m = 0.5 * (a + b)
    return dict(beta_o=float(m), R0_achieved=float(R0_at(m)), warn=None)


def sample_nb(mean_arr: np.ndarray, k: float, rng: np.random.Generator) -> np.ndarray:
    """NegBin2 sampling: Var = μ + μ²/k. numpy uses n=k, p=k/(k+μ)."""
    mu = np.maximum(mean_arr, 1e-6)
    p = k / (k + mu)
    return rng.negative_binomial(k, p).astype(np.float64)


def main():
    print("=" * 78)
    print(f"C2: work identifiability floor under NB noise  —  {SEASON_LABEL}")
    print(f"  work_levels = {WORK_LEVELS}   ref work = {WORK_REF}")
    print(f"  reps per level = {N_REPLICATES}   phi_nb_true = {PHI_NB_TRUE}")
    print(f"  φ = ones(15), γ = CDC, HOLIDAY on, AMP={AMP}")
    print("=" * 78)

    setup = build_setup()
    print(f"  n_weeks = {setup['n_weeks']}   real obs sum = "
          f"{float(np.asarray(setup['obs_real_j']).sum()):,.0f}")

    # ─── STEP 1: fit real data ─────────────────────────────
    print(f"\n[STEP 1] free-fit real HIRA 2019-2020  (multi-start {N_STARTS_STEP1})")
    t0 = time.perf_counter()
    best1, nlls1 = fit_free(setup["obs_real_j"], setup["w_real_j"],
                             N_STARTS_STEP1, STEP1_SEED, setup)
    wall1 = time.perf_counter() - t0
    beta_fit = np.array(best1["x"][:4]); phi_nb_fit = float(best1["x"][4])
    r0_fit = R0_of(beta_fit, setup)
    print(f"  wall {wall1:.1f}s  best NLL={best1['nll']:.4e}  "
          f"start_idx={best1['start_idx']}")
    print(f"  β_fit = {[round(x,4) for x in beta_fit]}")
    print(f"  phi_nb_fit = {phi_nb_fit:.3f}   R0_fit = {r0_fit:.3f}")
    print(f"  NLL 12 starts: min={nlls1.min():.4e}  max={nlls1.max():.4e}  "
          f"std={nlls1.std():.3e}")

    # ─── STEP 2: R0_ref + build β_test grid ────────────────
    print(f"\n[STEP 2] R0_ref = R0 at β_h_fit, work={WORK_REF}, β_s_fit, β_o_fit")
    beta_ref = np.array([beta_fit[0], WORK_REF, beta_fit[2], beta_fit[3]])
    r0_ref = R0_of(beta_ref, setup)
    print(f"  β_ref = {[round(x,4) for x in beta_ref]}   R0_ref = {r0_ref:.4f}")

    beta_tests = []
    for wl in WORK_LEVELS:
        sol = solve_beta_o_for_R0(beta_fit[0], wl, beta_fit[2], r0_ref, setup)
        if sol["beta_o"] is None:
            print(f"  ⚠️  work_level={wl}: {sol['warn']}")
            continue
        b_test = np.array([beta_fit[0], wl, beta_fit[2], sol["beta_o"]])
        r0_check = R0_of(b_test, setup)
        beta_tests.append(dict(work_level=wl, beta_test=b_test.tolist(),
                                beta_o_solved=sol["beta_o"],
                                R0_achieved=r0_check))
        print(f"  work_level={wl:.4f}  β_o solved={sol['beta_o']:.4f}   "
              f"R0 achieved={r0_check:.4f}")

    # ─── STEP 3+4: replicates and recovery ──────────────────
    print(f"\n[STEP 3+4] replicate NB obs (phi_nb={PHI_NB_TRUE}) + recover β")
    all_results = []
    for bt in beta_tests:
        wl = bt["work_level"]
        beta_test = np.array(bt["beta_test"])
        pred_clean = np.asarray(forward_pred_hira(beta_test, setup))
        print(f"\n  ── work_level={wl:.4f}  β_test={[round(x,4) for x in beta_test]} ──")
        print(f"     pred sum = {pred_clean.sum():,.0f}")

        beta_recovered = []
        phi_nb_recovered = []
        nll_recovered = []
        collapsed_flags = []
        w_j = jnp.ones_like(jnp.asarray(pred_clean))

        t0 = time.perf_counter()
        for rep in range(N_REPLICATES):
            rng = np.random.default_rng(STEP4_BASE_SEED + rep +
                                         int(wl * 1e6))
            obs_rep = sample_nb(pred_clean, PHI_NB_TRUE, rng)
            best_rep, _ = fit_free(jnp.asarray(obs_rep), w_j,
                                    N_STARTS_STEP4, STEP4_BASE_SEED + rep,
                                    setup)
            b_rec = np.array(best_rep["x"][:4])
            pn_rec = float(best_rep["x"][4])
            beta_recovered.append(b_rec.tolist())
            phi_nb_recovered.append(pn_rec)
            nll_recovered.append(best_rep["nll"])
            collapsed_flags.append(b_rec[1] <= LOWER_BOUND_TOL)
        wall = time.perf_counter() - t0

        beta_recovered = np.array(beta_recovered)   # (n_rep, 4)
        bw_rec = beta_recovered[:, 1]
        bh_rec = beta_recovered[:, 0]
        bo_rec = beta_recovered[:, 3]
        bs_rec = beta_recovered[:, 2]
        collapse_frac = float(np.mean(collapsed_flags))
        bw_true_in_ci = bool(
            (wl >= np.quantile(bw_rec, 0.025)) and
            (wl <= np.quantile(bw_rec, 0.975))
        )

        rec = dict(
            work_level=wl, beta_test=beta_test.tolist(),
            beta_o_solved=bt["beta_o_solved"],
            R0_achieved=bt["R0_achieved"],
            n_replicates=N_REPLICATES,
            beta_w_recovered=bw_rec.tolist(),
            beta_h_recovered=bh_rec.tolist(),
            beta_s_recovered=bs_rec.tolist(),
            beta_o_recovered=bo_rec.tolist(),
            phi_nb_recovered=phi_nb_recovered,
            nll_recovered=nll_recovered,
            bw_median=float(np.median(bw_rec)),
            bw_q025=float(np.quantile(bw_rec, 0.025)),
            bw_q975=float(np.quantile(bw_rec, 0.975)),
            bh_median=float(np.median(bh_rec)),
            bo_median=float(np.median(bo_rec)),
            bs_median=float(np.median(bs_rec)),
            collapse_fraction=collapse_frac,
            true_in_ci=bw_true_in_ci,
            wall_sec=float(wall),
        )
        all_results.append(rec)
        print(f"     β_w recovered median={rec['bw_median']:.4f}  "
              f"[{rec['bw_q025']:.4f}, {rec['bw_q975']:.4f}]")
        print(f"     collapse (β_w≤{LOWER_BOUND_TOL}) = "
              f"{int(collapse_frac*N_REPLICATES)}/{N_REPLICATES} = "
              f"{collapse_frac*100:.0f}%")
        print(f"     true={wl:.4f} in CI? {rec['true_in_ci']}")
        print(f"     β_h median={rec['bh_median']:.4f}  "
              f"β_o median={rec['bo_median']:.4f} (true β_o={bt['beta_o_solved']:.4f})  "
              f"β_s median={rec['bs_median']:.4f}")
        print(f"     wall {wall:.1f}s")

    # ─── Console summary table ─────────────────────────────
    print("\n" + "=" * 78)
    print("  work_level sweep summary")
    print("  work_true  β_o_true   β_w_med   β_w_[q025, q975]         collapse%  β_h_med  β_o_med")
    for r in all_results:
        print(f"  {r['work_level']:>8.4f}  {r['beta_o_solved']:>8.4f}  "
              f"{r['bw_median']:>8.4f}  "
              f"[{r['bw_q025']:.4f}, {r['bw_q975']:.4f}]     "
              f"{r['collapse_fraction']*100:>6.0f}%    "
              f"{r['bh_median']:>7.4f}  {r['bo_median']:>7.4f}")

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "beta_fit_real": beta_fit.tolist(),
            "phi_nb_fit_real": phi_nb_fit,
            "R0_fit_real": r0_fit,
            "R0_ref": r0_ref,
            "work_ref_for_R0_ref": WORK_REF,
            "phi_nb_true_for_sampling": PHI_NB_TRUE,
            "n_replicates": N_REPLICATES,
            "n_starts_step1": N_STARTS_STEP1,
            "n_starts_step4": N_STARTS_STEP4,
            "lower_bound_tol_for_collapse": LOWER_BOUND_TOL,
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Panel 1: boxplot β_w recovery vs true work_level
    true_wls = [r["work_level"] for r in all_results]
    bw_data = [r["beta_w_recovered"] for r in all_results]
    positions = np.arange(len(all_results))
    bp = axes[0].boxplot(bw_data, positions=positions, widths=0.6,
                          showfliers=True, patch_artist=True,
                          boxprops=dict(facecolor="#1a5490", alpha=0.35),
                          medianprops=dict(color="#1a5490", lw=2))
    axes[0].plot(positions, true_wls, "kD-", ms=8, label="true work_level")
    for i, r in enumerate(all_results):
        axes[0].text(i, max(r["beta_w_recovered"]) + 0.005,
                     f"{int(r['collapse_fraction']*100)}%↓",
                     ha="center", fontsize=8, color="#c0392b")
    axes[0].axhline(BETA_BOUNDS[0][0], color="grey", ls=":", lw=1,
                     label="lower bound 0.001")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([f"{wl:.3f}" for wl in true_wls])
    axes[0].set_xlabel("true work_level")
    axes[0].set_ylabel("recovered β_w  (boxplot over 20 reps)")
    axes[0].set_title("Panel 1: β_w recovery vs truth  "
                       f"(NB obs, phi_nb={PHI_NB_TRUE})\n"
                       f"red % = replicate collapse rate (β_w ≤ {LOWER_BOUND_TOL})")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)

    # Panel 2: β_o recovery vs work_level (mass leak indicator)
    bo_data = [r["beta_o_recovered"] for r in all_results]
    bo_true = [r["beta_o_solved"] for r in all_results]
    axes[1].boxplot(bo_data, positions=positions, widths=0.6,
                     showfliers=True, patch_artist=True,
                     boxprops=dict(facecolor="#c0392b", alpha=0.35),
                     medianprops=dict(color="#c0392b", lw=2))
    axes[1].plot(positions, bo_true, "kD-", ms=8, label="true β_o (R0-preserving)")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"{wl:.3f}" for wl in true_wls])
    axes[1].set_xlabel("true work_level")
    axes[1].set_ylabel("recovered β_o")
    axes[1].set_title("Panel 2: β_o recovery — does work-mass leak into other?")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    fig.suptitle(f"C2 work-floor test — β_h/β_s fixed at β_fit, β_o adjusts to hold R0  ({SEASON_LABEL})")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
