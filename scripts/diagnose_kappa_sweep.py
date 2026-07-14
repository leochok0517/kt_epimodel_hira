"""Diagnose: does home spillover κ suppress β_w? Sweep κ_scale and refit β + φ.

For each κ_scale ∈ {0.0, 0.5, 1.0, 1.5}:
    κ_swept = kappa_default * κ_scale
    free β_4 + φ_14 (anchor idx 5 ≡ 1.0) + phi_nb
    objective = NB-NLL + 0.1 · Σ (φ[i+1] − 2·φ[i] + φ[i-1])²    (interior 13 pts)
Compare β_w, β_h trajectories to test whether home-spillover dominance is
what pins work at 0.

Setup mirrors diagnose_phi_2ndorder.py (2019-2020, NB, HOLIDAY realloc=1
amp=0.7, AMP=0.9, γ=CDC). n_admdong=1 aggregated (mobility identity — same as
production calibration path).

Decision guide (comments only — script does not interpret):
- Some κ_scale gives β_w>0.01 with unchanged/improved NLL → κ dominance was
  masking work; work is recoverable.
- κ=0 (spillover off) → β_h alone grows, β_w still 0 → home independent of
  spillover, work has separate root cause.
- All scales → β_w≈0 → κ is not the cause; investigate seed / immunity /
  age-binning resolution instead.
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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "kappa_sweep.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "kappa_sweep.png"
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

KAPPA_SCALES = [0.0, 0.5, 1.0, 1.5]
LAMBDA_PHI = 0.1
N_STARTS = 12
START_SEED = 17
PEAK_HALF_WIN = 2
REF_AGE_IDX = 5

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_BOUNDS = [(0.1, 5.0)] * 14
PHI_NB_BOUNDS = (1e-3, 1e6)


def phi14_to_phi_full(phi14: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([phi14[:REF_AGE_IDX], jnp.array([1.0]),
                             phi14[REF_AGE_IDX:]])


def peak_window_sum(arr_1d, peak_w, half):
    lo = max(0, peak_w - half)
    hi = min(arr_1d.shape[0], peak_w + half + 1)
    return float(arr_1d[lo:hi].sum())


def build_setup():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination
    policy = ModelParameters().policy

    kappa_default = np.asarray(disease.kappa_array, dtype=np.float64)  # (15,)

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
        # kappa injected per scale below
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
    shared_base.update(HOLIDAY)
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

    ngm_fn = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )

    return dict(
        shared_base=shared_base, kappa_default=kappa_default,
        gamma_15=gamma_15, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_fn=ngm_fn,
    )


def predict_hira(beta_4, phi14, *, shared, setup):
    phi_full = phi14_to_phi_full(phi14)
    kw = dict(shared)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc_15 = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc_15, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def build_loss(kappa_swept: jnp.ndarray, setup):
    """x = [β_4, phi_14, phi_nb]. penalty is 2nd-order on full φ(15), λ=0.1."""
    shared = dict(setup["shared_base"])
    shared["kappa"] = kappa_swept

    def loss(x):
        beta_4 = x[:4]
        phi14 = x[4:18]
        phi_nb = x[18]
        pred = predict_hira(beta_4, phi14, shared=shared, setup=setup)
        nll = nb_nll_jax(setup["obs_j"], pred, setup["w_j"],
                         concentration=phi_nb, min_rate=0.01)
        phi_full = phi14_to_phi_full(phi14)
        curv = phi_full[2:] - 2.0 * phi_full[1:-1] + phi_full[:-2]
        penalty = LAMBDA_PHI * jnp.sum(curv ** 2)
        return nll + penalty, nll, penalty

    loss_j = jax.jit(lambda x: loss(x)[0])
    parts_j = jax.jit(loss)
    grad_j = jax.jit(jax.grad(lambda x: loss(x)[0]))

    def fg_np(x_np):
        x = jnp.asarray(x_np)
        v = float(loss_j(x))
        g = np.asarray(grad_j(x))
        if not np.isfinite(v):
            v = 1e15
            g = np.where(np.isfinite(g), g, 0.0)
        return v, g

    def parts_np(x_np):
        x = jnp.asarray(x_np)
        tot, nll, pen = parts_j(x)
        return float(tot), float(nll), float(pen)

    return fg_np, parts_np, shared


def make_starts(n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = [
        (np.array([0.10, 0.10, 0.10, 0.10]), np.ones(14)),
        (np.array([0.05, 0.05, 0.05, 0.15]), np.ones(14)),
        (np.array([0.07, 0.07, 0.20, 0.10]), np.ones(14) * 0.5),
        (np.array([0.07, 0.07, 0.05, 0.20]), np.ones(14) * 1.5),
        (np.array([0.15, 0.15, 0.05, 0.05]), np.ones(14)),
        (np.array([0.02, 0.20, 0.05, 0.10]), np.ones(14)),
    ]
    starts = []
    for b, p in base:
        starts.append(np.concatenate([b, p, np.array([10.0])]))
    while len(starts) < n_starts:
        b = rng.uniform(0.02, 0.20, 4)
        p = rng.uniform(0.3, 2.5, 14)
        starts.append(np.concatenate([b, p, np.array([rng.uniform(2.0, 20.0)])]))
    return starts[:n_starts]


def fit_scale(kappa_scale: float, setup) -> dict:
    kappa_swept = jnp.asarray(setup["kappa_default"] * kappa_scale)
    fg, parts, shared_used = build_loss(kappa_swept, setup)
    bounds = BETA_BOUNDS + PHI_BOUNDS + [PHI_NB_BOUNDS]
    starts = make_starts(N_STARTS, START_SEED)

    per_start_nll = []
    best = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                            bounds=bounds,
                            options=dict(maxiter=300, ftol=1e-9, gtol=1e-6))
            tot, nll, pen = parts(res.x)
        except Exception as e:
            print(f"      [warn] start {i} failed: {e}")
            continue
        per_start_nll.append(nll)
        if best is None or tot < best["total"]:
            best = {"total": tot, "nll": nll, "penalty": pen,
                    "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0

    beta_4 = np.array(best["x"][:4])
    phi14 = np.array(best["x"][4:18])
    phi_nb = float(best["x"][18])
    phi_full = np.asarray(phi14_to_phi_full(jnp.asarray(phi14)))

    pred = np.asarray(predict_hira(beta_4, phi14, shared=shared_used,
                                    setup=setup))
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    obs_masked = np.where(mask[:, None], obs, -1e18)
    pred_masked = np.where(mask[:, None], pred, -1e18)
    per_age = {}
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        obs_pw = int(np.argmax(obs_masked[:, ai]))
        mdl_pw = int(np.argmax(pred_masked[:, ai]))
        obs_sum = peak_window_sum(obs[:, ai], obs_pw, PEAK_HALF_WIN)
        mdl_sum = peak_window_sum(pred[:, ai], mdl_pw, PEAK_HALF_WIN)
        per_age[ag] = dict(
            obs_peak_week=obs_pw, model_peak_week=mdl_pw,
            phase_offset_weeks=mdl_pw - obs_pw,
            ratio=obs_sum / max(mdl_sum, 1.0),
        )

    r0 = float(setup["ngm_fn"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        jnp.asarray(phi_full),
    ))
    nll_arr = np.array(per_start_nll)

    return dict(
        kappa_scale=kappa_scale,
        kappa_swept=[float(x) for x in kappa_swept.tolist()],
        beta_4=[float(x) for x in beta_4],
        phi_full_15=[float(p) for p in phi_full],
        phi_nb=phi_nb,
        R0_ngm=r0,
        nll=best["nll"], penalty=best["penalty"], total=best["total"],
        best_start_idx=best["start_idx"],
        nll_per_start=[float(x) for x in nll_arr.tolist()],
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        nll_min=float(np.min(nll_arr)) if len(nll_arr) else float("nan"),
        nll_max=float(np.max(nll_arr)) if len(nll_arr) else float("nan"),
        per_age=per_age,
        wall_sec=float(wall),
    )


def main():
    print("=" * 78)
    print(f"DIAGNOSE: κ (home spillover) sweep with fair β+φ refit  —  {SEASON_LABEL}")
    print(f"  κ_scales = {KAPPA_SCALES}   (scale=1.0 = production default)")
    print(f"  free: β_4 + φ_14 (anchor idx {REF_AGE_IDX} ≡ 1.0) + phi_nb")
    print(f"  penalty = {LAMBDA_PHI} · Σ (φ[i+1]-2φ[i]+φ[i-1])²  (2nd-order)")
    print(f"  multi-start {N_STARTS}, β bounds {BETA_BOUNDS[0]}, φ bounds {PHI_BOUNDS[0]}")
    print(f"  shared: HOLIDAY realloc=1 amp=0.7, AMP={AMP}, γ=CDC[0.40/0.18/0.25]")
    print("=" * 78)

    setup = build_setup()
    print(f"  κ_default (15) = {[round(float(x),3) for x in setup['kappa_default']]}")
    print(f"  obs shape: {setup['obs_j'].shape}  n_weeks={setup['n_weeks']}")

    all_results = []
    for sc in KAPPA_SCALES:
        print(f"\n── fit κ_scale={sc} ──")
        r = fit_scale(sc, setup)
        all_results.append(r)
        b = r["beta_4"]
        print(f"  NLL={r['nll']:.4e}  pen={r['penalty']:.4e}  total={r['total']:.4e}"
              f"  wall={r['wall_sec']:.1f}s  best_start={r['best_start_idx']}")
        print(f"  β_h/w/s/o = {[round(x,4) for x in b]}   ★ β_h={b[0]:.4f}  ★ β_w={b[1]:.4f}")
        print(f"  R0_ngm={r['R0_ngm']:.3f}   phi_nb={r['phi_nb']:.2f}")
        print(f"  NLL std across starts = {r['nll_std']:.3e}  "
              f"(min={r['nll_min']:.4e}, max={r['nll_max']:.4e})")
        print(f"  φ_full(15) = {[round(p,3) for p in r['phi_full_15']]}")

    # Cross-scale summary
    print("\n" + "=" * 78)
    print("  κ_scale sweep summary")
    print("  scale   β_h      β_w      β_s      β_o      NLL          R0     phi_nb")
    print("-" * 78)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['kappa_scale']:>5.2f}  {b[0]:>7.4f}  {b[1]:>7.4f}  "
              f"{b[2]:>7.4f}  {b[3]:>7.4f}  {r['nll']:.4e}  "
              f"{r['R0_ngm']:>5.3f}  {r['phi_nb']:>5.2f}")

    print("\n  ★ β_h and β_w trajectories:")
    for r in all_results:
        print(f"    κ_scale={r['kappa_scale']:>5.2f}  "
              f"β_h={r['beta_4'][0]:.4f}   β_w={r['beta_4'][1]:.4f}")

    print("\n" + "=" * 78)
    print(f"  Per-age r  (obs_peak±{PEAK_HALF_WIN}w / model_peak±{PEAK_HALF_WIN}w)")
    print("=" * 78)
    header = "  κ_scale  " + "  ".join(f"{ag:>8s}" for ag in HIRA_AGE_GROUPS)
    print(header)
    for r in all_results:
        row = f"  {r['kappa_scale']:>5.2f}    "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>8.2f}"
        print(row)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3": [0.40, 0.18, 0.25],
                "bounds_beta": BETA_BOUNDS[0],
                "bounds_phi": PHI_BOUNDS[0],
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "kappa_scales": KAPPA_SCALES,
                "kappa_default": [float(x) for x in setup["kappa_default"]],
                "lambda_phi_2ndorder": LAMBDA_PHI,
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scales = [r["kappa_scale"] for r in all_results]
    for ch_idx, ch_name, col in [
        (0, "β_h (home)", "#1a5490"),
        (1, "β_w (work)", "#c0392b"),
        (2, "β_s (school)", "#27ae60"),
        (3, "β_o (other)", "#f39c12"),
    ]:
        vals = [r["beta_4"][ch_idx] for r in all_results]
        axes[0].plot(scales, vals, "-o", label=ch_name, color=col, lw=1.8)
    axes[0].axhline(0.001, color="grey", ls=":", lw=1, label="lower bound 0.001")
    axes[0].set_xlabel("κ_scale (multiplier on default κ)")
    axes[0].set_ylabel("best β")
    axes[0].set_title("β vs κ_scale (β+φ refit each scale)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)

    nlls = [r["nll"] for r in all_results]
    axes[1].plot(scales, nlls, "-o", color="#333333", lw=1.8)
    axes[1].axvline(1.0, color="grey", ls=":", lw=1, label="production κ (1.0)")
    axes[1].set_xlabel("κ_scale")
    axes[1].set_ylabel("NLL (best)")
    axes[1].set_title("Fit cost vs κ_scale")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    fig.suptitle(f"κ (home spillover) sweep with fair β+φ refit  —  {SEASON_LABEL}")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
