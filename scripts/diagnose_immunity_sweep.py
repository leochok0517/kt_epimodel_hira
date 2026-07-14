"""Diagnose: does adult R(0) (initial immunity) suppress β_w? Sweep adult band.

For each adult_scale ∈ {0.5, 1.0, 1.5}:
    R(0)_adult = default_adult × adult_scale (clipped to < 0.95)
    R(0) children (0-19) unchanged, R(0) elder (65+) unchanged
    free β_4 + φ_14 (anchor idx 5 ≡ 1.0) + phi_nb
    objective = NB-NLL + 0.1 · Σ (φ[i+1] − 2·φ[i] + φ[i-1])²

Default R0_IMMUNITY_PROFILE (from simple_model.py):
  idx  0-3  (0-19)   = 0.10
  idx  4-9  (20-49)  = 0.30   ← adult band
  idx 10-12 (50-64)  = 0.45   ← adult band
  idx 13-14 (65+)    = 0.65

Adult band = idx 4-12. Scale multiplies these values only.

Decision guide (comments only — script does not interpret):
- β_w > 0.05 with unchanged/improved NLL → adult immunity was suppressing work.
- β_w in 0.01–0.03, multi-start std large → excess-parameter noise, not real.
- All scales → β_w<0.05 → immunity is not the cause; channel non-id is deeper.
- Adult S(0) does change with scale, but β_w unresponsive → channel non-id
  dominates immunity.
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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "immunity_sweep.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "immunity_sweep.png"
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

# Adult band = NIMS idx 4-12 (20-64y). Children (0-3) and elder (13-14) untouched.
ADULT_IDX = list(range(4, 13))
IMMUNITY_CAP = 0.95     # clip to keep S(0) > 0

ADULT_SCALES = [0.5, 1.0, 1.5]
LAMBDA_PHI = 0.1
N_STARTS = 12
START_SEED = 17
PEAK_HALF_WIN = 2
REF_AGE_IDX = 5

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_BOUNDS = [(0.1, 5.0)] * 14
PHI_NB_BOUNDS = (1e-3, 1e6)


def scale_adult_immunity(profile_15: np.ndarray, adult_scale: float) -> np.ndarray:
    """Multiply adult-band values by adult_scale; clip to IMMUNITY_CAP."""
    prof = np.array(profile_15, dtype=np.float64, copy=True)
    prof[ADULT_IDX] = np.minimum(prof[ADULT_IDX] * adult_scale, IMMUNITY_CAP)
    return prof


def phi14_to_phi_full(phi14: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([phi14[:REF_AGE_IDX], jnp.array([1.0]),
                             phi14[REF_AGE_IDX:]])


def peak_window_sum(arr_1d, peak_w, half):
    lo = max(0, peak_w - half)
    hi = min(arr_1d.shape[0], peak_w + half + 1)
    return float(arr_1d[lo:hi].sum())


def build_setup_base():
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

    tgt = load_hira_target_by_age(
        SEASON_LABEL, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]

    # NGM ρ note: R0_immunity is baked into NGM at build time; we rebuild per scale.
    disease_gamma = disease.gamma
    C_h, C_w, C_s, C_o = matrices["C_home"], matrices["C_work"], \
                          matrices["C_school"], matrices["C_other"]

    return dict(
        shared=shared, gamma_15=gamma_15, seed_15=seed_15,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks,
        pop_15=pop_15, rho_emp=rho_emp,
        C_h=C_h, C_w=C_w, C_s=C_s, C_o=C_o,
        disease_gamma=disease_gamma,
    )


def build_state0(immunity_15: np.ndarray, setup) -> jnp.ndarray:
    return jnp.asarray(_build_initial_state_with_age_seed(
        setup["pop_15"], setup["seed_15"], seed_e_factor=0.5,
        initial_immunity=immunity_15, initial_vaccinated_fraction=0.0,
    ))


def predict_hira(beta_4, phi14, state0, *, setup):
    phi_full = phi14_to_phi_full(phi14)
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    st = simulate_jax(state0, **kw, discretize_time=False)
    inc_15 = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc_15, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def build_loss(state0: jnp.ndarray, setup):
    def loss(x):
        beta_4 = x[:4]
        phi14 = x[4:18]
        phi_nb = x[18]
        pred = predict_hira(beta_4, phi14, state0, setup=setup)
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

    return fg_np, parts_np


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


def compute_adult_S0(state0: jnp.ndarray) -> float:
    """Sum S at t=0 over adult ages (idx 4-12), across admdong axis."""
    # state shape (5, 15, n_admdong); IDX_S = 0
    S = np.asarray(state0[0])           # (15, n_admdong)
    return float(S[ADULT_IDX].sum())


def fit_scale(adult_scale: float, setup) -> dict:
    imm = scale_adult_immunity(R0_IMMUNITY_PROFILE, adult_scale)
    state0 = build_state0(imm, setup)
    adult_S0 = compute_adult_S0(state0)

    # NGM closure uses this scale's immunity
    ngm_fn = make_ngm_eigvalue_fn(
        pop_15=setup["pop_15"], rho=setup["rho_emp"],
        C_home=setup["C_h"], C_work=setup["C_w"],
        C_school=setup["C_s"], C_other=setup["C_o"],
        R0_immunity=imm,
        gamma=setup["disease_gamma"], seasonal_factor=1.0 + AMP,
    )

    fg, parts = build_loss(state0, setup)
    bounds = BETA_BOUNDS + PHI_BOUNDS + [PHI_NB_BOUNDS]
    starts = make_starts(N_STARTS, START_SEED)

    per_start_nll = []
    best = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
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

    pred = np.asarray(predict_hira(beta_4, phi14, state0, setup=setup))
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

    r0 = float(ngm_fn(
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        jnp.asarray(phi_full),
    ))
    nll_arr = np.array(per_start_nll)

    return dict(
        adult_scale=adult_scale,
        immunity_15=[float(x) for x in imm],
        adult_S0=adult_S0,
        beta_4=[float(x) for x in beta_4],
        phi_full_15=[float(p) for p in phi_full],
        phi_nb=phi_nb, R0_ngm=r0,
        nll=best["nll"], penalty=best["penalty"], total=best["total"],
        best_start_idx=best["start_idx"],
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        nll_min=float(np.min(nll_arr)) if len(nll_arr) else float("nan"),
        nll_max=float(np.max(nll_arr)) if len(nll_arr) else float("nan"),
        per_age=per_age,
        wall_sec=float(wall),
    )


def main():
    print("=" * 78)
    print(f"DIAGNOSE: adult R(0) immunity sweep with fair β+φ refit  —  {SEASON_LABEL}")
    print(f"  adult_scales = {ADULT_SCALES}   (scale=1.0 = production)")
    print(f"  adult band idx {ADULT_IDX} (20-64y),  clip cap {IMMUNITY_CAP}")
    print(f"  children (0-19) and elder (65+) R(0) unchanged")
    print(f"  free: β_4 + φ_14 (anchor idx {REF_AGE_IDX} ≡ 1.0) + phi_nb")
    print(f"  penalty = {LAMBDA_PHI} · Σ 2nd-order,  multi-start {N_STARTS}")
    print("=" * 78)

    setup = build_setup_base()
    print(f"  R(0) default (15) = "
          f"{[round(float(x),3) for x in R0_IMMUNITY_PROFILE]}")
    print(f"  obs shape: {setup['obs_j'].shape}  n_weeks={setup['n_weeks']}")

    all_results = []
    for sc in ADULT_SCALES:
        print(f"\n── fit adult_scale={sc} ──")
        r = fit_scale(sc, setup)
        all_results.append(r)
        b = r["beta_4"]
        print(f"  immunity(15) = {[round(x,3) for x in r['immunity_15']]}")
        print(f"  adult S(0) sum = {r['adult_S0']:,.0f}")
        print(f"  NLL={r['nll']:.4e}  pen={r['penalty']:.4e}  "
              f"total={r['total']:.4e}  wall={r['wall_sec']:.1f}s  "
              f"best_start={r['best_start_idx']}")
        print(f"  β_h/w/s/o = {[round(x,4) for x in b]}   "
              f"★ β_h={b[0]:.4f}  ★ β_w={b[1]:.4f}")
        print(f"  R0_ngm={r['R0_ngm']:.3f}   phi_nb={r['phi_nb']:.2f}")
        print(f"  NLL std across starts = {r['nll_std']:.3e}  "
              f"(min={r['nll_min']:.4e}, max={r['nll_max']:.4e})")
        print(f"  φ_full(15) = {[round(p,3) for p in r['phi_full_15']]}")

    # Summary
    print("\n" + "=" * 78)
    print("  adult_scale sweep summary")
    print("  scale   adult_S(0)     β_h     β_w      β_s      β_o      NLL          R0     phi_nb")
    print("-" * 78)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['adult_scale']:>5.2f}  {r['adult_S0']:>12,.0f}  "
              f"{b[0]:>7.4f}  {b[1]:>7.4f}  {b[2]:>7.4f}  {b[3]:>7.4f}  "
              f"{r['nll']:.4e}  {r['R0_ngm']:>5.3f}  {r['phi_nb']:>5.2f}")

    print("\n  ★ β_h and β_w trajectories:")
    for r in all_results:
        print(f"    adult_scale={r['adult_scale']:>5.2f}  "
              f"S_adult(0)={r['adult_S0']:>12,.0f}  "
              f"β_h={r['beta_4'][0]:.4f}   β_w={r['beta_4'][1]:.4f}")

    print("\n" + "=" * 78)
    print(f"  Per-age r  (obs_peak±{PEAK_HALF_WIN}w / model_peak±{PEAK_HALF_WIN}w)")
    print("=" * 78)
    header = "  scale    " + "  ".join(f"{ag:>8s}" for ag in HIRA_AGE_GROUPS)
    print(header)
    for r in all_results:
        row = f"  {r['adult_scale']:>5.2f}    "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>8.2f}"
        print(row)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3": [0.40, 0.18, 0.25],
                "bounds_beta": BETA_BOUNDS[0], "bounds_phi": PHI_BOUNDS[0],
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "adult_scales": ADULT_SCALES,
                "adult_idx": ADULT_IDX,
                "immunity_cap": IMMUNITY_CAP,
                "R0_immunity_default": [float(x) for x in R0_IMMUNITY_PROFILE],
                "lambda_phi_2ndorder": LAMBDA_PHI,
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scales = [r["adult_scale"] for r in all_results]
    s0s = [r["adult_S0"] for r in all_results]

    for ch_idx, ch_name, col in [
        (0, "β_h (home)", "#1a5490"),
        (1, "β_w (work)", "#c0392b"),
        (2, "β_s (school)", "#27ae60"),
        (3, "β_o (other)", "#f39c12"),
    ]:
        vals = [r["beta_4"][ch_idx] for r in all_results]
        axes[0].plot(scales, vals, "-o", label=ch_name, color=col, lw=1.8)
    axes[0].axhline(0.001, color="grey", ls=":", lw=1, label="lower bound 0.001")
    axes[0].set_xlabel("adult_scale (R(0) multiplier on 20-64y band)")
    axes[0].set_ylabel("best β")
    axes[0].set_title("β vs adult R(0) scale (β+φ refit)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)

    # annotate adult S(0) per point
    for x, y_max, s in zip(scales, [max(r["beta_4"]) for r in all_results], s0s):
        axes[0].annotate(f"S_adult(0)={s/1e6:.2f}M",
                          (x, y_max), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=8, color="grey")

    nlls = [r["nll"] for r in all_results]
    axes[1].plot(scales, nlls, "-o", color="#333333", lw=1.8)
    axes[1].axvline(1.0, color="grey", ls=":", lw=1,
                    label="production scale (1.0)")
    axes[1].set_xlabel("adult_scale")
    axes[1].set_ylabel("NLL (best)")
    axes[1].set_title("Fit cost vs adult R(0) scale")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    fig.suptitle(f"Adult R(0) immunity sweep with fair β+φ refit  —  {SEASON_LABEL}")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
