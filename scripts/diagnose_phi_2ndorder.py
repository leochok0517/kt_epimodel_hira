"""Diagnose φ with 2nd-order (curvature) smoothing penalty — variant of
diagnose_phi_free_vs_misspec.py. Only the penalty form differs; everything
else is identical.

1st-order (previous script):
    penalty = λ · Σ_{i=0..13} (φ_full[i+1] − φ_full[i])²
2nd-order (this script):
    penalty = λ · Σ_{i=1..13}
              (φ_full[i+1] − 2·φ_full[i] + φ_full[i−1])²
(interior 13 curvature points; boundary i=0 and i=14 excluded)

Decision guide (comments only — script does not interpret):
- 2nd-order lifts φ[3] (15-19) to neighbor level (0.6–0.75) with unchanged NLL
  → 2nd-order penalty succeeds.
- φ[3] < 0.4 persists → 2nd-order also insufficient; local school tie needed.
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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "phi_2ndorder.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "phi_2ndorder_diag.png"
PREV_JSON = REPO_ROOT / "outputs" / "eda" / "phi_free_vs_misspec.json"
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
LAMBDA_LIST = [0.0, 0.1, 1.0, 10.0, 100.0]     # ← 100 added vs 1st-order
N_STARTS = 8
START_SEED = 13
PEAK_HALF_WIN = 2

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_BOUNDS = [(0.1, 5.0)] * 14
PHI_NB_BOUNDS = (1e-3, 1e6)
REF_AGE_IDX = 5


def phi14_to_phi_full(phi14: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([phi14[:REF_AGE_IDX], jnp.array([1.0]),
                             phi14[REF_AGE_IDX:]])


def peak_window_sum(arr_1d: np.ndarray, peak_w: int, half: int) -> float:
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

    ngm_fn = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + AMP,
    )

    return dict(
        shared=shared, gamma_15=gamma_15, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_fn=ngm_fn,
    )


def predict_hira(beta_4, phi14, *, setup):
    phi_full = phi14_to_phi_full(phi14)
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc_15 = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc_15, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def build_loss(lambda_phi: float, setup):
    """x = [β_4, phi_14, phi_nb] (19 dim).
    ★ 2nd-order penalty on FULL 15-vec:
        Σ_{i=1..13} (φ[i+1] − 2·φ[i] + φ[i-1])²
      = Σ (second-difference)² over 13 interior points.
    """
    def loss(x):
        beta_4 = x[:4]
        phi14 = x[4:18]
        phi_nb = x[18]
        pred = predict_hira(beta_4, phi14, setup=setup)
        nll = nb_nll_jax(setup["obs_j"], pred, setup["w_j"],
                         concentration=phi_nb, min_rate=0.01)
        phi_full = phi14_to_phi_full(phi14)
        # 2nd-order: φ[2:] − 2·φ[1:-1] + φ[:-2]  gives the 13 curvatures
        curv = phi_full[2:] - 2.0 * phi_full[1:-1] + phi_full[:-2]
        penalty = lambda_phi * jnp.sum(curv ** 2)
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
    ]
    starts = []
    for b, p in base:
        starts.append(np.concatenate([b, p, np.array([10.0])]))
    while len(starts) < n_starts:
        b = rng.uniform(0.02, 0.20, 4)
        p = rng.uniform(0.3, 2.5, 14)
        starts.append(np.concatenate([b, p, np.array([rng.uniform(2.0, 20.0)])]))
    return starts[:n_starts]


def fit_lambda(lambda_phi: float, setup) -> dict:
    fg, parts = build_loss(lambda_phi, setup)
    bounds = BETA_BOUNDS + PHI_BOUNDS + [PHI_NB_BOUNDS]
    starts = make_starts(N_STARTS, START_SEED)

    per_start_nll = []
    best = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(
                fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                options=dict(maxiter=300, ftol=1e-9, gtol=1e-6),
            )
            tot, nll, pen = parts(res.x)
        except Exception as e:
            print(f"    [warn] start {i} failed: {e}")
            continue
        per_start_nll.append(nll)
        if best is None or tot < best["total"]:
            best = {"total": tot, "nll": nll, "penalty": pen,
                    "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0
    nll_arr = np.array(per_start_nll)
    return {
        "lambda_phi": lambda_phi, "best": best, "wall_sec": float(wall),
        "n_starts_ok": int(len(per_start_nll)),
        "nll_per_start": [float(x) for x in nll_arr.tolist()],
        "nll_std_across_starts": float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        "nll_min_across_starts": float(np.min(nll_arr)) if len(nll_arr) else float("nan"),
        "nll_max_across_starts": float(np.max(nll_arr)) if len(nll_arr) else float("nan"),
    }


def diagnose(fit: dict, setup) -> dict:
    best = fit["best"]
    x = best["x"]
    beta_4 = x[:4]; phi14 = x[4:18]; phi_nb = x[18]
    phi_full = np.asarray(phi14_to_phi_full(phi14))
    pred = np.asarray(predict_hira(beta_4, phi14, setup=setup))
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
        per_age[ag] = {
            "obs_peak_week": obs_pw, "model_peak_week": mdl_pw,
            "phase_offset_weeks": mdl_pw - obs_pw,
            "ratio": obs_sum / max(mdl_sum, 1.0),
        }

    try:
        r0 = float(setup["ngm_fn"](
            jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
            jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
            jnp.asarray(phi_full),
        ))
    except Exception as e:
        r0 = float("nan")
        print(f"    [warn] NGM R0 failed: {e}")

    return {
        "beta_4": [float(b) for b in beta_4],
        "phi_full_15": [float(p) for p in phi_full],
        "phi_nb": float(phi_nb),
        "R0_ngm": r0,
        "pred_total_per_age": pred.tolist(),
        "obs_total_per_age": obs.tolist(),
        "per_age": per_age,
        "best_start_idx": best["start_idx"],
    }


def load_prev_1st_order():
    """Load 1st-order results for dashed overlay in the φ plot."""
    if not PREV_JSON.exists():
        return None
    with open(PREV_JSON) as f:
        prev = json.load(f)
    return prev


def main():
    print("=" * 78)
    print(f"DIAGNOSE: φ with 2nd-ORDER (curvature) penalty — {SEASON_LABEL}, "
          f"multi-start {N_STARTS}")
    print(f"  free: β_4 + φ_14 (anchor idx {REF_AGE_IDX} ≡ 1.0) + phi_nb")
    print(f"  penalty = λ · Σ (φ[i+1] − 2·φ[i] + φ[i-1])²   (interior 13 pts)")
    print(f"  λ_phi ∈ {LAMBDA_LIST}   bounds β=(0.001,1.0), φ=(0.1,5.0)")
    print(f"  shared: HOLIDAY realloc=1, amp={HOLIDAY['school_holiday_amp']}  "
          f"AMP={AMP}  γ=CDC[0.40/0.18/0.25]")
    print("=" * 78)

    setup = build_setup()
    print(f"  obs shape: {setup['obs_j'].shape}  n_weeks={setup['n_weeks']}")

    all_results = []
    for lam in LAMBDA_LIST:
        print(f"\n── fit λ_phi={lam} ──")
        fit = fit_lambda(lam, setup)
        diag = diagnose(fit, setup)
        rec = {**fit, "diagnostic": diag}
        rec["best"] = {k: (v if k != "x" else v.tolist())
                        for k, v in fit["best"].items()}
        all_results.append(rec)

        b = diag["beta_4"]
        phi15 = diag["phi_full_15"]
        print(f"  best NLL={fit['best']['nll']:.4e}  pen={fit['best']['penalty']:.4e}  "
              f"total={fit['best']['total']:.4e}")
        print(f"  best start idx={diag['best_start_idx']}  "
              f"NLL std={fit['nll_std_across_starts']:.3e}  "
              f"(min={fit['nll_min_across_starts']:.4e}, "
              f"max={fit['nll_max_across_starts']:.4e})")
        print(f"  β_4 = {[round(x,4) for x in b]}  R0_ngm={diag['R0_ngm']:.3f}  "
              f"phi_nb={diag['phi_nb']:.2f}")
        print(f"  ★ φ_full[3] (15-19) = {phi15[3]:.4f}   "
              f"(neighbors idx2={phi15[2]:.3f}, idx4={phi15[4]:.3f})")
        print(f"  φ_full(15) = {[round(p,3) for p in phi15]}")

    # Cross-λ summary
    print("\n" + "=" * 78)
    print("  λ-sweep summary (2nd-order penalty)")
    print("  λ        NLL          penalty        total         NLL_std       φ[3]")
    print("-" * 78)
    for r in all_results:
        b = r["best"]; p3 = r["diagnostic"]["phi_full_15"][3]
        print(f"  {r['lambda_phi']:>6.2f}  {b['nll']:.4e}  {b['penalty']:.4e}  "
              f"{b['total']:.4e}  {r['nll_std_across_starts']:.3e}   {p3:.4f}")

    # φ[3] trajectory
    print("\n  φ_full[3] (15-19) across λ (2nd-order):")
    for r in all_results:
        print(f"    λ={r['lambda_phi']:>6.2f} → φ[3] = "
              f"{r['diagnostic']['phi_full_15'][3]:.4f}")

    # Per-age r table
    print("\n" + "=" * 78)
    print(f"  Age ratio r = obs_peak_sum(±{PEAK_HALF_WIN}w) / model_peak_sum(±{PEAK_HALF_WIN}w)")
    print("=" * 78)
    header = "  λ         " + "  ".join(f"{ag:>8s}" for ag in HIRA_AGE_GROUPS) \
             + "    phase(off)"
    print(header)
    for r in all_results:
        row = f"  {r['lambda_phi']:>6.2f}   "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['diagnostic']['per_age'][ag]['ratio']:>8.2f}"
        offs = [r['diagnostic']['per_age'][ag]['phase_offset_weeks']
                for ag in HIRA_AGE_GROUPS]
        row += f"     [{','.join(f'{o:+d}' for o in offs)}]"
        print(row)

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "penalty_form": "2nd-order (curvature) Σ (φ[i+1]-2φ[i]+φ[i-1])²",
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3": [0.40, 0.18, 0.25],
                "bounds_beta": BETA_BOUNDS[0], "bounds_phi": PHI_BOUNDS[0],
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "lambda_list": LAMBDA_LIST, "ref_age_idx": REF_AGE_IDX,
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure ─────────────────────────────────────────────
    prev = load_prev_1st_order()
    fig, axes = plt.subplots(2, 6, figsize=(22, 8),
                              gridspec_kw=dict(height_ratios=[2, 1.3]))
    cmap = plt.get_cmap("viridis")
    n_lam = len(LAMBDA_LIST)
    colors = [cmap(i / max(1, n_lam - 1)) for i in range(n_lam)]
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    weeks_fit = np.arange(obs.shape[0])[mask]

    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        ax = axes[0, ai]
        ax.plot(weeks_fit, obs[mask, ai], "k-o", ms=3, label="obs")
        for li, r in enumerate(all_results):
            pred = np.asarray(r["diagnostic"]["pred_total_per_age"])
            ax.plot(weeks_fit, pred[mask, ai], "-", color=colors[li],
                    label=f"λ={r['lambda_phi']}", lw=1.2)
        ax.set_title(ag)
        ax.grid(True, alpha=0.3)
        if ai == 0:
            ax.set_ylabel("HIRA weekly count")
            ax.legend(fontsize=7)

    for ai in range(1, 6):
        axes[1, ai].axis("off")
    ax_phi = axes[1, 0]
    ax_phi.set_position([0.06, 0.07, 0.88, 0.22])
    ages_idx = np.arange(15)
    # 2nd-order solid
    for li, r in enumerate(all_results):
        phi15 = r["diagnostic"]["phi_full_15"]
        ax_phi.plot(ages_idx, phi15, "-o", color=colors[li],
                    label=f"2nd λ={r['lambda_phi']}", lw=1.6, ms=4)
    # 1st-order dashed (if available)
    if prev is not None:
        prev_lams = [pr["lambda_phi"] for pr in prev["results"]]
        prev_cmap = plt.get_cmap("plasma")
        for pi, pr in enumerate(prev["results"]):
            phi15 = pr["diagnostic"]["phi_full_15"]
            ax_phi.plot(ages_idx, phi15, "--", lw=1.0,
                        color=prev_cmap(pi / max(1, len(prev_lams) - 1)),
                        alpha=0.75, label=f"1st λ={pr['lambda_phi']}")
    ax_phi.axhline(1.0, color="grey", ls=":", lw=1)
    ax_phi.axvline(REF_AGE_IDX, color="red", ls=":", lw=1)
    ax_phi.axvline(3, color="magenta", ls=":", lw=1)   # highlight idx 3 (15-19)
    ax_phi.set_xticks(ages_idx)
    ax_phi.set_xticklabels([f"{5*i}-{5*i+4}" for i in range(14)] + ["70+"],
                            rotation=45, ha="right", fontsize=8)
    ax_phi.set_ylabel("φ_full (15)")
    ax_phi.set_title("best φ per λ  (solid = 2nd-order this run;  "
                     "dashed = 1st-order previous)  vertical magenta = idx 3 (15-19)")
    ax_phi.grid(True, alpha=0.3)
    ax_phi.legend(fontsize=7, ncol=3, loc="upper right")

    fig.suptitle(f"φ-free ({SEASON_LABEL}) 2nd-order penalty λ sweep  vs  "
                  f"1st-order overlay")
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
