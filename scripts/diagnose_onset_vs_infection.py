"""Diagnose: fitting target — infection Δ(E+I+R) vs onset Δ(I+R).

Hypothesis: HIRA records claims at symptom onset (not infection). The current
production target is infection incidence Δ(E+I+R) (daily_new_infection_by_age_jax),
which is shifted ~1/σ days earlier than onset. With σ=0.5 → 2-day latent period,
weekly aggregation may absorb this, but residual peak misalignment / lag pattern
should reveal whether the shift matters.

Test: single-season (2019-20) point estimation, free β_4 (φ=1.0, γ fixed), with
2 observation models × 2 targets = 4 fits:
    {Poisson, NB-2} × {infection Δ(E+I+R), onset Δ(I+R)}

Decision rule (in code comments):
- onset shows lower NLL + better peak alignment + smaller residual lag
  → adopt onset as the calibration target.
- Improvement under Poisson but flat under NB → NB dispersion absorbs the
  lag (phi_nb 폭주 connection).
- All four indistinguishable → 2-day shift absorbed by weekly aggregation;
  pursue other root causes for peak over-prediction.

This script DOES NOT modify production code. It imports daily_new_infection_by_age_jax
for the infection target and defines a local daily_new_onset_by_age_jax for onset.
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
from kt_epimodel_hira.calibration.param_vector import ParameterBounds
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax, IDX_I, IDX_R,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, poisson_nll_jax, nb_nll_jax,
    gamma_triple_to_15,
)


# ─── Setup (mirror m2_policy_compare_AB.py / m2_production_chprior.py) ─────
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR_EDA = REPO_ROOT / "outputs" / "eda"
OUTDIR_EDA.mkdir(parents=True, exist_ok=True)
FIG_PATH = REPO_ROOT / "presentations" / "figures" / "onset_vs_infection.png"

SEASON_LABEL = "2019-2020"
SEASON_IDX = 2
AMP = 0.9
HOLIDAY = dict(
    school_holiday_amp=0.7,
    school_holiday_start_day=113.0,
    school_holiday_min_start_day=127.0,
    school_holiday_min_end_day=162.0,
    school_holiday_end_day=183.0,
    school_holiday_realloc=1.0,
)
# γ: production CDC absolute (matches m2_production_chprior.py)
GAMMA_3 = np.array([0.40, 0.18, 0.25])    # child, adult, elder

N_STARTS = 4
START_SEED = 7

# Bounds for β_4 — ParameterBounds default is (0.001, 5.0); we cap upper at 1.0
# here to avoid the R_eff-cliff region (β >> 1 → R0 explosion → point-estimate
# can land on the suppression edge, distorting NLL comparisons).
BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)               # NB dispersion: only positive, wide
BETA_UPPER_WARN_FRAC = 0.99               # warn when best β ≥ 99% of upper bound


# ─── Incidence definitions (★ the only one-line difference between targets) ──
def daily_new_infection_by_age(states: jnp.ndarray) -> jnp.ndarray:
    """Infection incidence: Δ(E+I+R). Same as production helper.
    Imported daily_new_infection_by_age_jax is reused for sanity comparison."""
    # ★ only difference vs onset: include E.
    sum_states = (
        states[:, 2, :, :].sum(axis=-1)   # E
        + states[:, IDX_I, :, :].sum(axis=-1)
        + states[:, IDX_R, :, :].sum(axis=-1)
    )
    return jnp.diff(sum_states, axis=0)   # (n_days-1, 15)


def daily_new_onset_by_age(states: jnp.ndarray) -> jnp.ndarray:
    """Onset (symptom onset) incidence: Δ(I+R). One-line difference from
    daily_new_infection_by_age: E is omitted (transitions E→I are onsets)."""
    # ★ only difference vs infection: omit E.
    sum_states = (
        states[:, IDX_I, :, :].sum(axis=-1)
        + states[:, IDX_R, :, :].sum(axis=-1)
    )
    return jnp.diff(sum_states, axis=0)   # (n_days-1, 15)


# ─── Setup shared simulation kwargs (identical for both targets) ──────────
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
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]
        w[:, i] = tgt["weights"][ag]
    obs_j = jnp.asarray(obs); w_j = jnp.asarray(w)

    gamma_15 = gamma_triple_to_15(GAMMA_3[0], GAMMA_3[1], GAMMA_3[2])

    return shared, phi_full, state0, obs_j, w_j, gamma_15, n_weeks, obs


def predict_hira(beta_4, *, shared, phi_full, state0, gamma_15, n_weeks,
                 incidence_fn):
    """Run forward sim + chosen incidence + HIRA conversion. (n_weeks, 6) pred."""
    kw = dict(shared)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    states = simulate_jax(state0, **kw, discretize_time=False)
    inc_15 = incidence_fn(states)
    pred_hira = simulation_to_hira_by_age_jax(inc_15, gamma_15, n_weeks=n_weeks)
    return pred_hira


def build_nll(*, obs_model: str, incidence_fn, shared, phi_full, state0,
              obs_j, w_j, gamma_15, n_weeks):
    """Return (jit'd loss(x) → scalar, x_dim) for {Poisson, NB} × incidence."""
    if obs_model == "poisson":
        def loss(x):  # x = β_4
            pred = predict_hira(
                x, shared=shared, phi_full=phi_full, state0=state0,
                gamma_15=gamma_15, n_weeks=n_weeks, incidence_fn=incidence_fn,
            )
            return poisson_nll_jax(obs_j, pred, w_j, min_rate=0.01)
        x_dim = 4
    elif obs_model == "nb":
        def loss(x):  # x = [β_4, phi_nb]
            beta_4 = x[:4]; phi_nb = x[4]
            pred = predict_hira(
                beta_4, shared=shared, phi_full=phi_full, state0=state0,
                gamma_15=gamma_15, n_weeks=n_weeks, incidence_fn=incidence_fn,
            )
            return nb_nll_jax(obs_j, pred, w_j, concentration=phi_nb,
                              min_rate=0.01)
        x_dim = 5
    else:
        raise ValueError(obs_model)

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

    return fg_np, x_dim


def make_init_set(x_dim: int, n_starts: int, seed: int) -> list[np.ndarray]:
    """N fixed-seed starting points, x_dim ∈ {4, 5}. Same across all 4 fits."""
    rng = np.random.default_rng(seed)
    # Reasonable β_4 range: 0.05 ~ 0.5 (covers production NIMS/literature mixes)
    starts = []
    base_betas = [
        np.array([0.10, 0.10, 0.10, 0.10]),
        np.array([0.20, 0.20, 0.20, 0.20]),
        np.array([0.05, 0.05, 0.25, 0.25]),
        np.array([0.15, 0.05, 0.15, 0.25]),
    ][:n_starts]
    for b in base_betas:
        x0 = b.copy()
        if x_dim == 5:
            x0 = np.concatenate([x0, np.array([10.0])])   # phi_nb init mid-range
        starts.append(x0)
    return starts


def fit_one(obs_model: str, incidence_fn, setup) -> dict:
    """4 starts L-BFGS-B; return best fit."""
    shared, phi_full, state0, obs_j, w_j, gamma_15, n_weeks, _ = setup
    fg, x_dim = build_nll(
        obs_model=obs_model, incidence_fn=incidence_fn,
        shared=shared, phi_full=phi_full, state0=state0,
        obs_j=obs_j, w_j=w_j, gamma_15=gamma_15, n_weeks=n_weeks,
    )
    if x_dim == 4:
        bounds = BETA_BOUNDS
    else:
        bounds = BETA_BOUNDS + [PHI_NB_BOUNDS]

    starts = make_init_set(x_dim, N_STARTS, START_SEED)
    best = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(
                fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                options=dict(maxiter=200, ftol=1e-9, gtol=1e-6),
            )
            nll = float(res.fun)
        except Exception as e:
            print(f"  [warn] start {i} failed: {e}")
            continue
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0
    best["wall_sec"] = float(wall)
    best["obs_model"] = obs_model
    return best


def diagnose_fit(best, incidence_fn, setup) -> dict:
    """Compute pred curve, peak alignment, residual lag pattern."""
    shared, phi_full, state0, obs_j, w_j, gamma_15, n_weeks, obs_np = setup
    beta_4 = best["x"][:4]
    pred_hira = np.asarray(predict_hira(
        beta_4, shared=shared, phi_full=phi_full, state0=state0,
        gamma_15=gamma_15, n_weeks=n_weeks, incidence_fn=incidence_fn,
    ))   # (n_weeks, 6)

    # peak = argmax over weeks of total across age groups
    obs_total = obs_np.sum(axis=1)
    pred_total = pred_hira.sum(axis=1)
    obs_peak_w = int(np.argmax(obs_total))
    pred_peak_w = int(np.argmax(pred_total))
    peak_offset = pred_peak_w - obs_peak_w     # +ve = pred peak later

    # residuals: obs - pred (weekly total)
    resid = obs_total - pred_total

    # per-age peak ratios (pred/obs)
    obs_peak_per_age = obs_np.max(axis=0)
    pred_peak_per_age = pred_hira.max(axis=0)
    peak_ratio_per_age = np.where(
        obs_peak_per_age > 0, pred_peak_per_age / obs_peak_per_age, np.nan,
    )

    return {
        "beta_4": [float(x) for x in beta_4],
        "phi_nb": float(best["x"][4]) if len(best["x"]) == 5 else None,
        "pred_total": pred_total.tolist(),
        "obs_total": obs_total.tolist(),
        "residual_total": resid.tolist(),
        "obs_peak_week": obs_peak_w,
        "pred_peak_week": pred_peak_w,
        "peak_offset_weeks": peak_offset,
        "peak_ratio_per_age": peak_ratio_per_age.tolist(),
        "nll": best["nll"],
        "best_start_idx": best["start_idx"],
        "wall_sec": best["wall_sec"],
    }


def main():
    print("=" * 70)
    print("DIAGNOSE: onset Δ(I+R) vs infection Δ(E+I+R) target — 2019-20")
    print(f"  obs models: Poisson, NB(free phi_nb)")
    print(f"  free params: β_4 (+ phi_nb for NB), φ=1, γ=CDC absolute")
    print(f"  multi-start: {N_STARTS} (fixed seed {START_SEED})")
    print("=" * 70)

    setup = build_setup()
    _, _, _, obs_j, _, _, n_weeks, _ = setup
    print(f"  obs shape: {obs_j.shape}  n_weeks={n_weeks}")
    print(f"  HOLIDAY realloc={HOLIDAY['school_holiday_realloc']} "
          f"amp={HOLIDAY['school_holiday_amp']}  AMP={AMP}")
    print(f"  GAMMA_3 (child, adult, elder) = {GAMMA_3.tolist()}")
    print(f"  bounds β_4: {BETA_BOUNDS}")

    targets = {
        "infection": daily_new_infection_by_age,
        "onset":     daily_new_onset_by_age,
    }
    obs_models = ["poisson", "nb"]

    results = {}
    for tname, inc_fn in targets.items():
        for omodel in obs_models:
            key = f"{omodel}_{tname}"
            print(f"\n── fit [{key}] ──")
            best = fit_one(omodel, inc_fn, setup)
            diag = diagnose_fit(best, inc_fn, setup)
            print(f"  NLL={diag['nll']:.4e}  peak obs/pred = "
                  f"{diag['obs_peak_week']}/{diag['pred_peak_week']} "
                  f"(offset {diag['peak_offset_weeks']:+d}w)  "
                  f"β_4={[round(x,3) for x in diag['beta_4']]}"
                  + (f"  phi_nb={diag['phi_nb']:.2f}" if diag['phi_nb'] else ""))
            # ★ Warn if best β touches upper bound (cliff risk indicator)
            for ch, (b, (_, hi)) in enumerate(zip(diag["beta_4"], BETA_BOUNDS)):
                if b >= BETA_UPPER_WARN_FRAC * hi:
                    print(f"  [WARN] β[{ch}]={b:.4f} ≥ {BETA_UPPER_WARN_FRAC*100:.0f}% "
                          f"of upper bound {hi} — cliff region; result may be unreliable")
            results[key] = diag

    # Cross comparison table
    print("\n" + "=" * 70)
    print("  fit                    NLL          peak(obs/pred)  offset   |resid|_max")
    print("-" * 70)
    for key, d in results.items():
        max_abs_resid = float(np.max(np.abs(d["residual_total"])))
        print(f"  {key:22s}  {d['nll']:.4e}     {d['obs_peak_week']:>2d}/"
              f"{d['pred_peak_week']:>2d}        {d['peak_offset_weeks']:+d}     "
              f"{max_abs_resid:>10,.0f}")
    print("=" * 70)

    # NLL gaps
    for omodel in obs_models:
        nll_inf = results[f"{omodel}_infection"]["nll"]
        nll_onset = results[f"{omodel}_onset"]["nll"]
        gap = nll_inf - nll_onset
        print(f"  Δ NLL ({omodel}): infection − onset = {gap:+.4e}  "
              f"({'onset better' if gap > 0 else 'infection better'})")

    # ─── Save summary JSON (no large arrays beyond curves) ────────────────
    out_json = OUTDIR_EDA / "onset_vs_infection_diag.json"
    with open(out_json, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3": GAMMA_3.tolist(),
                "bounds_beta": BETA_BOUNDS,
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "phi": 1.0,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nsaved {out_json}")

    # ─── Figure: obs vs pred + residuals, Poisson / NB panels ─────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    for col, omodel in enumerate(obs_models):
        ax_top = axes[0, col]; ax_bot = axes[1, col]
        d_inf = results[f"{omodel}_infection"]
        d_ons = results[f"{omodel}_onset"]
        weeks = np.arange(len(d_inf["obs_total"]))
        ax_top.plot(weeks, d_inf["obs_total"], "k-o", label="obs", ms=4)
        ax_top.plot(weeks, d_inf["pred_total"], "-",
                    color="#1a5490", label="pred (infection)")
        ax_top.plot(weeks, d_ons["pred_total"], "-",
                    color="#c0392b", label="pred (onset)")
        ax_top.set_ylabel("weekly HIRA total (all ages)")
        ax_top.set_title(
            f"{omodel.upper()}  NLL inf={d_inf['nll']:.2e} / "
            f"onset={d_ons['nll']:.2e}\n"
            f"peak offset inf={d_inf['peak_offset_weeks']:+d}w / "
            f"onset={d_ons['peak_offset_weeks']:+d}w"
        )
        ax_top.legend(); ax_top.grid(True, alpha=0.3)

        ax_bot.axhline(0, color="grey", ls=":")
        ax_bot.plot(weeks, d_inf["residual_total"], "-",
                    color="#1a5490", label="infection resid")
        ax_bot.plot(weeks, d_ons["residual_total"], "-",
                    color="#c0392b", label="onset resid")
        ax_bot.set_xlabel("week")
        ax_bot.set_ylabel("obs − pred")
        ax_bot.legend(); ax_bot.grid(True, alpha=0.3)

    fig.suptitle(f"Onset Δ(I+R) vs infection Δ(E+I+R) — {SEASON_LABEL} "
                 f"single-season point estimate (φ=1, γ=CDC)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=130, bbox_inches="tight")
    print(f"saved {FIG_PATH}")


if __name__ == "__main__":
    main()
