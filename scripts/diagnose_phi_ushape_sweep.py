"""φ U-shape shape sweep — 3 axes (child_peak, adult_level, elder_rise).

Base U-shape (docs/parameter_justification.md A.2):
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5]

Anchor idx 5 = 1.0 always fixed.

Axes (one-at-a-time; center = base, others fixed at base):
    child_peak  ∈ {0.75, 1.0, 1.25}
      DIRECT multiplier on idx 0-3
      (child_peak=0.75 → idx 0 value 2.0×0.75 = 1.5,   1.9×0.75 = 1.425, ...)
    adult_level ∈ {0.85, 1.0, 1.15}
      DIRECT multiplier on idx 4, 6, 7, 8   (idx 5 anchor stays at 1.0)
    elder_rise  ∈ {0.7, 1.0, 1.3}
      Multiplier on the DEVIATION from 1.0 for idx 9-14 :
      φ_new[k] = 1.0 + elder_rise × (φ_base[k] − 1.0)
      (elder_rise=0.7, base 1.5 → 1.0 + 0.7×0.5 = 1.35 for idx 14)

7 unique combos + reference base. β_4 (+ phi_nb) free L-BFGS × 12,
NB obs, no channel prior (pin off — the point is to see whether work / home
revive as φ shape shifts).

Decision guide (comments only):
- Some shape brings β_w above lower bound (>0.005) → work sensitive to φ shape;
  pin-alternative possible.
- adult_level down → per-age r for 18-44/45-64 shifts (over→under prediction).
- elder_rise down → per-age r for 65+ shifts (over→under).
- child_peak → fine-tune r(0-5).

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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "phi_ushape_sweep.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "phi_ushape_sweep.png"
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

PHI_BASE_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5],
    dtype=np.float64,
)
CHILD_PEAK_GRID = [0.75, 1.0, 1.25]
ADULT_LEVEL_GRID = [0.85, 1.0, 1.15]
ELDER_RISE_GRID = [0.7, 1.0, 1.3]

N_STARTS = 12
START_SEED = 23
PEAK_HALF_WIN = 2
COVERAGE_CAP = 0.99

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_phi(child_peak: float, adult_level: float,
               elder_rise: float) -> np.ndarray:
    phi = PHI_BASE_USHAPE.copy()
    # Child: direct scale idx 0-3
    phi[0:4] = PHI_BASE_USHAPE[0:4] * child_peak
    # Adult: direct scale idx 4, 6, 7, 8 (skip anchor idx 5)
    for i in [4, 6, 7, 8]:
        phi[i] = PHI_BASE_USHAPE[i] * adult_level
    phi[5] = 1.0                                    # anchor
    # Elder: rise-multiplier on deviation from 1.0 for idx 9-14
    for i in range(9, 15):
        phi[i] = 1.0 + elder_rise * (PHI_BASE_USHAPE[i] - 1.0)
    return phi


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


def predict_hira(beta_4, phi_full_j, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_j
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


def fit_combo(sweep_var: str, child_peak: float, adult_level: float,
               elder_rise: float, setup) -> dict:
    phi = build_phi(child_peak, adult_level, elder_rise)
    phi_j = jnp.asarray(phi)
    fg = build_loss(phi_j, setup)
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
    pred = np.asarray(predict_hira(beta_4, phi_j, setup))
    ratios = per_age_ratios(pred, setup)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        phi_j,
    ))

    nll_arr = np.array(per_start_nll)
    return dict(
        sweep_var=sweep_var,
        child_peak=child_peak, adult_level=adult_level, elder_rise=elder_rise,
        phi_full_15=[float(x) for x in phi],
        beta_4=[float(x) for x in beta_4], phi_nb=phi_nb,
        nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios, wall_sec=float(wall),
    )


def main():
    print("=" * 78, flush=True)
    print(f"DIAGNOSE: φ U-shape shape sweep  —  {SEASON_LABEL}", flush=True)
    print(f"  base U-shape = {PHI_BASE_USHAPE.tolist()}", flush=True)
    print(f"  child_peak (idx 0-3 direct) ∈ {CHILD_PEAK_GRID}", flush=True)
    print(f"  adult_level (idx 4,6-8 direct, anchor 5 fixed) ∈ {ADULT_LEVEL_GRID}",
          flush=True)
    print(f"  elder_rise (deviation-scale idx 9-14) ∈ {ELDER_RISE_GRID}",
          flush=True)
    print(f"  free: β_4 + phi_nb   no channel prior", flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    combos = []
    # Center (base) — reference. Also enumerate the per-axis 3 levels; note
    # centers repeat; dedupe at reporting.
    for c in CHILD_PEAK_GRID:
        combos.append(("child_peak", c, 1.0, 1.0))
    for a in ADULT_LEVEL_GRID:
        combos.append(("adult_level", 1.0, a, 1.0))
    for e in ELDER_RISE_GRID:
        combos.append(("elder_rise", 1.0, 1.0, e))

    all_results = []
    seen_center = False
    for sw, cp, al, er in combos:
        if cp == 1.0 and al == 1.0 and er == 1.0:
            # This is the base — only run once, tag sweep_var as "base"
            if seen_center:
                # Copy from prior center run
                center_rec = next(r for r in all_results
                                    if r["sweep_var"] == "base")
                rec = dict(center_rec); rec["sweep_var"] = sw
                all_results.append(rec)
                print(f"\n── sweep={sw}  (cp,al,er)=(1.0,1.0,1.0) = base "
                      f"[reuse] ──", flush=True)
                continue
            sw_use = "base"
            print(f"\n── sweep=base  (cp,al,er)=(1.0,1.0,1.0) ──", flush=True)
            seen_center = True
        else:
            sw_use = sw
            print(f"\n── sweep={sw}  (cp,al,er)=({cp},{al},{er}) ──", flush=True)
        r = fit_combo(sw_use, cp, al, er, setup)
        all_results.append(r)
        print(f"    NLL={r['nll']:.4e}  β_4={[round(x,4) for x in r['beta_4']]}"
              f"  R0={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
              f"std={r['nll_std']:.2e}  wall={r['wall_sec']:.1f}s", flush=True)
        print(f"    φ_full = {[round(x,3) for x in r['phi_full_15']]}",
              flush=True)
        print(f"    ★ β_w = {r['beta_4'][1]:.4f}   β_h = {r['beta_4'][0]:.4f}",
              flush=True)
        print(f"    per-age r:", flush=True)
        for ag in HIRA_AGE_GROUPS:
            print(f"      {ag:>6s}: {r['per_age'][ag]['ratio']:.2f}",
                  flush=True)

    print("\n" + "=" * 78, flush=True)
    print("  Summary — β/NLL + β_w focus", flush=True)
    print(f"  {'sweep':>12s}  {'cp':>4s} {'al':>4s} {'er':>4s}  "
          f"{'β_h':>7s} {'β_w':>7s} {'β_s':>7s} {'β_o':>7s}  "
          f"{'NLL':>10s}  {'R0':>5s}  {'std':>8s}", flush=True)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['sweep_var']:>12s}  {r['child_peak']:>4.2f} "
              f"{r['adult_level']:>4.2f} {r['elder_rise']:>4.2f}  "
              f"{b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f}  {b[3]:.4f}  "
              f"{r['nll']:.4e}  {r['R0_ngm']:.3f}  "
              f"{r['nll_std']:.2e}", flush=True)

    print("\n  Per-age r  (obs_peak±2w / model_peak±2w)", flush=True)
    header = f"  {'sweep':>12s}  {'cp':>4s} {'al':>4s} {'er':>4s}  " + \
             "  ".join(f"{ag:>7s}" for ag in HIRA_AGE_GROUPS)
    print(header, flush=True)
    for r in all_results:
        row = (f"  {r['sweep_var']:>12s}  {r['child_peak']:>4.2f} "
               f"{r['adult_level']:>4.2f} {r['elder_rise']:>4.2f}  ")
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
                "PHI_BASE_USHAPE": PHI_BASE_USHAPE.tolist(),
                "child_peak_grid": CHILD_PEAK_GRID,
                "adult_level_grid": ADULT_LEVEL_GRID,
                "elder_rise_grid": ELDER_RISE_GRID,
                "vaccine": "A_fix cov_eff = -ln(1-cov)",
                "channel_prior": "NONE",
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # Figure: 2 panels — β_4 per combo, per-age r per combo
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels = [f"{r['sweep_var']} cp={r['child_peak']} al={r['adult_level']} "
              f"er={r['elder_rise']}" for r in all_results]

    # Panel 1: β_4 stacked bars per combo
    ax = axes[0]
    x = np.arange(len(all_results))
    bw = 0.18
    for k, (ch, col) in enumerate([("β_h", "#1a5490"), ("β_w", "#c0392b"),
                                     ("β_s", "#27ae60"), ("β_o", "#f39c12")]):
        vals = [r["beta_4"][k] for r in all_results]
        ax.bar(x + (k - 1.5) * bw, vals, bw, label=ch, color=col)
    ax.axhline(0.001, color="grey", ls=":", lw=1, label="lower bound 0.001")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right",
                                             fontsize=8)
    ax.set_ylabel("β")
    ax.set_title("β_4 per combo  (free channel, no pin)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: per-age r heatmap
    ax = axes[1]
    ages = HIRA_AGE_GROUPS
    xa = np.arange(len(ages))
    combo_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_results)))
    bw2 = 0.09
    for i, r in enumerate(all_results):
        offset = (i - len(all_results) / 2) * bw2
        vals = [r["per_age"][ag]["ratio"] for ag in ages]
        ax.bar(xa + offset, vals, bw2,
                color=combo_colors[i], label=labels[i], alpha=0.85)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(ages)
    ax.set_ylabel("r = obs / model")
    ax.set_title("Per-age r per combo")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"φ U-shape shape sweep  —  {SEASON_LABEL}")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
