"""Diagnose vaccine bug fixes A + B together.

Bug A: v_rate should be −ln(1 − coverage) × density (not coverage × density),
       so that ∫v_rate dt = −ln(1 − coverage) → 1 − exp(−∫) = coverage
       exactly. In production: coverage=0.82 → 1 − exp(−0.82) = 0.56 (wrong).
Bug B: R(0) elder 0.65 double-counts vaccine (also modelled dynamically via
       V compartment). Remove vaccine share from R(0), keep only natural /
       cross-reactive immunity.

Production dynamics_jax.vax_rate_vector_jax computes v_rate = coverage × density.
Trick: pass annual_coverage_eff = −ln(1 − coverage_true) into the ODE. Then
production computes coverage_eff × density = −ln(1 − coverage) × density,
which is the corrected rate. No source code modification.

Conditions (all with φ = ones(15) FIXED):
  a_baseline   : R(0) default,               coverage_eff = coverage (buggy)
  d_A_only     : R(0) default,               coverage_eff = −ln(1 − cov) (A fix)
  e_AB         : R(0) elder ≤ 0.35 (natural), coverage_eff = −ln(1 − cov) (A+B)

R(0) in condition e (removing vaccine share):
  0-19 : 0.10                (unchanged — no vaccine share in original)
  20-49: 0.30                (unchanged — no vaccine share)
  50-64: 0.20                (was 0.45 : "regular vaccine + accumulation")
  65+  : 0.35                (was 0.65 : "vaccine 82% + cross-imm";
                               0.35 = trial cross-imm/natural only)

Decision guide (comments only):
- e r(65+) ≈ 0.97 (baseline level) → single-count vaccine already fits elder;
  fix is complete.
- e r(65+) < 0.7 → 0.35 natural too low; try 0.4–0.5.
- d shows S→V matches coverage (verifies A fix).
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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "vaccine_fixed_v2.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "vaccine_fixed_v2.png"
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
COVERAGE_CAP = 0.99   # avoid log(0) blow-up in −ln(1−cov)

BETA_BOUNDS = [(0.001, 1.0)] * 4
PHI_NB_BOUNDS = (1e-3, 1e6)

# Condition e: R(0) with vaccine share removed
R0_IMMUNITY_E = np.array(
    [0.10] * 4 + [0.30] * 6 + [0.20] * 3 + [0.35] * 2,
    dtype=np.float64,
)


def correct_coverage(cov_15: np.ndarray) -> np.ndarray:
    """Bug A fix: return −ln(1 − cov) so that production's
    v_rate = coverage_eff × density matches the intended rate."""
    cov_clip = np.minimum(cov_15, COVERAGE_CAP)
    return -np.log(1.0 - cov_clip)


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
        pop_15=pop_15, seed_15=seed_15, vax=vax, matrices=matrices,
        rho_emp=rho_emp, disease=disease,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
    )


def build_state0(immunity_15: np.ndarray, setup) -> jnp.ndarray:
    return jnp.asarray(_build_initial_state_with_age_seed(
        setup["pop_15"], setup["seed_15"], seed_e_factor=0.5,
        initial_immunity=immunity_15, initial_vaccinated_fraction=0.0,
    ))


def build_shared(annual_coverage_eff_15: np.ndarray, setup):
    kw = dict(setup["shared_base"])
    kw["annual_coverage"] = jnp.asarray(annual_coverage_eff_15)
    return kw


def predict_hira(beta_4, phi_full_15, state0, shared, setup):
    kw = dict(shared)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_15
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


def elder_svr_trajectory(states: jnp.ndarray, age_idx: int = 14) -> dict:
    S = np.asarray(states[:, IDX_S, age_idx, :].sum(axis=-1))
    V = np.asarray(states[:, IDX_V, age_idx, :].sum(axis=-1))
    R = np.asarray(states[:, IDX_R, age_idx, :].sum(axis=-1))
    E = np.asarray(states[:, 2, age_idx, :].sum(axis=-1))
    I = np.asarray(states[:, 3, age_idx, :].sum(axis=-1))
    N = S + V + E + I + R
    n_t = states.shape[0]
    end = n_t - 1
    S0, V0, R0 = float(S[0]), float(V[0]), float(R[0])
    Se, Ve, Re = float(S[end]), float(V[end]), float(R[end])
    N0 = float(N[0])

    def pct(x, y):
        return 100.0 * x / max(y, 1e-9)

    return dict(
        age_idx=age_idx, N0=N0,
        day0=dict(S=S0, V=V0, R=R0, S_pct=pct(S0, N0),
                  V_pct=pct(V0, N0), R_pct=pct(R0, N0)),
        day_end=dict(day=end, S=Se, V=Ve, R=Re, S_pct=pct(Se, N0),
                     V_pct=pct(Ve, N0), R_pct=pct(Re, N0)),
        frac_S0_to_V_by_end=pct(Ve - V0, S0),
        frac_S0_to_R_by_end=pct(Re - R0, S0),
        frac_S0_remaining=pct(Se, S0),
        S_loss_pct=pct(S0 - Se, S0),
    )


def fit_condition(name: str, immunity_15, cov_eff_15, setup) -> dict:
    state0 = build_state0(immunity_15, setup)
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
    elder = elder_svr_trajectory(states, age_idx=14)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        jnp.asarray(phi_full),
    ))

    nll_arr = np.array(per_start_nll)
    return dict(
        condition=name,
        immunity_15=[float(x) for x in immunity_15],
        cov_eff_15=[float(x) for x in cov_eff_15],
        beta_4=[float(x) for x in beta_4],
        phi_nb=phi_nb, nll=best["nll"],
        R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios,
        elderly_70p_trajectory=elder,
        wall_sec=float(wall),
        pred_hira=pred.tolist(),
    )


def main():
    print("=" * 78, flush=True)
    print(f"DIAGNOSE: vaccine bug A + B fixes  —  {SEASON_LABEL}", flush=True)
    print(f"  free: β_4 + phi_nb   φ = ones(15) FIXED", flush=True)
    print(f"  A fix: coverage_eff = −ln(1 − cov)  (cap {COVERAGE_CAP})", flush=True)
    print(f"  B fix: R(0)_e = [.10×4,.30×6,.20×3,.35×2]  (vaccine share out)",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()
    cov_default = np.asarray(setup["vax"].annual_coverage, dtype=np.float64)
    cov_A = correct_coverage(cov_default)

    print(f"  R0_IMMUNITY default (15) = "
          f"{[round(float(x),2) for x in R0_IMMUNITY_PROFILE]}", flush=True)
    print(f"  R0_IMMUNITY (e) natural-only = "
          f"{[round(float(x),2) for x in R0_IMMUNITY_E]}", flush=True)
    print(f"  cov default                  = "
          f"{[round(float(x),2) for x in cov_default]}", flush=True)
    print(f"  cov_eff (A fix) = -ln(1-cov) = "
          f"{[round(float(x),4) for x in cov_A]}", flush=True)
    for i in [0, 4, 14]:
        print(f"    idx {i:>2d} cov={cov_default[i]:.2f} → cov_eff={cov_A[i]:.4f}   "
              f"1-exp(-cov_eff)={1-np.exp(-cov_A[i]):.4f} (=cov target)",
              flush=True)
    print(f"  n_weeks = {setup['n_weeks']}", flush=True)

    conditions = [
        ("a_baseline",  R0_IMMUNITY_PROFILE, cov_default),   # buggy prod
        ("d_A_only",    R0_IMMUNITY_PROFILE, cov_A),
        ("e_AB",        R0_IMMUNITY_E,       cov_A),
    ]

    all_results = []
    for name, imm, cov_eff in conditions:
        print(f"\n── fit condition [{name}] ──", flush=True)
        r = fit_condition(name, imm, cov_eff, setup)
        all_results.append(r)
        elder = r["elderly_70p_trajectory"]
        print(f"    NLL={r['nll']:.4e}  β_4={[round(x,4) for x in r['beta_4']]}  "
              f"R0_ngm={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
              f"wall={r['wall_sec']:.1f}s", flush=True)
        print(f"    NLL 12 starts: min={r['nll_min']:.4e}  max={r['nll_max']:.4e}"
              f"  std={r['nll_std']:.3e}", flush=True)
        print(f"    ★ elder (idx 14 = 70+):", flush=True)
        print(f"       day 0  S/N={elder['day0']['S_pct']:.2f}%   "
              f"day end S/N={elder['day_end']['S_pct']:.2f}%  "
              f"V/N={elder['day_end']['V_pct']:.2f}%  "
              f"R/N={elder['day_end']['R_pct']:.2f}%", flush=True)
        print(f"       S(0)→V: {elder['frac_S0_to_V_by_end']:.2f}%   "
              f"S(0)→R: {elder['frac_S0_to_R_by_end']:.2f}%   "
              f"S remaining: {elder['frac_S0_remaining']:.2f}%", flush=True)
        print(f"    per-age r:", flush=True)
        for ag in HIRA_AGE_GROUPS:
            print(f"       {ag:>6s}: {r['per_age'][ag]['ratio']:.2f}"
                  f"  (peak offset {r['per_age'][ag]['phase_offset_weeks']:+d}w)",
                  flush=True)

    print("\n" + "=" * 78, flush=True)
    print("  Condition sweep — β_4 + NLL", flush=True)
    print("  cond          β_h     β_w     β_s     β_o     NLL          R0", flush=True)
    for r in all_results:
        b = r["beta_4"]
        print(f"  {r['condition']:>12s}  {b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f}  "
              f"{b[3]:.4f}  {r['nll']:.4e}  {r['R0_ngm']:.3f}", flush=True)

    print("\n  Elderly (70+) — S(0)→V + S remaining", flush=True)
    print("  cond          S(0)%   S(end)%  V(end)%  R(end)%  →V%    S rem%",
          flush=True)
    for r in all_results:
        e = r["elderly_70p_trajectory"]
        print(f"  {r['condition']:>12s}  {e['day0']['S_pct']:5.2f}  "
              f"{e['day_end']['S_pct']:5.2f}   {e['day_end']['V_pct']:5.2f}   "
              f"{e['day_end']['R_pct']:5.2f}   {e['frac_S0_to_V_by_end']:5.2f}  "
              f"{e['frac_S0_remaining']:5.2f}", flush=True)

    print("\n  Per-age r (obs_peak±2w / model_peak±2w)", flush=True)
    print("  cond          " + "  ".join(f"{ag:>7s}" for ag in HIRA_AGE_GROUPS),
          flush=True)
    for r in all_results:
        row = f"  {r['condition']:>12s}  "
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
                "R0_IMMUNITY_E_natural_only": R0_IMMUNITY_E.tolist(),
                "annual_coverage_default": cov_default.tolist(),
                "annual_coverage_eff_A_fix": cov_A.tolist(),
                "coverage_cap": COVERAGE_CAP,
                "phi": "ones(15) FIXED",
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [r["condition"] for r in all_results]
    x = np.arange(len(labels))
    w = 0.25
    s_pct = [r["elderly_70p_trajectory"]["day_end"]["S_pct"] for r in all_results]
    v_pct = [r["elderly_70p_trajectory"]["day_end"]["V_pct"] for r in all_results]
    r_pct = [r["elderly_70p_trajectory"]["day_end"]["R_pct"] for r in all_results]

    axes[0].bar(x - w, s_pct, w, label="S end%", color="#1a5490")
    axes[0].bar(x, v_pct, w, label="V end%", color="#27ae60")
    axes[0].bar(x + w, r_pct, w, label="R end%", color="#c0392b")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("% of N (70+)")
    axes[0].set_title("Elderly 70+ end-of-season S/V/R (%)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].set_title("Per-age r  (obs_peak±2w / model_peak±2w)")
    n_ages = len(HIRA_AGE_GROUPS)
    xa = np.arange(n_ages)
    bw = 0.25
    cond_colors = ["#888888", "#1a5490", "#c0392b"]
    for i, r in enumerate(all_results):
        rvals = [r["per_age"][ag]["ratio"] for ag in HIRA_AGE_GROUPS]
        offset = (i - 1) * bw
        axes[1].bar(xa + offset, rvals, bw, color=cond_colors[i],
                     label=r["condition"], alpha=0.85)
    axes[1].axhline(1.0, color="grey", ls=":", lw=1)
    axes[1].set_xticks(xa); axes[1].set_xticklabels(HIRA_AGE_GROUPS)
    axes[1].set_ylabel("r = obs / model")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Vaccine fix v2 (A: coverage=-ln(1-cov); B: R(0) natural)  "
                  f"—  {SEASON_LABEL}  (φ = ones(15) fixed)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
