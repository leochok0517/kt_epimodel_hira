"""γ_report all-age sensitivity + policy robustness under bug A fix.

One-at-a-time sweep around center (child=0.40, adult=0.18, elder=0.25):
    child ∈ {0.30, 0.40, 0.50}  (adult 0.18, elder 0.25 fixed)
    adult ∈ {0.12, 0.18, 0.30}  (child 0.40, elder 0.25 fixed)
    elder ∈ {0.25, 0.35, 0.45}  (child 0.40, adult 0.18 fixed)
= 9 combos (center 0.40/0.18/0.25 appears in all 3 sweeps → 7 unique + reference).

Setup: φ=ones(15), R(0)=default [.10×4,.30×6,.45×3,.65×2], vaccine A-fix on
(cov_eff = −ln(1 − cov)), single season 2019-2020. β_4 (+phi_nb) free
L-BFGS × 12 starts, NB obs.

Measurement A: β_4, NLL, R0, per-age r(6-group), multistart std.
Measurement B: policy sick_leave — using the fitted β_4, forward with
    baseline (p_work=1.0) vs sick_leave (p_work=0.4)
    averted_total  = 100 · (base − sick) / base   (total daily new infections)
    averted_by_age = 100 · (base_age − sick_age) / base_age   per HIRA age.

Baseline production reference (buggy vaccine, γ_center):
    r = 2.15 / 1.05 / 0.72 / 0.86 / 0.86 / 0.97 (from vaccine_doublecount.py)

Decision guide (comments only):
- Each γ_group monotonically moves its own age's r (γ_elder → 65+, etc.).
- averted% stable across γ range → policy conclusion robust.
- γ_adult especially moves work-side averted (18-44, 45-64) → sensitive.

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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "gamma_allage.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "gamma_allage.png"
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

CENTER = dict(child=0.40, adult=0.18, elder=0.25)
CHILD_GRID = [0.30, 0.40, 0.50]
ADULT_GRID = [0.12, 0.18, 0.30]
ELDER_GRID = [0.25, 0.35, 0.45]

P_WORK_SICK = 0.4       # sick_leave scenario
P_SCHOOL_BASE = 1.0
P_WORK_BASE = 1.0

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
        VE=vax.VE,
        annual_coverage=jnp.asarray(cov_eff),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=AMP,
        seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )
    shared_base.update(HOLIDAY)

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
        shared_base=shared_base, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
    )


def run_forward(beta_4, phi_full, gamma_15_j, p_school, p_work, setup):
    kw = dict(setup["shared_base"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    kw["p_school"] = p_school; kw["p_work"] = p_work
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc_15 = daily_new_infection_by_age_jax(st)   # (n_days-1, 15)
    pred_hira = simulation_to_hira_by_age_jax(
        inc_15, gamma_15_j, n_weeks=setup["n_weeks"],
    )
    return inc_15, pred_hira


def build_loss(gamma_15_j, setup):
    phi_full = jnp.ones(15)

    def loss(x):
        beta_4 = x[:4]; phi_nb = x[4]
        _, pred = run_forward(beta_4, phi_full, gamma_15_j,
                                P_SCHOOL_BASE, P_WORK_BASE, setup)
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


def fit_combo(child, adult, elder, sweep_var, setup) -> dict:
    gamma_15 = build_gamma_15(child, adult, elder)
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

    # (A) fit measurement
    _, pred_base = run_forward(beta_4, phi_full, gamma_15_j,
                                 P_SCHOOL_BASE, P_WORK_BASE, setup)
    pred_base_np = np.asarray(pred_base)
    ratios = per_age_ratios(pred_base_np, setup)

    # (B) policy sick_leave — same β_4, p_work=0.4
    inc_base, pred_base_hira = run_forward(
        beta_4, phi_full, gamma_15_j, P_SCHOOL_BASE, P_WORK_BASE, setup)
    inc_sick, pred_sick_hira = run_forward(
        beta_4, phi_full, gamma_15_j, P_SCHOOL_BASE, P_WORK_SICK, setup)

    inc_base_np = np.asarray(inc_base)   # (n_days-1, 15)
    inc_sick_np = np.asarray(inc_sick)
    total_base = float(inc_base_np.sum())
    total_sick = float(inc_sick_np.sum())
    averted_total_pct = 100.0 * (total_base - total_sick) / max(total_base, 1.0)

    # per-HIRA-age averted from the HIRA-mapped predictions (weekly totals)
    pred_base_np_h = np.asarray(pred_base_hira)   # (n_weeks, 6)
    pred_sick_np_h = np.asarray(pred_sick_hira)
    per_age_averted = {}
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        b = float(pred_base_np_h[:, ai].sum())
        s = float(pred_sick_np_h[:, ai].sum())
        per_age_averted[ag] = 100.0 * (b - s) / max(b, 1e-9)

    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        jnp.asarray(phi_full),
    ))

    nll_arr = np.array(per_start_nll)
    return dict(
        sweep_var=sweep_var,
        gamma_child=child, gamma_adult=adult, gamma_elder=elder,
        beta_4=[float(x) for x in beta_4], phi_nb=phi_nb,
        nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios,
        averted_total_pct=averted_total_pct,
        averted_by_age_pct=per_age_averted,
        wall_sec=float(wall),
    )


def main():
    print("=" * 78, flush=True)
    print(f"DIAGNOSE: γ_report all-age sweep + policy robustness  —  "
          f"{SEASON_LABEL}", flush=True)
    print(f"  center γ = child={CENTER['child']} adult={CENTER['adult']} "
          f"elder={CENTER['elder']}", flush=True)
    print(f"  child ∈ {CHILD_GRID}   adult ∈ {ADULT_GRID}   elder ∈ {ELDER_GRID}",
          flush=True)
    print(f"  free: β_4 + phi_nb   φ = ones(15)   R(0) default   A fix on",
          flush=True)
    print(f"  policy sick_leave: p_work {P_WORK_BASE} → {P_WORK_SICK} "
          f"(β_4 held from fit)", flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    print(f"\n  Baseline (buggy prod, γ = center):", flush=True)
    print(f"    β_4 = {BASELINE_BETA}   NLL = {BASELINE_NLL:.4e}", flush=True)
    print(f"    r   = {BASELINE_R}", flush=True)

    combos = []
    for c in CHILD_GRID:
        combos.append(("child", c, CENTER["adult"], CENTER["elder"]))
    for a in ADULT_GRID:
        combos.append(("adult", CENTER["child"], a, CENTER["elder"]))
    for e in ELDER_GRID:
        combos.append(("elder", CENTER["child"], CENTER["adult"], e))

    all_results = []
    for sw, c, a, e in combos:
        print(f"\n── sweep={sw}  γ=({c}, {a}, {e}) ──", flush=True)
        r = fit_combo(c, a, e, sw, setup)
        all_results.append(r)
        print(f"    NLL={r['nll']:.4e}  β_4={[round(x,4) for x in r['beta_4']]}"
              f"  R0={r['R0_ngm']:.3f}  phi_nb={r['phi_nb']:.2f}  "
              f"std={r['nll_std']:.2e}  wall={r['wall_sec']:.1f}s", flush=True)
        print(f"    r per age: " + "  ".join(
            f"{ag}={r['per_age'][ag]['ratio']:.2f}" for ag in HIRA_AGE_GROUPS),
              flush=True)
        print(f"    ★ averted_total = {r['averted_total_pct']:+.2f}%   "
              f"per-age averted: "
              + "  ".join(f"{ag}={r['averted_by_age_pct'][ag]:+.2f}%"
                          for ag in HIRA_AGE_GROUPS), flush=True)

    print("\n" + "=" * 78, flush=True)
    print("  Summary — β/NLL + averted", flush=True)
    print(f"  {'sw':>5s}  {'child':>5s} {'adult':>5s} {'elder':>5s}  "
          f"{'NLL':>10s}  {'R0':>5s}  {'std':>8s}  {'avert_tot%':>10s}",
          flush=True)
    for r in all_results:
        print(f"  {r['sweep_var']:>5s}  {r['gamma_child']:>5.2f} "
              f"{r['gamma_adult']:>5.2f} {r['gamma_elder']:>5.2f}  "
              f"{r['nll']:.4e}  {r['R0_ngm']:.3f}  {r['nll_std']:.2e}  "
              f"{r['averted_total_pct']:>+10.2f}", flush=True)

    print("\n  Per-age r  (obs_peak±2w / model_peak±2w)", flush=True)
    header = f"  {'sw':>5s}  {'child':>5s} {'adult':>5s} {'elder':>5s}  " + \
             "  ".join(f"{ag:>7s}" for ag in HIRA_AGE_GROUPS)
    print(header, flush=True)
    print(f"  {'base':>5s}   {'—':>4s}  {'—':>4s}  {'—':>4s}    " +
          "  ".join(f"{BASELINE_R[ag]:>7.2f}" for ag in HIRA_AGE_GROUPS),
          flush=True)
    for r in all_results:
        row = (f"  {r['sweep_var']:>5s}  {r['gamma_child']:>5.2f} "
               f"{r['gamma_adult']:>5.2f} {r['gamma_elder']:>5.2f}  ")
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>7.2f}"
        print(row, flush=True)

    print("\n  Per-age averted % (sick_leave: p_work=0.4)", flush=True)
    print(header, flush=True)
    for r in all_results:
        row = (f"  {r['sweep_var']:>5s}  {r['gamma_child']:>5.2f} "
               f"{r['gamma_adult']:>5.2f} {r['gamma_elder']:>5.2f}  ")
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['averted_by_age_pct'][ag]:>+7.2f}"
        print(row, flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "center": CENTER,
                "child_grid": CHILD_GRID, "adult_grid": ADULT_GRID,
                "elder_grid": ELDER_GRID,
                "R0_IMMUNITY_default": [float(x) for x in R0_IMMUNITY_PROFILE],
                "vaccine": "A_fix cov_eff = -ln(1-cov)",
                "phi": "ones(15) FIXED",
                "policy_sick_leave_p_work": P_WORK_SICK,
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "peak_half_window_weeks": PEAK_HALF_WIN,
                "baseline_reference": {
                    "beta_4": BASELINE_BETA, "NLL": BASELINE_NLL,
                    "r": BASELINE_R,
                },
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # Figure: 2 panels — γ vs per-age r, γ vs per-age averted
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sweep_colors = {"child": "#1a5490", "adult": "#27ae60", "elder": "#c0392b"}
    ages = HIRA_AGE_GROUPS
    xa = np.arange(len(ages))

    # Panel 1: per-age r for each combo (grouped bars by sweep)
    ax = axes[0]
    combos_by_sweep = {sw: [] for sw in ["child", "adult", "elder"]}
    for r in all_results:
        combos_by_sweep[r["sweep_var"]].append(r)
    bw = 0.09
    n_combos_per_sw = 3
    total_slots = len(ages) * (n_combos_per_sw * 3 + 2)
    idx_offset = -bw * (n_combos_per_sw * 3) / 2
    ax.bar(xa - bw * 4, [BASELINE_R[a] for a in ages], bw, color="#333333",
            label="baseline (buggy prod)")
    for i, sw in enumerate(["child", "adult", "elder"]):
        for j, r in enumerate(combos_by_sweep[sw]):
            offset = (i * 3 + j - 4) * bw
            val = getattr(r, "gamma_" + sw) if hasattr(r, "gamma_" + sw) \
                  else r[f"gamma_{sw}"]
            label = f"{sw}={val}"
            ax.bar(xa + offset, [r["per_age"][ag]["ratio"] for ag in ages],
                    bw, color=sweep_colors[sw],
                    alpha=0.35 + 0.3 * j, label=label)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(ages)
    ax.set_ylabel("r = obs / model")
    ax.set_title("Per-age r per γ combo")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: per-age averted %
    ax = axes[1]
    for i, sw in enumerate(["child", "adult", "elder"]):
        for j, r in enumerate(combos_by_sweep[sw]):
            offset = (i * 3 + j - 4) * bw
            val = r[f"gamma_{sw}"]
            label = f"{sw}={val}"
            ax.bar(xa + offset,
                    [r["averted_by_age_pct"][ag] for ag in ages], bw,
                    color=sweep_colors[sw], alpha=0.35 + 0.3 * j, label=label)
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(ages)
    ax.set_ylabel("averted % (sick_leave)")
    ax.set_title("Policy robustness — per-age averted % per γ combo")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"γ_report all-age sweep + sick_leave averted  —  "
                  f"{SEASON_LABEL}  (φ fixed, R(0) default, A fix on)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
