"""Pin fit point-estimate batch — A/B target × [loose] + A [school-tight].

Motivation: A-a NUTS gave posterior β ≈ point estimate → skip NUTS, use
point-only. Compare A vs B on policy, and probe whether school over-share
(π_school 0.06 target → ~0.15 realised) drives the 95% school_absence
averted result.

Combos (3):
  A_loose   : target_A NIMS  σ_per_channel = [0.01, 0.01, 0.30, 0.30]
  B_loose   : target_B lit    σ_per_channel = [0.01, 0.01, 0.30, 0.30]
  A_school_tight : target_A NIMS  σ_per_channel = [0.01, 0.01, 0.05, 0.30]
                   ← school pin tightened 0.30 → 0.05

Setup: single season 2019-2020, φ U-shape fixed, R(0) default, A-fix cov,
HOLIDAY on, γ_report = (0.40, 0.18, 0.25).

Point estimate: L-BFGS × 12 starts over (log_R0, logit_pi, phi_nb).
Policy at posterior/best β:
  sick_leave     : p_work  1.0 → 0.4  (p_school = 1.0)
  school_absence : p_school 1.0 → 0.4  (p_work   = 1.0)

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
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "pin_point_batch.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "pin_point_batch.png"
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
UNIT_R0 = np.array([8.70, 6.21, 25.40, 9.33])
PHI_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64,
)
GAMMA_CENTER = (0.40, 0.18, 0.25)

TARGETS = {
    "A": np.array([23.26, 17.03, 14.24, 31.77]),   # NIMS contact
    "B": np.array([0.40, 0.10, 0.27, 0.23]),        # Italy 2009 R0 contrib
}

# Sigma per channel: (home, work, school, other)
SIGMA_LOOSE = np.array([0.01, 0.01, 0.30, 0.30])
SIGMA_A_SCHOOL_TIGHT = np.array([0.01, 0.01, 0.05, 0.30])

COMBOS = [
    ("A_loose", "A", SIGMA_LOOSE),
    ("B_loose", "B", SIGMA_LOOSE),
    ("A_school_tight", "A", SIGMA_A_SCHOOL_TIGHT),
]

N_STARTS = 12
POINT_START_SEED = 23
LOG_R0_BOUNDS = (float(np.log(0.8)), float(np.log(3.0)))
LOGIT_PI_BOUNDS = (-10.0, 10.0)
PHI_NB_BOUNDS = (1e-3, 1e6)

P_WORK_BASE = 1.0
P_SCHOOL_BASE = 1.0
P_WORK_SICK = 0.4
P_SCHOOL_ABSENCE = 0.4

COVERAGE_CAP = 0.99
PEAK_HALF_WIN = 2


def correct_coverage(cov_15):
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_gamma_15(c, a, e):
    return np.concatenate([np.full(4, c), np.full(9, a), np.full(2, e)])


def r0contrib_to_pi(r0c):
    r0c_n = r0c / r0c.sum()
    beta_share = r0c_n / UNIT_R0
    return beta_share / beta_share.sum()


def logit_centered_target(pi_target):
    lp = np.log(np.clip(pi_target, 1e-6, None))
    return lp - lp.mean()


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


def run_forward(beta_4, phi_full_j, gamma_15_j, p_school, p_work, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_j
    kw["p_school"] = p_school; kw["p_work"] = p_work
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    pred = simulation_to_hira_by_age_jax(inc, gamma_15_j,
                                          n_weeks=setup["n_weeks"])
    return inc, pred


def build_point_loss(logit_target, sigma_per_channel, phi_full_j, gamma_15_j,
                       setup):
    lt = jnp.asarray(logit_target)
    inv_var = 1.0 / (sigma_per_channel ** 2)

    def loss(x):
        log_R0 = x[0]
        logit_pi = x[1:5]
        phi_nb = x[5]
        R0 = jnp.exp(log_R0)
        pi = jax.nn.softmax(logit_pi)
        beta_4 = derive_beta_from_R0_simplex(setup["ngm_default"], R0, pi,
                                               phi_full_j)
        _, pred = run_forward(beta_4, phi_full_j, gamma_15_j,
                                P_SCHOOL_BASE, P_WORK_BASE, setup)
        nll = nb_nll_jax(setup["obs_j"], pred, setup["w_j"],
                         concentration=phi_nb, min_rate=0.01)
        centered = logit_pi - jnp.mean(logit_pi)
        dev = centered - lt
        ch_pen = 0.5 * jnp.sum(dev * dev * inv_var)
        return nll + ch_pen

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


def make_starts(logit_target, n_starts, seed):
    rng = np.random.default_rng(seed)
    starts = []
    for delta in [0.0, 0.2, -0.2, 0.4, -0.4]:
        x0 = np.concatenate([[np.log(2.0) + delta], np.asarray(logit_target),
                              [10.0]])
        starts.append(x0)
    while len(starts) < n_starts:
        starts.append(np.concatenate([
            [rng.uniform(*LOG_R0_BOUNDS)],
            np.asarray(logit_target) + rng.normal(0, 0.5, 4),
            [rng.uniform(2.0, 20.0)],
        ]))
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


def point_fit(pi_target, sigma_per_channel, phi_full, gamma_15, setup):
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(phi_full)
    gamma_15_j = jnp.asarray(gamma_15)
    fg = build_point_loss(logit_target, sigma_per_channel, phi_full_j,
                            gamma_15_j, setup)
    bounds = [LOG_R0_BOUNDS] + [LOGIT_PI_BOUNDS] * 4 + [PHI_NB_BOUNDS]
    starts = make_starts(logit_target, N_STARTS, POINT_START_SEED)

    best = None
    per_start_nll = []
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
            nll = float(res.fun)
        except Exception:
            continue
        per_start_nll.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0

    x = best["x"]
    log_R0 = float(x[0]); logit_pi = np.array(x[1:5])
    phi_nb = float(x[5])
    R0 = float(np.exp(log_R0))
    pi = np.array(jax.nn.softmax(jnp.asarray(logit_pi)))
    beta_4 = np.array(derive_beta_from_R0_simplex(
        setup["ngm_default"], jnp.asarray(R0), jnp.asarray(pi), phi_full_j,
    ))
    _, pred_j = run_forward(beta_4, phi_full_j, gamma_15_j,
                              P_SCHOOL_BASE, P_WORK_BASE, setup)
    pred = np.asarray(pred_j)
    ratios = per_age_ratios(pred, setup)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        phi_full_j,
    ))
    nll_arr = np.array(per_start_nll)
    return dict(
        log_R0=log_R0, R0=R0,
        pi=[float(p) for p in pi],
        beta_4=[float(b) for b in beta_4],
        phi_nb=phi_nb, nll=best["nll"], R0_ngm=r0_ngm,
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios, wall_sec=float(wall),
    )


def policy_averted(beta_4, phi_full, gamma_15, setup):
    phi_full_j = jnp.asarray(phi_full)
    gamma_15_j = jnp.asarray(gamma_15)
    beta_arr = np.array(beta_4)

    def fwd(p_school, p_work):
        inc, pred_hira = run_forward(beta_arr, phi_full_j, gamma_15_j,
                                       p_school, p_work, setup)
        return np.asarray(inc), np.asarray(pred_hira)

    inc_b, pred_b = fwd(P_SCHOOL_BASE, P_WORK_BASE)
    inc_s, pred_s = fwd(P_SCHOOL_BASE, P_WORK_SICK)
    inc_c, pred_c = fwd(P_SCHOOL_ABSENCE, P_WORK_BASE)

    tot_b = float(inc_b.sum())
    tot_s = float(inc_s.sum())
    tot_c = float(inc_c.sum())

    def by_age(pred_base, pred_sc):
        out = {}
        for ai, ag in enumerate(HIRA_AGE_GROUPS):
            b = float(pred_base[:, ai].sum())
            s = float(pred_sc[:, ai].sum())
            out[ag] = 100.0 * (b - s) / max(b, 1e-9)
        return out

    return dict(
        baseline_total=tot_b,
        sick_total=tot_s, school_total=tot_c,
        averted_sick_total_pct=100.0 * (tot_b - tot_s) / max(tot_b, 1.0),
        averted_school_total_pct=100.0 * (tot_b - tot_c) / max(tot_b, 1.0),
        averted_sick_by_age_pct=by_age(pred_b, pred_s),
        averted_school_by_age_pct=by_age(pred_b, pred_c),
    )


def main():
    print("=" * 78, flush=True)
    print(f"PIN FIT POINT BATCH  —  {SEASON_LABEL}", flush=True)
    print(f"  3 combos = {{A_loose, B_loose, A_school_tight}}", flush=True)
    print(f"  γ = {GAMMA_CENTER}   φ FIXED U-shape   R(0) default", flush=True)
    print(f"  A-fix cov  HOLIDAY on  channel pin via ch_prior factor",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()
    gamma_15 = build_gamma_15(*GAMMA_CENTER)

    all_results = dict(
        season=SEASON_LABEL,
        config=dict(
            AMP=AMP, HOLIDAY=HOLIDAY, UNIT_R0=UNIT_R0.tolist(),
            PHI_USHAPE=PHI_USHAPE.tolist(),
            GAMMA_CENTER=list(GAMMA_CENTER),
            targets={k: v.tolist() for k, v in TARGETS.items()},
            sigma_loose=SIGMA_LOOSE.tolist(),
            sigma_A_school_tight=SIGMA_A_SCHOOL_TIGHT.tolist(),
            n_starts=N_STARTS,
            p_work_sick=P_WORK_SICK, p_school_absence=P_SCHOOL_ABSENCE,
        ),
        combos={},
    )

    for combo_name, target_key, sigma_per_channel in COMBOS:
        print(f"\n{'#' * 60}", flush=True)
        print(f"#  COMBO {combo_name}   target={target_key}   σ={sigma_per_channel.tolist()}",
              flush=True)
        print(f"{'#' * 60}", flush=True)

        pi_target = r0contrib_to_pi(TARGETS[target_key])
        print(f"  target π = {[round(float(x),4) for x in pi_target]}", flush=True)

        pt = point_fit(pi_target, sigma_per_channel, PHI_USHAPE, gamma_15, setup)
        pol = policy_averted(pt["beta_4"], PHI_USHAPE, gamma_15, setup)

        rec = dict(
            combo=combo_name, target=target_key,
            sigma_per_channel=sigma_per_channel.tolist(),
            pi_target=pi_target.tolist(),
            point=pt, policy=pol,
        )
        all_results["combos"][combo_name] = rec
        with open(OUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  NLL={pt['nll']:.4e}  R0={pt['R0']:.3f}  R0_ngm={pt['R0_ngm']:.3f}"
              f"  phi_nb={pt['phi_nb']:.2f}  wall={pt['wall_sec']:.1f}s",
              flush=True)
        print(f"  NLL 12 starts std={pt['nll_std']:.3e}", flush=True)
        print(f"  π recovered vs target:", flush=True)
        for c, ch in enumerate(["home", "work", "school", "other"]):
            print(f"    π_{ch}: recovered={pt['pi'][c]:.4f}  "
                  f"target={pi_target[c]:.4f}  σ={sigma_per_channel[c]}",
                  flush=True)
        print(f"  β_4 = {[round(x,4) for x in pt['beta_4']]}", flush=True)
        print(f"  per-age r: " + "  ".join(
            f"{ag}={pt['per_age'][ag]['ratio']:.2f}" for ag in HIRA_AGE_GROUPS),
              flush=True)
        print(f"  ★ averted sick_leave total = {pol['averted_sick_total_pct']:+.2f}%"
              f"   school_absence total = {pol['averted_school_total_pct']:+.2f}%",
              flush=True)
        print(f"  averted sick per age: " + "  ".join(
            f"{ag}={pol['averted_sick_by_age_pct'][ag]:+.2f}%"
            for ag in HIRA_AGE_GROUPS), flush=True)
        print(f"  averted school per age: " + "  ".join(
            f"{ag}={pol['averted_school_by_age_pct'][ag]:+.2f}%"
            for ag in HIRA_AGE_GROUPS), flush=True)

    # Console summary
    print("\n" + "=" * 78, flush=True)
    print("  Summary — β_4, NLL, averted", flush=True)
    print(f"  {'combo':>18s}  {'β_h':>7s} {'β_w':>7s} {'β_s':>7s} {'β_o':>7s}  "
          f"{'NLL':>10s}  {'R0':>5s}  {'sick%':>7s}  {'school%':>8s}",
          flush=True)
    for name in ["A_loose", "B_loose", "A_school_tight"]:
        rec = all_results["combos"][name]
        pt = rec["point"]; pol = rec["policy"]
        b = pt["beta_4"]
        print(f"  {name:>18s}  {b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f}  {b[3]:.4f}  "
              f"{pt['nll']:.4e}  {pt['R0_ngm']:.3f}  "
              f"{pol['averted_sick_total_pct']:>+7.2f}  "
              f"{pol['averted_school_total_pct']:>+8.2f}", flush=True)

    print("\n  π recovered vs target", flush=True)
    print(f"  {'combo':>18s}  {'π_h':>6s} {'π_w':>6s} {'π_s':>6s} {'π_o':>6s}   "
          f"{'target_h':>8s} {'target_w':>8s} {'target_s':>8s} {'target_o':>8s}",
          flush=True)
    for name in ["A_loose", "B_loose", "A_school_tight"]:
        rec = all_results["combos"][name]
        pi_r = rec["point"]["pi"]; pi_t = rec["pi_target"]
        print(f"  {name:>18s}  {pi_r[0]:.3f}  {pi_r[1]:.3f}  {pi_r[2]:.3f}  "
              f"{pi_r[3]:.3f}   {pi_t[0]:>8.3f} {pi_t[1]:>8.3f} "
              f"{pi_t[2]:>8.3f} {pi_t[3]:>8.3f}", flush=True)

    print("\n  Per-age r", flush=True)
    header = f"  {'combo':>18s}  " + "  ".join(f"{ag:>7s}"
                                                    for ag in HIRA_AGE_GROUPS)
    print(header, flush=True)
    for name in ["A_loose", "B_loose", "A_school_tight"]:
        rec = all_results["combos"][name]
        row = f"  {name:>18s}  "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {rec['point']['per_age'][ag]['ratio']:>7.2f}"
        print(row, flush=True)

    print("\n  averted sick_leave per age", flush=True)
    print(header, flush=True)
    for name in ["A_loose", "B_loose", "A_school_tight"]:
        rec = all_results["combos"][name]
        row = f"  {name:>18s}  "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {rec['policy']['averted_sick_by_age_pct'][ag]:>+7.2f}"
        print(row, flush=True)

    print("\n  averted school_absence per age", flush=True)
    print(header, flush=True)
    for name in ["A_loose", "B_loose", "A_school_tight"]:
        rec = all_results["combos"][name]
        row = f"  {name:>18s}  "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {rec['policy']['averted_school_by_age_pct'][ag]:>+7.2f}"
        print(row, flush=True)

    print(f"\nsaved {OUT_JSON}", flush=True)

    # Figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    combo_colors = {"A_loose": "#1a5490", "B_loose": "#c0392b",
                     "A_school_tight": "#27ae60"}
    ages = HIRA_AGE_GROUPS
    xa = np.arange(len(ages))
    bw = 0.25

    # Panel 1: per-age r
    ax = axes[0]
    for i, name in enumerate(["A_loose", "B_loose", "A_school_tight"]):
        rec = all_results["combos"][name]
        vals = [rec["point"]["per_age"][ag]["ratio"] for ag in ages]
        ax.bar(xa + (i - 1) * bw, vals, bw, color=combo_colors[name],
                label=name, alpha=0.85)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(ages)
    ax.set_ylabel("r = obs / model")
    ax.set_title("Per-age r  (point estimate)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: averted sick_leave per age (A vs B)
    ax = axes[1]
    for i, name in enumerate(["A_loose", "B_loose"]):
        rec = all_results["combos"][name]
        vals = [rec["policy"]["averted_sick_by_age_pct"][ag] for ag in ages]
        ax.bar(xa + (i - 0.5) * bw, vals, bw, color=combo_colors[name],
                label=name, alpha=0.85)
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(ages)
    ax.set_ylabel("averted % (sick_leave)")
    ax.set_title("sick_leave averted per age  (A vs B)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: averted school_absence per age (A_loose vs A_school_tight)
    ax = axes[2]
    for i, name in enumerate(["A_loose", "A_school_tight"]):
        rec = all_results["combos"][name]
        vals = [rec["policy"]["averted_school_by_age_pct"][ag] for ag in ages]
        ax.bar(xa + (i - 0.5) * bw, vals, bw, color=combo_colors[name],
                label=name, alpha=0.85)
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(ages)
    ax.set_ylabel("averted % (school_absence)")
    ax.set_title("school_absence averted  (loose vs school-tight)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Pin fit point batch — {SEASON_LABEL}  "
                  f"(φ U-shape, R(0) default, A-fix cov)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
