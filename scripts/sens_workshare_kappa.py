"""work_share × κ 2D sensitivity — sick_leave averted% sign map.

work is not identifiable from HIRA; scan its share directly and cross with κ.
Fit is independent of κ (spillover=0 at p_work=p_school=1 baseline), so we
fit once per work_share and evaluate the policy at each κ value. JSON stores
all 15 (work_share × κ) combos for reporting symmetry.

Setup: single season 2019-2020, φ U-shape fixed, R(0) default, A-fix cov,
HOLIDAY on, γ_report = (0.40, 0.18, 0.25).

Grid:
  work_share ∈ {0.03, 0.06, 0.09, 0.12, 0.15}
  κ (scalar, broadcast to all 15 ages) ∈ {0.2, 0.4, 0.6}
  → 15 combos.

Pin (A relative-share for home:school:other = 0.29:0.06:0.36):
  π_home   = (1 − work_share) × 0.29/0.71
  π_work   = work_share                   (strong pin, σ = 0.01)
  π_school = (1 − work_share) × 0.06/0.71
  π_other  = (1 − work_share) × 0.36/0.71
  σ_per_channel = [0.15, 0.01, 0.15, 0.15]  (loose on non-work)

Policy: sick_leave (p_work 1.0 → 0.4) at each κ.  School averted also
reported for pin-slack diagnosis.

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
import matplotlib.colors as mcolors

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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "sens_workshare_kappa.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "sens_workshare_kappa.png"
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
PHI_USHAPE = np.array(
    [2.0, 1.9, 1.7, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0,
     1.05, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64,
)
GAMMA_CENTER = (0.40, 0.18, 0.25)

WORK_SHARE_GRID = [0.03, 0.06, 0.09, 0.12, 0.15]
KAPPA_GRID = [0.2, 0.4, 0.6]

# A relative share for non-work (home:school:other) — used to distribute (1 - work_share)
A_REL = np.array([0.29, 0.06, 0.36])          # (h, s, o) from A NIMS-derived π
A_REL_NORM = A_REL / A_REL.sum()               # → [0.408, 0.085, 0.507]

SIGMA_PER_CHANNEL = np.array([0.15, 0.01, 0.15, 0.15])  # (h, w, s, o)

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

ITALY_WORK_SHARE = 0.033   # literature reference marker


def correct_coverage(cov_15):
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_gamma_15(c, a, e):
    return np.concatenate([np.full(4, c), np.full(9, a), np.full(2, e)])


def build_pi_target(work_share: float) -> np.ndarray:
    remaining = 1.0 - work_share
    pi_h = remaining * A_REL_NORM[0]
    pi_s = remaining * A_REL_NORM[1]
    pi_o = remaining * A_REL_NORM[2]
    return np.array([pi_h, work_share, pi_s, pi_o])


def logit_centered_target(pi_target):
    lp = np.log(np.clip(pi_target, 1e-6, None))
    return lp - lp.mean()


def build_setup():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination

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


def make_shared(kappa_scalar: float, setup):
    """Return shared dict with kappa broadcast to (15,)."""
    kw = dict(setup["shared_base"])
    kw["kappa"] = jnp.full(15, kappa_scalar, dtype=jnp.float64)
    return kw


def run_forward(beta_4, phi_full_j, gamma_15_j, p_school, p_work, shared, setup):
    kw = dict(shared)
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
                       shared_fit, setup):
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
                                P_SCHOOL_BASE, P_WORK_BASE, shared_fit, setup)
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
        starts.append(np.concatenate([[np.log(2.0) + delta],
                                        np.asarray(logit_target),
                                        [10.0]]))
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


def point_fit_at_ws(work_share, setup):
    """Fit once per work_share (κ irrelevant at baseline forward).
    Uses a middle κ=0.4 for the fit; result is independent of κ.
    """
    pi_target = build_pi_target(work_share)
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15 = build_gamma_15(*GAMMA_CENTER)
    gamma_15_j = jnp.asarray(gamma_15)
    shared_fit = make_shared(0.4, setup)   # κ arbitrary here — no spillover at baseline

    fg = build_point_loss(logit_target, SIGMA_PER_CHANNEL, phi_full_j,
                            gamma_15_j, shared_fit, setup)
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
                              P_SCHOOL_BASE, P_WORK_BASE, shared_fit, setup)
    pred = np.asarray(pred_j)
    ratios = per_age_ratios(pred, setup)
    r0_ngm = float(setup["ngm_default"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        phi_full_j,
    ))
    nll_arr = np.array(per_start_nll)
    return dict(
        work_share=work_share, pi_target=pi_target.tolist(),
        log_R0=log_R0, R0=R0, R0_ngm=r0_ngm,
        pi=[float(p) for p in pi],
        beta_4=[float(b) for b in beta_4],
        phi_nb=phi_nb, nll=best["nll"],
        best_start_idx=int(best["start_idx"]),
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        per_age=ratios, wall_sec=float(wall),
    )


def policy_averted_at_kappa(beta_4, kappa_scalar, setup):
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15_j = jnp.asarray(build_gamma_15(*GAMMA_CENTER))
    shared_pol = make_shared(kappa_scalar, setup)
    beta_arr = np.array(beta_4)

    def fwd(p_school, p_work):
        inc, pred_hira = run_forward(beta_arr, phi_full_j, gamma_15_j,
                                       p_school, p_work, shared_pol, setup)
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
        kappa=kappa_scalar,
        baseline_total=tot_b,
        sick_total=tot_s, school_total=tot_c,
        averted_sick_total_pct=100.0 * (tot_b - tot_s) / max(tot_b, 1.0),
        averted_school_total_pct=100.0 * (tot_b - tot_c) / max(tot_b, 1.0),
        averted_sick_by_age_pct=by_age(pred_b, pred_s),
        averted_school_by_age_pct=by_age(pred_b, pred_c),
    )


def main():
    print("=" * 78, flush=True)
    print(f"SENSITIVITY: work_share × κ  —  {SEASON_LABEL}", flush=True)
    print(f"  work_share ∈ {WORK_SHARE_GRID}   κ ∈ {KAPPA_GRID}"
          f"   = 15 combos", flush=True)
    print(f"  φ FIXED U-shape   γ_report = {GAMMA_CENTER}   R(0) default",
          flush=True)
    print(f"  σ_per_channel = {SIGMA_PER_CHANNEL.tolist()}  "
          f"(loose h/s/o, strong pin on work)", flush=True)
    print(f"  A relative-share for non-work: h/s/o = "
          f"{[round(float(x),3) for x in A_REL_NORM]}", flush=True)
    print(f"  Italy literature marker: work_share = {ITALY_WORK_SHARE}",
          flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    all_results = dict(
        season=SEASON_LABEL,
        config=dict(
            AMP=AMP, HOLIDAY=HOLIDAY,
            PHI_USHAPE=PHI_USHAPE.tolist(),
            GAMMA_CENTER=list(GAMMA_CENTER),
            work_share_grid=WORK_SHARE_GRID,
            kappa_grid=KAPPA_GRID,
            A_REL=A_REL.tolist(),
            A_REL_NORM=A_REL_NORM.tolist(),
            SIGMA_PER_CHANNEL=SIGMA_PER_CHANNEL.tolist(),
            n_starts=N_STARTS,
            p_work_sick=P_WORK_SICK, p_school_absence=P_SCHOOL_ABSENCE,
        ),
        combos=[],
    )

    # Fit once per work_share
    fits_by_ws = {}
    for ws in WORK_SHARE_GRID:
        print(f"\n── fit work_share = {ws:.2f} "
              f"(pi_target = {[round(float(x),4) for x in build_pi_target(ws)]}) ──",
              flush=True)
        rec = point_fit_at_ws(ws, setup)
        fits_by_ws[ws] = rec
        print(f"    NLL={rec['nll']:.4e}  R0={rec['R0']:.3f}  "
              f"phi_nb={rec['phi_nb']:.2f}  wall={rec['wall_sec']:.1f}s  "
              f"std={rec['nll_std']:.2e}", flush=True)
        print(f"    π = {[round(x,4) for x in rec['pi']]}"
              f"   target = {[round(x,4) for x in rec['pi_target']]}",
              flush=True)
        print(f"    β = {[round(b,4) for b in rec['beta_4']]}", flush=True)
        print(f"    per-age r: " + "  ".join(
            f"{ag}={rec['per_age'][ag]['ratio']:.2f}"
            for ag in HIRA_AGE_GROUPS), flush=True)

    # Policy at each (ws, κ)
    for ws in WORK_SHARE_GRID:
        fit = fits_by_ws[ws]
        for kappa_val in KAPPA_GRID:
            pol = policy_averted_at_kappa(fit["beta_4"], kappa_val, setup)
            combo_rec = dict(
                work_share=ws, kappa=kappa_val,
                fit=fit, policy=pol,
            )
            all_results["combos"].append(combo_rec)
            print(f"\n  ── ws={ws:.2f}  κ={kappa_val}: "
                  f"avert(sick_total) = "
                  f"{pol['averted_sick_total_pct']:+.2f}%   "
                  f"avert(school_total) = "
                  f"{pol['averted_school_total_pct']:+.2f}%",
                  flush=True)
            print(f"    per-age sick: " + "  ".join(
                f"{ag}={pol['averted_sick_by_age_pct'][ag]:+.2f}%"
                for ag in HIRA_AGE_GROUPS), flush=True)

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)

    # ─── Console summary ─────
    print("\n" + "=" * 78, flush=True)
    print("  15-combo summary", flush=True)
    print(f"  {'ws':>5s}  {'κ':>4s}  {'π_w':>6s}  {'π_s':>6s}  {'R0':>5s}  "
          f"{'NLL':>10s}  {'avert_sick':>10s}  {'avert_school':>12s}",
          flush=True)
    for rec in all_results["combos"]:
        ws = rec["work_share"]; k = rec["kappa"]
        fit = rec["fit"]; pol = rec["policy"]
        pi_w = fit["pi"][1]; pi_s = fit["pi"][2]
        print(f"  {ws:>5.2f}  {k:>4.1f}  {pi_w:>6.3f}  {pi_s:>6.3f}  "
              f"{fit['R0_ngm']:>5.3f}  {fit['nll']:>.4e}  "
              f"{pol['averted_sick_total_pct']:>+10.2f}%  "
              f"{pol['averted_school_total_pct']:>+12.2f}%", flush=True)

    # ─── Figure ─────
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # Panel 1 (top-left): Heatmap ws × κ, cell = averted_sick_total_pct
    ax = fig.add_subplot(gs[0, 0])
    n_ws = len(WORK_SHARE_GRID); n_k = len(KAPPA_GRID)
    M = np.zeros((n_k, n_ws))   # rows=κ, cols=ws
    for i, k in enumerate(KAPPA_GRID):
        for j, ws in enumerate(WORK_SHARE_GRID):
            for rec in all_results["combos"]:
                if rec["work_share"] == ws and rec["kappa"] == k:
                    M[i, j] = rec["policy"]["averted_sick_total_pct"]
                    break
    vmax = max(abs(np.nanmin(M)), abs(np.nanmax(M)))
    im = ax.imshow(M, cmap="coolwarm_r",
                    norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                    aspect="auto")
    ax.set_xticks(range(n_ws))
    ax.set_xticklabels([f"{v:.2f}" for v in WORK_SHARE_GRID])
    ax.set_yticks(range(n_k))
    ax.set_yticklabels([f"{v:.1f}" for v in KAPPA_GRID])
    ax.set_xlabel("work_share")
    ax.set_ylabel("κ")
    ax.set_title("sick_leave averted %  (blue=positive, red=negative)")
    for i in range(n_k):
        for j in range(n_ws):
            ax.text(j, i, f"{M[i, j]:+.1f}%", ha="center", va="center",
                    fontsize=9,
                    color="black" if abs(M[i, j]) < 0.6 * vmax else "white")
    # Italy marker (0.033)
    ax.axvline(np.interp(ITALY_WORK_SHARE, WORK_SHARE_GRID,
                          np.arange(n_ws)), color="k", ls=":", lw=1.5)
    ax.text(np.interp(ITALY_WORK_SHARE, WORK_SHARE_GRID,
                       np.arange(n_ws)) + 0.05, n_k - 0.4,
             "Italy 0.033", fontsize=8, color="k")
    fig.colorbar(im, ax=ax, fraction=0.045)

    # Panel 2 (top-middle): π_school realised heatmap (pin slack diagnosis)
    ax = fig.add_subplot(gs[0, 1])
    M2 = np.zeros((n_k, n_ws))
    for i, k in enumerate(KAPPA_GRID):
        for j, ws in enumerate(WORK_SHARE_GRID):
            for rec in all_results["combos"]:
                if rec["work_share"] == ws and rec["kappa"] == k:
                    M2[i, j] = rec["fit"]["pi"][2]
                    break
    im = ax.imshow(M2, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_ws))
    ax.set_xticklabels([f"{v:.2f}" for v in WORK_SHARE_GRID])
    ax.set_yticks(range(n_k))
    ax.set_yticklabels([f"{v:.1f}" for v in KAPPA_GRID])
    ax.set_xlabel("work_share")
    ax.set_ylabel("κ  (identical rows — fit κ-independent)")
    ax.set_title("π_school realised (target ≈ 0.085 × (1 − ws))")
    for i in range(n_k):
        for j in range(n_ws):
            ax.text(j, i, f"{M2[i, j]:.3f}", ha="center", va="center",
                     fontsize=9, color="white")
    fig.colorbar(im, ax=ax, fraction=0.045)

    # Panel 3 (top-right): school averted heatmap (pin-slack indicator)
    ax = fig.add_subplot(gs[0, 2])
    M3 = np.zeros((n_k, n_ws))
    for i, k in enumerate(KAPPA_GRID):
        for j, ws in enumerate(WORK_SHARE_GRID):
            for rec in all_results["combos"]:
                if rec["work_share"] == ws and rec["kappa"] == k:
                    M3[i, j] = rec["policy"]["averted_school_total_pct"]
                    break
    im = ax.imshow(M3, cmap="magma", aspect="auto")
    ax.set_xticks(range(n_ws))
    ax.set_xticklabels([f"{v:.2f}" for v in WORK_SHARE_GRID])
    ax.set_yticks(range(n_k))
    ax.set_yticklabels([f"{v:.1f}" for v in KAPPA_GRID])
    ax.set_xlabel("work_share")
    ax.set_ylabel("κ")
    ax.set_title("school_absence averted %  (pin-slack indicator)")
    for i in range(n_k):
        for j in range(n_ws):
            ax.text(j, i, f"{M3[i, j]:.1f}%", ha="center", va="center",
                     fontsize=9, color="white")
    fig.colorbar(im, ax=ax, fraction=0.045)

    # Panel 4 (bottom row): per-age sick_leave averted at each ws, aggregating κ=0.4
    ax = fig.add_subplot(gs[1, :])
    xa = np.arange(len(HIRA_AGE_GROUPS))
    bw = 0.15
    for j, ws in enumerate(WORK_SHARE_GRID):
        for rec in all_results["combos"]:
            if rec["work_share"] == ws and rec["kappa"] == 0.4:
                vals = [rec["policy"]["averted_sick_by_age_pct"][ag]
                        for ag in HIRA_AGE_GROUPS]
                ax.bar(xa + (j - 2) * bw, vals, bw,
                        color=plt.cm.viridis(j / (len(WORK_SHARE_GRID) - 1)),
                        label=f"ws={ws:.2f}", alpha=0.85)
                break
    ax.axhline(0.0, color="grey", ls=":", lw=1)
    ax.set_xticks(xa); ax.set_xticklabels(HIRA_AGE_GROUPS)
    ax.set_ylabel("averted % (sick_leave)")
    ax.set_title(f"per-age sick_leave averted at κ=0.4")
    ax.legend(fontsize=9, ncol=5); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"work_share × κ sensitivity — {SEASON_LABEL}  "
                  f"(φ U-shape, R(0) default, A-fix cov, γ CDC)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
