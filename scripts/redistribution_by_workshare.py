"""Sick-leave age-redistribution analysis over work_share × κ.

Goal: Show HOW sick_leave shifts infections BETWEEN age groups, not just the
total averted %. For each (work_share, κ) combo we compare baseline vs
sick_leave (p_work 1.0 → 0.4) and report:

  1) attack rate per HIRA-6 age group (baseline & sick_leave)
  2) Δattack = sick − baseline (positive → group receives redistributed load)
  3) transfer ratio = (Σ positive Δ) / (Σ |negative Δ|)
  4) working-age (18-64) vs non-working (0-17, 65+) net Δ

Grid: work_share ∈ {0.03, 0.06, 0.09, 0.12, 0.15} × κ ∈ {0.2, 0.4, 0.6}.

Setup: φ U-shape fixed, R(0) default, A-fix coverage, HOLIDAY on,
γ_report = (0.40, 0.18, 0.25).  Point estimate L-BFGS multistart 12.
Fit is independent of κ (spillover=0 at baseline p_work=p_school=1); fit once
per work_share, evaluate at each κ.

Crash-safe: each (ws, κ) evaluation wrapped in try/except; partial JSON saved
after every combo. No user input required at any point.
"""
from __future__ import annotations
import os, json, time, traceback
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
from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target_by_age, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters, ModelParameters,
)
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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "redistribution_workshare.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "redistribution_workshare.png"
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

A_REL = np.array([0.29, 0.06, 0.36])
A_REL_NORM = A_REL / A_REL.sum()  # [0.408, 0.085, 0.507]
SIGMA_PER_CHANNEL = np.array([0.15, 0.01, 0.05, 0.15])

N_STARTS = 12
POINT_START_SEED = 23
LOG_R0_BOUNDS = (float(np.log(0.8)), float(np.log(3.0)))
LOGIT_PI_BOUNDS = (-10.0, 10.0)
PHI_NB_BOUNDS = (1e-3, 1e6)

P_WORK_BASE = 1.0
P_SCHOOL_BASE = 1.0
P_WORK_SICK = 0.4

COVERAGE_CAP = 0.99

WORKING_AGE_HIRA = ["18-44", "45-64"]
NONWORKING_HIRA = ["0-5", "6-11", "12-17", "65+"]


def build_hira_matrix() -> np.ndarray:
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for nims_idx, w in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, nims_idx] = w
    return H


def correct_coverage(cov_15):
    return -np.log(1.0 - np.minimum(cov_15, COVERAGE_CAP))


def build_gamma_15(c, a, e):
    return np.concatenate([np.full(4, c), np.full(9, a), np.full(2, e)])


def build_pi_target(work_share: float) -> np.ndarray:
    remaining = 1.0 - work_share
    return np.array([
        remaining * A_REL_NORM[0],
        work_share,
        remaining * A_REL_NORM[1],
        remaining * A_REL_NORM[2],
    ])


def logit_centered_target(pi_target):
    lp = np.log(np.clip(pi_target, 1e-6, None))
    return lp - lp.mean()


def build_setup():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]
    rho_emp = inputs["rho"]
    matrices = inputs["matrices"]
    mobility = inputs["mobility"]
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
    H = build_hira_matrix()
    # pop_15 is (15, n_admdong); reduce to (15,) per-age total population
    pop_15_arr = np.asarray(pop_15)
    if pop_15_arr.ndim == 2:
        # find which axis has length 15
        if pop_15_arr.shape[0] == 15:
            pop_15_flat = pop_15_arr.sum(axis=1)
        else:
            pop_15_flat = pop_15_arr.sum(axis=0)
    else:
        pop_15_flat = pop_15_arr
    assert pop_15_flat.shape == (15,), f"pop_15_flat has wrong shape {pop_15_flat.shape}"
    pop_6 = H @ pop_15_flat
    return dict(
        shared_base=shared_base, state0=state0,
        obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
        obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
        H=H, pop_6=pop_6, pop_15_flat=pop_15_flat,
    )


def make_shared(kappa_scalar, setup):
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
    pred = simulation_to_hira_by_age_jax(inc, gamma_15_j, n_weeks=setup["n_weeks"])
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


def point_fit_at_ws(work_share, setup):
    pi_target = build_pi_target(work_share)
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15 = build_gamma_15(*GAMMA_CENTER)
    gamma_15_j = jnp.asarray(gamma_15)
    shared_fit = make_shared(0.4, setup)

    fg = build_point_loss(logit_target, SIGMA_PER_CHANNEL, phi_full_j,
                          gamma_15_j, shared_fit, setup)
    bounds = [LOG_R0_BOUNDS] + [LOGIT_PI_BOUNDS] * 4 + [PHI_NB_BOUNDS]
    starts = make_starts(logit_target, N_STARTS, POINT_START_SEED)

    best = None; per_start_nll = []
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
    if best is None:
        raise RuntimeError("all L-BFGS starts failed at work_share=%s" % work_share)

    x = best["x"]
    log_R0 = float(x[0]); logit_pi = np.array(x[1:5]); phi_nb = float(x[5])
    R0 = float(np.exp(log_R0))
    pi = np.array(jax.nn.softmax(jnp.asarray(logit_pi)))
    beta_4 = np.array(derive_beta_from_R0_simplex(
        setup["ngm_default"], jnp.asarray(R0), jnp.asarray(pi), phi_full_j,
    ))
    return dict(
        work_share=work_share, pi_target=pi_target.tolist(),
        R0=R0, pi=[float(p) for p in pi],
        beta_4=[float(b) for b in beta_4],
        phi_nb=phi_nb, nll=best["nll"],
        fit_wall_sec=float(wall),
    )


def infections_per_hira6(inc, setup):
    """inc: (T-1, 15).  Returns (6,) total infections per HIRA age group."""
    tot_15 = np.asarray(inc).sum(axis=0)         # (15,)
    return setup["H"] @ tot_15                    # (6,)


def redistribution_at_kappa(beta_4, kappa_scalar, setup):
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15_j = jnp.asarray(build_gamma_15(*GAMMA_CENTER))
    shared_pol = make_shared(kappa_scalar, setup)
    beta_arr = np.array(beta_4)

    inc_b, _ = run_forward(beta_arr, phi_full_j, gamma_15_j,
                            P_SCHOOL_BASE, P_WORK_BASE, shared_pol, setup)
    inc_s, _ = run_forward(beta_arr, phi_full_j, gamma_15_j,
                            P_SCHOOL_BASE, P_WORK_SICK, shared_pol, setup)

    infb_6 = infections_per_hira6(inc_b, setup)   # (6,)
    infs_6 = infections_per_hira6(inc_s, setup)
    pop_6 = setup["pop_6"]

    attack_b = infb_6 / pop_6
    attack_s = infs_6 / pop_6
    d_attack = attack_s - attack_b       # + = redistributed INTO this group

    d_by_age = {ag: float(d_attack[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)}
    ab_by_age = {ag: float(attack_b[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)}
    as_by_age = {ag: float(attack_s[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)}

    # Transfer ratio = Σ(positive Δ · pop) / Σ(|negative Δ| · pop)
    #   → absolute counts, so units cancel and it's dimensionless.
    d_counts = d_attack * pop_6
    pos_sum = float(d_counts[d_counts > 0].sum())
    neg_abs_sum = float(-d_counts[d_counts < 0].sum())
    transfer_ratio = (pos_sum / neg_abs_sum) if neg_abs_sum > 1e-9 else float("inf")

    idx_work = [HIRA_AGE_GROUPS.index(a) for a in WORKING_AGE_HIRA]
    idx_non = [HIRA_AGE_GROUPS.index(a) for a in NONWORKING_HIRA]
    d_work_abs = float((d_attack[idx_work] * pop_6[idx_work]).sum())
    d_non_abs = float((d_attack[idx_non] * pop_6[idx_non]).sum())

    tot_b_infections = float(infb_6.sum())
    tot_s_infections = float(infs_6.sum())
    averted_total_pct = 100.0 * (tot_b_infections - tot_s_infections) \
        / max(tot_b_infections, 1.0)

    return dict(
        kappa=kappa_scalar,
        attack_baseline_by_age=ab_by_age,
        attack_sick_by_age=as_by_age,
        d_attack_by_age=d_by_age,
        infections_baseline_by_age={ag: float(infb_6[i])
                                    for i, ag in enumerate(HIRA_AGE_GROUPS)},
        infections_sick_by_age={ag: float(infs_6[i])
                                for i, ag in enumerate(HIRA_AGE_GROUPS)},
        pop_by_age={ag: float(pop_6[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)},
        transfer_ratio=transfer_ratio,
        d_infections_working_age=d_work_abs,
        d_infections_nonworking=d_non_abs,
        total_infections_baseline=tot_b_infections,
        total_infections_sick=tot_s_infections,
        averted_total_pct=averted_total_pct,
    )


def save_json(payload):
    tmp = OUT_JSON.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(OUT_JSON)


def make_plot(all_results, setup):
    combos = all_results["combos"]
    if not combos:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # (1) Δattack curves per age at κ=0.4 vs work_share
    ax = axes[0]
    ws_grid = sorted({c["work_share"] for c in combos if c.get("policy")})
    for ag in HIRA_AGE_GROUPS:
        ys = []
        for ws in ws_grid:
            rec = next((c for c in combos
                        if c["work_share"] == ws and c["kappa"] == 0.4
                        and c.get("policy")), None)
            if rec is None:
                ys.append(np.nan); continue
            ys.append(rec["policy"]["d_attack_by_age"][ag] * 100.0)
        ax.plot(ws_grid, ys, marker="o", label=ag)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("work_share")
    ax.set_ylabel("Δ attack rate  (sick − baseline)  [% pts]")
    ax.set_title("Δattack per age  (κ = 0.4)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # (2) Transfer ratio heatmap ws × κ
    ax = axes[1]
    ws_arr = sorted({c["work_share"] for c in combos})
    k_arr = sorted({c["kappa"] for c in combos})
    M = np.full((len(k_arr), len(ws_arr)), np.nan)
    for c in combos:
        if not c.get("policy"):
            continue
        i = k_arr.index(c["kappa"])
        j = ws_arr.index(c["work_share"])
        M[i, j] = c["policy"]["transfer_ratio"]
    Mplot = np.where(np.isfinite(M), M, np.nan)
    im = ax.imshow(Mplot, aspect="auto", cmap="RdYlBu_r", origin="lower",
                    vmin=0, vmax=2.0)
    for i in range(len(k_arr)):
        for j in range(len(ws_arr)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(ws_arr))); ax.set_xticklabels([f"{x:.2f}" for x in ws_arr])
    ax.set_yticks(range(len(k_arr))); ax.set_yticklabels([f"{k:.1f}" for k in k_arr])
    ax.set_xlabel("work_share"); ax.set_ylabel("κ")
    ax.set_title("Transfer ratio  (Σ+Δ / Σ|−Δ|)")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # (3) Attack bar at ws=0.06, κ=0.4 baseline vs sick
    ax = axes[2]
    rec = next((c for c in combos
                if c["work_share"] == 0.06 and c["kappa"] == 0.4
                and c.get("policy")), None)
    if rec:
        pol = rec["policy"]
        xs = np.arange(len(HIRA_AGE_GROUPS))
        b = np.array([pol["attack_baseline_by_age"][a] for a in HIRA_AGE_GROUPS])
        s = np.array([pol["attack_sick_by_age"][a] for a in HIRA_AGE_GROUPS])
        ax.bar(xs - 0.2, b * 100, 0.4, label="baseline")
        ax.bar(xs + 0.2, s * 100, 0.4, label="sick_leave")
        ax.set_xticks(xs); ax.set_xticklabels(HIRA_AGE_GROUPS)
        ax.set_ylabel("attack rate [%]")
        ax.set_title("Attack rate  ws=0.06  κ=0.4")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Sick-leave redistribution — {SEASON_LABEL}   "
        f"(p_work: 1.0 → {P_WORK_SICK})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 78, flush=True)
    print(f"REDISTRIBUTION: work_share × κ  —  {SEASON_LABEL}", flush=True)
    print(f"  work_share ∈ {WORK_SHARE_GRID}   κ ∈ {KAPPA_GRID} = 15 combos",
          flush=True)
    print(f"  φ U-shape fixed   γ = {GAMMA_CENTER}   R(0) default   HOLIDAY on",
          flush=True)
    print(f"  σ_channel = {SIGMA_PER_CHANNEL.tolist()}",
          flush=True)
    print(f"  policy: p_work {P_WORK_BASE} → {P_WORK_SICK}", flush=True)
    print("=" * 78, flush=True)

    try:
        setup = build_setup()
    except Exception:
        print("[FATAL] build_setup failed:\n" + traceback.format_exc(),
              flush=True)
        payload = dict(season=SEASON_LABEL, combos=[],
                        errors=[f"build_setup: {traceback.format_exc()}"])
        save_json(payload)
        return

    print(f"pop_6 = {[round(float(p),0) for p in setup['pop_6']]}", flush=True)

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
            p_work_sick=P_WORK_SICK,
        ),
        combos=[], errors=[], fits=[],
    )
    save_json(all_results)

    # Phase 1: Fit once per work_share (crash-safe)
    fits_by_ws = {}
    for ws in WORK_SHARE_GRID:
        print(f"\n── fit work_share = {ws:.2f} "
              f"(pi_target = {[round(float(x),4) for x in build_pi_target(ws)]}) ──",
              flush=True)
        try:
            fit = point_fit_at_ws(ws, setup)
            fits_by_ws[ws] = fit
            all_results["fits"].append(fit)
            print(f"  R0={fit['R0']:.3f}  NLL={fit['nll']:.4e}  "
                  f"β={[round(b,4) for b in fit['beta_4']]}  "
                  f"π={[round(p,4) for p in fit['pi']]}  "
                  f"(fit {fit['fit_wall_sec']:.1f}s)", flush=True)
        except Exception:
            err = traceback.format_exc()
            print(f"  [FAIL fit ws={ws}] {err}", flush=True)
            all_results["errors"].append(f"fit ws={ws}: {err}")
        save_json(all_results)

    # Phase 2: Redistribution per (ws, κ) — crash-safe per combo
    print("\n" + "=" * 78, flush=True)
    print("Phase 2: redistribution at each (ws, κ)", flush=True)
    print("=" * 78, flush=True)
    for ws in WORK_SHARE_GRID:
        fit = fits_by_ws.get(ws)
        if fit is None:
            print(f"\n[skip ws={ws}] no fit available", flush=True)
            for k in KAPPA_GRID:
                all_results["combos"].append(dict(
                    work_share=ws, kappa=k, policy=None, fit=None,
                    error="no_fit_available",
                ))
            save_json(all_results)
            continue
        for k in KAPPA_GRID:
            print(f"\n▶ ws={ws:.2f}  κ={k:.1f}", flush=True)
            t0 = time.perf_counter()
            combo_rec = dict(work_share=ws, kappa=k,
                              fit_R0=fit["R0"], fit_beta_4=fit["beta_4"],
                              fit_pi=fit["pi"], fit_nll=fit["nll"])
            try:
                pol = redistribution_at_kappa(fit["beta_4"], k, setup)
                combo_rec["policy"] = pol
                combo_rec["policy_wall_sec"] = time.perf_counter() - t0
                print(f"  averted total = {pol['averted_total_pct']:+.2f}%  "
                      f"transfer ratio = {pol['transfer_ratio']:.3f}",
                      flush=True)
                print(f"  Δattack (% pts):", flush=True)
                for ag in HIRA_AGE_GROUPS:
                    da = pol['d_attack_by_age'][ag] * 100
                    ab = pol['attack_baseline_by_age'][ag] * 100
                    asr = pol['attack_sick_by_age'][ag] * 100
                    print(f"    {ag:>6s}: base={ab:6.3f}%  sick={asr:6.3f}%  "
                          f"Δ={da:+7.4f}%pt", flush=True)
                print(f"  ΔInfections work(18-64) = {pol['d_infections_working_age']:+.0f}   "
                      f"non-work = {pol['d_infections_nonworking']:+.0f}",
                      flush=True)
            except Exception:
                err = traceback.format_exc()
                combo_rec["policy"] = None
                combo_rec["error"] = err
                print(f"  [FAIL] {err}", flush=True)
                all_results["errors"].append(f"combo ws={ws} k={k}: {err}")
            all_results["combos"].append(combo_rec)
            save_json(all_results)

    # Phase 3: plot & final save
    print("\n" + "=" * 78, flush=True)
    print("Phase 3: plot & final save", flush=True)
    try:
        make_plot(all_results, setup)
        print(f"saved {OUT_FIG}", flush=True)
    except Exception:
        print(f"[plot fail] {traceback.format_exc()}", flush=True)
        all_results["errors"].append(f"plot: {traceback.format_exc()}")
    save_json(all_results)
    print(f"saved {OUT_JSON}", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[TOP-LEVEL EXC]\n" + traceback.format_exc(), flush=True)
