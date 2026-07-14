"""B point-estimate: sweep channel prior target × strength, refit R0/π/φ/phi_nb.

Reproduces production's hira_model_nb_chprior β structure (β = R0×π via NGM
1-homogeneous inversion) as a point estimate.  Sweeps 2 targets × 3 σ_hw:
    targets = { T_lit  (R0 contrib home 0.35 / school 0.20 / work 0.05 / other 0.40),
                T_B    (production B, Italy 2009 R0 contrib [0.40, 0.10, 0.27, 0.23]) }
    σ_hw    = { 0.01, 0.10, 0.30 }    (school / other σ fixed at 0.30)

Free params (20): log_R0(1), logit_pi(4), phi_14(14, idx5 anchor≡1.0), phi_nb(1).
Fixed: γ=CDC[0.40/0.18/0.25], R(0) immunity default, κ default, HOLIDAY realloc=1
amp=0.7, AMP=0.9, σ=0.5, γ=0.25 (disease), NB observation.

Objective:
    total = NB_NLL
          + ch_prior_penalty = 0.5·Σ ((centered_logit_pi − logit_target)² / σ²)
          + phi_smooth       = 0.1·Σ (φ_full[i+1] − 2·φ_full[i] + φ_full[i-1])²

Decision guide (comments only — script does not interpret):
- σ_hw = 0.01 pin: work ≈ target (0.05 for T_lit / 0.17 for T_B). If φ becomes
  physically U-shaped and per-age r improves under pin, target is promising.
- Larger σ_hw → work drifts back to 0 → data still refuses work; pin is required.
- Compare T_lit vs T_B: which target improves per-age r most?
- Does channel pin remove the 15-19 φ spike seen in free diagnostics?
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
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "chprior_sweep.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "chprior_sweep.png"
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
UNIT_R0 = np.array([8.70, 6.21, 25.40, 9.33])   # (h, w, s, o) — production convention

# ── Target R0-contribution (channel order h, w, s, o) ─────────
# T_lit user-specified: home 0.35, school 0.20, work 0.05, other 0.40
#   → reordered (h, w, s, o) = [0.35, 0.05, 0.20, 0.40]
T_LIT_R0CONTRIB = np.array([0.35, 0.05, 0.20, 0.40])
# T_B production B (Italy 2009 H1N1)
T_B_R0CONTRIB = np.array([0.40, 0.10, 0.27, 0.23])

SIGMA_HW_LIST = [0.01, 0.10, 0.30]
SIGMA_SO_FIXED = 0.30

LAMBDA_PHI = 0.1
N_STARTS = 12
START_SEED = 29
PEAK_HALF_WIN = 2
REF_AGE_IDX = 5

# Bounds
LOG_R0_BOUNDS = (float(np.log(0.8)), float(np.log(3.0)))
LOGIT_PI_BOUNDS = (-10.0, 10.0)
PHI_BOUNDS = [(0.1, 5.0)] * 14
PHI_NB_BOUNDS = (1e-3, 1e6)


def r0contrib_to_pi(r0c: np.ndarray) -> np.ndarray:
    """Same production pipe: R0 contrib → π (β-share) via ÷ unit_R0 then normalize."""
    r0c_n = r0c / r0c.sum()
    beta_share = r0c_n / UNIT_R0
    return beta_share / beta_share.sum()


def logit_centered_target(pi_target: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(pi_target, 1e-6, None))
    return lp - lp.mean()


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


def predict_hira(log_R0, logit_pi, phi14, phi_nb, setup):
    R0 = jnp.exp(log_R0)
    pi = jax.nn.softmax(logit_pi, axis=-1)
    phi_full = phi14_to_phi_full(phi14)
    beta_4 = derive_beta_from_R0_simplex(setup["ngm_fn"], R0, pi, phi_full)
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    pred = simulation_to_hira_by_age_jax(inc, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])
    return pred, R0, pi, beta_4, phi_full


def build_loss(logit_target: jnp.ndarray, sigma_per_channel: jnp.ndarray, setup):
    """x layout (20):
      x[0]     = log_R0
      x[1:5]   = logit_pi
      x[5:19]  = phi_14
      x[19]    = phi_nb
    """
    inv_var = 1.0 / (sigma_per_channel ** 2)

    def loss(x):
        log_R0 = x[0]
        logit_pi = x[1:5]
        phi14 = x[5:19]
        phi_nb = x[19]
        pred, _, _, _, phi_full = predict_hira(
            log_R0, logit_pi, phi14, phi_nb, setup,
        )
        nll = nb_nll_jax(setup["obs_j"], pred, setup["w_j"],
                         concentration=phi_nb, min_rate=0.01)
        centered = logit_pi - jnp.mean(logit_pi)
        dev = centered - logit_target
        ch_pen = 0.5 * jnp.sum(dev * dev * inv_var)
        curv = phi_full[2:] - 2.0 * phi_full[1:-1] + phi_full[:-2]
        phi_pen = LAMBDA_PHI * jnp.sum(curv ** 2)
        return nll + ch_pen + phi_pen, nll, ch_pen, phi_pen

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
        tot, nll, chp, php = parts_j(x)
        return float(tot), float(nll), float(chp), float(php)

    return fg_np, parts_np


def make_starts(logit_target: np.ndarray, n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = []
    lp_base = list(logit_target)
    for delta_R in [np.log(2.0), np.log(1.5), np.log(2.5), np.log(1.8), np.log(2.2)]:
        starts.append(np.concatenate([[delta_R], lp_base, np.ones(14), [10.0]]))
    for _ in range(3):
        pi_pert = np.array(lp_base) + rng.normal(0, 0.5, 4)
        starts.append(np.concatenate([[np.log(2.0)], pi_pert,
                                       np.ones(14), [10.0]]))
    while len(starts) < n_starts:
        starts.append(np.concatenate([
            [rng.uniform(*LOG_R0_BOUNDS)],
            rng.normal(0, 1.0, 4),
            rng.uniform(0.5, 2.0, 14),
            [rng.uniform(2.0, 20.0)],
        ]))
    return starts[:n_starts]


def fit_combo(target_name: str, target_r0c: np.ndarray, sigma_hw: float,
               setup) -> dict:
    pi_target = r0contrib_to_pi(target_r0c)
    logit_target = logit_centered_target(pi_target)
    sigma_per_channel = jnp.asarray([sigma_hw, sigma_hw,
                                       SIGMA_SO_FIXED, SIGMA_SO_FIXED])

    fg, parts = build_loss(jnp.asarray(logit_target), sigma_per_channel, setup)
    bounds = ([LOG_R0_BOUNDS]
              + [LOGIT_PI_BOUNDS] * 4
              + PHI_BOUNDS
              + [PHI_NB_BOUNDS])
    starts = make_starts(logit_target, N_STARTS, START_SEED)

    per_start_nll = []
    best = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
            tot, nll, chp, php = parts(res.x)
        except Exception as e:
            print(f"      [warn] start {i} failed: {e}")
            continue
        per_start_nll.append(nll)
        if best is None or tot < best["total"]:
            best = {"total": tot, "nll": nll, "ch_pen": chp, "phi_pen": php,
                    "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0
    nll_arr = np.array(per_start_nll)

    # Diagnose best
    x = best["x"]
    log_R0 = float(x[0])
    logit_pi = np.array(x[1:5])
    phi14 = np.array(x[5:19])
    phi_nb = float(x[19])
    R0 = float(np.exp(log_R0))
    pi = np.array(jax.nn.softmax(jnp.asarray(logit_pi), axis=-1))
    phi_full = np.asarray(phi14_to_phi_full(jnp.asarray(phi14)))
    beta_4 = np.array(derive_beta_from_R0_simplex(
        setup["ngm_fn"], jnp.asarray(R0), jnp.asarray(pi),
        jnp.asarray(phi_full),
    ))

    pred, _, _, _, _ = predict_hira(
        jnp.asarray(log_R0), jnp.asarray(logit_pi), jnp.asarray(phi14),
        jnp.asarray(phi_nb), setup,
    )
    pred = np.asarray(pred)

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

    return dict(
        target_name=target_name, sigma_hw=sigma_hw,
        target_r0contrib=target_r0c.tolist(),
        pi_target=[float(x) for x in pi_target],
        sigma_per_channel=[float(x) for x in sigma_per_channel],
        R0=R0, pi=[float(x) for x in pi],
        beta_4=[float(x) for x in beta_4],
        phi_full_15=[float(p) for p in phi_full],
        phi_nb=phi_nb,
        nll=best["nll"], ch_pen=best["ch_pen"], phi_pen=best["phi_pen"],
        total=best["total"],
        best_start_idx=best["start_idx"],
        nll_std=float(np.std(nll_arr)) if len(nll_arr) > 1 else 0.0,
        nll_min=float(np.min(nll_arr)), nll_max=float(np.max(nll_arr)),
        per_age=per_age, wall_sec=float(wall),
        pred_hira=pred.tolist(),
    )


def main():
    print("=" * 78)
    print(f"chprior sweep — 2 targets × 3 σ_hw = 6 fits  ({SEASON_LABEL})")
    print(f"  T_lit (R0 contrib h,w,s,o) = {T_LIT_R0CONTRIB.tolist()}")
    print(f"  T_B   (R0 contrib h,w,s,o) = {T_B_R0CONTRIB.tolist()}")
    print(f"  σ_hw ∈ {SIGMA_HW_LIST}   (school/other σ = {SIGMA_SO_FIXED})")
    print(f"  free params: log_R0(1) + logit_pi(4) + phi_14(14) + phi_nb(1) = 20")
    print(f"  multi-start {N_STARTS}, λ_phi_2nd = {LAMBDA_PHI}")
    print("=" * 78)

    # Convert targets and print
    pi_T_lit = r0contrib_to_pi(T_LIT_R0CONTRIB)
    pi_T_B = r0contrib_to_pi(T_B_R0CONTRIB)
    print(f"  π target T_lit = {[round(float(x),3) for x in pi_T_lit]}  (h,w,s,o)")
    print(f"  π target T_B   = {[round(float(x),3) for x in pi_T_B]}  (h,w,s,o)")

    setup = build_setup()
    print(f"  n_weeks = {setup['n_weeks']}   obs sum = "
          f"{float(np.asarray(setup['obs_j']).sum()):,.0f}")

    combos = [
        ("T_lit", T_LIT_R0CONTRIB, s) for s in SIGMA_HW_LIST
    ] + [
        ("T_B", T_B_R0CONTRIB, s) for s in SIGMA_HW_LIST
    ]

    all_results = []
    for name, tgt, sig in combos:
        print(f"\n── fit target={name}  σ_hw={sig} ──")
        r = fit_combo(name, tgt, sig, setup)
        all_results.append(r)
        print(f"  NLL={r['nll']:.4e}  ch_pen={r['ch_pen']:.4e}  "
              f"phi_pen={r['phi_pen']:.4e}  total={r['total']:.4e}  "
              f"wall={r['wall_sec']:.1f}s  best_start={r['best_start_idx']}")
        print(f"  π recovered = {[round(x,4) for x in r['pi']]}   "
              f"π target = {[round(x,4) for x in r['pi_target']]}")
        print(f"  ★ π_work = {r['pi'][1]:.4f}   (target π_work = {r['pi_target'][1]:.4f})")
        print(f"  β_4 = {[round(x,4) for x in r['beta_4']]}")
        print(f"  R0 = {r['R0']:.3f}   phi_nb = {r['phi_nb']:.2f}")
        print(f"  NLL 12 starts: min={r['nll_min']:.4e}  max={r['nll_max']:.4e}"
              f"  std={r['nll_std']:.3e}")
        print(f"  φ_full(15) = {[round(p,3) for p in r['phi_full_15']]}")

    # ─── Summary ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  chprior sweep summary")
    print("  target  σ_hw    π_h    π_w    π_s    π_o     R0    NLL         "
          "φ[3]  phi_nb")
    print("-" * 78)
    for r in all_results:
        pi = r["pi"]; ph = r["phi_full_15"]
        print(f"  {r['target_name']:>5s}  {r['sigma_hw']:.2f}  "
              f"{pi[0]:.3f}  {pi[1]:.3f}  {pi[2]:.3f}  {pi[3]:.3f}  "
              f"{r['R0']:.3f}  {r['nll']:.4e}  {ph[3]:.3f}  {r['phi_nb']:.2f}")

    print("\n" + "=" * 78)
    print(f"  Per-age r  (obs_peak±{PEAK_HALF_WIN}w / model_peak±{PEAK_HALF_WIN}w)")
    print("=" * 78)
    header = "  target σ_hw  " + "  ".join(f"{ag:>8s}" for ag in HIRA_AGE_GROUPS) \
             + "    phase(off)"
    print(header)
    for r in all_results:
        row = f"  {r['target_name']:>5s}  {r['sigma_hw']:.2f}  "
        for ag in HIRA_AGE_GROUPS:
            row += f"  {r['per_age'][ag]['ratio']:>8.2f}"
        offs = [r['per_age'][ag]['phase_offset_weeks'] for ag in HIRA_AGE_GROUPS]
        row += f"     [{','.join(f'{o:+d}' for o in offs)}]"
        print(row)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {
                "AMP": AMP, "HOLIDAY": HOLIDAY,
                "GAMMA_3": [0.40, 0.18, 0.25],
                "UNIT_R0": UNIT_R0.tolist(),
                "T_lit_R0contrib_hwso": T_LIT_R0CONTRIB.tolist(),
                "T_B_R0contrib_hwso": T_B_R0CONTRIB.tolist(),
                "pi_target_T_lit": pi_T_lit.tolist(),
                "pi_target_T_B": pi_T_B.tolist(),
                "sigma_hw_list": SIGMA_HW_LIST,
                "sigma_so_fixed": SIGMA_SO_FIXED,
                "n_starts": N_STARTS, "start_seed": START_SEED,
                "lambda_phi_2ndorder": LAMBDA_PHI,
                "log_R0_bounds": LOG_R0_BOUNDS,
                "logit_pi_bounds": LOGIT_PI_BOUNDS,
                "phi_bounds": PHI_BOUNDS[0],
                "peak_half_window_weeks": PEAK_HALF_WIN,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.4, 1.4])

    # Panel 1: π_4 bars (6 combos side-by-side per channel) + target dashed
    ax1 = fig.add_subplot(gs[0, 0])
    channels = ["home", "work", "school", "other"]
    n_combos = len(all_results)
    bar_w = 0.11
    x_ch = np.arange(4)
    combo_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_combos))
    for i, r in enumerate(all_results):
        offset = (i - n_combos / 2 + 0.5) * bar_w
        ax1.bar(x_ch + offset, r["pi"], bar_w,
                 label=f"{r['target_name']} σ={r['sigma_hw']:.2f}",
                 color=combo_colors[i], edgecolor="k", lw=0.4)
    # target markers
    for tname, r0c, sty in [("T_lit", T_LIT_R0CONTRIB, "D"),
                              ("T_B", T_B_R0CONTRIB, "s")]:
        pi_t = r0contrib_to_pi(r0c)
        ax1.scatter(x_ch, pi_t, marker=sty, s=90, edgecolor="k",
                    facecolor="none", lw=1.6, label=f"target {tname}")
    ax1.set_xticks(x_ch); ax1.set_xticklabels(channels)
    ax1.set_ylabel("π (β-share simplex)")
    ax1.set_title("π_4 recovered per combo vs target")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=7, ncol=2, loc="upper right")

    # Panel 2: φ_full overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ages_idx = np.arange(15)
    for i, r in enumerate(all_results):
        ax2.plot(ages_idx, r["phi_full_15"], "-o",
                 color=combo_colors[i], lw=1.4, ms=4,
                 label=f"{r['target_name']} σ={r['sigma_hw']:.2f}")
    ax2.axhline(1.0, color="grey", ls=":", lw=1)
    ax2.axvline(REF_AGE_IDX, color="red", ls=":", lw=1, label="anchor idx 5")
    ax2.axvline(3, color="magenta", ls=":", lw=1, label="idx 3 (15-19)")
    ax2.set_xticks(ages_idx)
    ax2.set_xticklabels([f"{5*i}-{5*i+4}" for i in range(14)] + ["70+"],
                        rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("φ_full (15)")
    ax2.set_title("φ per combo (anchor idx 5 ≡ 1.0)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=2, loc="upper right")

    # Panel 3: representative combo (T_lit σ=0.01) obs vs pred per age
    rep = next((r for r in all_results
                 if r["target_name"] == "T_lit" and r["sigma_hw"] == SIGMA_HW_LIST[0]),
                 all_results[0])
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    weeks = np.arange(obs.shape[0])[mask]
    pred_rep = np.asarray(rep["pred_hira"])
    ax3 = fig.add_subplot(gs[0, 2])
    age_colors = plt.cm.tab10.colors
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        ax3.plot(weeks, obs[mask, ai], "o", color=age_colors[ai],
                 ms=3, alpha=0.8, label=f"{ag} obs")
        ax3.plot(weeks, pred_rep[mask, ai], "-", color=age_colors[ai], lw=1.5)
    ax3.set_yscale("log")
    ax3.set_xlabel("week")
    ax3.set_ylabel("HIRA weekly count (log)")
    ax3.set_title(f"epi curves ({rep['target_name']} σ={rep['sigma_hw']:.2f})"
                   f"  NLL={rep['nll']:.3e}")
    ax3.legend(fontsize=6, ncol=2, loc="lower right")
    ax3.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"channel prior sweep — 2 targets × 3 σ_hw  ({SEASON_LABEL})")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
