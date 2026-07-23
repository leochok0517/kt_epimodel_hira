"""Diagnostic — quantify β bias from holiday spillover error (realloc=1).

4 conditions, single (ws=0.06, κ=0.4) point fit each:
  (1) BASE       realloc=1.0, amp=0.7   — current buggy structure
  (2) NO_SPILL   realloc=0.0, amp=0.7   — β_s scaled by mult, spillover 0
  (3) NO_HOLIDAY amp=0.0                — school channel unmodified all season
  (4) MASK_FIT   realloc=1.0, amp=0.7,
                 fitting weights zeroed on weeks 16-26 (holiday overlap)

Reuses sens_workshare_kappa_v2 setup/fit/forward primitives (no code fork).
Point estimate L-BFGS multistart 12. Output: outputs/eda/diag_holiday_bias.json.
"""
from __future__ import annotations
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

from sens_workshare_kappa_v2 import (
    build_setup, make_shared, run_forward, build_gamma_15,
    build_pi_target, logit_centered_target, make_starts,
    PHI_USHAPE, GAMMA_CENTER, SIGMA_PER_CHANNEL, HIRA_AGE_GROUPS,
    P_WORK_BASE, P_SCHOOL_BASE, SEASON_LABEL,
    N_STARTS, POINT_START_SEED, LOG_R0_BOUNDS, LOGIT_PI_BOUNDS,
    PHI_NB_BOUNDS,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import (
    simulate_jax, daily_new_infection_by_age_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "diag_holiday_bias.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

WORK_SHARE = 0.06
KAPPA = 0.4

# Fitting-region mask for MASK_FIT: zero weights on epi-weeks that overlap the
# holiday period.  Holiday = season days 113..183 = weeks ~16.1..26.1.
# We mask weeks with index ≥ 15 (0-indexed → week 16+ in 1-indexed).
MASK_WEEK_START = 15   # 0-indexed → epi-week 16


CONDITIONS = [
    dict(name="BASE",       amp=0.7, realloc=1.0, mask_weeks=False),
    dict(name="NO_SPILL",   amp=0.7, realloc=0.0, mask_weeks=False),
    dict(name="NO_HOLIDAY", amp=0.0, realloc=0.0, mask_weeks=False),  # realloc irrelevant when amp=0
    dict(name="MASK_FIT",   amp=0.7, realloc=1.0, mask_weeks=True),
]


def override_holiday(shared, amp, realloc):
    kw = dict(shared)
    kw["school_holiday_amp"] = amp
    kw["school_holiday_realloc"] = realloc
    return kw


def run_forward_local(beta_4, phi_full_j, gamma_15_j, p_school, p_work,
                      shared, setup):
    kw = dict(shared)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_j
    kw["p_school"] = p_school; kw["p_work"] = p_work
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc = daily_new_infection_by_age_jax(st)
    pred = simulation_to_hira_by_age_jax(inc, gamma_15_j, n_weeks=setup["n_weeks"])
    return inc, pred


def build_loss(logit_target, phi_full_j, gamma_15_j, shared_fit, setup, w_j):
    lt = jnp.asarray(logit_target)
    inv_var = 1.0 / (SIGMA_PER_CHANNEL ** 2)

    def loss(x):
        log_R0 = x[0]; logit_pi = x[1:5]; phi_nb = x[5]
        R0 = jnp.exp(log_R0)
        pi = jax.nn.softmax(logit_pi)
        beta_4 = derive_beta_from_R0_simplex(setup["ngm_default"], R0, pi,
                                              phi_full_j)
        _, pred = run_forward_local(beta_4, phi_full_j, gamma_15_j,
                                     P_SCHOOL_BASE, P_WORK_BASE,
                                     shared_fit, setup)
        nll = nb_nll_jax(setup["obs_j"], pred, w_j,
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


def fit_one(cond, setup):
    pi_target = build_pi_target(WORK_SHARE)
    logit_target = logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(PHI_USHAPE)
    gamma_15_j = jnp.asarray(build_gamma_15(*GAMMA_CENTER))

    # Build shared with holiday overrides
    base_shared = make_shared(KAPPA, setup)
    shared_fit = override_holiday(base_shared, cond["amp"], cond["realloc"])

    # Mask fit weights if requested
    w_np = setup["w_np"].copy()
    if cond["mask_weeks"]:
        w_np[MASK_WEEK_START:, :] = 0.0
    w_j = jnp.asarray(w_np)
    n_masked = int((setup["w_np"] > 0).sum() - (w_np > 0).sum())

    fg = build_loss(logit_target, phi_full_j, gamma_15_j, shared_fit, setup, w_j)
    bounds = [LOG_R0_BOUNDS] + [LOGIT_PI_BOUNDS]*4 + [PHI_NB_BOUNDS]
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

    x = best["x"]
    log_R0 = float(x[0]); logit_pi = np.array(x[1:5]); phi_nb = float(x[5])
    R0 = float(np.exp(log_R0))
    pi = np.array(jax.nn.softmax(jnp.asarray(logit_pi)))
    beta_4 = np.array(derive_beta_from_R0_simplex(
        setup["ngm_default"], jnp.asarray(R0), jnp.asarray(pi), phi_full_j,
    ))

    return dict(
        condition=cond["name"], amp=cond["amp"], realloc=cond["realloc"],
        mask_weeks=cond["mask_weeks"], n_weeks_masked=n_masked,
        beta_4=[float(b) for b in beta_4],
        pi=[float(p) for p in pi], pi_target=pi_target.tolist(),
        R0=R0, phi_nb=phi_nb, nll=best["nll"], wall_sec=wall,
        nll_min=float(np.min(per_start_nll)) if per_start_nll else float("nan"),
        nll_std=float(np.std(per_start_nll)) if len(per_start_nll) > 1 else 0.0,
        n_starts_ok=len(per_start_nll),
    )


def main():
    print("=" * 78, flush=True)
    print(f"HOLIDAY SPILLOVER BIAS DIAGNOSTIC  —  ws={WORK_SHARE}  κ={KAPPA}",
          flush=True)
    print(f"  Mask weeks (MASK_FIT): 0-indexed {MASK_WEEK_START}+ "
          f"(epi week 16+, holiday day 113-183 overlap)", flush=True)
    print("=" * 78, flush=True)

    setup = build_setup()

    rows = []
    for cond in CONDITIONS:
        print(f"\n▶ {cond['name']}  amp={cond['amp']}  realloc={cond['realloc']}  "
              f"mask={cond['mask_weeks']}", flush=True)
        rec = fit_one(cond, setup)
        rows.append(rec)
        print(f"   β=[h={rec['beta_4'][0]:.5f}, w={rec['beta_4'][1]:.5f}, "
              f"s={rec['beta_4'][2]:.5f}, o={rec['beta_4'][3]:.5f}]  "
              f"R0={rec['R0']:.3f}  NLL={rec['nll']:.4e}  wall={rec['wall_sec']:.1f}s",
              flush=True)
        print(f"   π=[h={rec['pi'][0]:.4f}, w={rec['pi'][1]:.4f}, "
              f"s={rec['pi'][2]:.4f}, o={rec['pi'][3]:.4f}]  "
              f"φ_nb={rec['phi_nb']:.2f}", flush=True)

    # β_home change vs BASE
    base_bh = rows[0]["beta_4"][0]
    print("\n" + "=" * 78, flush=True)
    print(f"{'cond':>12s} {'β_home':>9s} {'Δ%vsBASE':>10s} "
          f"{'β_work':>9s} {'β_school':>10s} {'β_other':>9s} "
          f"{'R0':>6s} {'NLL':>10s}", flush=True)
    print("-" * 78, flush=True)
    for r in rows:
        bh = r["beta_4"][0]
        dpct = 100.0 * (bh - base_bh) / base_bh if base_bh else float("nan")
        marker = ""
        if r["condition"] != "BASE":
            if abs(dpct) < 5:   marker = " (<5% negl)"
            elif abs(dpct) < 20: marker = " (5-20% mod)"
            else:               marker = " (>20% SEVERE)"
        print(f"{r['condition']:>12s} {bh:>9.5f} {dpct:>+9.2f}%{marker:>0s}",
              flush=True)
        print(f"{'':>12s}          {'':>10s} "
              f"{r['beta_4'][1]:>9.5f} {r['beta_4'][2]:>10.5f} "
              f"{r['beta_4'][3]:>9.5f} {r['R0']:>6.3f} {r['nll']:>10.4e}",
              flush=True)

    # NO_HOLIDAY − NO_SPILL 차이: 방학 학교β 감소 자체 효과
    bh_no_spill = rows[1]["beta_4"][0]
    bh_no_hol = rows[2]["beta_4"][0]
    print(f"\nNO_HOLIDAY vs NO_SPILL β_home diff: "
          f"{100*(bh_no_hol - bh_no_spill)/bh_no_spill:+.2f}%   "
          f"(방학 학교β 감소 자체 효과)", flush=True)

    payload = dict(
        setup=dict(work_share=WORK_SHARE, kappa=KAPPA,
                    season=SEASON_LABEL,
                    mask_week_start_idx=MASK_WEEK_START,
                    phi_ushape=PHI_USHAPE.tolist(),
                    gamma_center=list(GAMMA_CENTER),
                    sigma_per_channel=SIGMA_PER_CHANNEL.tolist(),
                    n_starts=N_STARTS),
        rows=rows,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nsaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
