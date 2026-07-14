"""Diagnose whether β_o (other channel) absorbs β_h and β_w.

Test: hold β_o at fixed levels ∈ {free, 0.0, 0.03, 0.06, 0.10} and refit the
remaining channels. If (β_h + β_w) rises as β_o drops while NLL is nearly
flat, other is absorbing home/work — a substitution non-identifiability, and
home/work≈0 in production is an artefact of the substitution rather than a
real zero signal.

Repeat under 3 φ conditions to check whether the absorption is φ-independent:
  (1) φ = ones(15)                            (production choice)
  (2) φ = λ=0.1 best from phi_2ndorder.json   (data-driven, mild smoothing)
  (3) φ = λ=100 best from phi_2ndorder.json   (physical monotone shape)

Single season 2019-2020, NB obs, HOLIDAY realloc=1 amp=0.7, AMP=0.9, γ=CDC.
φ_full is FIXED per condition (not estimated). β_o is FIXED per sweep level
(free case is a normal 4-channel fit). L-BFGS multi-start 8.

Decision guide (comments only — script does not interpret):
- β_o↓ → (β_h+β_w)↑ with NLL nearly flat  → substitution non-identifiability
  (other absorbing home/work; the home/work≈0 result is not a real zero).
- β_o↓ → NLL sharply rises  → other is genuinely needed; home/work≈0 is real.
- 3 φ curves identical → absorption is φ-independent.  Different → confound.
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
PREV_JSON = REPO_ROOT / "outputs" / "eda" / "phi_2ndorder.json"
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "other_absorption.json"
OUT_FIG = REPO_ROOT / "presentations" / "figures" / "other_absorption.png"
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
BETA_BOUNDS_SINGLE = (0.001, 1.0)
PHI_NB_BOUNDS = (1e-3, 1e6)
N_STARTS = 8
START_SEED = 13
PEAK_HALF_WIN = 2

BETA_O_LEVELS = ["free", 0.00, 0.03, 0.06, 0.10]

PHI_LAM100_FALLBACK = [1.786, 1.406, 1.202, 1.046, 0.979, 1.000, 1.028,
                        1.065, 1.083, 1.101, 1.139, 1.186, 1.207, 1.250, 1.301]


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


def predict_hira(beta_4, phi_full_15, *, setup):
    kw = dict(setup["shared"])
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = phi_full_15
    st = simulate_jax(setup["state0"], **kw, discretize_time=False)
    inc_15 = daily_new_infection_by_age_jax(st)
    return simulation_to_hira_by_age_jax(inc_15, setup["gamma_15"],
                                          n_weeks=setup["n_weeks"])


def peak_window_sum(arr_1d, peak_w, half):
    lo = max(0, peak_w - half)
    hi = min(arr_1d.shape[0], peak_w + half + 1)
    return float(arr_1d[lo:hi].sum())


def per_age_ratios(pred, setup):
    obs = setup["obs_np"]
    mask = setup["w_np"].sum(axis=1) > 0
    obs_masked = np.where(mask[:, None], obs, -1e18)
    pred_masked = np.where(mask[:, None], pred, -1e18)
    rows = {}
    for ai, ag in enumerate(HIRA_AGE_GROUPS):
        obs_pw = int(np.argmax(obs_masked[:, ai]))
        mdl_pw = int(np.argmax(pred_masked[:, ai]))
        obs_sum = peak_window_sum(obs[:, ai], obs_pw, PEAK_HALF_WIN)
        mdl_sum = peak_window_sum(pred[:, ai], mdl_pw, PEAK_HALF_WIN)
        rows[ag] = dict(
            obs_peak_week=obs_pw, model_peak_week=mdl_pw,
            phase_offset_weeks=mdl_pw - obs_pw,
            ratio=obs_sum / max(mdl_sum, 1.0),
        )
    return rows


def build_loss(beta_o_level, phi_full_15: jnp.ndarray, setup):
    """Build (fg, x_dim, unpack).
    beta_o_level = 'free': x = [β_h, β_w, β_s, β_o, phi_nb]  (5 dim)
    beta_o_level = float:  x = [β_h, β_w, β_s, phi_nb]        (4 dim)  β_o fixed
    """
    free = (beta_o_level == "free")

    def loss(x):
        if free:
            beta_4 = x[:4]; phi_nb = x[4]
        else:
            beta_4 = jnp.array([x[0], x[1], x[2], beta_o_level])
            phi_nb = x[3]
        pred = predict_hira(beta_4, phi_full_15, setup=setup)
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

    x_dim = 5 if free else 4
    return fg_np, x_dim


def make_starts(x_dim: int, n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    # β_free portion + phi_nb=10.0
    base_5 = [
        np.array([0.10, 0.10, 0.10, 0.10, 10.0]),
        np.array([0.05, 0.05, 0.05, 0.15, 10.0]),
        np.array([0.07, 0.07, 0.20, 0.10, 10.0]),
        np.array([0.07, 0.07, 0.05, 0.20, 10.0]),
    ]
    starts = []
    for b in base_5:
        if x_dim == 5:
            starts.append(b.copy())
        else:
            # drop the β_o slot (index 3), keep phi_nb (index 4)
            starts.append(np.array([b[0], b[1], b[2], b[4]]))
    while len(starts) < n_starts:
        b = rng.uniform(0.02, 0.20, 4)
        pn = rng.uniform(2.0, 20.0)
        if x_dim == 5:
            starts.append(np.concatenate([b, np.array([pn])]))
        else:
            starts.append(np.array([b[0], b[1], b[2], pn]))
    return starts[:n_starts]


def fit_one(beta_o_level, phi_full_15: jnp.ndarray, setup) -> dict:
    fg, x_dim = build_loss(beta_o_level, phi_full_15, setup)
    if beta_o_level == "free":
        bounds = [BETA_BOUNDS_SINGLE] * 4 + [PHI_NB_BOUNDS]
    else:
        bounds = [BETA_BOUNDS_SINGLE] * 3 + [PHI_NB_BOUNDS]
    starts = make_starts(x_dim, N_STARTS, START_SEED)
    best = None
    per_start_nll = []
    t0 = time.perf_counter()
    for i, x0 in enumerate(starts):
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                            bounds=bounds,
                            options=dict(maxiter=300, ftol=1e-9, gtol=1e-6))
            nll = float(res.fun)
        except Exception as e:
            print(f"      [warn] start {i} failed: {e}")
            continue
        per_start_nll.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x), "start_idx": i}
    wall = time.perf_counter() - t0

    if beta_o_level == "free":
        beta_4 = np.array(best["x"][:4]); phi_nb = float(best["x"][4])
    else:
        beta_4 = np.array([best["x"][0], best["x"][1], best["x"][2],
                            float(beta_o_level)])
        phi_nb = float(best["x"][3])

    pred = np.asarray(predict_hira(beta_4, phi_full_15, setup=setup))
    r = per_age_ratios(pred, setup)
    r0 = float(setup["ngm_fn"](
        jnp.asarray(beta_4[0]), jnp.asarray(beta_4[1]),
        jnp.asarray(beta_4[2]), jnp.asarray(beta_4[3]),
        jnp.asarray(phi_full_15),
    ))
    return dict(
        beta_o_level=beta_o_level, nll=best["nll"], wall_sec=float(wall),
        beta_4=[float(x) for x in beta_4],
        beta_h_plus_w=float(beta_4[0] + beta_4[1]),
        phi_nb=phi_nb, R0_ngm=r0,
        best_start_idx=best["start_idx"],
        nll_per_start=per_start_nll,
        per_age=r,
    )


def load_phi_conditions():
    conds = {"phi1": np.ones(15).tolist()}
    if PREV_JSON.exists():
        with open(PREV_JSON) as f:
            prev = json.load(f)
        for r in prev["results"]:
            lam = r["lambda_phi"]
            if lam == 0.1:
                conds["phi_l01"] = r["diagnostic"]["phi_full_15"]
            if lam == 100.0:
                conds["phi_l100"] = r["diagnostic"]["phi_full_15"]
    if "phi_l100" not in conds:
        conds["phi_l100"] = list(PHI_LAM100_FALLBACK)
    return conds


def main():
    print("=" * 78)
    print(f"DIAGNOSE: β_o absorption of β_h/β_w  —  {SEASON_LABEL}")
    print(f"  3 φ conditions × 5 β_o levels = 15 fits, multi-start {N_STARTS}")
    print(f"  β_o levels: {BETA_O_LEVELS}")
    print("=" * 78)

    setup = build_setup()
    phi_conds = load_phi_conditions()
    print("  φ conditions:")
    for k, v in phi_conds.items():
        print(f"    {k}: {[round(float(p),3) for p in v]}")

    all_results = {}
    for cond_name, phi_list in phi_conds.items():
        phi_j = jnp.asarray(phi_list)
        print(f"\n── φ = {cond_name} ──")
        rows = []
        for lvl in BETA_O_LEVELS:
            print(f"    fit β_o={lvl}")
            r = fit_one(lvl, phi_j, setup)
            rows.append(r)
            print(f"      NLL={r['nll']:.4e}  β_h/w/s/o={[round(x,4) for x in r['beta_4']]}"
                  f"  β_h+w={r['beta_h_plus_w']:.4f}  R0={r['R0_ngm']:.3f}"
                  f"  phi_nb={r['phi_nb']:.2f}  start={r['best_start_idx']}"
                  f"  wall={r['wall_sec']:.1f}s")
        all_results[cond_name] = rows

    # ─── Console tables per φ condition ───────────────
    print("\n" + "=" * 78)
    print("  Per-φ tables: rows = β_o levels, columns = β_h/w/s + NLL + R0")
    print("=" * 78)
    for cond_name, rows in all_results.items():
        print(f"\n  [φ = {cond_name}]")
        print("  β_o        β_h      β_w      β_s      β_h+w    NLL          R0     phi_nb")
        for r in rows:
            lvl = r["beta_o_level"]
            lvl_s = "free" if lvl == "free" else f"{lvl:.2f}"
            b = r["beta_4"]
            print(f"  {lvl_s:>6s}  {b[0]:>7.4f}  {b[1]:>7.4f}  {b[2]:>7.4f}"
                  f"  {r['beta_h_plus_w']:>7.4f}  {r['nll']:.4e}"
                  f"  {r['R0_ngm']:>5.3f}  {r['phi_nb']:>5.2f}")

    # ─── Per-age r tables ────────────────────────────
    print("\n" + "=" * 78)
    print(f"  Per-age r (obs_peak±{PEAK_HALF_WIN}w / model_peak±{PEAK_HALF_WIN}w)")
    print("=" * 78)
    header = "  β_o     " + "  ".join(f"{ag:>8s}" for ag in HIRA_AGE_GROUPS)
    for cond_name, rows in all_results.items():
        print(f"\n  [φ = {cond_name}]  {header[8:]}")
        for r in rows:
            lvl = r["beta_o_level"]
            lvl_s = "free" if lvl == "free" else f"{lvl:.2f}"
            row = f"  {lvl_s:>6s}  "
            for ag in HIRA_AGE_GROUPS:
                row += f"  {r['per_age'][ag]['ratio']:>8.2f}"
            print(row)

    # ─── Save JSON ───────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump({
            "season": SEASON_LABEL,
            "setup": {"AMP": AMP, "HOLIDAY": HOLIDAY,
                       "GAMMA_3": [0.40, 0.18, 0.25],
                       "beta_bounds": list(BETA_BOUNDS_SINGLE),
                       "beta_o_levels": [str(x) for x in BETA_O_LEVELS],
                       "n_starts": N_STARTS, "start_seed": START_SEED,
                       "peak_half_window_weeks": PEAK_HALF_WIN},
            "phi_conditions": phi_conds,
            "results": all_results,
        }, f, indent=2)
    print(f"\nsaved {OUT_JSON}")

    # ─── Figure: (left) β_o vs β_h+w  (right) β_o vs NLL ─
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cond_colors = {"phi1": "#888888", "phi_l01": "#c0392b", "phi_l100": "#1a5490"}
    cond_labels = {"phi1": "φ = 1", "phi_l01": "φ = λ0.1",
                    "phi_l100": "φ = λ100"}

    # For plotting, treat 'free' as the resulting β_o value along x
    for cond_name, rows in all_results.items():
        xs, hw, nlls = [], [], []
        # order: fixed levels then free
        fixed_rows = [r for r in rows if r["beta_o_level"] != "free"]
        for r in sorted(fixed_rows, key=lambda x: x["beta_o_level"]):
            xs.append(float(r["beta_o_level"]))
            hw.append(r["beta_h_plus_w"])
            nlls.append(r["nll"])
        # add free point using its fitted β_o
        for r in rows:
            if r["beta_o_level"] == "free":
                axes[0].scatter([r["beta_4"][3]], [r["beta_h_plus_w"]],
                                 marker="*", s=180, color=cond_colors[cond_name],
                                 edgecolor="k", zorder=5,
                                 label=f"{cond_labels[cond_name]} free")
                axes[1].scatter([r["beta_4"][3]], [r["nll"]],
                                 marker="*", s=180, color=cond_colors[cond_name],
                                 edgecolor="k", zorder=5,
                                 label=f"{cond_labels[cond_name]} free")
        axes[0].plot(xs, hw, "-o", color=cond_colors[cond_name],
                     label=f"{cond_labels[cond_name]} sweep")
        axes[1].plot(xs, nlls, "-o", color=cond_colors[cond_name],
                     label=f"{cond_labels[cond_name]} sweep")

    axes[0].set_xlabel("β_o (fixed level;  ★ = free)")
    axes[0].set_ylabel("β_h + β_w")
    axes[0].set_title("Absorption: β_o vs (β_h + β_w)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].set_xlabel("β_o (fixed level;  ★ = free)")
    axes[1].set_ylabel("NLL")
    axes[1].set_title("Cost of substitution: β_o vs NLL")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)

    fig.suptitle(f"β_o absorption sweep  —  {SEASON_LABEL}  "
                  f"(3 φ conditions × 5 β_o levels)")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"saved {OUT_FIG}")


if __name__ == "__main__":
    main()
