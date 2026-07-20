"""Policy map under Step A (C(t) term↔vacation + κ 3-way base) + Step B (v(t)
finite-season normalization). 2D grid: work_share (12) × μ κ-scale (4) = 48.

Per combo: pin-fit β_4 (baseline, κ-independent → fit once per work_share, reuse
across μ), then sick-leave policy (p_work=0.4) redistribution:
  averted % total, per-age Δattack (%pt, HIRA-6), transfer ratio, age flip.

κ_a = μ · κ_base_a,  κ_base = [학생0.29×4, 직장인0.30×10, 70+0].
Reuses sens_workshare_kappa_v2 harness (reparam A pin fit) + redistribution attack
math. build_setup: RAW coverage (Step B owns A-fix), C_*_vac injected, no HOLIDAY.

Output: outputs/eda/policy_map_stepAB.json + console grid. Point estimate.
"""
from __future__ import annotations
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

from kt_data import SUDOGWON_SIDO_CODES
from kt_data.data.load_contact import load_contact_matrices
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs, estimate_initial_infected_from_hira,
    _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE,
)
from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target_by_age, HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)
import sens_workshare_kappa_v2 as S   # harness reuse (main-guarded)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "policy_map_stepAB.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
VAC_NPZ = REPO_ROOT.parent / "kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz"

WORK_SHARE_GRID = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36]
MU_GRID = [0.5, 1.0, 1.5, 2.0]
KAPPA_BASE = np.array([0.29]*4 + [0.30]*10 + [0.0])
WORKING_AGE_HIRA = ["18-44", "45-64"]
NONWORKING_HIRA = ["0-5", "6-11", "12-17", "65+"]
LIT_POINTS = {"Italy": 0.03, "B": 0.17, "A": 0.29}
# documented old-base sick-leave reference (docs/PROJECT_REFERENCE.md §, β_home≈0.036)
OLD_REF = {"sick_averted_range_pct": [-2.8, 2.2], "B_sign_flip_kappa": 0.49}


def build_hira_matrix() -> np.ndarray:
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for nims_idx, w in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, nims_idx] = w
    return H


def build_setup_AB():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease
    vax = ModelParameters().vaccination

    seed_15 = estimate_initial_infected_from_hira(
        S.SEASON_LABEL, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15,
    )
    tgt = load_hira_target_by_age(
        S.SEASON_LABEL, sido_codes=list(SUDOGWON_SIDO_CODES),
        first_peak_only=True, first_peak_end_week=26,
    )
    n_weeks = tgt["n_weeks"]
    obs = np.zeros((n_weeks, 6)); w = np.zeros((n_weeks, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]; w[:, i] = tgt["weights"][ag]

    vac = load_contact_matrices(path=VAC_NPZ)
    shared_base = dict(
        C_home=jnp.asarray(matrices["C_home"]), C_school=jnp.asarray(matrices["C_school"]),
        C_work=jnp.asarray(matrices["C_work"]), C_other=jnp.asarray(matrices["C_other"]),
        C_home_vac=jnp.asarray(vac["C_home"]), C_school_vac=jnp.asarray(vac["C_school"]),
        C_work_vac=jnp.asarray(vac["C_work"]), C_other_vac=jnp.asarray(vac["C_other"]),
        M_home=jnp.asarray(mobility["home"]), M_school=jnp.asarray(mobility["school"]),
        M_work=jnp.asarray(mobility["work"]), M_other=jnp.asarray(mobility["other"]),
        pop_15=jnp.asarray(pop_15), rho=jnp.asarray(rho_emp),
        sigma=disease.sigma, gamma=disease.gamma, VE=vax.VE,
        annual_coverage=jnp.asarray(np.asarray(vax.annual_coverage, dtype=np.float64)),  # RAW
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=S.AMP, seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day,
        seasonality_period=disease.seasonality_period,
    )   # no HOLIDAY — C(t) owns winter break
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop_15, seed_15, seed_e_factor=0.5,
        initial_immunity=R0_IMMUNITY_PROFILE, initial_vaccinated_fraction=0.0,
    ))
    ngm_default = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp,
        C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE, gamma=disease.gamma, seasonal_factor=1.0 + S.AMP,
    )
    H = build_hira_matrix()
    pop_15_flat = np.asarray(pop_15).sum(axis=1) if np.asarray(pop_15).ndim == 2 else np.asarray(pop_15)
    pop_6 = H @ pop_15_flat
    return dict(shared_base=shared_base, state0=state0,
               obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
               obs_np=obs, w_np=w, n_weeks=n_weeks, ngm_default=ngm_default,
               H=H, pop_6=pop_6)


def shared_with_kappa(setup, kappa_vec):
    kw = dict(setup["shared_base"]); kw["kappa"] = jnp.asarray(kappa_vec); return kw


def fit_beta(work_share, setup):
    """Pin-fit β_4 at baseline (κ-independent). 12-start L-BFGS-B."""
    pi_target = S.build_pi_target(work_share)
    logit_target = S.logit_centered_target(pi_target)
    phi_full_j = jnp.asarray(S.PHI_USHAPE)
    gamma_15_j = jnp.asarray(S.build_gamma_15(*S.GAMMA_CENTER))
    shared_fit = shared_with_kappa(setup, KAPPA_BASE)   # κ irrelevant at baseline
    fg = S.build_point_loss(logit_target, S.SIGMA_PER_CHANNEL, phi_full_j,
                            gamma_15_j, shared_fit, setup)
    bounds = [S.LOG_R0_BOUNDS] + [S.LOGIT_PI_BOUNDS] * 4 + [S.PHI_NB_BOUNDS]
    starts = S.make_starts(logit_target, S.N_STARTS, S.POINT_START_SEED)
    best = None; nlls = []
    for x0 in starts:
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        nll = float(res.fun); nlls.append(nll)
        if best is None or nll < best["nll"]:
            best = {"nll": nll, "x": np.asarray(res.x)}
    x = best["x"]
    R0 = float(np.exp(x[0])); pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    beta_4 = np.asarray(derive_beta_from_R0_simplex(
        setup["ngm_default"], jnp.asarray(R0), jnp.asarray(pi), phi_full_j))
    return dict(pi_target=pi_target.tolist(), R0=R0, pi=pi.tolist(),
                beta_4=[float(b) for b in beta_4], nll=best["nll"],
                nll_std=float(np.std(nlls)), n_starts=len(nlls))


def infections_per_hira6(inc, setup):
    return setup["H"] @ np.asarray(inc).sum(axis=0)


def policy_at_mu(beta_4, mu, setup):
    phi_full_j = jnp.asarray(S.PHI_USHAPE)
    gamma_15_j = jnp.asarray(S.build_gamma_15(*S.GAMMA_CENTER))
    kappa_vec = mu * KAPPA_BASE
    shared_pol = shared_with_kappa(setup, kappa_vec)
    b = np.array(beta_4)
    inc_b, _ = S.run_forward(b, phi_full_j, gamma_15_j, S.P_SCHOOL_BASE, S.P_WORK_BASE, shared_pol, setup)
    inc_s, _ = S.run_forward(b, phi_full_j, gamma_15_j, S.P_SCHOOL_BASE, S.P_WORK_SICK, shared_pol, setup)
    infb = infections_per_hira6(inc_b, setup); infs = infections_per_hira6(inc_s, setup)
    pop6 = setup["pop_6"]
    attack_b = infb / pop6; attack_s = infs / pop6
    d_attack = attack_s - attack_b                       # fraction; + = load received
    d_counts = d_attack * pop6
    pos = float(d_counts[d_counts > 0].sum()); neg = float(-d_counts[d_counts < 0].sum())
    transfer = (pos / neg) if neg > 1e-9 else float("inf")
    iw = [HIRA_AGE_GROUPS.index(a) for a in WORKING_AGE_HIRA]
    inn = [HIRA_AGE_GROUPS.index(a) for a in NONWORKING_HIRA]
    d_work = float(d_counts[iw].sum()); d_non = float(d_counts[inn].sum())
    totb = float(infb.sum()); tots = float(infs.sum())
    averted = 100.0 * (totb - tots) / max(totb, 1.0)
    # age flip: adults averted (d_work<0) but any school-age receives load (d_attack>0)
    school_up = (d_attack[HIRA_AGE_GROUPS.index("6-11")] > 0) or (d_attack[HIRA_AGE_GROUPS.index("12-17")] > 0)
    age_flip = bool(d_work < 0 and school_up)
    return dict(
        mu=mu, kappa_worker=float(0.30*mu), kappa_student=float(0.29*mu),
        averted_total_pct=averted,
        d_attack_pct_by_age={ag: float(100.0*d_attack[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)},
        attack_baseline_by_age={ag: float(attack_b[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)},
        transfer_ratio=transfer, d_infections_working=d_work, d_infections_nonworking=d_non,
        age_flip=age_flip, total_infections_baseline=totb, total_infections_sick=tots,
    )


def main():
    print("=" * 96)
    print("POLICY MAP Step A+B — work_share × μ (κ-scale)  sick-leave (p_work=0.4)  2019-2020")
    print(f"  κ_base=[학생0.29, 직장인0.30, 70+0]  μ∈{MU_GRID}  ws∈{WORK_SHARE_GRID}")
    print(f"  C(t) term↔vacation | v(t) norm | RAW cov | φ U-shape | γ CDC Reed | {S.N_STARTS} starts")
    print("=" * 96)
    t0 = time.perf_counter()
    setup = build_setup_AB()
    print(f"[setup] {time.perf_counter()-t0:.1f}s\n")

    results = []   # list over ws: {work_share, fit, per_mu:[...]}
    beta_home_by_ws = {}
    for ws in WORK_SHARE_GRID:
        tw = time.perf_counter()
        fit = fit_beta(ws, setup)
        beta_home_by_ws[ws] = fit["beta_4"][0]
        per_mu = [policy_at_mu(fit["beta_4"], mu, setup) for mu in MU_GRID]
        results.append(dict(work_share=ws, fit=fit, per_mu=per_mu))
        print(f"  ws={ws:.2f} fit(R0={fit['R0']:.3f} βh={fit['beta_4'][0]:.4f} βw={fit['beta_4'][1]:.4f} "
              f"nll={fit['nll']:.1f}) {time.perf_counter()-tw:.1f}s")
    print()

    # ---- console: averted grid ws × μ ----
    print("AVERTED % (sick-leave, total infections)   rows=work_share  cols=μ")
    print("  ws \\ μ  " + "".join(f"{mu:>9.1f}" for mu in MU_GRID))
    for r in results:
        row = "".join(f"{m['averted_total_pct']:>+9.2f}" for m in r["per_mu"])
        print(f"  {r['work_share']:>5.2f}  {row}")
    print()

    # ---- sign boundary per μ (first ws where averted turns >=0) ----
    print("SIGN BOUNDARY  (min work_share with averted ≥ 0, per μ)")
    for j, mu in enumerate(MU_GRID):
        cross = None
        for r in results:
            if r["per_mu"][j]["averted_total_pct"] >= 0:
                cross = r["work_share"]; break
        allpos = all(r["per_mu"][j]["averted_total_pct"] >= 0 for r in results)
        allneg = all(r["per_mu"][j]["averted_total_pct"] < 0 for r in results)
        tag = (f"ws≥{cross:.2f}" if cross is not None else "none(all<0)") if not allpos else "all≥0"
        print(f"  μ={mu:.1f} (κ_worker={0.30*mu:.2f}): {tag}")
    print()

    # ---- representative per-age Δattack: (ws=0.06,μ=1) (ws=0.29,μ=2) ----
    print("REPRESENTATIVE per-age Δattack (%pt)  [+ = load received]")
    def find(ws, mu):
        r = next(x for x in results if abs(x["work_share"]-ws) < 1e-9)
        m = next(x for x in r["per_mu"] if abs(x["mu"]-mu) < 1e-9)
        return r, m
    for ws, mu in [(0.06, 1.0), (0.17 if 0.17 in WORK_SHARE_GRID else 0.18, 1.0), (0.29 if 0.29 in WORK_SHARE_GRID else 0.30, 2.0)]:
        r, m = find(ws, mu)
        da = m["d_attack_pct_by_age"]
        print(f"  ws={ws:.2f} μ={mu:.1f}  averted={m['averted_total_pct']:+.2f}%  transfer={m['transfer_ratio']:.3f}  flip={m['age_flip']}")
        print("     " + "  ".join(f"{ag}:{da[ag]:+.3f}" for ag in HIRA_AGE_GROUPS))
    print()

    # ---- literature points averted (nearest ws, μ=1) ----
    print("LITERATURE POINTS averted (μ=1.0)")
    for name, ws_lit in LIT_POINTS.items():
        r = min(results, key=lambda x: abs(x["work_share"]-ws_lit))
        m = next(x for x in r["per_mu"] if abs(x["mu"]-1.0) < 1e-9)
        print(f"  {name:>5s} (ws≈{ws_lit:.2f}→{r['work_share']:.2f}): averted={m['averted_total_pct']:+.2f}%  flip={m['age_flip']}")
    print()

    # ---- vs old base ----
    new_range = [min(m["averted_total_pct"] for r in results for m in r["per_mu"]),
                 max(m["averted_total_pct"] for r in results for m in r["per_mu"])]
    print("VS OLD BASE (docs: β_home≈0.036, sick averted −2.8~+2.2%, B flip κ≈0.49)")
    print(f"  new β_home≈{np.mean(list(beta_home_by_ws.values())):.4f} (+{(np.mean(list(beta_home_by_ws.values()))/0.036-1)*100:.0f}%)")
    print(f"  new averted range: [{new_range[0]:+.2f}, {new_range[1]:+.2f}]%  vs old [−2.8, +2.2]%")
    print("=" * 96)

    out = dict(
        meta=dict(season=S.SEASON_LABEL, work_share_grid=WORK_SHARE_GRID, mu_grid=MU_GRID,
                  kappa_base=KAPPA_BASE.tolist(), sigma_pin=S.SIGMA_PER_CHANNEL.tolist(),
                  n_starts=S.N_STARTS, sick_p_work=S.P_WORK_SICK,
                  step_A="C(t) term-vacation + kappa 3-way", step_B="v(t) -ln(1-C)/Z",
                  coverage="raw", phi="U-shape fixed", gamma_15=list(S.GAMMA_CENTER),
                  lit_points=LIT_POINTS, old_ref=OLD_REF),
        results=results, beta_home_by_ws=beta_home_by_ws,
        new_averted_range_pct=new_range,
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}  ({len(WORK_SHARE_GRID)*len(MU_GRID)} combos)")


if __name__ == "__main__":
    main()
