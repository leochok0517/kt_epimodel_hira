"""Re-calibration with realistic presenteeism baseline p_work=0.6 (vs old 1.0).

Old baseline assumed sick workers attend 100% (p_work=1.0). Realistic: ~40%
stay home when sick (p_work=0.6). Re-estimate β_4 at this baseline (implies real
work transmission was larger → β_work should rise), then recompute policy under
the new baseline (interventions p_work=0.4, 0.2 relative to 0.6).

Setup: Step A+B — C(t) term↔vacation, κ 3-way [0.29,0.30,0], v(t) norm, RAW cov.
φ U-shape, γ CDC Reed. β_4 fit DIRECTLY (φ/γ/κ fixed), first_peak_only, 10 starts.
Representative season 2019-2020. Point estimate (NOT NUTS).

Output: outputs/eda/refit_baseline_p06.json + console.
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
jax.devices()
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
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import simulate_jax, daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn, derive_beta_from_R0_simplex
import sens_workshare_kappa_v2 as S

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "refit_baseline_p06.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
VAC_NPZ = REPO_ROOT.parent / "kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz"

SEASON = "2019-2020"
KAPPA_BASE = np.array([0.29]*4 + [0.30]*10 + [0.0])
PHI = np.array(S.PHI_USHAPE)
GAMMA15 = jnp.asarray(np.array([0.40]*4 + [0.18]*9 + [0.25]*2))
# π FIXED at multi-season joint (free β_4 floors the work channel — non-identifiable;
# fixing the channel mix lets β_work respond to the baseline change cleanly).
PI_FIXED = np.array([0.357, 0.255, 0.067, 0.321])
LOG_R0_B = (float(np.log(0.8)), float(np.log(3.5))); PHI_NB_B = (1e-3, 1e6)
N_STARTS = 10; SEED = 21
TERM_WIN = (70.0, 113.0); VAC_WIN = (113.0, 183.0); WHOLE = (-1.0e9, 1.0e9)
CHILD = ["0-5", "6-11", "12-17"]; ADULT = ["18-44", "45-64"]; SCHOOL = ["6-11", "12-17"]
BASE_P06 = 0.6


def build():
    inp = build_aggregated_inputs()
    pop15 = inp["pop_15"]; rho = inp["rho"]; M = inp["matrices"]; mob = inp["mobility"]
    disease = ModelParameters().disease; vax = ModelParameters().vaccination
    vac = load_contact_matrices(path=VAC_NPZ)
    shared = dict(
        C_home=jnp.asarray(M["C_home"]), C_school=jnp.asarray(M["C_school"]),
        C_work=jnp.asarray(M["C_work"]), C_other=jnp.asarray(M["C_other"]),
        C_home_vac=jnp.asarray(vac["C_home"]), C_school_vac=jnp.asarray(vac["C_school"]),
        C_work_vac=jnp.asarray(vac["C_work"]), C_other_vac=jnp.asarray(vac["C_other"]),
        M_home=jnp.asarray(mob["home"]), M_school=jnp.asarray(mob["school"]),
        M_work=jnp.asarray(mob["work"]), M_other=jnp.asarray(mob["other"]),
        pop_15=jnp.asarray(pop15), rho=jnp.asarray(rho), kappa=jnp.asarray(KAPPA_BASE),
        sigma=disease.sigma, gamma=disease.gamma, VE=vax.VE,
        annual_coverage=jnp.asarray(np.asarray(vax.annual_coverage, dtype=np.float64)),
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=S.AMP, seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day, seasonality_period=disease.seasonality_period,
    )
    seed = estimate_initial_infected_from_hira(
        SEASON, pop15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
        gamma_15_assumed=CalibrationParameters().gamma_15)
    state0 = jnp.asarray(_build_initial_state_with_age_seed(
        pop15, seed, seed_e_factor=0.5, initial_immunity=R0_IMMUNITY_PROFILE,
        initial_vaccinated_fraction=0.0))
    tgt = load_hira_target_by_age(SEASON, sido_codes=list(SUDOGWON_SIDO_CODES),
                                  first_peak_only=True, first_peak_end_week=26)
    nw = tgt["n_weeks"]
    obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        obs[:, i] = tgt["hira_counts"][ag]; w[:, i] = tgt["weights"][ag]
    ngm = make_ngm_eigvalue_fn(
        pop_15=pop15, rho=rho, C_home=M["C_home"], C_work=M["C_work"],
        C_school=M["C_school"], C_other=M["C_other"], R0_immunity=R0_IMMUNITY_PROFILE,
        gamma=disease.gamma, seasonal_factor=1.0 + S.AMP)
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for idx, wt in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, idx] = wt
    pflat = np.asarray(pop15).sum(axis=1) if np.asarray(pop15).ndim == 2 else np.asarray(pop15)
    return dict(shared=shared, state0=state0, obs_j=jnp.asarray(obs), w_j=jnp.asarray(w),
                obs=obs, w=w, nw=nw, ngm=ngm, H=H, pop6=H @ pflat)


def run_inc(C, beta_4, p_work, p_school=1.0, work_win=WHOLE, work_base=1.0):
    kw = dict(C["shared"])
    beta_4 = jnp.asarray(beta_4)
    kw["beta_h"] = beta_4[0]; kw["beta_w"] = beta_4[1]
    kw["beta_s"] = beta_4[2]; kw["beta_o"] = beta_4[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_work_baseline"] = work_base
    st = simulate_jax(C["state0"], **kw, discretize_time=False)
    return daily_new_infection_by_age_jax(st)


def _beta_from_R0(C, R0):
    return derive_beta_from_R0_simplex(C["ngm"], jnp.asarray(R0), jnp.asarray(PI_FIXED), jnp.asarray(PHI))


def fit_beta(C, p_work_base):
    """π FIXED at joint; fit log_R0 + phi_nb at whole-season baseline p_work=base.
    β_4 = derive(R0, π_fixed, φ); β_work rises if the baseline needs higher R0."""
    def loss(x):
        beta = _beta_from_R0(C, jnp.exp(x[0]))
        inc = run_inc(C, beta, p_work_base)
        pred = simulation_to_hira_by_age_jax(inc, GAMMA15, n_weeks=C["nw"])
        return nb_nll_jax(C["obs_j"], pred, C["w_j"], concentration=x[1], min_rate=0.01)
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(SEED)
    bounds = [LOG_R0_B, PHI_NB_B]
    best = None; r0s = []
    for k in range(N_STARTS):
        x0 = np.array([np.log(rng.uniform(1.5, 2.8)), 10.0])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        r0s.append(float(np.exp(r.x[0])))
        if best is None or r.fun < best.fun: best = r
    R0 = float(np.exp(best.x[0]))
    b = np.asarray(_beta_from_R0(C, R0))
    return dict(beta_4=[float(x) for x in b], phi_nb=float(best.x[1]), nll=float(best.fun),
                R0=R0, R0_std=float(np.std(r0s)))


def attack6(C, inc):
    return C["H"] @ np.asarray(inc).sum(axis=0)


def policy_eval(C, beta_4, p_work, base_inf6, p_school=1.0, work_win=WHOLE, work_base=1.0):
    inc = run_inc(C, beta_4, p_work, p_school, work_win, work_base)
    inf6 = attack6(C, inc); pop6 = C["pop6"]
    averted = 100.0 * (base_inf6.sum() - inf6.sum()) / max(base_inf6.sum(), 1.0)
    d = (inf6 - base_inf6) / pop6
    da = {ag: float(100.0*d[i]) for i, ag in enumerate(HIRA_AGE_GROUPS)}
    dc = d * pop6; pos = float(dc[dc > 0].sum()); neg = float(-dc[dc < 0].sum())
    return dict(averted=float(averted), d_attack=da, transfer=(pos/neg if neg > 1e-9 else float("inf")))


def main():
    print("=" * 90)
    print("REFIT with presenteeism baseline p_work=0.6  (2019-2020, point estimate)")
    print(f"  Step A+B | κ3-way | φ U-shape | γ CDC | first_peak | {N_STARTS} starts")
    print("=" * 90)
    t0 = time.perf_counter(); C = build()
    print(f"[setup] {time.perf_counter()-t0:.1f}s\n")

    # (1) β_4 fit at both baselines
    print("[fit] baseline p_work=1.0 (old) ...")
    f10 = fit_beta(C, 1.0)
    print("[fit] baseline p_work=0.6 (new presenteeism) ...")
    f06 = fit_beta(C, 0.6)
    b10, b06 = f10["beta_4"], f06["beta_4"]
    print("\n(1) β_4  [home, work, school, other]   (obs/model, R0, NLL)")
    om10 = C["obs"][C["w"].sum(1) > 0].sum() / max(np.asarray(simulation_to_hira_by_age_jax(
        run_inc(C, b10, 1.0), GAMMA15, n_weeks=C["nw"]))[C["w"].sum(1) > 0].sum(), 1)
    om06 = C["obs"][C["w"].sum(1) > 0].sum() / max(np.asarray(simulation_to_hira_by_age_jax(
        run_inc(C, b06, 0.6), GAMMA15, n_weeks=C["nw"]))[C["w"].sum(1) > 0].sum(), 1)
    print(f"  p_work=1.0 (old): [{b10[0]:.4f} {b10[1]:.4f} {b10[2]:.4f} {b10[3]:.4f}]  obs/model={om10:.2f} R0={f10['R0']:.3f} nll={f10['nll']:.1f}")
    print(f"  p_work=0.6 (new): [{b06[0]:.4f} {b06[1]:.4f} {b06[2]:.4f} {b06[3]:.4f}]  obs/model={om06:.2f} R0={f06['R0']:.3f} nll={f06['nll']:.1f}")
    print(f"  ★ β_work: {b10[1]:.4f} → {b06[1]:.4f}  ({(b06[1]/b10[1]-1)*100:+.1f}%)   [R0 {f10['R0']:.3f}→{f06['R0']:.3f}; 순수보상 상한 1/0.6={1/0.6:.2f}×]")
    print(f"    (π FIXED {list(PI_FIXED)}; R0 multistart std new={f06['R0_std']:.4f})")

    # (2)+(3) whole-season policy under 0.6 baseline
    base06 = attack6(C, run_inc(C, b06, 0.6))       # baseline = p_work 0.6 whole season
    pol = {}
    for p in (0.4, 0.2):
        pol[p] = policy_eval(C, b06, p, base06)
    print("\n(2) 정책 averted% (baseline p_work=0.6 대비, 전 기간)")
    for p in (0.4, 0.2):
        print(f"  p_work={p} (추가 {int((0.6-p)*100)}%p 결근): averted={pol[p]['averted']:+.2f}%  transfer={pol[p]['transfer']:.3f}")
    print("\n(3) 연령별 Δattack (%pt, +=부하유입) — 재분배 방향")
    for p in (0.4, 0.2):
        da = pol[p]["d_attack"]
        ad = all(da[a] < 0 for a in ADULT); sc = any(da[s] > 0 for s in SCHOOL)
        print(f"  p_work={p}: " + " ".join(f"{ag}:{da[ag]:+.3f}" for ag in HIRA_AGE_GROUPS))
        print(f"           성인↓={ad}  학령기↑={sc}  재분배방향={ad and sc}")

    # (4) holiday reversal under 0.6 baseline (windowed intervention 0.6→0.4)
    print("\n(4) 방학 부호반전 (baseline 0.6, 창내 p_work=0.4 개입, 창밖 0.6)")
    holiday = {}
    for wname, wwin in (("term", TERM_WIN), ("vacation", VAC_WIN)):
        r = policy_eval(C, b06, 0.4, base06, work_win=wwin, work_base=0.6)
        da = r["d_attack"]; cs = sum(da[c] for c in CHILD)
        holiday[wname] = dict(averted=r["averted"], child_d_attack={c: da[c] for c in CHILD},
                              all_d_attack=da, child_sum=cs)
        print(f"  {wname:>8} 창: averted={r['averted']:+.2f}%  아동 " +
              " ".join(f"{c}:{da[c]:+.3f}" for c in CHILD) + f"  child_sum={cs:+.3f}")
    rev = (holiday["term"]["child_sum"] < 0) and (holiday["vacation"]["child_sum"] > 0)
    print(f"  ★ 부호반전(학기− → 방학+): {'Y' if rev else 'N'}")
    print("=" * 90)

    out = dict(
        meta=dict(season=SEASON, baseline_new=0.6, baseline_old=1.0, kappa_base=KAPPA_BASE.tolist(),
                  n_starts=N_STARTS, first_peak_only=True, note="direct β_4 fit, point estimate"),
        fit_p10=dict(**f10, obs_model=float(om10)), fit_p06=dict(**f06, obs_model=float(om06)),
        beta_work_change_pct=float((b06[1]/b10[1]-1)*100),
        policy=pol, holiday_reversal=dict(reversal=bool(rev), **holiday),
    )
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
