"""Experiment 2 — shared-π joint fit across 6 normal seasons.

Tests whether ONE channel partition π=(home,work,school,other) explains all
seasons (size axis R0_s per season), and whether the multi-season constraint
identifies π_work when the work pin is loosened (σ 0.01→0.10) — the channel
that was silent/non-identified under single-season fits.

Structure (make_shared_pi_joint_loss_nb):
  shared: π (channel mix), κ 3-way, φ U-shape, γ CDC Reed
  per-season: R0_s, initial state (seed·immunity)
  β_{c,s} = (R0_s / ρ(π,φ)) · π_c

For a clean "does forcing shared-π hurt each season" test, each season is also
fit INDEPENDENTLY with the same machinery/pin (π free per season). Compare
per-season DATA NLL (pure nb_nll, no pin): joint vs independent.

Step A+B setup. Output: outputs/eda/multiseason_joint_sharedpi.json + console.
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
from kt_epimodel_hira.calibration.hira_target import load_hira_target_by_age
from kt_epimodel_hira.model.parameters import CalibrationParameters, ModelParameters
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, make_shared_pi_joint_loss_nb, simulation_to_hira_by_age_jax,
    nb_nll_jax,
)
from kt_epimodel_hira.jax_model.solver_jax import simulate_jax, daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn
import sens_workshare_kappa_v2 as S
import policy_map_stepAB as P

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "eda" / "multiseason_joint_sharedpi.json"
EXP1_JSON = REPO_ROOT / "outputs" / "eda" / "multiseason_independent.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
VAC_NPZ = REPO_ROOT.parent / "kt_data/data/external/contact_matrices/empirical_matrices_15_vacation.npz"

SEASONS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020", "2022-2023"]
N_STARTS = 12
SEED = 23
# loosened work pin: [h, w, s, o]  (work 0.01 → 0.10 to test multi-season identifiability)
SIGMA_PIN = np.array([0.15, 0.10, 0.05, 0.15])
PI_REF = np.array(S.build_pi_target(0.29))          # A reference center (π_work≈0.29)
LOGIT_REF = S.logit_centered_target(PI_REF)
LOG_R0_B = S.LOG_R0_BOUNDS; LOGIT_B = S.LOGIT_PI_BOUNDS; PHI_NB_B = S.PHI_NB_BOUNDS
PHI_USHAPE = np.array(S.PHI_USHAPE)
KAPPA_3 = np.array([0.29]*4 + [0.30]*10 + [0.0])


def build_common():
    inputs = build_aggregated_inputs()
    pop_15 = inputs["pop_15"]; rho_emp = inputs["rho"]
    matrices = inputs["matrices"]; mobility = inputs["mobility"]
    disease = ModelParameters().disease; vax = ModelParameters().vaccination
    vac = load_contact_matrices(path=VAC_NPZ)
    shared = dict(
        C_home=jnp.asarray(matrices["C_home"]), C_school=jnp.asarray(matrices["C_school"]),
        C_work=jnp.asarray(matrices["C_work"]), C_other=jnp.asarray(matrices["C_other"]),
        C_home_vac=jnp.asarray(vac["C_home"]), C_school_vac=jnp.asarray(vac["C_school"]),
        C_work_vac=jnp.asarray(vac["C_work"]), C_other_vac=jnp.asarray(vac["C_other"]),
        M_home=jnp.asarray(mobility["home"]), M_school=jnp.asarray(mobility["school"]),
        M_work=jnp.asarray(mobility["work"]), M_other=jnp.asarray(mobility["other"]),
        pop_15=jnp.asarray(pop_15), rho=jnp.asarray(rho_emp), kappa=jnp.asarray(KAPPA_3),
        p_school=1.0, p_work=1.0,   # baseline (no policy) — fit is baseline
        sigma=disease.sigma, gamma=disease.gamma, VE=vax.VE,
        annual_coverage=jnp.asarray(np.asarray(vax.annual_coverage, dtype=np.float64)),  # RAW
        vax_peak_iso_week=vax.peak_iso_week, vax_spread_weeks=vax.spread_weeks,
        seasonality_amp=S.AMP, seasonality_base=disease.seasonality_base,
        seasonality_peak_day=disease.seasonality_peak_day, seasonality_period=disease.seasonality_period,
    )
    ngm = make_ngm_eigvalue_fn(
        pop_15=pop_15, rho=rho_emp, C_home=matrices["C_home"], C_work=matrices["C_work"],
        C_school=matrices["C_school"], C_other=matrices["C_other"],
        R0_immunity=R0_IMMUNITY_PROFILE, gamma=disease.gamma, seasonal_factor=1.0 + S.AMP)
    states, obs_list, w_list, nweeks = [], [], [], None
    for s in SEASONS:
        tgt = load_hira_target_by_age(s, sido_codes=list(SUDOGWON_SIDO_CODES),
                                      first_peak_only=True, first_peak_end_week=26)
        nw = tgt["n_weeks"]; nweeks = nw
        obs = np.zeros((nw, 6)); w = np.zeros((nw, 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            obs[:, i] = tgt["hira_counts"][ag]; w[:, i] = tgt["weights"][ag]
        seed = estimate_initial_infected_from_hira(
            s, pop_15.flatten(), sido_codes=list(SUDOGWON_SIDO_CODES),
            gamma_15_assumed=CalibrationParameters().gamma_15)
        st0 = jnp.asarray(_build_initial_state_with_age_seed(
            pop_15, seed, seed_e_factor=0.5, initial_immunity=R0_IMMUNITY_PROFILE,
            initial_vaccinated_fraction=0.0))
        states.append(st0); obs_list.append(jnp.asarray(obs)); w_list.append(jnp.asarray(w))
    return dict(shared=shared, ngm=ngm, states=states, obs=obs_list, w=w_list,
                nweeks=nweeks, gamma_15=jnp.asarray(S.build_gamma_15(*S.GAMMA_CENTER)))


def pin_pen(logit_pi):
    centered = logit_pi - np.mean(logit_pi)
    return 0.5 * float(np.sum((centered - LOGIT_REF)**2 / SIGMA_PIN**2))


def per_season_data_nll(log_R0_vec, logit_pi, phi_nb, C):
    """Pure nb_nll per season (no pin) at given params."""
    phi_j = jnp.asarray(PHI_USHAPE)
    pi = jax.nn.softmax(jnp.asarray(logit_pi))
    rho_pi = C["ngm"](pi[0], pi[1], pi[2], pi[3], phi_j)
    out = []
    for i in range(len(C["states"])):
        R0 = float(np.exp(log_R0_vec[i])); beta = (R0/float(rho_pi))*np.asarray(pi)
        kw = dict(C["shared"]); kw["beta_h"], kw["beta_w"], kw["beta_s"], kw["beta_o"] = beta
        kw["phi_susc"] = phi_j
        st = simulate_jax(C["states"][i], **kw, discretize_time=False)
        inc = daily_new_infection_by_age_jax(st)
        pred = simulation_to_hira_by_age_jax(inc, C["gamma_15"], n_weeks=C["nweeks"])
        out.append(float(nb_nll_jax(C["obs"][i], pred, C["w"][i], concentration=phi_nb, min_rate=0.01)))
    return out


# ---------- independent per-season fit (π free per season, same loose pin) ----------
def fit_independent(i, C):
    loss_fn = make_shared_pi_joint_loss_nb(
        initial_states=[C["states"][i]], obs_hira_list=[C["obs"][i]],
        weights_hira_list=[C["w"][i]], shared_static=C["shared"], ngm_eigval_fn=C["ngm"],
        phi_full=jnp.asarray(PHI_USHAPE), gamma_15=C["gamma_15"], n_weeks=C["nweeks"])
    def obj(x):
        lr0 = jnp.asarray(x[0:1]); lpi = jnp.asarray(x[1:5]); pnb = x[5]
        return loss_fn(lr0, lpi, pnb)
    ov = jax.jit(obj); og = jax.jit(jax.grad(obj))
    def fg(xn):
        x = jnp.asarray(xn); v = float(ov(x)) + pin_pen(xn[1:5]); g = np.array(og(x))
        gp = (xn[1:5]-np.mean(xn[1:5])-LOGIT_REF)/SIGMA_PIN**2; g[1:5] += gp
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    bounds = [LOG_R0_B] + [LOGIT_B]*4 + [PHI_NB_B]
    rng = np.random.default_rng(SEED + i)
    best = None
    for k in range(N_STARTS):
        x0 = np.concatenate([[np.log(2.0)+rng.normal(0,0.2)], LOGIT_REF+rng.normal(0,0.5,4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun: best = r
    x = best.x
    pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    data_nll = per_season_data_nll([x[0]], x[1:5], x[5],
                 dict(shared=C["shared"], ngm=C["ngm"], states=[C["states"][i]],
                      obs=[C["obs"][i]], w=[C["w"][i]], gamma_15=C["gamma_15"], nweeks=C["nweeks"]))[0]
    return dict(R0=float(np.exp(x[0])), pi=pi.tolist(), pi_work=float(pi[1]),
                obj=float(best.fun), data_nll=data_nll)


# ---------- joint shared-π fit ----------
def fit_joint(C):
    loss_fn = make_shared_pi_joint_loss_nb(
        initial_states=C["states"], obs_hira_list=C["obs"], weights_hira_list=C["w"],
        shared_static=C["shared"], ngm_eigval_fn=C["ngm"],
        phi_full=jnp.asarray(PHI_USHAPE), gamma_15=C["gamma_15"], n_weeks=C["nweeks"])
    n = len(SEASONS)
    def obj(x):
        lr0 = jnp.asarray(x[0:n]); lpi = jnp.asarray(x[n:n+4]); pnb = x[n+4]
        return loss_fn(lr0, lpi, pnb)
    ov = jax.jit(obj); og = jax.jit(jax.grad(obj))
    def fg(xn):
        x = jnp.asarray(xn); v = float(ov(x)) + pin_pen(xn[n:n+4]); g = np.array(og(x))
        gp = (xn[n:n+4]-np.mean(xn[n:n+4])-LOGIT_REF)/SIGMA_PIN**2; g[n:n+4] += gp
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    bounds = [LOG_R0_B]*n + [LOGIT_B]*4 + [PHI_NB_B]
    rng = np.random.default_rng(SEED)
    best = None; pi_work_starts = []
    t0 = time.perf_counter()
    for k in range(N_STARTS):
        x0 = np.concatenate([np.log(2.0)+rng.normal(0,0.2,n), LOGIT_REF+rng.normal(0,0.5,4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=600, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        piw = float(jax.nn.softmax(jnp.asarray(r.x[n:n+4]))[1])
        pi_work_starts.append(piw)
        if best is None or r.fun < best.fun: best = r
    wall = time.perf_counter()-t0
    x = best.x
    pi = np.asarray(jax.nn.softmax(jnp.asarray(x[n:n+4])))
    logit_rail = any(abs(v-LOGIT_B[0]) < 0.05 or abs(v-LOGIT_B[1]) < 0.05 for v in x[n:n+4])
    data_nll_s = per_season_data_nll(x[0:n], x[n:n+4], x[n+4], C)
    return dict(pi=pi.tolist(), pi_work=float(pi[1]), R0=[float(np.exp(x[i])) for i in range(n)],
                phi_nb=float(x[n+4]), obj=float(best.fun), data_nll_by_season=data_nll_s,
                pi_work_starts=pi_work_starts, pi_work_std=float(np.std(pi_work_starts)),
                logit_railing=bool(logit_rail), wall_sec=wall)


def fit_quality(R0, pi, i, C):
    phi_j = jnp.asarray(PHI_USHAPE)
    rho_pi = C["ngm"](*[jnp.asarray(pi[k]) for k in range(4)], phi_j)
    beta = (R0/float(rho_pi))*np.asarray(pi)
    setup = dict(obs_np=np.asarray(C["obs"][i]), w_np=np.asarray(C["w"][i]),
                 state0=C["states"][i], n_weeks=C["nweeks"])
    _, pred = S.run_forward(beta, phi_j, C["gamma_15"], S.P_SCHOOL_BASE, S.P_WORK_BASE,
                            C["shared"], setup)
    r = S.per_age_ratios(np.asarray(pred), setup)
    return {ag: round(r[ag]["ratio"], 3) for ag in HIRA_AGE_GROUPS}


def main():
    print("=" * 100)
    print("EXPERIMENT 2 — shared-π joint fit (6 seasons)   work pin σ=0.10 (loosened from 0.01)")
    print(f"  seasons={SEASONS}")
    print(f"  shared: π, κ3-way, φ U-shape, γ CDC Reed | per-season: R0_s | pin center=A(π_w≈0.29)")
    print("=" * 100)
    t0 = time.perf_counter(); C = build_common()
    print(f"[common setup] {time.perf_counter()-t0:.1f}s\n")

    print("[1/2] independent per-season fits (π free, same loose pin)...")
    indep = {}
    for i, s in enumerate(SEASONS):
        ti = time.perf_counter(); indep[s] = fit_independent(i, C)
        print(f"   {s}: R0={indep[s]['R0']:.3f} π_work={indep[s]['pi_work']:.3f} "
              f"data_nll={indep[s]['data_nll']:.1f} ({time.perf_counter()-ti:.1f}s)")

    print("\n[2/2] joint shared-π fit (12 starts)...")
    J = fit_joint(C)
    print(f"   done {J['wall_sec']:.1f}s\n")

    piN = ["home", "work", "school", "other"]
    print("SHARED π (joint):  " + "  ".join(f"{n}={J['pi'][k]:.3f}" for k, n in enumerate(piN)))
    print(f"  κ 3-way=[0.29,0.30,0]  φ=U-shape  γ=[0.40,0.18,0.25]  phi_nb={J['phi_nb']:.2f}")
    print()
    print("PER-SEASON R0 + NLL (joint vs independent DATA NLL)")
    print(f"  {'season':>11} | {'R0_joint':>8} {'R0_indep':>8} | {'nll_joint':>9} {'nll_indep':>9} {'Δ%':>7} | π_work_indep")
    dpcts = []
    for i, s in enumerate(SEASONS):
        nj = J["data_nll_by_season"][i]; ni = indep[s]["data_nll"]
        dp = 100.0*(nj-ni)/max(ni, 1.0); dpcts.append(dp)
        print(f"  {s:>11} | {J['R0'][i]:>8.3f} {indep[s]['R0']:>8.3f} | {nj:>9.1f} {ni:>9.1f} {dp:>+6.1f}% | {indep[s]['pi_work']:.3f}")
    print(f"  {'Σ / mean':>11} | {'':>8} {'':>8} | {sum(J['data_nll_by_season']):>9.1f} "
          f"{sum(indep[s]['data_nll'] for s in SEASONS):>9.1f} {np.mean(dpcts):>+6.1f}%")
    print()
    print("FIT QUALITY (joint, per-age obs/model peak ratio)")
    fq = {}
    for i, s in enumerate(SEASONS):
        q = fit_quality(J["R0"][i], J["pi"], i, C); fq[s] = q
        print(f"  {s}: " + " ".join(f"{k}={v}" for k, v in q.items()))
    print()
    print("WORK IDENTIFIABILITY (loose pin σ_work=0.10)")
    print(f"  joint π_work={J['pi_work']:.3f}  across 12 starts: std={J['pi_work_std']:.4f} "
          f"range=[{min(J['pi_work_starts']):.3f},{max(J['pi_work_starts']):.3f}]  railing={J['logit_railing']}")
    identified = J['pi_work_std'] < 0.02 and not J['logit_railing']
    print(f"  → π_work {'IDENTIFIED (multi-season constraint pins it)' if identified else 'still weak (scatter/rail)'}")
    print()

    worse = max(dpcts); mean_worse = np.mean(dpcts)
    success = mean_worse < 20.0 and worse < 50.0
    print("=" * 100)
    print(f"VERDICT: shared-π {'SUCCESS' if success else 'STRAINED'} — "
          f"per-season data NLL mean {mean_worse:+.1f}% (max {worse:+.1f}%) vs independent")
    if worse >= 50.0:
        ws_bad = SEASONS[int(np.argmax(dpcts))]
        print(f"  season needing own π: {ws_bad} (+{worse:.1f}%)")
    print("=" * 100)

    out = dict(
        meta=dict(seasons=SEASONS, sigma_pin=SIGMA_PIN.tolist(), pi_ref=PI_REF.tolist(),
                  n_starts=N_STARTS, step="A+B", shared="pi/kappa3/phi-Ushape/gammaCDC",
                  per_season="R0 + initial state", work_pin_sigma=0.10),
        shared_pi=dict(zip(piN, J["pi"])), phi_nb=J["phi_nb"],
        R0_by_season=dict(zip(SEASONS, J["R0"])),
        data_nll_joint=dict(zip(SEASONS, J["data_nll_by_season"])),
        data_nll_independent={s: indep[s]["data_nll"] for s in SEASONS},
        nll_delta_pct=dict(zip(SEASONS, dpcts)),
        independent_pi={s: indep[s]["pi"] for s in SEASONS},
        fit_quality=fq,
        work_identifiability=dict(pi_work=J["pi_work"], pi_work_std=J["pi_work_std"],
                                  pi_work_starts=J["pi_work_starts"], railing=J["logit_railing"],
                                  identified=bool(identified)),
        verdict=("SUCCESS" if success else "STRAINED"),
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] {OUT_JSON}")


if __name__ == "__main__":
    main()
