"""Parametric U-shape φ estimation → improve age-resolved fit at baseline p=0.6.

Problem: 12-17 over-estimated (obs/model~0.45), 0-5 under-estimated (>1), all
seasons, structural. φ and γ_report are observed only as a product (eLife 2020),
so we constrain φ to a parametric U-shape (shape fixed, 3 free heights) and let
the data pick the heights — partially breaking the φ-γ degeneracy.

φ params (piecewise-linear U): φ_young (idx0, 0-4), φ_mid (idx2, 10-14; the kink
controlling 12-17), φ_old (idx14, 70+). Flat φ=1 on idx5-8 (25-44). Linear links.

γ_report FIXED at CDC Reed [0.40,0.18,0.25] (test φ alone first). Common π fixed
(joint p06). Baseline p_work=0.6. Per-season R0. 2019-2020 warm-up + 6-season
joint (φ shared, R0 per season). Point estimate.

Output: outputs/eda/parametric_phi_fit.json + console.
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

from kt_epimodel_hira.calibration.hira_target import HIRA_GROUP_TO_NIMS_WEIGHTED
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax
from kt_epimodel_hira.jax_model.solver_jax import simulate_jax, daily_new_infection_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
import sens_workshare_kappa_v2 as S
import multiseason_joint_sharedpi as MJ

REPO_ROOT = Path(__file__).resolve().parent.parent
ED = REPO_ROOT / "outputs" / "eda"
OUT = ED / "parametric_phi_fit.json"
OLD_JSON = ED / "per_season_fit_p06.json"     # fixed U-shape obs/model, baseline 0.6

SEASONS = MJ.SEASONS
BASE_PWORK = 0.6
PI_FIXED = np.array([0.322, 0.282, 0.069, 0.327])   # joint p06 shared π
GAMMA15 = jnp.asarray(np.array([0.40]*4 + [0.18]*9 + [0.25]*2))
PHI_USHAPE = np.array(S.PHI_USHAPE)
# literature ranges (warn if outside)
RNG_YOUNG = (1.5, 2.5); RNG_MID = (1.0, 1.8); RNG_OLD = (1.0, 2.0)
LOG_R0_B = (float(np.log(0.8)), float(np.log(3.5))); PHI_NB_B = (1e-3, 1e6)
N_STARTS = 8; SEED = 31
TERM_WIN = (70.0, 113.0); WHOLE = (-1.0e9, 1.0e9)
ADULT = ["18-44", "45-64"]; SCHOOL = ["6-11", "12-17"]


def phi_from_params(y, m, o):
    """15-vec piecewise-linear U: idx0=y, idx2=m, idx5-8=1, idx14=o, linear links."""
    phi = jnp.ones(15)
    phi = phi.at[0].set(y)
    phi = phi.at[1].set(0.5*(y+m))
    phi = phi.at[2].set(m)
    phi = phi.at[3].set(m + (1.0-m)*(1.0/3.0))   # idx3 (12-17 core)
    phi = phi.at[4].set(m + (1.0-m)*(2.0/3.0))
    # idx5..8 = 1.0 (already)
    for i in range(9, 15):
        phi = phi.at[i].set(1.0 + (o-1.0)*(i-8)/6.0)
    return phi


def build():
    C = MJ.build_common(); C["shared"]["p_work"] = BASE_PWORK
    C["nw"] = C["nweeks"]
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for idx, wt in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, idx] = wt
    pflat = np.asarray(C["shared"]["pop_15"]); pflat = pflat.sum(1) if pflat.ndim == 2 else pflat
    C["H"] = H; C["pop6"] = H @ pflat
    return C


def run_inc(C, i, R0, phi_vec, p_work=BASE_PWORK, work_win=WHOLE, work_base=1.0):
    beta = derive_beta_from_R0_simplex(C["ngm"], jnp.asarray(R0), jnp.asarray(PI_FIXED), phi_vec)
    kw = dict(C["shared"])
    kw["beta_h"] = beta[0]; kw["beta_w"] = beta[1]; kw["beta_s"] = beta[2]; kw["beta_o"] = beta[3]
    kw["phi_susc"] = phi_vec; kw["p_school"] = 1.0; kw["p_work"] = p_work
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_work_baseline"] = work_base
    st = simulate_jax(C["states"][i], **kw, discretize_time=False)
    return daily_new_infection_by_age_jax(st)


def obs_model(C, i, R0, phi_vec):
    pred = np.asarray(simulation_to_hira_by_age_jax(run_inc(C, i, R0, phi_vec), GAMMA15, n_weeks=C["nw"]))
    obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i]); mask = w.sum(1) > 0
    om = {ag: float(obs[mask, a].sum()/max(pred[mask, a].sum(), 1.0)) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return om, float(obs[mask].sum()/max(pred[mask].sum(), 1.0))


def fit_single(C, i):
    """φ(3)+R0+phi_nb for one season."""
    def loss(x):
        phi = phi_from_params(x[1], x[2], x[3])
        inc = run_inc(C, i, jnp.exp(x[0]), phi)
        pred = simulation_to_hira_by_age_jax(inc, GAMMA15, n_weeks=C["nw"])
        return nb_nll_jax(C["obs_j"][i] if isinstance(C["obs_j"], list) else C["obs_j"],
                          pred, C["w_j"][i] if isinstance(C["w_j"], list) else C["w_j"],
                          concentration=x[4], min_rate=0.01)
    # per-season obs_j/w_j
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i])
    def loss2(x):
        phi = phi_from_params(x[1], x[2], x[3])
        inc = run_inc(C, i, jnp.exp(x[0]), phi)
        pred = simulation_to_hira_by_age_jax(inc, GAMMA15, n_weeks=C["nw"])
        return nb_nll_jax(obsj, pred, wj, concentration=x[4], min_rate=0.01)
    lj = jax.jit(loss2); gj = jax.jit(jax.grad(loss2))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(SEED+i)
    bounds = [LOG_R0_B, RNG_YOUNG, RNG_MID, RNG_OLD, PHI_NB_B]
    best = None
    for k in range(N_STARTS):
        x0 = [np.log(rng.uniform(1.6, 2.6)), rng.uniform(*RNG_YOUNG), rng.uniform(*RNG_MID), rng.uniform(*RNG_OLD), 10.0]
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun: best = r
    x = best.x
    return dict(R0=float(np.exp(x[0])), phi_young=float(x[1]), phi_mid=float(x[2]),
                phi_old=float(x[3]), phi_nb=float(x[4]), nll=float(best.fun))


def fit_joint(C):
    """φ(3 shared) + R0(6) + phi_nb, sum NB-NLL over seasons."""
    n = len(SEASONS)
    obsj = [jnp.asarray(C["obs"][i]) for i in range(n)]; wj = [jnp.asarray(C["w"][i]) for i in range(n)]
    def loss(x):
        phi = phi_from_params(x[n], x[n+1], x[n+2])
        tot = 0.0
        for i in range(n):
            inc = run_inc(C, i, jnp.exp(x[i]), phi)
            pred = simulation_to_hira_by_age_jax(inc, GAMMA15, n_weeks=C["nw"])
            tot = tot + nb_nll_jax(obsj[i], pred, wj[i], concentration=x[n+3], min_rate=0.01)
        return tot
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(SEED)
    bounds = [LOG_R0_B]*n + [RNG_YOUNG, RNG_MID, RNG_OLD, PHI_NB_B]
    best = None
    for k in range(N_STARTS):
        x0 = list(np.log(rng.uniform(1.8, 2.4, n))) + [rng.uniform(*RNG_YOUNG), rng.uniform(*RNG_MID), rng.uniform(*RNG_OLD), 10.0]
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=500, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun: best = r
    x = best.x
    return dict(R0=[float(np.exp(x[i])) for i in range(n)], phi_young=float(x[n]),
                phi_mid=float(x[n+1]), phi_old=float(x[n+2]), phi_nb=float(x[n+3]), nll=float(best.fun))


def redistribution(C, i, R0, phi_vec):
    base = C["H"] @ np.asarray(run_inc(C, i, R0, phi_vec, p_work=BASE_PWORK)).sum(0)
    inc = run_inc(C, i, R0, phi_vec, p_work=0.4)   # whole-season extra sick-leave from 0.6
    d = (C["H"] @ np.asarray(inc).sum(0) - base) / C["pop6"]
    da = {ag: float(100.0*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    ad = all(da[a] < 0 for a in ADULT); sc = any(da[s] > 0 for s in SCHOOL)
    return da, bool(ad), bool(sc)


def warn_range(y, m, o):
    w = []
    if not (RNG_YOUNG[0] <= y <= RNG_YOUNG[1]): w.append(f"φ_young={y:.2f}∉{RNG_YOUNG}")
    if not (RNG_MID[0] <= m <= RNG_MID[1]): w.append(f"φ_mid={m:.2f}∉{RNG_MID}")
    if not (RNG_OLD[0] <= o <= RNG_OLD[1]): w.append(f"φ_old={o:.2f}∉{RNG_OLD}")
    return w


def main():
    print("=" * 92)
    print("PARAMETRIC φ U-shape FIT — baseline p_work=0.6, γ CDC fixed, π joint fixed")
    print("=" * 92)
    t0 = time.perf_counter(); C = build(); print(f"[setup] {time.perf_counter()-t0:.1f}s\n")
    old = json.load(open(OLD_JSON)) if OLD_JSON.exists() else None

    # (A) 2019-2020 warm-up
    i19 = SEASONS.index("2019-2020")
    print("── (A) 2019-2020 single-season φ fit ──")
    f19 = fit_single(C, i19)
    phi19 = phi_from_params(f19["phi_young"], f19["phi_mid"], f19["phi_old"])
    om19, omt19 = obs_model(C, i19, f19["R0"], phi19)
    om_old19, omt_old19 = obs_model(C, i19, f19["R0"], jnp.asarray(PHI_USHAPE))
    print(f"  φ_young={f19['phi_young']:.2f} φ_mid={f19['phi_mid']:.2f} φ_old={f19['phi_old']:.2f} R0={f19['R0']:.2f} nll={f19['nll']:.1f}")
    w19 = warn_range(f19["phi_young"], f19["phi_mid"], f19["phi_old"])
    print(f"  문헌범위: {'OK' if not w19 else '⚠ '+', '.join(w19)}")
    print(f"  obs/model 12-17: fixed={om_old19['12-17']:.2f} → param={om19['12-17']:.2f}   0-5: {om_old19['0-5']:.2f} → {om19['0-5']:.2f}")

    # (B) 6-season joint φ (shared) + R0
    print("\n── (B) 6-season joint φ (shared) + per-season R0 ──")
    tj = time.perf_counter(); J = fit_joint(C)
    phiJ = phi_from_params(J["phi_young"], J["phi_mid"], J["phi_old"])
    phiJ_np = np.asarray(phiJ)
    print(f"  ({time.perf_counter()-tj:.1f}s)")
    wJ = warn_range(J["phi_young"], J["phi_mid"], J["phi_old"])
    print(f"\n(1) φ params: young={J['phi_young']:.2f} mid={J['phi_mid']:.2f} old={J['phi_old']:.2f}   문헌범위: {'OK' if not wJ else '⚠ '+', '.join(wJ)}")
    ages = ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70+']
    print("    15군 φ (param vs 기존 U-shape):")
    print("      " + " ".join(f"{a}:{phiJ_np[k]:.2f}/{PHI_USHAPE[k]:.2f}" for k, a in enumerate(ages) if k in (0,2,3,5,9,14)))

    print("\n(2) 시즌×연령 obs/model (param φ, γ CDC 고정):")
    print(f"  {'season':>11} " + " ".join(f"{ag:>7}" for ag in HIRA_AGE_GROUPS))
    om_new_all = {}; om_old_all = {}
    for i, s in enumerate(SEASONS):
        om, omt = obs_model(C, i, J["R0"][i], phiJ)
        om_new_all[s] = om
        if old: om_old_all[s] = old["fits"][s]["obs_model_by_age"]
        print(f"  {s:>11} " + " ".join(f"{om[ag]:>7.2f}" for ag in HIRA_AGE_GROUPS))
    if old:
        print("  [기존 U-shape 대비 12-17 / 0-5]:")
        for s in SEASONS:
            print(f"    {s}: 12-17 {om_old_all[s]['12-17']:.2f}→{om_new_all[s]['12-17']:.2f}   0-5 {om_old_all[s]['0-5']:.2f}→{om_new_all[s]['0-5']:.2f}")
    sa_new = np.mean([np.mean([om_new_all[s][ag] for ag in ('6-11','12-17')]) for s in SEASONS])
    print(f"  학령기(6-17) obs/model mean: param={sa_new:.2f}" + (f" (기존={old['school_age_obs_model']:.2f})" if old else ""))

    print("\n(3) 재분배 방향 (param φ, 병가 0.6→0.4 전기간):")
    rev_ok = 0
    redist = {}
    for i, s in enumerate(SEASONS):
        da, ad, sc = redistribution(C, i, J["R0"][i], phiJ)
        redist[s] = dict(d_attack=da, adult_down=ad, school_up=sc)
        print(f"  {s}: 성인↓={ad} 학령기↑={sc}  (18-44:{da['18-44']:+.3f} 12-17:{da['12-17']:+.3f})")

    print(f"\n(4) NLL: joint φ param={J['nll']:.1f}" + (f"  (참고: 기존 U-shape per-season NLL 합={sum(old['fits'][s]['data_nll'] for s in SEASONS):.1f})" if old else ""))

    verdict = "φ로 12-17 과대 개선" if sa_new > 0.75 else "φ만으론 12-17 과대 잔존 → γ_report 조정 필요"
    print(f"\n★ 판정: {verdict}")
    print("=" * 92)

    OUT.write_text(json.dumps(dict(
        meta=dict(baseline_pwork=BASE_PWORK, pi_fixed=PI_FIXED.tolist(), gamma_fixed=[0.40,0.18,0.25],
                  phi_ranges=dict(young=RNG_YOUNG, mid=RNG_MID, old=RNG_OLD)),
        single_2019=dict(**f19, phi_vec=[float(x) for x in np.asarray(phi19)],
                         obs_model_new=om19, obs_model_fixed=om_old19, range_warn=w19),
        joint=dict(phi_young=J["phi_young"], phi_mid=J["phi_mid"], phi_old=J["phi_old"],
                   phi_nb=J["phi_nb"], R0_by_season=dict(zip(SEASONS, J["R0"])), nll=J["nll"],
                   phi_vec=[float(x) for x in phiJ_np], range_warn=wJ),
        obs_model_new=om_new_all, obs_model_fixed=om_old_all if old else None,
        school_age_obs_model_new=float(sa_new),
        redistribution=redist, verdict=verdict), indent=2, default=float))
    print(f"[json] {OUT}")


if __name__ == "__main__":
    main()
