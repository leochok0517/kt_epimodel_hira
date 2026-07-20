"""φ linear U-shape FIXED + γ_report 12-17/6-11 estimated. baseline p=0.6.

φ (fully fixed, linear-interpolated between literature anchors):
  0-4=2.0, 5-9=1.75, 10-14=1.5, 15-19=1.25, 20-44 (idx4-8)=1.0 flat,
  45-49..70+ (idx9-14) = 1.0→1.5 linear.

γ_report (reporting fraction per infection), estimate 6-11 & 12-17 only:
  0-5 (idx0-1) = 0.40 FIXED; 6-11 (idx2) ∈ [0.25,0.40]; 12-17 (idx3) ∈ [0.15,0.30];
  adult (idx4-12) = 0.18 FIXED; elder (idx13-14) = 0.25 FIXED.
  Monotone constraint 0.40 ≥ γ_6-11 ≥ γ_12-17 ≥ 0.18 (penalty if violated).

Hypothesis: 12-17 over-estimation (obs/model~0.4) is a reporting issue — 12-17
report fewer episodes per infection than the child 0.40. Lowering γ_12-17 toward
adult level should pull obs/model → 1.

Common π fixed (joint p06), per-season R0, 6-season joint. Point estimate.
Output: outputs/eda/gamma_refit_linear_phi.json + console.
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
import multiseason_joint_sharedpi as MJ

REPO_ROOT = Path(__file__).resolve().parent.parent
ED = REPO_ROOT / "outputs" / "eda"
OUT = ED / "gamma_refit_linear_phi.json"
OLD_JSON = ED / "per_season_fit_p06.json"     # fixed-U-shape + child-0.40 obs/model

SEASONS = MJ.SEASONS
BASE_PWORK = 0.6
PI_FIXED = np.array([0.322, 0.282, 0.069, 0.327])

# φ LINEAR U-shape (fully fixed): anchors 0-4=2.0, min=1.0 (idx4-8), 70+=1.5
PHI_LINEAR = np.array([2.0, 1.75, 1.5, 1.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0 + 0.5*1/6, 1.0 + 0.5*2/6, 1.0 + 0.5*3/6,
                       1.0 + 0.5*4/6, 1.0 + 0.5*5/6, 1.5])   # idx9..14 = 1.083..1.5

# γ_report fixed parts
G05 = 0.40; G_ADULT = 0.18; G_ELDER = 0.25
RNG_611 = (0.25, 0.40); RNG_1217 = (0.15, 0.30)
LOG_R0_B = (float(np.log(0.8)), float(np.log(3.5))); PHI_NB_B = (1e-3, 1e6)
MONO_PEN = 1e4
N_STARTS = 10; SEED = 41
ADULT = ["18-44", "45-64"]; SCHOOL = ["6-11", "12-17"]


def gamma15(g611, g1217):
    """γ_15: idx0-1=0.40, idx2=g611, idx3=g1217, idx4-12=0.18, idx13-14=0.25."""
    g = jnp.array([G05, G05, g611, g1217] + [G_ADULT]*9 + [G_ELDER]*2)
    return g


def build():
    C = MJ.build_common(); C["shared"]["p_work"] = BASE_PWORK; C["nw"] = C["nweeks"]
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for idx, wt in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, idx] = wt
    pflat = np.asarray(C["shared"]["pop_15"]); pflat = pflat.sum(1) if pflat.ndim == 2 else pflat
    C["H"] = H; C["pop6"] = H @ pflat
    return C


def run_inc(C, i, R0, p_work=BASE_PWORK, work_win=(-1e9, 1e9), work_base=1.0):
    phi = jnp.asarray(PHI_LINEAR)
    beta = derive_beta_from_R0_simplex(C["ngm"], jnp.asarray(R0), jnp.asarray(PI_FIXED), phi)
    kw = dict(C["shared"])
    kw["beta_h"] = beta[0]; kw["beta_w"] = beta[1]; kw["beta_s"] = beta[2]; kw["beta_o"] = beta[3]
    kw["phi_susc"] = phi; kw["p_school"] = 1.0; kw["p_work"] = p_work
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_work_baseline"] = work_base
    st = simulate_jax(C["states"][i], **kw, discretize_time=False)
    return daily_new_infection_by_age_jax(st)


def obs_model(C, i, R0, g611, g1217):
    inc = run_inc(C, i, R0)
    pred = np.asarray(simulation_to_hira_by_age_jax(inc, gamma15(g611, g1217), n_weeks=C["nw"]))
    obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i]); mask = w.sum(1) > 0
    om = {ag: float(obs[mask, a].sum()/max(pred[mask, a].sum(), 1.0)) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return om


def fit_joint(C):
    """R0(6) + γ_611 + γ_1217 + phi_nb.  φ fixed linear."""
    n = len(SEASONS)
    obsj = [jnp.asarray(C["obs"][i]) for i in range(n)]; wj = [jnp.asarray(C["w"][i]) for i in range(n)]
    def loss(x):
        g611 = x[n]; g1217 = x[n+1]; pnb = x[n+2]
        gv = gamma15(g611, g1217)
        tot = 0.0
        for i in range(n):
            inc = run_inc(C, i, jnp.exp(x[i]))
            pred = simulation_to_hira_by_age_jax(inc, gv, n_weeks=C["nw"])
            tot = tot + nb_nll_jax(obsj[i], pred, wj[i], concentration=pnb, min_rate=0.01)
        # monotone: 0.40 ≥ g611 ≥ g1217 ≥ 0.18
        tot = tot + MONO_PEN * (jnp.maximum(g1217 - g611, 0.0)**2
                                + jnp.maximum(g611 - G05, 0.0)**2
                                + jnp.maximum(G_ADULT - g1217, 0.0)**2)
        return tot
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v): v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(SEED)
    bounds = [LOG_R0_B]*n + [RNG_611, RNG_1217, PHI_NB_B]
    best = None; g_starts = []
    for k in range(N_STARTS):
        x0 = list(np.log(rng.uniform(1.8, 2.4, n))) + [rng.uniform(*RNG_611), rng.uniform(*RNG_1217), 10.0]
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=500, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        g_starts.append((float(r.x[n]), float(r.x[n+1])))
        if best is None or r.fun < best.fun: best = r
    x = best.x
    return dict(R0=[float(np.exp(x[i])) for i in range(n)], g611=float(x[n]), g1217=float(x[n+1]),
                phi_nb=float(x[n+2]), nll=float(best.fun),
                g1217_std=float(np.std([g[1] for g in g_starts])),
                g611_std=float(np.std([g[0] for g in g_starts])))


def redistribution(C, i, R0):
    base = C["H"] @ np.asarray(run_inc(C, i, R0, p_work=BASE_PWORK)).sum(0)
    inc = run_inc(C, i, R0, p_work=0.4)
    d = (C["H"] @ np.asarray(inc).sum(0) - base) / C["pop6"]
    da = {ag: float(100.0*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return da, all(da[a] < 0 for a in ADULT), any(da[s] > 0 for s in SCHOOL)


def main():
    print("=" * 92)
    print("γ_report REFIT (12-17, 6-11) + φ LINEAR fixed — baseline p_work=0.6")
    print("=" * 92)
    t0 = time.perf_counter(); C = build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
    old = json.load(open(OLD_JSON)) if OLD_JSON.exists() else None

    ages = ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70+']
    print("\n(1) φ LINEAR (15군, 고정):")
    print("    " + " ".join(f"{a}:{PHI_LINEAR[k]:.2f}" for k, a in enumerate(ages) if k in (0,1,2,3,4,8,9,11,14)))

    print("\n── joint fit: R0(6) + γ(6-11,12-17) + phi_nb, φ fixed linear ──")
    tj = time.perf_counter(); J = fit_joint(C); print(f"  ({time.perf_counter()-tj:.1f}s)")
    gv = np.asarray(gamma15(J["g611"], J["g1217"]))
    print(f"\n(2) 추정 γ_report:  6-11={J['g611']:.3f} (std {J['g611_std']:.3f})   12-17={J['g1217']:.3f} (std {J['g1217_std']:.3f})")
    print(f"    γ_15 = [0-5:{G05:.2f}, 6-11:{J['g611']:.3f}, 12-17:{J['g1217']:.3f}, adult:{G_ADULT:.2f}, elder:{G_ELDER:.2f}]")
    # railing warnings
    warns = []
    if abs(J["g1217"] - RNG_1217[0]) < 0.005: warns.append(f"⚠ 12-17 하한 railing ({RNG_1217[0]}) — γ로도 부족, 구조문제 신호")
    if abs(J["g1217"] - RNG_1217[1]) < 0.005: warns.append(f"⚠ 12-17 상한 railing ({RNG_1217[1]})")
    if abs(J["g611"] - RNG_611[0]) < 0.005: warns.append(f"⚠ 6-11 하한 railing ({RNG_611[0]})")
    mono_ok = (G05 >= J["g611"] >= J["g1217"] >= G_ADULT - 1e-6)
    print(f"    단조감소(0.40≥6-11≥12-17≥0.18): {'OK' if mono_ok else '⚠ 위반'}   {' | '.join(warns) if warns else '범위 내 안착'}")

    print("\n(3) 시즌×연령 obs/model (φ선형 + γ추정):")
    print(f"  {'season':>11} " + " ".join(f"{ag:>7}" for ag in HIRA_AGE_GROUPS))
    om_new = {}
    for i, s in enumerate(SEASONS):
        om = obs_model(C, i, J["R0"][i], J["g611"], J["g1217"]); om_new[s] = om
        print(f"  {s:>11} " + " ".join(f"{om[ag]:>7.2f}" for ag in HIRA_AGE_GROUPS))
    if old:
        print("  [기존(child0.40) 대비 12-17 / 6-11 / 0-5]:")
        for s in SEASONS:
            o = old["fits"][s]["obs_model_by_age"]
            print(f"    {s}: 12-17 {o['12-17']:.2f}→{om_new[s]['12-17']:.2f}   6-11 {o['6-11']:.2f}→{om_new[s]['6-11']:.2f}   0-5 {o['0-5']:.2f}→{om_new[s]['0-5']:.2f}")
    m1217 = np.mean([om_new[s]['12-17'] for s in SEASONS]); m611 = np.mean([om_new[s]['6-11'] for s in SEASONS])
    print(f"  12-17 obs/model mean={m1217:.2f} (target~1.0)   6-11 mean={m611:.2f}")

    print("\n(4) 재분배 방향 (병가 0.6→0.4 전기간) + NLL:")
    redist = {}
    for i, s in enumerate(SEASONS):
        da, ad, sc = redistribution(C, i, J["R0"][i]); redist[s] = dict(d_attack=da, adult_down=ad, school_up=sc)
    n_ad = sum(1 for s in SEASONS if redist[s]["adult_down"])
    print(f"  성인↓ {n_ad}/6 시즌 (γ는 관측이라 동역학 무관 — 유지 당연)")
    print(f"  NLL(γ refit)={J['nll']:.1f}" + (f"  참고: 기존 per-season 합={sum(old['fits'][s]['data_nll'] for s in SEASONS):.1f}" if old else ""))

    improved = (m1217 > 0.75) and not any("하한 railing" in w for w in warns)
    verdict = ("γ_report 12-17 세분으로 과대추정 개선" if improved
               else ("12-17 하한 railing — γ로도 부족, 구조문제" if any("하한" in w for w in warns)
                     else "12-17 부분개선(mean %.2f)"%m1217))
    print(f"\n★ 판정: {verdict}")
    print("=" * 92)

    OUT.write_text(json.dumps(dict(
        meta=dict(baseline_pwork=BASE_PWORK, phi_linear=PHI_LINEAR.tolist(), pi_fixed=PI_FIXED.tolist(),
                  gamma_fixed=dict(g05=G05, adult=G_ADULT, elder=G_ELDER),
                  ranges=dict(g611=RNG_611, g1217=RNG_1217)),
        gamma_est=dict(g611=J["g611"], g1217=J["g1217"], g611_std=J["g611_std"], g1217_std=J["g1217_std"],
                       gamma_15=gv.tolist(), monotone_ok=bool(mono_ok), warnings=warns),
        R0_by_season=dict(zip(SEASONS, J["R0"])), nll=J["nll"],
        obs_model_new=om_new, obs_model_old=(old["fits"] if old else None),
        obs_model_12_17_mean=float(m1217), obs_model_6_11_mean=float(m611),
        redistribution=redist, verdict=verdict), indent=2, default=float))
    print(f"[json] {OUT}")


if __name__ == "__main__":
    main()
