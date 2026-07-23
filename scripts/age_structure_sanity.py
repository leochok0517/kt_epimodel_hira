"""연령구조 반영 sanity check — v4 point est.

방법: 2023-01 인구를 시즌별 approximated growth factor 로 scale,
      3시즌 fit + 정책(병가/학교, p=0.4, term) 재계산 → 기존 v4 결과와 대조.

Scaling factors (KOSIS 공표 근사, 2023 → 시즌 연도):
  0-4:   2016 ×1.30  |  2017 ×1.25  |  2019 ×1.15   (저출산 시대 반영)
  5-9:   2016 ×1.18  |  2017 ×1.15  |  2019 ×1.09
  10-14: 2016 ×1.13  |  2017 ×1.11  |  2019 ×1.07
  15-19: 2016 ×1.05  |  2017 ×1.04  |  2019 ×1.02
  20-24: 2016 ×1.11  |  2017 ×1.09  |  2019 ×1.05
  25-29: 2016 ×1.05  |  2017 ×1.04  |  2019 ×1.02
  30-34: 2016 ×1.02  |  2017 ×1.02  |  2019 ×1.01
  35-39: 2016 ×0.98  |  2017 ×0.99  |  2019 ×0.99
  40-44: 2016 ×0.95  |  2017 ×0.96  |  2019 ×0.98
  45-64: 2016 ×0.90  |  2017 ×0.92  |  2019 ×0.96   (4 그룹 공통)
  65+:   2016 ×0.72  |  2017 ×0.77  |  2019 ×0.87   (2 그룹 공통, 고령화)
"""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import jax.numpy as jnp
from scipy.optimize import minimize
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import (
    estimate_initial_infected_from_hira, _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.jax_model.loss_jax import (
    HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax,
)
from kt_epimodel_hira.jax_model.numpyro_model import (
    make_ngm_eigvalue_fn, derive_beta_from_R0_simplex,
)
from kt_epimodel_hira.jax_model.erlang_presymp import (
    simulate_jax_erlang_presymp, daily_new_onset_by_age_erlang_presymp,
    split_seed_to_erlang, ngm_factor, W_PRESYMP,
)
import final_pipeline_confirmed as F

ED = Path(__file__).resolve().parent.parent/"outputs"/"eda"
SEAS = ["2016-2017","2017-2018","2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEAS]

PHI = np.array(F.PHI); BASE = 0.6
GAMMA = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)
KAP = np.array([0.34]*4+[0.40]*10+[0.0])
W = W_PRESYMP; NGM_F = ngm_factor(W)
TERM = (70.0, 113.0); WH = (-1e9, 1e9)

# NIMS-15 → season scaling factors
SCALE = {
    "2016-2017": np.array([1.30,1.18,1.13,1.05, 1.11,1.05,1.02,0.98,0.95, 0.90,0.90,0.90,0.90, 0.72,0.72]),
    "2017-2018": np.array([1.25,1.15,1.11,1.04, 1.09,1.04,1.02,0.99,0.96, 0.92,0.92,0.92,0.92, 0.77,0.77]),
    "2019-2020": np.array([1.15,1.09,1.07,1.02, 1.05,1.02,1.01,0.99,0.98, 0.96,0.96,0.96,0.96, 0.87,0.87]),
}


def build_with_scale(scale=None):
    C = F.build()
    pf = np.asarray(C["shared"]["pop_15"])
    pf_flat = pf.sum(1) if pf.ndim == 2 else pf   # (15,)
    if scale is not None:
        pf_scaled = pf_flat * scale
    else:
        pf_scaled = pf_flat
    C["pf"] = pf_scaled
    # override pop_15 in shared (used inside FOI as N)
    C["shared"]["pop_15"] = jnp.asarray(pf_scaled.reshape(15,1))
    M = C["shared"]
    C["ngm3"] = make_ngm_eigvalue_fn(
        pop_15=np.asarray(M["pop_15"]), rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
        C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]),
        R0_immunity=IMM, gamma=float(M["gamma"]), seasonal_factor=1.0+F.S.AMP)
    return C


def build_setup_per_season(C_base, s):
    """C_base with pop_15 pre-scaled. Add seed·init_state for season s."""
    sd = estimate_initial_infected_from_hira(s, C_base["pf"],
        sido_codes=list(SUDOGWON_SIDO_CODES), gamma_15_assumed=GAMMA)
    return jnp.asarray(_build_initial_state_with_age_seed(
        C_base["pf"], sd, seed_e_factor=0.5, initial_immunity=IMM,
        initial_vaccinated_fraction=0.0))


def beta_from(C, R0, pi):
    b0 = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                       jnp.asarray(pi), jnp.asarray(PHI))
    return b0 / NGM_F


def sim(C, init_state, R0, pi, p_school=BASE, p_work=BASE,
         sch_win=WH, work_win=WH):
    beta = beta_from(C, R0, pi)
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang_presymp(split_seed_to_erlang(init_state),
                                      w_presymp=W, **kw, discretize_time=False)
    return daily_new_onset_by_age_erlang_presymp(st)


def att6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


def fit_season(C, init_state, s, i):
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i])
    def loss(x):
        R0 = jnp.exp(x[0]); pi = jax.nn.softmax(x[1:5])
        beta = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, jnp.asarray(PHI)) / NGM_F
        kw = dict(C["shared"])
        kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
        kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang_presymp(split_seed_to_erlang(init_state),
                                          w_presymp=W, **kw, discretize_time=False)
        pred = simulation_to_hira_by_age_jax(
            daily_new_onset_by_age_erlang_presymp(st),
            jnp.asarray(GAMMA), n_weeks=C["nw"])
        c = x[1:5] - jnp.mean(x[1:5])
        return nb_nll_jax(obsj, pred, wj, concentration=x[5], min_rate=0.01) \
            + 0.5 * jnp.sum((c - jnp.asarray(F.LOGIT_REF))**2
                             / jnp.asarray(F.SIGMA_PIN)**2)
    lj = jax.jit(loss); gj = jax.jit(jax.grad(loss))
    def fg(xn):
        x = jnp.asarray(xn); v = float(lj(x)); g = np.array(gj(x))
        if not np.isfinite(v):
            v = 1e15; g = np.where(np.isfinite(g), g, 0.0)
        return v, g
    rng = np.random.default_rng(200 + i)
    bounds = [F.LOG_R0_B] + [(-10,10)]*4 + [F.PHI_NB_B]
    best = None
    for k in range(8):
        x0 = np.concatenate([[np.log(rng.uniform(1.8,2.5))],
                              F.LOGIT_REF + rng.normal(0,0.5,4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    R0 = float(np.exp(x[0])); pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    beta = np.asarray(beta_from(C, R0, pi))
    return dict(R0=R0, pi=[float(p) for p in pi],
                 beta_4=[float(b) for b in beta])


def eval_policy(C, init_state, R0, pi):
    """averted total + connctd per-age Δattack for p=0.4 term."""
    base_inc = sim(C, init_state, R0, pi)
    base6 = att6(C, base_inc); pop6 = np.asarray(C["H"] @ np.asarray(C["pf"]))
    tot_b = float(base6.sum())
    sick_inc = sim(C, init_state, R0, pi, p_work=0.4, work_win=TERM)
    sick6 = att6(C, sick_inc)
    school_inc = sim(C, init_state, R0, pi, p_school=0.4, sch_win=TERM)
    school6 = att6(C, school_inc)
    av_sick = 100.0*(tot_b - float(sick6.sum()))/max(tot_b,1.0)
    av_school = 100.0*(tot_b - float(school6.sum()))/max(tot_b,1.0)
    d_sick = (np.asarray(sick6) - np.asarray(base6))/pop6*100.0
    d_school = (np.asarray(school6) - np.asarray(base6))/pop6*100.0
    return dict(
        av_sick=av_sick, av_school=av_school,
        ratio=av_school/av_sick if av_sick != 0 else float("nan"),
        d_sick_by_age={ag: float(d_sick[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)},
        d_school_by_age={ag: float(d_school[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)},
        pop6=pop6.tolist(),
    )


def run_case(scale_dict_or_none, label):
    """scale_dict_or_none: None = 2023 baseline / dict = per-season scaling."""
    print(f"\n══════════ CASE: {label} ══════════", flush=True)
    if scale_dict_or_none is None:
        C = build_with_scale(None)
        print(f"  pop_15 (2023): {[int(x) for x in np.asarray(C['pf'])]}", flush=True)
        results = {}
        for s, i in zip(SEAS, IDX):
            init = build_setup_per_season(C, s)
            fit = fit_season(C, init, s, i); pol = eval_policy(C, init, fit["R0"], fit["pi"])
            results[s] = dict(fit=fit, policy=pol)
            print(f"    {s}: R0={fit['R0']:.3f}  π={[round(p,4) for p in fit['pi']]}",
                  flush=True)
            print(f"       av: sick={pol['av_sick']:+.2f}%  school={pol['av_school']:+.2f}%  배율={pol['ratio']:+.2f}×",
                  flush=True)
    else:
        results = {}
        for s, i in zip(SEAS, IDX):
            scale = scale_dict_or_none[s]
            C = build_with_scale(scale)
            init = build_setup_per_season(C, s)
            fit = fit_season(C, init, s, i); pol = eval_policy(C, init, fit["R0"], fit["pi"])
            results[s] = dict(fit=fit, policy=pol,
                                pop_used=[int(x) for x in np.asarray(C['pf'])])
            print(f"    {s}: R0={fit['R0']:.3f}  π={[round(p,4) for p in fit['pi']]}"
                  f"  pop_total={int(np.asarray(C['pf']).sum()):,}", flush=True)
            print(f"       av: sick={pol['av_sick']:+.2f}%  school={pol['av_school']:+.2f}%  배율={pol['ratio']:+.2f}×",
                  flush=True)
    return results


def main():
    print("="*94, flush=True)
    print("연령구조 sanity check — 2023 인구 vs 시즌별 scaling", flush=True)
    print("  방법: 3시즌 각각 point-fit + policy (sick/school p=0.4 term)", flush=True)
    print("="*94, flush=True)
    t0 = time.perf_counter()
    baseline = run_case(None, "baseline (2023 pop, 3시즌 공통)")
    per_season = run_case(SCALE, "per-season scaling (2016/17/19)")
    print(f"\n[wall] {time.perf_counter()-t0:.1f}s", flush=True)

    print("\n══════════ 비교 ══════════", flush=True)
    print(f"{'시즌':>10s}  {'metric':>18s}  {'2023 baseline':>15s}  {'per-season':>13s}  {'Δ %':>8s}", flush=True)
    for s in SEAS:
        b = baseline[s]; p = per_season[s]
        pairs = [
            ("R0", b["fit"]["R0"], p["fit"]["R0"]),
            ("π_home", b["fit"]["pi"][0], p["fit"]["pi"][0]),
            ("π_work", b["fit"]["pi"][1], p["fit"]["pi"][1]),
            ("π_school", b["fit"]["pi"][2], p["fit"]["pi"][2]),
            ("π_other", b["fit"]["pi"][3], p["fit"]["pi"][3]),
            ("averted sick %", b["policy"]["av_sick"], p["policy"]["av_sick"]),
            ("averted school %", b["policy"]["av_school"], p["policy"]["av_school"]),
            ("ratio", b["policy"]["ratio"], p["policy"]["ratio"]),
        ]
        for name, bv, pv in pairs:
            dp = 100*(pv-bv)/bv if bv != 0 else float("nan")
            print(f"{s:>10s}  {name:>18s}  {bv:>15.4f}  {pv:>13.4f}  {dp:>+7.2f}%",
                  flush=True)
        # per-age direction check
        for ag in ["6-11","12-17","18-44","45-64"]:
            bv = b["policy"]["d_sick_by_age"][ag]
            pv = p["policy"]["d_sick_by_age"][ag]
            print(f"{s:>10s}  {'Δ_sick '+ag:>18s}  {bv:>15.4f}  {pv:>13.4f}  {'부호'+'유지' if bv*pv>0 else '부호변화':>9s}",
                  flush=True)
        print(flush=True)

    # 판정
    print("══════════ 판정 ══════════", flush=True)
    max_change = 0.0
    sign_flip = False
    for s in SEAS:
        b = baseline[s]; p = per_season[s]
        for name, bv, pv in [("R0",b["fit"]["R0"],p["fit"]["R0"]),
                              ("π_work",b["fit"]["pi"][1],p["fit"]["pi"][1]),
                              ("av_sick",b["policy"]["av_sick"],p["policy"]["av_sick"]),
                              ("av_school",b["policy"]["av_school"],p["policy"]["av_school"])]:
            dp = abs(100*(pv-bv)/bv) if bv != 0 else 0
            max_change = max(max_change, dp)
        # 재분배 방향 check (sick: 학령기+, 성인-)
        for ag in ["6-11","12-17"]:
            if b["policy"]["d_sick_by_age"][ag] * p["policy"]["d_sick_by_age"][ag] < 0:
                sign_flip = True
        for ag in ["18-44","45-64"]:
            if b["policy"]["d_sick_by_age"][ag] * p["policy"]["d_sick_by_age"][ag] < 0:
                sign_flip = True
    print(f"  최대 상대변화 (R0/π_work/av): {max_change:.1f}%", flush=True)
    print(f"  재분배 방향 부호변화: {'YES (문제)' if sign_flip else 'NO (유지)'}", flush=True)
    if not sign_flip and max_change < 15:
        print(f"  ★ 판정: 방향 유지 + 변화 미미(<15%) → 기존 결과 유지 OK, 연령구조 반영은 나중에",
              flush=True)
    elif sign_flip:
        print(f"  ★ 판정: 부호변화 발생 → 연령구조 지금 반영 필요", flush=True)
    else:
        print(f"  ★ 판정: 변화 큼(>15%) → 연령구조 반영 권장", flush=True)

    out = dict(baseline=baseline, per_season=per_season,
                scale_factors={s: SCALE[s].tolist() for s in SEAS},
                max_change_pct=float(max_change), sign_flip=bool(sign_flip))
    (ED/"age_structure_sanity.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[json] {ED/'age_structure_sanity.json'}", flush=True)


if __name__ == "__main__":
    main()
