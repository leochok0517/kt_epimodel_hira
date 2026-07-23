"""κ 단순화 — η(겹침효율) 제거, κ = 체류시간 증가율만.

변경 유일: KAP
  before (v3): κ_학생 0.29, κ_성인 0.30, κ_70+ 0    (κ_체류 × η, η∈[0.75,0.86])
  after  (본): κ_학생 0.34, κ_성인 0.40, κ_70+ 0    (κ_체류 only)

나머지 확정 파라미터는 final_v3_pipeline 과 동일:
  baseline p=0.6, φ linear, γ=[0.40×2,0.25,0.18×10,0.25×2], Erlang I₃,
  R(0)=[0.10×4,0.40×5,0.60×4,0.65×2], C(t) term↔vacation, v(t) 정규화,
  seed γ-fix, μ=1.0, 3시즌 (16-17, 17-18, 19-20), first_peak_only.

Output: outputs/eda/kappa_no_eta_*.json (v3 대비 격리 저장, 기존 파일 무손실).
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
from kt_epimodel_hira.jax_model.erlang import (
    simulate_jax_erlang, daily_new_infection_by_age_erlang, split_seed_to_erlang,
)
import final_pipeline_confirmed as F

ED = Path(__file__).resolve().parent.parent / "outputs" / "eda"
SEAS = ["2016-2017", "2017-2018", "2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEAS]

# ---- 확정 파라미터 ----
PHI = np.array(F.PHI)
BASE = 0.6
GAMMA = np.array([0.40, 0.40, 0.25, 0.18] + [0.18]*9 + [0.25, 0.25])
IMM = np.array([0.10]*4 + [0.40]*5 + [0.60]*4 + [0.65]*2)
TERM = (70.0, 113.0); VAC = (113.0, 183.0); WH = (-1e9, 1e9)

# ★ κ 변경: η 제거, 체류시간 증가율만
KAP = np.array([0.34]*4 + [0.40]*10 + [0.0])

CHILD = ["0-5", "6-11", "12-17"]
ADULT = ["18-44", "45-64"]


def build():
    C = F.build()
    pf = np.asarray(C["shared"]["pop_15"])
    C["pf"] = pf.sum(1) if pf.ndim == 2 else pf
    M = C["shared"]
    C["ngm3"] = make_ngm_eigvalue_fn(
        pop_15=np.asarray(M["pop_15"]), rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
        C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]),
        R0_immunity=IMM, gamma=float(M["gamma"]), seasonal_factor=1.0 + F.S.AMP,
    )
    C["st"] = {}
    for s in SEAS:
        sd = estimate_initial_infected_from_hira(
            s, C["pf"], sido_codes=list(SUDOGWON_SIDO_CODES),
            gamma_15_assumed=GAMMA,
        )
        C["st"][s] = jnp.asarray(_build_initial_state_with_age_seed(
            C["pf"], sd, seed_e_factor=0.5, initial_immunity=IMM,
            initial_vaccinated_fraction=0.0,
        ))
    return C


def sim(C, s, R0, pi, p_school=BASE, p_work=BASE, sch_win=WH, work_win=WH):
    beta = derive_beta_from_R0_simplex(C["ngm3"], jnp.asarray(R0),
                                        jnp.asarray(pi), jnp.asarray(PHI))
    kw = dict(C["shared"])
    kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
    kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
    kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
    kw["p_school"] = p_school; kw["p_work"] = p_work
    kw["policy_school_start_day"], kw["policy_school_end_day"] = sch_win
    kw["policy_work_start_day"], kw["policy_work_end_day"] = work_win
    kw["policy_school_baseline"] = BASE; kw["policy_work_baseline"] = BASE
    st = simulate_jax_erlang(split_seed_to_erlang(C["st"][s]), **kw,
                              discretize_time=False)
    return daily_new_infection_by_age_erlang(st)


def pred_h(C, inc):
    return np.asarray(simulation_to_hira_by_age_jax(
        jnp.asarray(inc), jnp.asarray(GAMMA), n_weeks=52))


def att6(C, inc):
    return C["H"] @ np.asarray(inc).sum(0)


def fit(C, s, i):
    obsj = jnp.asarray(C["obs"][i]); wj = jnp.asarray(C["w"][i])

    def loss(x):
        R0 = jnp.exp(x[0]); pi = jax.nn.softmax(x[1:5])
        beta = derive_beta_from_R0_simplex(C["ngm3"], R0, pi, jnp.asarray(PHI))
        kw = dict(C["shared"])
        kw["beta_h"], kw["beta_w"] = beta[0], beta[1]
        kw["beta_s"], kw["beta_o"] = beta[2], beta[3]
        kw["phi_susc"] = jnp.asarray(PHI); kw["kappa"] = jnp.asarray(KAP)
        kw["p_school"] = BASE; kw["p_work"] = BASE
        st = simulate_jax_erlang(split_seed_to_erlang(C["st"][s]), **kw,
                                  discretize_time=False)
        pred = simulation_to_hira_by_age_jax(
            daily_new_infection_by_age_erlang(st), jnp.asarray(GAMMA),
            n_weeks=C["nw"])
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

    rng = np.random.default_rng(121 + i)
    bounds = [F.LOG_R0_B] + [(-10, 10)]*4 + [F.PHI_NB_B]
    best = None
    for k in range(10):
        x0 = np.concatenate([[np.log(rng.uniform(1.8, 2.5))],
                              F.LOGIT_REF + rng.normal(0, 0.5, 4), [10.0]])
        try:
            r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=400, ftol=1e-9, gtol=1e-6))
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    R0 = float(np.exp(x[0])); pi = np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    beta = np.asarray(derive_beta_from_R0_simplex(
        C["ngm3"], jnp.asarray(R0), jnp.asarray(pi), jnp.asarray(PHI)))
    return dict(R0=R0, pi=[float(p) for p in pi],
                 beta_4=[float(b) for b in beta], nll=float(best.fun))


def omw(C, i, pred):
    obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i])
    mask = w.sum(1) > 0; o = obs[mask].sum(1); m = pred[mask].sum(1)
    om = {ag: float(obs[mask, a].sum() / max(pred[mask, a].sum(), 1.0))
          for a, ag in enumerate(HIRA_AGE_GROUPS)}
    return om, float(o.sum() / max(m.sum(), 1)), \
        float((o.sum() / max(o.max(), 1)) / (m.sum() / max(m.max(), 1)))


def dattack(C, s, i, R0, pi, base6, **pol):
    inc = sim(C, s, R0, pi, **pol); inf6 = att6(C, inc)
    d = (inf6 - base6) / C["pop6"]
    return (float(100 * (base6.sum() - inf6.sum()) / max(base6.sum(), 1)),
            {ag: float(100*d[a]) for a, ag in enumerate(HIRA_AGE_GROUPS)})


def main():
    print("=" * 94)
    print(f"KAPPA NO-η — κ 학생={KAP[0]:.2f}  성인={KAP[4]:.2f}  70+={KAP[14]:.2f}   "
          f"(v3: 0.29/0.30/0)")
    print(f"  나머지 확정: baseline={BASE}, φ선형, γ CDC Reed(0-5·6-11=0.40, 12-17=0.25, 18-64=0.18, 65+=0.25),")
    print(f"  Erlang I₃, R(0)=[0.10×4,0.40×5,0.60×4,0.65×2], C(t) term↔vac, v(t) 정규화")
    print("=" * 94)
    t0 = time.perf_counter(); C = build()
    print(f"[setup] {time.perf_counter()-t0:.1f}s")

    # 1 fit
    print("\n[1] fit (baseline p=0.6):")
    t1 = {}; bp = {}
    for s, i in zip(SEAS, IDX):
        f = fit(C, s, i); bp[s] = (f["R0"], f["pi"])
        pr = pred_h(C, sim(C, s, f["R0"], f["pi"]))
        om, omt, wr = omw(C, i, pr)
        t1[s] = dict(**f, obs_model=om, om_total=omt, width_ratio=wr)
        print(f"  {s}: R0={f['R0']:.3f} β=[h{f['beta_4'][0]:.4f},w{f['beta_4'][1]:.4f},"
              f"s{f['beta_4'][2]:.4f},o{f['beta_4'][3]:.4f}] om={omt:.2f} wid={wr:.2f} "
              f"| 18-44={om['18-44']:.2f} 45-64={om['45-64']:.2f} 12-17={om['12-17']:.2f}")
    (ED/"kappa_no_eta_fit.json").write_text(json.dumps(
        dict(meta=dict(kap=KAP.tolist(), imm=IMM.tolist(), gamma=GAMMA.tolist(),
                       baseline=BASE), per_season=t1), indent=2, default=float))

    # 2 school vs sick
    print("\n[2] 학교 vs 병가 (term):")
    P = [0.6, 0.4, 0.2, 0.0]; t2 = {}
    for s, i in zip(SEAS, IDX):
        R0, pi = bp[s]; base = att6(C, sim(C, s, R0, pi))
        rec = {"sick": {}, "school": {}}
        for p in P:
            av, d = dattack(C, s, i, R0, pi, base, p_work=p, work_win=TERM)
            rec["sick"][str(p)] = dict(av=av, da=d)
            av2, d2 = dattack(C, s, i, R0, pi, base, p_school=p, sch_win=TERM)
            rec["school"][str(p)] = dict(av=av2, da=d2)
        t2[s] = rec
    sk = np.mean([t2[s]["sick"]["0.4"]["av"] for s in SEAS])
    sc = np.mean([t2[s]["school"]["0.4"]["av"] for s in SEAS])
    da_sk = np.mean([[t2[s]["sick"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS]
                     for s in SEAS], 0)
    da_sc = np.mean([[t2[s]["school"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS]
                     for s in SEAS], 0)
    print(f"  averted@p=0.4: 병가={sk:.2f}% 학교={sc:.2f}% → 배율={sc/sk:.2f}배")
    print(f"  병가 Δ: " + " ".join(f"{HIRA_AGE_GROUPS[a]}:{da_sk[a]:+.2f}" for a in range(6)))
    print(f"  학교 Δ: " + " ".join(f"{HIRA_AGE_GROUPS[a]}:{da_sc[a]:+.2f}" for a in range(6)))
    (ED/"kappa_no_eta_school_vs_sick.json").write_text(json.dumps(
        dict(meta=dict(P=P), results=t2, ratio=float(sc/sk),
             sick_04=float(sk), school_04=float(sc)),
        indent=2, default=float))

    # 3 rate vs number
    pop6 = np.asarray(C["pop6"])
    num_sk = -da_sk / 100 * pop6; num_sc = -da_sc / 100 * pop6
    print("\n[3] rate vs number (p=0.4):")
    print(f"  rate 최대: 병가={HIRA_AGE_GROUPS[np.argmin(da_sk)]} "
          f"학교={HIRA_AGE_GROUPS[np.argmin(da_sc)]}")
    print(f"  number 최대: 병가={HIRA_AGE_GROUPS[np.argmax(num_sk)]}({num_sk.max():.0f}명) "
          f"학교={HIRA_AGE_GROUPS[np.argmax(num_sc)]}({num_sc.max():.0f}명)")
    (ED/"kappa_no_eta_rate_vs_number.json").write_text(json.dumps(
        dict(pop6=pop6.tolist(), rate_sick=da_sk.tolist(),
             rate_school=da_sc.tolist(), num_sick=num_sk.tolist(),
             num_school=num_sc.tolist()), indent=2, default=float))

    # 4 policy intensity + redistribution
    print("\n[4] 병가 강도 + 재분배 (성인↓ 확인):")
    PL = [0.6, 0.4, 0.2, 0.0]; t4 = {}; nad = 0
    for s, i in zip(SEAS, IDX):
        R0, pi = bp[s]; base = att6(C, sim(C, s, R0, pi)); pr = {}
        for p in PL:
            av, d = dattack(C, s, i, R0, pi, base, p_work=p)
            ad = all(d[a] < 0 for a in ADULT)
            pr[str(p)] = dict(av=av, da=d, adult_down=bool(ad))
            if p in (0.4, 0.2, 0.0) and ad:
                nad += 1
        t4[s] = pr
    print("  averted%: " + " | ".join(
        f"{s}:" + ",".join(f"{t4[s][str(p)]['av']:+.1f}" for p in PL) for s in SEAS))
    print(f"  ★ 성인↓ {nad}/9   (p_work ∈ {{0.4,0.2,0.0}} × 3시즌)")
    (ED/"kappa_no_eta_policy_intensity.json").write_text(json.dumps(
        dict(meta=dict(levels=PL), results=t4, adult_down=f"{nad}/9"),
        indent=2, default=float))

    # 5 holiday reversal
    print("\n[5] 방학 부호반전 (κ↑ → spillover↑ → 아동 역효과 증가?):")
    t5 = {}; INT = [0.4, 0.2, 0.0]
    for s, i in zip(SEAS, IDX):
        R0, pi = bp[s]; base = att6(C, sim(C, s, R0, pi)); rec = {}
        for p in INT:
            _, dt = dattack(C, s, i, R0, pi, base, p_work=p, work_win=TERM)
            _, dv = dattack(C, s, i, R0, pi, base, p_work=p, work_win=VAC)
            cst = sum(dt[c] for c in CHILD); csv = sum(dv[c] for c in CHILD)
            rec[str(p)] = dict(tc=cst, vc=csv, vac=dv,
                                rev=bool(cst < 0 and csv > 0))
        t5[s] = rec
    for p in INT:
        nr = sum(1 for s in SEAS if t5[s][str(p)]["rev"])
        print(f"  p={p}: 반전 {nr}/3  vac_child=["
              + ",".join(f"{t5[s][str(p)]['vc']:+.2f}" for s in SEAS) + "]  "
              + "term_child=[" + ",".join(f"{t5[s][str(p)]['tc']:+.2f}" for s in SEAS) + "]")
    (ED/"kappa_no_eta_holiday.json").write_text(json.dumps(
        dict(meta=dict(intensities=INT), results=t5), indent=2, default=float))

    # 요약
    print("\n" + "=" * 94)
    print("SUMMARY vs v3:")
    print(f"  κ 변경: [0.29,0.30,0.30,0.30, 0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.30, 0.0]")
    print(f"       → [0.34,0.34,0.34,0.34, 0.40,0.40,0.40,0.40,0.40,0.40,0.40,0.40,0.40,0.40, 0.0]")
    print(f"  성인 κ  +33%   학생 κ  +17%")
    print(f"  → spillover 강화. 재분배·방학반전 크기 증가 예상.")
    print("=" * 94)


if __name__ == "__main__":
    main()
