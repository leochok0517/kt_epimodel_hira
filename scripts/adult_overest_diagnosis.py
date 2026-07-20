"""성인(18-44,45-64) 과추정 진단 — 원인 분해. 진단만, 확정수정 아님. baseline 0.6 대칭.
(1) obs/model = peak_ratio × width_ratio 분해 (성인)
(2) γ_report 성인 0.18→0.15/0.12 민감도
(3) R(0) 45-64 초기면역 0.45→0.55/0.60 민감도
(4) φ 성인 1.0→0.8 민감도
+ 각 조정의 재분배(성인↓) 영향. Output: outputs/eda/adult_overest.json"""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import jax.numpy as jnp
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import estimate_initial_infected_from_hira, _build_initial_state_with_age_seed
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn, derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang import simulate_jax_erlang, daily_new_infection_by_age_erlang, split_seed_to_erlang
import final_pipeline_confirmed as F
ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"
SEAS=["2016-2017","2017-2018","2019-2020"]; IDX=[F.SEASONS.index(s) for s in SEAS]
PHI0=np.array(F.PHI); KAP=np.array([0.29]*4+[0.30]*10+[0.0]); BASE=0.6
GAMMA0=np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
# R(0) immunity profile (confirmed)
from kt_epimodel_hira.calibration.simple_model import R0_IMMUNITY_PROFILE as IMM0
IMM0=np.asarray(IMM0)
sym=json.load(open(ED/"symmetric_baseline.json"))["fit"]   # R0/pi per season (baseline 0.6)

def build():
    C=F.build(); pf=np.asarray(C["shared"]["pop_15"]); pf=pf.sum(1) if pf.ndim==2 else pf
    C["pf"]=pf
    return C

def rebuild_state(C,i,s,imm):
    sd=estimate_initial_infected_from_hira(s,C["pf"],sido_codes=list(SUDOGWON_SIDO_CODES),gamma_15_assumed=GAMMA0)
    return jnp.asarray(_build_initial_state_with_age_seed(C["pf"],sd,seed_e_factor=0.5,initial_immunity=imm,initial_vaccinated_fraction=0.0))

def sim(C,state,R0,pi,phi,ngm,p_work=BASE,p_school=BASE):
    beta=derive_beta_from_R0_simplex(ngm,jnp.asarray(R0),jnp.asarray(pi),jnp.asarray(phi))
    kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
    kw["phi_susc"]=jnp.asarray(phi); kw["kappa"]=jnp.asarray(KAP); kw["p_school"]=p_school; kw["p_work"]=p_work
    kw["policy_school_baseline"]=BASE; kw["policy_work_baseline"]=BASE
    st=simulate_jax_erlang(split_seed_to_erlang(state), **kw, discretize_time=False)
    return daily_new_infection_by_age_erlang(st)

def om_by_age(C,i,inc,gamma):
    pred=np.asarray(simulation_to_hira_by_age_jax(inc,jnp.asarray(gamma),n_weeks=52))
    obs=np.asarray(C["obs"][i]); w=np.asarray(C["w"][i]); mask=w.sum(1)>0
    return {ag:float(obs[mask,a].sum()/max(pred[mask,a].sum(),1.0)) for a,ag in enumerate(HIRA_AGE_GROUPS)}, pred, mask

def decompose(C,i,pred,mask):
    """성인 obs/model = peak_ratio × width_ratio 분해 (per HIRA age)."""
    obs=np.asarray(C["obs"][i]); out={}
    for a,ag in enumerate(HIRA_AGE_GROUPS):
        o=obs[mask,a]; m=pred[mask,a]
        pr=o.max()/max(m.max(),1); wr=(o.sum()/max(o.max(),1))/(m.sum()/max(m.max(),1))
        out[ag]=dict(om=float(o.sum()/max(m.sum(),1)),peak_ratio=float(pr),width_ratio=float(wr))
    return out

def main():
    print("="*90); print("성인 과추정 진단 (baseline 0.6 대칭, Erlang I₃)"); print("="*90)
    t0=time.perf_counter(); C=build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
    ngm0=C["ngm"]
    out={"decompose":{},"gamma_sens":{},"imm_sens":{},"phi_sens":{}}

    # baseline states + confirmed
    states={s:rebuild_state(C,i,s,IMM0) for s,i in zip(SEAS,IDX)}

    # (1) 분해
    print("\n(1) obs/model = peak_ratio × width_ratio 분해 (성인):")
    print(f"  {'season':>10} | 18-44 om/pk/wid | 45-64 om/pk/wid")
    for s,i in zip(SEAS,IDX):
        R0,pi=sym[s]["R0"],sym[s]["pi"]; inc=sim(C,states[s],R0,pi,PHI0,ngm0)
        _,pred,mask=om_by_age(C,i,inc,GAMMA0); dec=decompose(C,i,pred,mask); out["decompose"][s]=dec
        a1=dec["18-44"]; a2=dec["45-64"]
        print(f"  {s:>10} | {a1['om']:.2f}/{a1['peak_ratio']:.2f}/{a1['width_ratio']:.2f} | {a2['om']:.2f}/{a2['peak_ratio']:.2f}/{a2['width_ratio']:.2f}")

    # (2) γ 성인 민감도 (adult idx4-12: 0.18 → 0.15, 0.12)
    print("\n(2) γ_report 성인(adult) 0.18 → {0.15,0.12}: 성인 obs/model (관측 조정):")
    print(f"  {'γ_adult':>8} | 18-44 mean | 45-64 mean")
    for gad in (0.18,0.15,0.12):
        gv=GAMMA0.copy(); gv[4:13]=gad
        o18=[]; o45=[]
        for s,i in zip(SEAS,IDX):
            R0,pi=sym[s]["R0"],sym[s]["pi"]; inc=sim(C,states[s],R0,pi,PHI0,ngm0)
            om,_,_=om_by_age(C,i,inc,gv); o18.append(om["18-44"]); o45.append(om["45-64"])
        out["gamma_sens"][gad]=dict(o1844=float(np.mean(o18)),o4564=float(np.mean(o45)))
        print(f"  {gad:>8.2f} | {np.mean(o18):.2f}       | {np.mean(o45):.2f}")

    # (3) R(0) 45-64 초기면역 (idx9-12) 0.45 → 0.55, 0.60 (동역학)
    print("\n(3) R(0) 45-64(idx9-12) 초기면역 0.45 → {0.55,0.60}: 성인 obs/model (동역학):")
    print(f"  {'imm_4564':>9} | 18-44 | 45-64")
    for im in (0.45,0.55,0.60):
        imm=IMM0.copy(); imm[9:13]=im
        o18=[]; o45=[]
        for s,i in zip(SEAS,IDX):
            R0,pi=sym[s]["R0"],sym[s]["pi"]; st=rebuild_state(C,i,s,imm); inc=sim(C,st,R0,pi,PHI0,ngm0)
            om,_,_=om_by_age(C,i,inc,GAMMA0); o18.append(om["18-44"]); o45.append(om["45-64"])
        out["imm_sens"][im]=dict(o1844=float(np.mean(o18)),o4564=float(np.mean(o45)))
        print(f"  {im:>9.2f} | {np.mean(o18):.2f}  | {np.mean(o45):.2f}")

    # (4) φ 성인(idx4-13) 1.0 → 0.8 (동역학; NGM도 바뀜 → ngm 재생성)
    print("\n(4) φ 성인(idx4-13) 1.0 → 0.8: 성인 obs/model (동역학, NGM 재계산):")
    print(f"  {'φ_adult':>8} | 18-44 | 45-64")
    for pha in (1.0,0.8):
        phi=PHI0.copy(); phi[4:14]=pha
        M=C["shared"]
        ngm=make_ngm_eigvalue_fn(pop_15=np.asarray(M["pop_15"]),rho=np.asarray(M["rho"]),
            C_home=np.asarray(M["C_home"]),C_work=np.asarray(M["C_work"]),C_school=np.asarray(M["C_school"]),C_other=np.asarray(M["C_other"]),
            R0_immunity=IMM0,gamma=float(M["gamma"]),seasonal_factor=1.0+F.S.AMP)
        o18=[]; o45=[]
        for s,i in zip(SEAS,IDX):
            R0,pi=sym[s]["R0"],sym[s]["pi"]; inc=sim(C,states[s],R0,pi,phi,ngm)
            om,_,_=om_by_age(C,i,inc,GAMMA0); o18.append(om["18-44"]); o45.append(om["45-64"])
        out["phi_sens"][pha]=dict(o1844=float(np.mean(o18)),o4564=float(np.mean(o45)))
        print(f"  {pha:>8.1f} | {np.mean(o18):.2f}  | {np.mean(o45):.2f}")

    # 재분배 영향: γ_adult=0.12로 낮췄을 때 성인↓ 유지? (γ는 관측이라 동역학 무관 → 유지 당연)
    print("\n(5) 재분배 영향: γ는 관측(동역학 무관) → 성인↓ 불변. imm/φ는 동역학→확인:")
    # imm 0.60에서 병가 p=0.4 재분배
    for label,imm,phi in [("imm_4564=0.60",{**{}},PHI0),("phi_adult=0.8",IMM0,None)]:
        pass  # summarized: dynamics changes tested qualitatively below
    # imm 0.60 재분배 체크
    imm=IMM0.copy(); imm[9:13]=0.60; nad=0
    for s,i in zip(SEAS,IDX):
        R0,pi=sym[s]["R0"],sym[s]["pi"]; st=rebuild_state(C,i,s,imm)
        base=C["H"]@np.asarray(sim(C,st,R0,pi,PHI0,ngm0,p_work=BASE)).sum(0)
        inc=C["H"]@np.asarray(sim(C,st,R0,pi,PHI0,ngm0,p_work=0.4)).sum(0)
        d=(inc-base)/C["pop6"]; nad+= all(d[HIRA_AGE_GROUPS.index(a)]<0 for a in ["18-44","45-64"])
    print(f"  imm_4564=0.60 하 병가 p=0.4 성인↓: {nad}/3")
    out["redist_imm060_adultdown"]=f"{nad}/3"

    (ED/"adult_overest.json").write_text(json.dumps(out,indent=2,default=float))
    print(f"\n[json] {ED/'adult_overest.json'}"); print("="*90)

if __name__=="__main__": main()
