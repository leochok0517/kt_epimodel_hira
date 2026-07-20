"""R(0) 성인 초기면역 재조정 (antigenic seniority) → 성인 과추정 개선. baseline 0.6 대칭.
면역 그리드 탐색 → 확정 → 3시즌 재fit. NGM+초기상태 둘 다 새 면역. γ/φ 불변.
Output: outputs/eda/immunity_refit.json + pres_*.png (ylim 100k)"""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})
from kt_data import SUDOGWON_SIDO_CODES
from kt_epimodel_hira.calibration.simple_model import estimate_initial_infected_from_hira, _build_initial_state_with_age_seed
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn, derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang import simulate_jax_erlang, daily_new_infection_by_age_erlang, split_seed_to_erlang
import final_symmetric_baseline as SB, final_pipeline_confirmed as F
ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"; FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS=SB.SEAS; IDX=SB.IDX; PHI=np.array(F.PHI); KAP=SB.KAP; BASE=0.6; GAMMA=np.array(SB.GAMMA)
ADULT=["18-44","45-64"]; CHILD=["0-5","6-11","12-17"]; AGE_C=SB.AGE_C; GRAY="#666"; MR="#B23A48"
sym=json.load(open(ED/"symmetric_baseline.json"))["fit"]

def immvec(i2044,i4564): return np.array([0.10]*4+[i2044]*5+[i4564]*4+[0.65]*2)

def make_ngm(C,imm):
    M=C["shared"]
    return make_ngm_eigvalue_fn(pop_15=np.asarray(M["pop_15"]),rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]),C_work=np.asarray(M["C_work"]),C_school=np.asarray(M["C_school"]),C_other=np.asarray(M["C_other"]),
        R0_immunity=imm,gamma=float(M["gamma"]),seasonal_factor=1.0+F.S.AMP)

def state(C,i,s,imm):
    sd=estimate_initial_infected_from_hira(s,C["pf"],sido_codes=list(SUDOGWON_SIDO_CODES),gamma_15_assumed=GAMMA)
    return jnp.asarray(_build_initial_state_with_age_seed(C["pf"],sd,seed_e_factor=0.5,initial_immunity=imm,initial_vaccinated_fraction=0.0))

def sim(C,st,R0,pi,ngm,p_school=BASE,p_work=BASE,sch_win=(-1e9,1e9),work_win=(-1e9,1e9)):
    beta=derive_beta_from_R0_simplex(ngm,jnp.asarray(R0),jnp.asarray(pi),jnp.asarray(PHI))
    kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
    kw["phi_susc"]=jnp.asarray(PHI); kw["kappa"]=jnp.asarray(KAP); kw["p_school"]=p_school; kw["p_work"]=p_work
    kw["policy_school_start_day"],kw["policy_school_end_day"]=sch_win
    kw["policy_work_start_day"],kw["policy_work_end_day"]=work_win
    kw["policy_school_baseline"]=BASE; kw["policy_work_baseline"]=BASE
    st2=simulate_jax_erlang(split_seed_to_erlang(st), **kw, discretize_time=False)
    return daily_new_infection_by_age_erlang(st2)

def omage(C,i,inc):
    pred=np.asarray(simulation_to_hira_by_age_jax(inc,jnp.asarray(GAMMA),n_weeks=52))
    obs=np.asarray(C["obs"][i]); w=np.asarray(C["w"][i]); mask=w.sum(1)>0
    return {ag:float(obs[mask,a].sum()/max(pred[mask,a].sum(),1.0)) for a,ag in enumerate(HIRA_AGE_GROUPS)}, pred

def fit_pi(C,i,ngm,st):
    obsj=jnp.asarray(C["obs"][i]); wj=jnp.asarray(C["w"][i])
    def loss(x):
        R0=jnp.exp(x[0]); pi=jax.nn.softmax(x[1:5]); beta=derive_beta_from_R0_simplex(ngm,R0,pi,jnp.asarray(PHI))
        kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
        kw["phi_susc"]=jnp.asarray(PHI); kw["kappa"]=jnp.asarray(KAP); kw["p_school"]=BASE; kw["p_work"]=BASE
        s2=simulate_jax_erlang(split_seed_to_erlang(st), **kw, discretize_time=False)
        pred=simulation_to_hira_by_age_jax(daily_new_infection_by_age_erlang(s2),jnp.asarray(GAMMA),n_weeks=C["nw"])
        c=x[1:5]-jnp.mean(x[1:5])
        return nb_nll_jax(obsj,pred,wj,concentration=x[5],min_rate=0.01)+0.5*jnp.sum((c-jnp.asarray(F.LOGIT_REF))**2/jnp.asarray(F.SIGMA_PIN)**2)
    lj=jax.jit(loss); gj=jax.jit(jax.grad(loss))
    def fg(xn):
        x=jnp.asarray(xn); v=float(lj(x)); g=np.array(gj(x))
        if not np.isfinite(v): v=1e15; g=np.where(np.isfinite(g),g,0.0)
        return v,g
    rng=np.random.default_rng(111+i); bounds=[F.LOG_R0_B]+[(-10,10)]*4+[F.PHI_NB_B]; best=None
    for k in range(10):
        x0=np.concatenate([[np.log(rng.uniform(1.8,2.5))],F.LOGIT_REF+rng.normal(0,0.5,4),[10.0]])
        try: r=minimize(fg,x0,jac=True,method="L-BFGS-B",bounds=bounds,options=dict(maxiter=400,ftol=1e-9,gtol=1e-6))
        except Exception: continue
        if best is None or r.fun<best.fun: best=r
    x=best.x; R0=float(np.exp(x[0])); pi=np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    return dict(R0=R0,pi=[float(p) for p in pi],beta_4=[float(b) for b in np.asarray(derive_beta_from_R0_simplex(ngm,jnp.asarray(R0),jnp.asarray(pi),jnp.asarray(PHI)))])

def main():
    print("="*90); print("R(0) 성인 초기면역 재조정 → 성인 과추정 개선 (baseline 0.6 대칭)"); print("="*90)
    t0=time.perf_counter(); C=SB.build(); pf=np.asarray(C["shared"]["pop_15"]); C["pf"]=pf.sum(1) if pf.ndim==2 else pf
    print(f"[setup] {time.perf_counter()-t0:.1f}s")

    # ── 탐색 (fixed sym R0/pi, 새 면역 NGM+state) ──
    print("\n(탐색) imm_2044 × imm_4564 → 성인 obs/model (fixed R0/pi 근사):")
    print(f"  {'i2044':>6} {'i4564':>6} | 18-44 | 45-64")
    grid=[]
    for i20 in (0.30,0.35,0.40):
        for i45 in (0.50,0.55,0.60):
            if i45<i20: continue
            imm=immvec(i20,i45); ngm=make_ngm(C,imm); o18=[];o45=[]
            for s,i in zip(SEAS,IDX):
                st=state(C,i,s,imm); inc=sim(C,st,sym[s]["R0"],sym[s]["pi"],ngm)
                om,_=omage(C,i,inc); o18.append(om["18-44"]); o45.append(om["45-64"])
            m18,m45=np.mean(o18),np.mean(o45); grid.append((i20,i45,m18,m45))
            print(f"  {i20:>6.2f} {i45:>6.2f} | {m18:.2f}  | {m45:.2f}")
    # 선택: 18-44,45-64 둘 다 1에 가장 근접 (|log| 합 최소, 단조 만족)
    best=min(grid,key=lambda g:abs(np.log(g[2]))+abs(np.log(g[3])))
    I2044,I4564=best[0],best[1]
    print(f"\n★ 확정 R(0): 20-44={I2044}, 45-64={I4564} (18-44~{best[2]:.2f}, 45-64~{best[3]:.2f} @fixed R0/pi)")
    IMM=immvec(I2044,I4564); ngm=make_ngm(C,IMM)
    print(f"  전체 R(0)_15: {[round(float(x),2) for x in IMM]}  (단조: {all(IMM[k]<=IMM[k+1]+1e-9 for k in range(14))})")

    # ── 재fit ──
    print("\n(재fit) 새 R(0)로 3시즌 π+R0:")
    t1={}; bp={}; preds={}; states={}
    for s,i in zip(SEAS,IDX):
        st=state(C,i,s,IMM); states[s]=st; f=fit_pi(C,i,ngm,st); bp[s]=(f["R0"],f["pi"])
        inc=sim(C,st,f["R0"],f["pi"],ngm); om,pred=omage(C,i,inc); preds[s]=pred
        omt=float(np.asarray(C["obs"][i])[np.asarray(C["w"][i]).sum(1)>0].sum()/max(pred[np.asarray(C["w"][i]).sum(1)>0].sum(),1))
        t1[s]=dict(**f,obs_model=om,om_total=omt)
        o=json.load(open(ED/"symmetric_baseline.json"))["fit"][s]["obs_model"]
        print(f"  {s}: R0={f['R0']:.3f} 18-44 {o['18-44']:.2f}→{om['18-44']:.2f}  45-64 {o['45-64']:.2f}→{om['45-64']:.2f}  om_tot={omt:.2f}")

    # ── 정책·재분배·배율 (baseline 0.6 대칭) ──
    def att6(inc): return C["H"]@np.asarray(inc).sum(0)
    def da(i,s,R0,pi,base6,**pol):
        inc=sim(C,states[s],R0,pi,ngm,**pol); inf6=att6(inc); d=(inf6-base6)/C["pop6"]
        return float(100*(base6.sum()-inf6.sum())/max(base6.sum(),1)), {ag:float(100*d[a]) for a,ag in enumerate(HIRA_AGE_GROUPS)}
    print("\n(정책) 학교 vs 병가 (p=0.4, baseline0.6) + 재분배:")
    sk=[];sc=[];nad=0; TERM=(70.0,113.0)
    da_sk=np.zeros(6); da_sc=np.zeros(6)
    for s,i in zip(SEAS,IDX):
        R0,pi=bp[s]; base=att6(sim(C,states[s],R0,pi,ngm))
        av,d1=da(i,s,R0,pi,base,p_work=0.4,work_win=TERM); sk.append(av); da_sk+=np.array([d1[a] for a in HIRA_AGE_GROUPS])/3
        av2,d2=da(i,s,R0,pi,base,p_school=0.4,sch_win=TERM); sc.append(av2); da_sc+=np.array([d2[a] for a in HIRA_AGE_GROUPS])/3
        _,dw=da(i,s,R0,pi,base,p_work=0.4)  # whole-season 성인↓ 체크
        nad+= all(dw[a]<0 for a in ADULT)
    print(f"  averted @p=0.4: 병가={np.mean(sk):.2f}% 학교={np.mean(sc):.2f}% → 배율 {np.mean(sc)/np.mean(sk):.1f}배")
    print(f"  Δattack 병가: "+" ".join(f"{ag}:{da_sk[a]:+.2f}" for a,ag in enumerate(HIRA_AGE_GROUPS)))
    print(f"  Δattack 학교: "+" ".join(f"{ag}:{da_sc[a]:+.2f}" for a,ag in enumerate(HIRA_AGE_GROUPS)))
    print(f"  ★ 재분배 성인↓(전기간 병가 p=0.4): {nad}/3")

    out=dict(meta=dict(immunity=IMM.tolist(),i2044=I2044,i4564=I4564,baseline=0.6),grid=[list(g) for g in grid],
             fit=t1,ratio=float(np.mean(sc)/np.mean(sk)),adult_down=f"{nad}/3",
             da_sick=da_sk.tolist(),da_school=da_sc.tolist())
    (ED/"immunity_refit.json").write_text(json.dumps(out,indent=2,default=float)); print(f"\n[json] {ED/'immunity_refit.json'}")

    # ── figures (ylim 100k) ──
    weeks=np.arange(52)
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for k,s in enumerate(SEAS):
        a=ax[k]; o=C["full_obs"][s].sum(1); a.plot(weeks,o,"o",color=GRAY,ms=3.5,alpha=0.7,label="데이터"); a.plot(weeks,preds[s].sum(1),"-",color=MR,lw=2,label="모델")
        a.set_title(f"{s}  R0={t1[s]['R0']:.2f}",fontsize=11,fontweight="bold"); a.set_xlabel("주차"); a.grid(alpha=0.25)
        a.text(0.03,0.86,f"om={t1[s]['om_total']:.2f}",transform=a.transAxes,fontsize=8,color="#333")
        if k==0: a.legend(fontsize=8); a.set_ylabel("주간 진료에피소드")
    fig.suptitle(f"3시즌 fit — R(0) 성인면역 재조정 (20-44={I2044},45-64={I4564})",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_fit_total.png",bbox_inches="tight"); plt.close(fig); print("[fig1]")
    fig,ax=plt.subplots(3,6,figsize=(16,7.5),sharex=True,sharey=True)
    for r,s in enumerate(SEAS):
        for c,ag in enumerate(HIRA_AGE_GROUPS):
            a=ax[r,c]; a.plot(weeks,C["full_obs"][s][:,c],"o",color=GRAY,ms=2,alpha=0.6); a.plot(weeks,preds[s][:,c],"-",color=AGE_C[c],lw=1.5)
            a.set_ylim(0,100000); a.grid(alpha=0.2); a.text(0.04,0.82,f"{t1[s]['obs_model'][ag]:.2f}",transform=a.transAxes,fontsize=7,color="#333")
            if r==0: a.set_title(f"{ag}세",fontsize=9,fontweight="bold")
            if c==0: a.set_ylabel(s,fontsize=8,fontweight="bold")
            a.tick_params(labelsize=6)
    fig.suptitle("3시즌 연령별 fit — R(0) 성인면역 재조정 (y통일 0~100k, 셀=obs/model)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"pres_fit_byage.png",bbox_inches="tight"); plt.close(fig); print("[fig2]")
    print("="*90)

if __name__=="__main__": main()
