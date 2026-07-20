"""FINAL v3 — 최종 확정 파라미터 (R(0) 성인면역 반영). 발표 최종본. point est.

확정: baseline p_work=p_school=0.6, φ선형, γ_15[0.40,0.40,0.25,0.18,0.18×9,0.25×2],
Erlang I₃, κ3, C(t), v(t), seed γ-fix, ★R(0)=[0.10×4,0.40×5,0.60×4,0.65×2].
시즌 16-17,17-18,19-20, first_peak, μ=1.0.

1 fit / 2 학교vs병가 / 3 rate vs number / 4 병가강도+재분배 / 5 방학반전 / 6 그림5.
Output: outputs/eda/v3_*.json + pres_*.png
"""
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
import final_pipeline_confirmed as F
ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"; FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS=["2016-2017","2017-2018","2019-2020"]; IDX=[F.SEASONS.index(s) for s in SEAS]
PHI=np.array(F.PHI); KAP=np.array([0.29]*4+[0.30]*10+[0.0]); BASE=0.6
GAMMA=np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM=np.array([0.10]*4+[0.40]*5+[0.60]*4+[0.65]*2)   # ★ 확정 성인면역
TERM=(70.0,113.0); VAC=(113.0,183.0); WH=(-1e9,1e9)
CHILD=["0-5","6-11","12-17"]; ADULT=["18-44","45-64"]; AGE_C=["#4575b4","#74add1","#fdae61","#f46d43","#d73027","#7b3294"]; GRAY="#666"; MR="#B23A48"

def build():
    C=F.build(); pf=np.asarray(C["shared"]["pop_15"]); C["pf"]=pf.sum(1) if pf.ndim==2 else pf
    M=C["shared"]
    C["ngm3"]=make_ngm_eigvalue_fn(pop_15=np.asarray(M["pop_15"]),rho=np.asarray(M["rho"]),
        C_home=np.asarray(M["C_home"]),C_work=np.asarray(M["C_work"]),C_school=np.asarray(M["C_school"]),C_other=np.asarray(M["C_other"]),
        R0_immunity=IMM,gamma=float(M["gamma"]),seasonal_factor=1.0+F.S.AMP)
    C["st"]={}
    for s,i in zip(SEAS,IDX):
        sd=estimate_initial_infected_from_hira(s,C["pf"],sido_codes=list(SUDOGWON_SIDO_CODES),gamma_15_assumed=GAMMA)
        C["st"][s]=jnp.asarray(_build_initial_state_with_age_seed(C["pf"],sd,seed_e_factor=0.5,initial_immunity=IMM,initial_vaccinated_fraction=0.0))
    return C

def sim(C,s,R0,pi,p_school=BASE,p_work=BASE,sch_win=WH,work_win=WH):
    beta=derive_beta_from_R0_simplex(C["ngm3"],jnp.asarray(R0),jnp.asarray(pi),jnp.asarray(PHI))
    kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
    kw["phi_susc"]=jnp.asarray(PHI); kw["kappa"]=jnp.asarray(KAP); kw["p_school"]=p_school; kw["p_work"]=p_work
    kw["policy_school_start_day"],kw["policy_school_end_day"]=sch_win; kw["policy_work_start_day"],kw["policy_work_end_day"]=work_win
    kw["policy_school_baseline"]=BASE; kw["policy_work_baseline"]=BASE
    st=simulate_jax_erlang(split_seed_to_erlang(C["st"][s]), **kw, discretize_time=False)
    return daily_new_infection_by_age_erlang(st)

def pred_h(C,inc): return np.asarray(simulation_to_hira_by_age_jax(jnp.asarray(inc),jnp.asarray(GAMMA),n_weeks=52))
def att6(C,inc): return C["H"]@np.asarray(inc).sum(0)

def fit(C,s,i):
    obsj=jnp.asarray(C["obs"][i]); wj=jnp.asarray(C["w"][i])
    def loss(x):
        R0=jnp.exp(x[0]); pi=jax.nn.softmax(x[1:5]); beta=derive_beta_from_R0_simplex(C["ngm3"],R0,pi,jnp.asarray(PHI))
        kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
        kw["phi_susc"]=jnp.asarray(PHI); kw["kappa"]=jnp.asarray(KAP); kw["p_school"]=BASE; kw["p_work"]=BASE
        st=simulate_jax_erlang(split_seed_to_erlang(C["st"][s]), **kw, discretize_time=False)
        pred=simulation_to_hira_by_age_jax(daily_new_infection_by_age_erlang(st),jnp.asarray(GAMMA),n_weeks=C["nw"])
        c=x[1:5]-jnp.mean(x[1:5])
        return nb_nll_jax(obsj,pred,wj,concentration=x[5],min_rate=0.01)+0.5*jnp.sum((c-jnp.asarray(F.LOGIT_REF))**2/jnp.asarray(F.SIGMA_PIN)**2)
    lj=jax.jit(loss); gj=jax.jit(jax.grad(loss))
    def fg(xn):
        x=jnp.asarray(xn); v=float(lj(x)); g=np.array(gj(x))
        if not np.isfinite(v): v=1e15; g=np.where(np.isfinite(g),g,0.0)
        return v,g
    rng=np.random.default_rng(121+i); bounds=[F.LOG_R0_B]+[(-10,10)]*4+[F.PHI_NB_B]; best=None
    for k in range(10):
        x0=np.concatenate([[np.log(rng.uniform(1.8,2.5))],F.LOGIT_REF+rng.normal(0,0.5,4),[10.0]])
        try: r=minimize(fg,x0,jac=True,method="L-BFGS-B",bounds=bounds,options=dict(maxiter=400,ftol=1e-9,gtol=1e-6))
        except Exception: continue
        if best is None or r.fun<best.fun: best=r
    x=best.x; R0=float(np.exp(x[0])); pi=np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))
    return dict(R0=R0,pi=[float(p) for p in pi],beta_4=[float(b) for b in np.asarray(derive_beta_from_R0_simplex(C["ngm3"],jnp.asarray(R0),jnp.asarray(pi),jnp.asarray(PHI)))],nll=float(best.fun))

def omw(C,i,pred):
    obs=np.asarray(C["obs"][i]); w=np.asarray(C["w"][i]); mask=w.sum(1)>0; o=obs[mask].sum(1); m=pred[mask].sum(1)
    om={ag:float(obs[mask,a].sum()/max(pred[mask,a].sum(),1.0)) for a,ag in enumerate(HIRA_AGE_GROUPS)}
    return om, float(o.sum()/max(m.sum(),1)), float((o.sum()/max(o.max(),1))/(m.sum()/max(m.max(),1)))

def dattack(C,s,i,R0,pi,base6,**pol):
    inc=sim(C,s,R0,pi,**pol); inf6=att6(C,inc); d=(inf6-base6)/C["pop6"]
    return float(100*(base6.sum()-inf6.sum())/max(base6.sum(),1)), {ag:float(100*d[a]) for a,ag in enumerate(HIRA_AGE_GROUPS)}

def main():
    print("="*94); print("FINAL v3 — 최종 확정 (R(0) 성인면역 0.40/0.60), baseline 0.6 대칭, Erlang I₃"); print("="*94)
    t0=time.perf_counter(); C=build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")

    # 1 fit
    print("\n[1] fit:")
    t1={}; bp={}; preds={}
    for s,i in zip(SEAS,IDX):
        f=fit(C,s,i); bp[s]=(f["R0"],f["pi"]); pr=pred_h(C,sim(C,s,f["R0"],f["pi"])); preds[s]=pr; om,omt,wr=omw(C,i,pr)
        t1[s]=dict(**f,obs_model=om,om_total=omt,width_ratio=wr)
        print(f"  {s}: R0={f['R0']:.3f} om={omt:.2f} wid={wr:.2f} | 18-44={om['18-44']:.2f} 45-64={om['45-64']:.2f} 12-17={om['12-17']:.2f}")
    pim=np.mean([t1[s]["pi"] for s in SEAS],0)
    (ED/"v3_fit.json").write_text(json.dumps(dict(meta=dict(imm=IMM.tolist(),gamma=GAMMA.tolist(),baseline=0.6),per_season=t1,pi_joint_mean=pim.tolist()),indent=2,default=float))

    # 2 school vs sick
    print("\n[2] 학교 vs 병가 (term, baseline 0.6):")
    P=[0.6,0.4,0.2,0.0]; t2={}
    for s,i in zip(SEAS,IDX):
        R0,pi=bp[s]; base=att6(C,sim(C,s,R0,pi)); rec={"sick":{},"school":{}}
        for p in P:
            av,d=dattack(C,s,i,R0,pi,base,p_work=p,work_win=TERM); rec["sick"][str(p)]=dict(av=av,da=d)
            av2,d2=dattack(C,s,i,R0,pi,base,p_school=p,sch_win=TERM); rec["school"][str(p)]=dict(av=av2,da=d2)
        t2[s]=rec
    sk=np.mean([t2[s]["sick"]["0.4"]["av"] for s in SEAS]); sc=np.mean([t2[s]["school"]["0.4"]["av"] for s in SEAS])
    da_sk=np.mean([[t2[s]["sick"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
    da_sc=np.mean([[t2[s]["school"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
    print(f"  averted@p=0.4: 병가={sk:.2f}% 학교={sc:.2f}% → 배율={sc/sk:.1f}배 (방향견고/크기 가정의존)")
    print(f"  병가 Δ: "+" ".join(f"{HIRA_AGE_GROUPS[a]}:{da_sk[a]:+.2f}" for a in range(6)))
    print(f"  학교 Δ: "+" ".join(f"{HIRA_AGE_GROUPS[a]}:{da_sc[a]:+.2f}" for a in range(6)))
    (ED/"v3_school_vs_sick.json").write_text(json.dumps(dict(meta=dict(P=P),results=t2,ratio=float(sc/sk)),indent=2,default=float))

    # 3 rate vs number
    pop6=np.asarray(C["pop6"]); num_sk=-da_sk/100*pop6; num_sc=-da_sc/100*pop6
    print("\n[3] rate vs number (p=0.4):")
    print(f"  rate 최대: 병가={HIRA_AGE_GROUPS[np.argmin(da_sk)]} 학교={HIRA_AGE_GROUPS[np.argmin(da_sc)]}")
    print(f"  number 최대: 병가={HIRA_AGE_GROUPS[np.argmax(num_sk)]}({num_sk.max():.0f}명) 학교={HIRA_AGE_GROUPS[np.argmax(num_sc)]}({num_sc.max():.0f}명)")
    (ED/"v3_rate_vs_number.json").write_text(json.dumps(dict(pop6=pop6.tolist(),rate_sick=da_sk.tolist(),rate_school=da_sc.tolist(),num_sick=num_sk.tolist(),num_school=num_sc.tolist()),indent=2,default=float))

    # 4 policy intensity + redistribution
    print("\n[4] 병가 강도 + 재분배:")
    PL=[0.6,0.4,0.2,0.0]; t4={}; nad=0
    for s,i in zip(SEAS,IDX):
        R0,pi=bp[s]; base=att6(C,sim(C,s,R0,pi)); pr={}
        for p in PL:
            av,d=dattack(C,s,i,R0,pi,base,p_work=p); ad=all(d[a]<0 for a in ADULT); pr[str(p)]=dict(av=av,da=d,adult_down=bool(ad))
            if p in (0.4,0.2,0.0) and ad: nad+=1
        t4[s]=pr
    print("  averted%: "+" | ".join(f"{s}:"+",".join(f"{t4[s][str(p)]['av']:+.1f}" for p in PL) for s in SEAS))
    print(f"  ★ 성인↓ {nad}/9")
    (ED/"v3_policy_intensity.json").write_text(json.dumps(dict(meta=dict(levels=PL),results=t4,adult_down=f"{nad}/9"),indent=2,default=float))

    # 5 holiday reversal
    print("\n[5] 방학 부호반전:")
    t5={}; INT=[0.4,0.2,0.0]
    for s,i in zip(SEAS,IDX):
        R0,pi=bp[s]; base=att6(C,sim(C,s,R0,pi)); rec={}
        for p in INT:
            _,dt=dattack(C,s,i,R0,pi,base,p_work=p,work_win=TERM); _,dv=dattack(C,s,i,R0,pi,base,p_work=p,work_win=VAC)
            cst=sum(dt[c] for c in CHILD); csv=sum(dv[c] for c in CHILD); rec[str(p)]=dict(tc=cst,vc=csv,vac=dv,rev=bool(cst<0 and csv>0))
        t5[s]=rec
    for p in INT:
        nr=sum(1 for s in SEAS if t5[s][str(p)]["rev"]); print(f"  p={p}: 반전 {nr}/3  vac_child=["+",".join(f"{t5[s][str(p)]['vc']:+.2f}" for s in SEAS)+"]")
    (ED/"v3_holiday.json").write_text(json.dumps(dict(meta=dict(intensities=INT),results=t5),indent=2,default=float))

    # 6 figures
    print("\n[6] figures:")
    weeks=np.arange(52)
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for k,s in enumerate(SEAS):
        a=ax[k]; a.plot(weeks,C["full_obs"][s].sum(1),"o",color=GRAY,ms=3.5,alpha=0.7,label="데이터"); a.plot(weeks,preds[s].sum(1),"-",color=MR,lw=2,label="모델")
        a.set_title(f"{s}  R0={t1[s]['R0']:.2f}",fontsize=11,fontweight="bold"); a.set_xlabel("주차"); a.grid(alpha=0.25); a.text(0.03,0.86,f"om={t1[s]['om_total']:.2f}",transform=a.transAxes,fontsize=8,color="#333")
        if k==0: a.legend(fontsize=8); a.set_ylabel("주간 진료에피소드")
    fig.suptitle("3시즌 fit — 최종 확정 (R(0) 성인면역 0.40/0.60, Erlang I₃, baseline 0.6)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_fit_total.png",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(3,6,figsize=(16,7.5),sharex=True,sharey=True)
    for r,s in enumerate(SEAS):
        for c,ag in enumerate(HIRA_AGE_GROUPS):
            a=ax[r,c]; a.plot(weeks,C["full_obs"][s][:,c],"o",color=GRAY,ms=2,alpha=0.6); a.plot(weeks,preds[s][:,c],"-",color=AGE_C[c],lw=1.5)
            a.set_ylim(0,100000); a.grid(alpha=0.2); a.text(0.04,0.82,f"{t1[s]['obs_model'][ag]:.2f}",transform=a.transAxes,fontsize=7,color="#333")
            if r==0: a.set_title(f"{ag}세",fontsize=9,fontweight="bold")
            if c==0: a.set_ylabel(s,fontsize=8,fontweight="bold")
            a.tick_params(labelsize=6)
    fig.suptitle("3시즌 연령별 fit — 최종 확정 (y통일 0~100k, 셀=obs/model)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"pres_fit_byage.png",bbox_inches="tight"); plt.close(fig)
    # school vs sick
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5)); xs=[BASE-p for p in P]
    a1.plot(xs,[np.mean([t2[s]["sick"][str(p)]["av"] for s in SEAS]) for p in P],"o-",color="#2166AC",lw=2,ms=7,label="병가")
    a1.plot(xs,[np.mean([t2[s]["school"][str(p)]["av"] for s in SEAS]) for p in P],"s-",color="#B2182B",lw=2,ms=7,label="학교결석")
    a1.axhline(0,color="k",lw=0.8,alpha=0.5); a1.annotate(f"×{sc/sk:.1f}배\n(방향견고,\n크기 가정의존)",xy=(0.2,sc),xytext=(0.24,sc-2.2),fontsize=9,fontweight="bold",color="#B2182B")
    a1.set_xlabel("p 감소량 (공통 baseline 0.6)"); a1.set_ylabel("averted % (3시즌평균)"); a1.set_title("학교 vs 병가 averted (μ=1, term)",fontsize=11,fontweight="bold"); a1.legend(fontsize=9); a1.grid(alpha=0.3)
    x=np.arange(6); bw=0.38
    a2.bar(x-bw/2,da_sk,bw,color="#2166AC",label="병가",edgecolor="k",lw=0.4); a2.bar(x+bw/2,da_sc,bw,color="#B2182B",label="학교결석",edgecolor="k",lw=0.4)
    a2.axhline(0,color="k",lw=1); a2.set_xticks(x); a2.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS],rotation=30,fontsize=8); a2.set_ylabel("Δattack (%pt)"); a2.set_title("연령 직격: 병가→성인, 학교→학령기",fontsize=11,fontweight="bold"); a2.legend(fontsize=9); a2.grid(axis="y",alpha=0.3)
    fig.suptitle("학교결석 vs 병가 — 최종 (대칭 baseline, 새 면역)",fontsize=13,fontweight="bold"); fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_school_vs_sick.png",bbox_inches="tight"); plt.close(fig)
    # rate vs number
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5)); x=np.arange(6); bw=0.38
    a1.bar(x-bw/2,-da_sk,bw,color="#2166AC",label="병가",edgecolor="k",lw=0.4); a1.bar(x+bw/2,-da_sc,bw,color="#B2182B",label="학교결석",edgecolor="k",lw=0.4)
    a1.set_xticks(x); a1.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS],rotation=30,fontsize=8); a1.set_ylabel("감소 (%pt)"); a1.set_title("rate (%pt) — 학령기 최대",fontsize=11,fontweight="bold"); a1.legend(fontsize=9); a1.grid(axis="y",alpha=0.3)
    a2.bar(x-bw/2,num_sk,bw,color="#2166AC",label="병가",edgecolor="k",lw=0.4); a2.bar(x+bw/2,num_sc,bw,color="#B2182B",label="학교결석",edgecolor="k",lw=0.4)
    a2.set_xticks(x); a2.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS],rotation=30,fontsize=8); a2.set_ylabel("감소 감염 수 (명)"); a2.set_title("number (명) — 성인(18-44) 최대",fontsize=11,fontweight="bold"); a2.legend(fontsize=9); a2.grid(axis="y",alpha=0.3)
    for xi in range(6): a2.text(xi-bw/2,num_sk[xi],f"{num_sk[xi]/1000:.0f}k",ha="center",va="bottom",fontsize=6.5,color="#2166AC"); a2.text(xi+bw/2,num_sc[xi],f"{num_sc[xi]/1000:.0f}k",ha="center",va="bottom",fontsize=6.5,color="#B2182B")
    fig.suptitle("★ rate vs number 역전 — rate는 학령기, number는 성인 지배 (p=0.4)",fontsize=13,fontweight="bold"); fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(FIG/"pres_rate_vs_number.png",bbox_inches="tight"); plt.close(fig)
    # policy intensity
    fig,axx=plt.subplots(1,1,figsize=(10,5.5)); x=np.arange(6); P2=[0.4,0.2,0.0]; cols={0.4:"#fdae61",0.2:"#f46d43",0.0:"#a50026"}; bw=0.26
    for j,p in enumerate(P2):
        vals=np.mean([[t4[s][str(p)]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0); axx.bar(x+(j-1)*bw,vals,bw,color=cols[p],label=f"p_work={p}",edgecolor="k",lw=0.4)
    axx.axhline(0,color="k",lw=1.2); axx.set_xticks(x); axx.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS]); axx.set_ylabel("Δattack (%pt)"); axx.set_title("병가 강도별 연령 Δattack (baseline 0.6, 성인↓ 9/9)",fontsize=12,fontweight="bold"); axx.legend(); axx.grid(axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"pres_policy_intensity.png",bbox_inches="tight"); plt.close(fig)
    # holiday
    labels=[f"{s[2:4]}-{s[7:9]}" for s in SEAS]; PS=0.0
    term=np.array([t5[s][str(PS)]["tc"] for s in SEAS]); vac=np.array([t5[s][str(PS)]["vc"] for s in SEAS])
    fig,axx=plt.subplots(figsize=(9,5)); x=np.arange(3); bw=0.38
    axx.bar(x-bw/2,term,bw,color="#2166AC",label="학기중 창",edgecolor="k"); axx.bar(x+bw/2,vac,bw,color="#B2182B",label="방학중 창",edgecolor="k")
    axx.axhline(0,color="k",lw=1.5); axx.set_xticks(x); axx.set_xticklabels(labels); axx.set_ylabel("아동(0-17) Δattack 합 (%pt)")
    nr=sum(1 for s in SEAS if t5[s][str(PS)]["rev"])
    axx.set_title(f"방학 부호반전 — 붕괴 확정 ({nr}/3, 강개입 p_work=0.0에서도, 대칭 baseline)\n방학중도 아동 이득(음수) → '방학 역효과' 주장 철회",fontsize=11,fontweight="bold"); axx.legend(); axx.grid(axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"pres_holiday.png",bbox_inches="tight"); plt.close(fig)
    print("  pres_fit_total/byage, school_vs_sick, rate_vs_number, policy_intensity, holiday")
    print("="*94)

if __name__=="__main__": main()
