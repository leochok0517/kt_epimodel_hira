"""FINAL presentation pipeline — confirmed params (Erlang I₃). 3 seasons.

CONFIRMED: baseline p_work=0.6, φ linear, γ_15 [0.40,0.40,0.25,0.18,0.18×9,0.25×2],
Erlang I₃ (3γ), κ 3-way, C(t) term↔vacation, v(t) norm, seed γ-fix.
Seasons 16-17,17-18,19-20, first_peak_only. Point estimate.

Tasks: (1) fit (per-season π + joint), (2) school-vs-sick symmetric, (3) policy
intensity, (4) holiday reversal, (5) 4 figures.
Outputs: outputs/eda/final_*.json + presentations/figures/pres_*.png
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
from kt_epimodel_hira.calibration.simple_model import estimate_initial_infected_from_hira, _build_initial_state_with_age_seed, R0_IMMUNITY_PROFILE
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang import simulate_jax_erlang, daily_new_infection_by_age_erlang, split_seed_to_erlang
import final_pipeline_confirmed as F

ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"; FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS=["2016-2017","2017-2018","2019-2020"]; IDX=[F.SEASONS.index(s) for s in SEAS]
GAMMA=jnp.asarray(np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25]))
KAPPA_BASE=np.array([0.29]*4+[0.30]*10+[0.0]); PHI=jnp.asarray(F.PHI)
TERM=(70.0,113.0); VAC=(113.0,183.0); WH=(-1e9,1e9)
CHILD=["0-5","6-11","12-17"]; SCHOOL_AG=["6-11","12-17"]; ADULT=["18-44","45-64"]
AGE_C=["#4575b4","#74add1","#fdae61","#f46d43","#d73027","#7b3294"]; GRAY="#666"; MR="#B23A48"


def build():
    C=F.build()
    pf=np.asarray(C["shared"]["pop_15"]); pf=pf.sum(1) if pf.ndim==2 else pf
    for s,i in zip(SEAS,IDX):   # seed γ-fix
        sd=estimate_initial_infected_from_hira(s,pf,sido_codes=list(SUDOGWON_SIDO_CODES),gamma_15_assumed=np.asarray(GAMMA))
        C["states"][i]=jnp.asarray(_build_initial_state_with_age_seed(pf,sd,seed_e_factor=0.5,initial_immunity=R0_IMMUNITY_PROFILE,initial_vaccinated_fraction=0.0))
    return C


def run(C,i,R0,pi,kappa=KAPPA_BASE,p_school=1.0,p_work=0.6,sch_win=WH,work_win=WH,work_base=1.0):
    beta=derive_beta_from_R0_simplex(C["ngm"],jnp.asarray(R0),jnp.asarray(pi),PHI)
    kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
    kw["phi_susc"]=PHI; kw["kappa"]=jnp.asarray(kappa); kw["p_school"]=p_school; kw["p_work"]=p_work
    kw["policy_school_start_day"],kw["policy_school_end_day"]=sch_win
    kw["policy_work_start_day"],kw["policy_work_end_day"]=work_win; kw["policy_work_baseline"]=work_base
    st=simulate_jax_erlang(split_seed_to_erlang(C["states"][i]), **kw, discretize_time=False)
    return daily_new_infection_by_age_erlang(st)


def pred_h(C,inc,nw=52): return np.asarray(simulation_to_hira_by_age_jax(jnp.asarray(inc),GAMMA,n_weeks=nw))
def att6(C,inc): return C["H"]@np.asarray(inc).sum(0)


def fit_pi(C,i):
    obsj=jnp.asarray(C["obs"][i]); wj=jnp.asarray(C["w"][i])
    def loss(x):
        R0=jnp.exp(x[0]); pi=jax.nn.softmax(x[1:5]); beta=derive_beta_from_R0_simplex(C["ngm"],R0,pi,PHI)
        kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]; kw["phi_susc"]=PHI; kw["kappa"]=jnp.asarray(KAPPA_BASE)
        st=simulate_jax_erlang(split_seed_to_erlang(C["states"][i]), **kw, discretize_time=False)
        pred=simulation_to_hira_by_age_jax(daily_new_infection_by_age_erlang(st),GAMMA,n_weeks=C["nw"])
        c=x[1:5]-jnp.mean(x[1:5])
        return nb_nll_jax(obsj,pred,wj,concentration=x[5],min_rate=0.01)+0.5*jnp.sum((c-jnp.asarray(F.LOGIT_REF))**2/jnp.asarray(F.SIGMA_PIN)**2)
    lj=jax.jit(loss); gj=jax.jit(jax.grad(loss))
    def fg(xn):
        x=jnp.asarray(xn); v=float(lj(x)); g=np.array(gj(x))
        if not np.isfinite(v): v=1e15; g=np.where(np.isfinite(g),g,0.0)
        return v,g
    rng=np.random.default_rng(91+i); bounds=[F.LOG_R0_B]+[(-10,10)]*4+[F.PHI_NB_B]; best=None
    for k in range(10):
        x0=np.concatenate([[np.log(rng.uniform(1.8,2.4))],F.LOGIT_REF+rng.normal(0,0.5,4),[10.0]])
        try: r=minimize(fg,x0,jac=True,method="L-BFGS-B",bounds=bounds,options=dict(maxiter=400,ftol=1e-9,gtol=1e-6))
        except Exception: continue
        if best is None or r.fun<best.fun: best=r
    x=best.x
    return dict(R0=float(np.exp(x[0])),pi=[float(p) for p in np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))],nll=float(best.fun))


def om_metrics(C,i,pred):
    obs=np.asarray(C["obs"][i]); w=np.asarray(C["w"][i]); mask=w.sum(1)>0; o=obs[mask].sum(1); m=pred[mask].sum(1)
    om={ag:float(obs[mask,a].sum()/max(pred[mask,a].sum(),1.0)) for a,ag in enumerate(HIRA_AGE_GROUPS)}
    return om, float(o.sum()/max(m.sum(),1.0)), float((o.sum()/max(o.max(),1))/(m.sum()/max(m.max(),1)))


def dattack(C,i,R0,pi,base6,**pol):
    inc=run(C,i,R0,pi,**pol); inf6=att6(C,inc); d=(inf6-base6)/C["pop6"]
    av=100.0*(base6.sum()-inf6.sum())/max(base6.sum(),1.0)
    return float(av), {ag:float(100*d[a]) for a,ag in enumerate(HIRA_AGE_GROUPS)}


def main():
    print("="*94); print("FINAL PRESENTATION PIPELINE — confirmed (Erlang I₃, baseline 0.6)"); print("="*94)
    t0=time.perf_counter(); C=build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")

    # ── TASK 1 ──
    print("\n── T1: 3시즌 fit ──")
    t1={}; bp={}; preds={}
    for s,i in zip(SEAS,IDX):
        f=fit_pi(C,i); bp[s]=(f["R0"],f["pi"]); pr=pred_h(C,run(C,i,f["R0"],f["pi"])); preds[s]=pr
        om,omt,wr=om_metrics(C,i,pr); t1[s]=dict(**f,obs_model=om,om_total=omt,width_ratio=wr)
        print(f"  {s}: R0={f['R0']:.3f} π=[{f['pi'][0]:.2f},{f['pi'][1]:.2f},{f['pi'][2]:.2f},{f['pi'][3]:.2f}] om_tot={omt:.2f} 12-17={om['12-17']:.2f} wid={wr:.2f}")
    pim=np.mean([t1[s]["pi"] for s in SEAS],0)
    print(f"  12-17 obs/model mean={np.mean([t1[s]['obs_model']['12-17'] for s in SEAS]):.2f}  supp공통π~mean=[{pim[0]:.2f},{pim[1]:.2f},{pim[2]:.2f},{pim[3]:.2f}]")
    (ED/"final_fit_confirmed.json").write_text(json.dumps(dict(meta=dict(baseline=0.6,stages="I3",gamma=np.asarray(GAMMA).tolist()),per_season=t1,pi_mean=pim.tolist()),indent=2,default=float))

    # ── TASK 2: school vs sick ──
    print("\n── T2: 학교 vs 병가 대칭 (term창, p×μ) ──")
    P=[1.0,0.8,0.6,0.4,0.2]; MU=[0.5,1.0,1.5]; t2={}
    for s,i in zip(SEAS,IDX):
        R0,pi=bp[s]; rec={"sick":{},"school":{}}
        for mu in MU:
            kap=mu*KAPPA_BASE
            base=att6(C,run(C,i,R0,pi,kappa=kap,p_school=1.0,p_work=0.6))  # baseline: presenteeism 0.6
            for p in P:
                av,da=dattack(C,i,R0,pi,base,kappa=kap,p_school=1.0,p_work=p,work_win=TERM,work_base=0.6)
                rec["sick"][f"{mu}_{p}"]=dict(av=av,da=da)
                av2,da2=dattack(C,i,R0,pi,base,kappa=kap,p_school=p,p_work=0.6,sch_win=TERM,work_base=1.0)
                rec["school"][f"{mu}_{p}"]=dict(av=av2,da=da2)
        t2[s]=rec
    # summary at p=0.4, mu=1.0 (avg over seasons)
    sk=np.mean([t2[s]["sick"]["1.0_0.4"]["av"] for s in SEAS]); sc=np.mean([t2[s]["school"]["1.0_0.4"]["av"] for s in SEAS])
    print(f"  averted @p=0.4,μ=1 (3시즌평균): 병가={sk:+.2f}% 학교결석={sc:+.2f}%  → 학교/병가 = {sc/sk if sk else 0:.1f}배")
    da_sk=np.mean([[t2[s]["sick"]["1.0_0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
    da_sc=np.mean([[t2[s]["school"]["1.0_0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
    print("  Δattack 병가 : "+" ".join(f"{ag}:{da_sk[a]:+.2f}" for a,ag in enumerate(HIRA_AGE_GROUPS)))
    print("  Δattack 학교 : "+" ".join(f"{ag}:{da_sc[a]:+.2f}" for a,ag in enumerate(HIRA_AGE_GROUPS)))
    (ED/"final_school_vs_sick.json").write_text(json.dumps(dict(meta=dict(P=P,MU=MU,window="term",baseline=0.6),results=t2),indent=2,default=float))

    # ── TASK 3: policy intensity ──
    print("\n── T3: 병가 강도 {0.6,0.4,0.2,0.0} 전기간 ──")
    PL=[0.6,0.4,0.2,0.0]; t3={}
    for s,i in zip(SEAS,IDX):
        R0,pi=bp[s]; base=att6(C,run(C,i,R0,pi,p_work=0.6)); pr={}
        for p in PL:
            av,da=dattack(C,i,R0,pi,base,p_work=p)
            adown=all(da[a]<0 for a in ADULT); pr[p]=dict(av=av,da=da,adult_down=bool(adown))
        t3[s]=pr
    print("  averted% by intensity (행=시즌):  " + " ".join(f"p={p}" for p in PL))
    for s in SEAS: print(f"    {s}: "+" ".join(f"{t3[s][p]['av']:+6.2f}" for p in PL))
    nad=sum(1 for s in SEAS for p in [0.4,0.2,0.0] if t3[s][p]["adult_down"])
    print(f"  ★ 성인↓: {nad}/{len(SEAS)*3} (개입강도 전부)")
    (ED/"final_policy_intensity.json").write_text(json.dumps(dict(meta=dict(levels=PL,baseline=0.6),results=t3),indent=2,default=float))

    # ── TASK 4: holiday reversal ──
    print("\n── T4: 방학 부호반전 (강도별 term vs vac) ──")
    t4={}; INT=[0.4,0.2,0.0]
    for s,i in zip(SEAS,IDX):
        R0,pi=bp[s]; base=att6(C,run(C,i,R0,pi,p_work=0.6)); rec={}
        for p in INT:
            _,dt=dattack(C,i,R0,pi,base,p_work=p,work_win=TERM,work_base=0.6)
            _,dv=dattack(C,i,R0,pi,base,p_work=p,work_win=VAC,work_base=0.6)
            cst=sum(dt[c] for c in CHILD); csv=sum(dv[c] for c in CHILD)
            rec[p]=dict(term_child=cst,vac_child=csv,term=dt,vac=dv,reversal=bool(cst<0 and csv>0))
        t4[s]=rec
    for p in INT:
        nr=sum(1 for s in SEAS if t4[s][p]["reversal"])
        print(f"  p_work={p}: 부호반전 {nr}/3  vac_child=["+",".join(f"{t4[s][p]['vac_child']:+.2f}" for s in SEAS)+"]")
    (ED/"final_holiday_reversal.json").write_text(json.dumps(dict(meta=dict(intensities=INT,baseline=0.6),results=t4),indent=2,default=float))

    # ── TASK 5: figures ──
    print("\n── T5: figures ──")
    weeks=np.arange(52)
    # fig1 total
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for k,s in enumerate(SEAS):
        a=ax[k]; o=C["full_obs"][s].sum(1)
        a.plot(weeks,o,"o",color=GRAY,ms=3.5,alpha=0.7,label="데이터"); a.plot(weeks,preds[s].sum(1),"-",color=MR,lw=2,label="모델(Erlang I₃)")
        a.set_title(f"{s}  R0={t1[s]['R0']:.2f}",fontsize=11,fontweight="bold"); a.set_xlabel("주차"); a.grid(alpha=0.25)
        a.text(0.03,0.86,f"om={t1[s]['om_total']:.2f}\nwid={t1[s]['width_ratio']:.2f}",transform=a.transAxes,fontsize=8,color="#333")
        if k==0: a.legend(fontsize=8); a.set_ylabel("주간 진료에피소드")
    fig.suptitle("3시즌 유행곡선 — 데이터 vs 모델 (확정 파라미터, Erlang I₃)",fontsize=13,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_fit_total.png",bbox_inches="tight"); plt.close(fig); print("  [fig1] pres_fit_total.png")
    # fig2 byage
    fig,ax=plt.subplots(3,6,figsize=(16,7.5),sharex=True)
    for r,s in enumerate(SEAS):
        for c,ag in enumerate(HIRA_AGE_GROUPS):
            a=ax[r,c]; a.plot(weeks,C["full_obs"][s][:,c],"o",color=GRAY,ms=2,alpha=0.6); a.plot(weeks,preds[s][:,c],"-",color=AGE_C[c],lw=1.5)
            a.grid(alpha=0.2); a.text(0.04,0.8,f"{t1[s]['obs_model'][ag]:.2f}",transform=a.transAxes,fontsize=7,color="#333")
            if r==0: a.set_title(f"{ag}세",fontsize=9,fontweight="bold")
            if c==0: a.set_ylabel(s,fontsize=8,fontweight="bold")
            a.tick_params(labelsize=6)
    fig.suptitle("3시즌 연령별 fitting — 확정 파라미터 (셀=obs/model)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"pres_fit_byage.png",bbox_inches="tight"); plt.close(fig); print("  [fig2] pres_fit_byage.png")
    # fig3 school vs sick
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
    xs=[1-p for p in P]  # 정책강도
    a1.plot(xs,[np.mean([t2[s]["sick"][f"1.0_{p}"]["av"] for s in SEAS]) for p in P],"o-",color="#2166AC",lw=2,label="병가(p_work)")
    a1.plot(xs,[np.mean([t2[s]["school"][f"1.0_{p}"]["av"] for s in SEAS]) for p in P],"s-",color="#B2182B",lw=2,label="학교결석(p_school)")
    a1.set_xlabel("정책강도 (1−p)"); a1.set_ylabel("averted % (3시즌 평균)"); a1.set_title("학교 vs 병가 — averted (μ=1, term창)",fontsize=11,fontweight="bold"); a1.legend(); a1.grid(alpha=0.3)
    x=np.arange(6); bw=0.38
    a2.bar(x-bw/2,da_sk,bw,color="#2166AC",label="병가",edgecolor="k",lw=0.4); a2.bar(x+bw/2,da_sc,bw,color="#B2182B",label="학교결석",edgecolor="k",lw=0.4)
    a2.axhline(0,color="k",lw=1); a2.set_xticks(x); a2.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS],rotation=30,fontsize=8)
    a2.set_ylabel("Δattack (%pt)"); a2.set_title("연령별 직격 (p=0.4, μ=1): 병가→성인, 학교→학령기",fontsize=11,fontweight="bold"); a2.legend(); a2.grid(axis="y",alpha=0.3)
    fig.suptitle("학교결석 vs 병가 — 정책 효과·연령 직격 (확정 파라미터)",fontsize=13,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_school_vs_sick.png",bbox_inches="tight"); plt.close(fig); print("  [fig3] pres_school_vs_sick.png")
    # fig4 policy intensity age Δattack (avg 3 seasons)
    fig,ax=plt.subplots(1,1,figsize=(10,5.5)); x=np.arange(6); PL2=[0.4,0.2,0.0]; cols={0.4:"#fdae61",0.2:"#f46d43",0.0:"#a50026"}; bw=0.26
    for j,p in enumerate(PL2):
        vals=np.mean([[t3[s][p]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
        ax.bar(x+(j-1)*bw,vals,bw,color=cols[p],label=f"p_work={p} (강도{int((0.6-p)/0.6*100)}%)",edgecolor="k",lw=0.4)
    ax.axhline(0,color="k",lw=1.2); ax.set_xticks(x); ax.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS]); ax.set_ylabel("Δattack (%pt)")
    ax.set_title("병가 강도별 연령 Δattack (baseline 0.6 대비, 3시즌 평균, 전 기간)\n성인 최대 감소 · 전 연령 이득(재분배 없음)",fontsize=12,fontweight="bold")
    ax.legend(); ax.grid(axis="y",alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG/"pres_policy_intensity.png",bbox_inches="tight"); plt.close(fig); print("  [fig4] pres_policy_intensity.png")
    print("="*94)


if __name__=="__main__": main()
