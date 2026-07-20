"""E₂I₃ (E Erlang(2) + I Erlang(3)) fit — 3 seasons. Does E-Erlang sharpen further?
확정 파라미터, baseline 0.6, seed γ-fix. Compare width_ratio/timing vs I3-only.
Output: outputs/eda/erlang_E_fit.json + viz_fit_erlangE_total.png"""
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
from kt_epimodel_hira.jax_model.erlang import simulate_jax_erlang_E2I3, daily_new_infection_by_age_erlang_E2I3, split_seed_to_erlang_E2I3
import final_pipeline_confirmed as F
ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"; FIGDIR=Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS=["2016-2017","2017-2018","2019-2020"]; IDX=[F.SEASONS.index(s) for s in SEAS]
GAMMA_NEW=jnp.asarray(np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25]))
GRAY="#666"; PUR="#762a83"; GRN="#1a9850"; ADULT=["18-44","45-64"]

def run(C,i,R0,pi,phi,p_work=0.6):
    beta=derive_beta_from_R0_simplex(C["ngm"],jnp.asarray(R0),jnp.asarray(pi),phi)
    kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
    kw["phi_susc"]=phi; kw["p_school"]=1.0; kw["p_work"]=p_work
    st=simulate_jax_erlang_E2I3(split_seed_to_erlang_E2I3(C["states"][i]), **kw, discretize_time=False)
    return daily_new_infection_by_age_erlang_E2I3(st)

def fit(C,i):
    obsj=jnp.asarray(C["obs"][i]); wj=jnp.asarray(C["w"][i]); phi=jnp.asarray(F.PHI)
    def loss(x):
        R0=jnp.exp(x[0]); pi=jax.nn.softmax(x[1:5])
        beta=derive_beta_from_R0_simplex(C["ngm"],R0,pi,phi)
        kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]; kw["phi_susc"]=phi
        st=simulate_jax_erlang_E2I3(split_seed_to_erlang_E2I3(C["states"][i]), **kw, discretize_time=False)
        pred=simulation_to_hira_by_age_jax(daily_new_infection_by_age_erlang_E2I3(st), GAMMA_NEW, n_weeks=C["nw"])
        c=x[1:5]-jnp.mean(x[1:5])
        return nb_nll_jax(obsj,pred,wj,concentration=x[5],min_rate=0.01)+0.5*jnp.sum((c-jnp.asarray(F.LOGIT_REF))**2/jnp.asarray(F.SIGMA_PIN)**2)
    lj=jax.jit(loss); gj=jax.jit(jax.grad(loss))
    def fg(xn):
        x=jnp.asarray(xn); v=float(lj(x)); g=np.array(gj(x))
        if not np.isfinite(v): v=1e15; g=np.where(np.isfinite(g),g,0.0)
        return v,g
    rng=np.random.default_rng(81+i); bounds=[F.LOG_R0_B]+[(-10,10)]*4+[F.PHI_NB_B]; best=None
    for k in range(10):
        x0=np.concatenate([[np.log(rng.uniform(1.8,2.4))],F.LOGIT_REF+rng.normal(0,0.5,4),[10.0]])
        try: r=minimize(fg,x0,jac=True,method="L-BFGS-B",bounds=bounds,options=dict(maxiter=400,ftol=1e-9,gtol=1e-6))
        except Exception: continue
        if best is None or r.fun<best.fun: best=r
    x=best.x
    return dict(R0=float(np.exp(x[0])),pi=[float(p) for p in np.asarray(jax.nn.softmax(jnp.asarray(x[1:5])))])

def metrics(C,i,pred):
    obs=np.asarray(C["obs"][i]); w=np.asarray(C["w"][i]); mask=w.sum(1)>0; wk=np.where(mask)[0]
    o=obs[mask].sum(1); m=pred[mask].sum(1)
    om={ag:float(obs[mask,a].sum()/max(pred[mask,a].sum(),1.0)) for a,ag in enumerate(HIRA_AGE_GROUPS)}
    return dict(obs_pk=int(wk[np.argmax(o)]),mdl_pk=int(wk[np.argmax(m)]),
                width_ratio=float((o.sum()/max(o.max(),1))/(m.sum()/max(m.max(),1))),
                om_total=float(o.sum()/max(m.sum(),1.0)),om_age=om)

def main():
    print("="*88); print("E₂I₃ (E Erlang2 + I Erlang3) fit — 3 seasons, baseline 0.6"); print("="*88)
    t0=time.perf_counter(); C=F.build()
    # seed γ-fix
    pf=np.asarray(C["shared"]["pop_15"]); pf=pf.sum(1) if pf.ndim==2 else pf
    for s,i in zip(SEAS,IDX):
        sd=estimate_initial_infected_from_hira(s,pf,sido_codes=list(SUDOGWON_SIDO_CODES),gamma_15_assumed=np.asarray(GAMMA_NEW))
        C["states"][i]=jnp.asarray(_build_initial_state_with_age_seed(pf,sd,seed_e_factor=0.5,initial_immunity=R0_IMMUNITY_PROFILE,initial_vaccinated_fraction=0.0))
    print(f"[setup] {time.perf_counter()-t0:.1f}s")
    old=json.load(open(ED/"erlang_fit.json"))["results"]  # I3-only
    phi=jnp.asarray(F.PHI); res={}; preds={}
    for s,i in zip(SEAS,IDX):
        f=fit(C,i); pred=np.asarray(simulation_to_hira_by_age_jax(run(C,i,f["R0"],f["pi"],phi),GAMMA_NEW,n_weeks=52)); preds[s]=pred
        me=metrics(C,i,pred)
        base=C["H"]@np.asarray(run(C,i,f["R0"],f["pi"],phi,p_work=0.6)).sum(0)
        inc2=run(C,i,f["R0"],f["pi"],phi,p_work=0.4); d=(C["H"]@np.asarray(inc2).sum(0)-base)/C["pop6"]
        adown=all(100*d[HIRA_AGE_GROUPS.index(a)]<0 for a in ADULT)
        res[s]=dict(R0=f["R0"],**me,adult_down=bool(adown))
        oi=old[s]["erlang"]
        print(f"\n  [{s}]  R0 I3={oi['R0']:.3f}→E2I3={f['R0']:.3f}")
        print(f"    width_ratio: I3={oi['width_ratio']:.2f} → E2I3={me['width_ratio']:.2f}  (과교정>1.1? {me['width_ratio']>1.1})")
        print(f"    peak mdl/obs: I3={oi['mdl_pk']}/{oi['obs_pk']} → E2I3={me['mdl_pk']}/{me['obs_pk']}")
        print(f"    obs/model total: I3={oi['om_total']:.2f} → E2I3={me['om_total']:.2f}   45-64={me['om_age']['45-64']:.2f} 65+={me['om_age']['65+']:.2f}")
        print(f"    재분배 성인↓={adown}")
    wr_i3=np.mean([old[s]["erlang"]["width_ratio"] for s in SEAS]); wr_e=np.mean([res[s]["width_ratio"] for s in SEAS])
    te=sum(abs(res[s]["mdl_pk"]-res[s]["obs_pk"]) for s in SEAS); te_i3=sum(abs(old[s]["erlang"]["mdl_pk"]-old[s]["erlang"]["obs_pk"]) for s in SEAS)
    print("\n"+"="*88)
    print(f"★ width_ratio 평균: I3={wr_i3:.2f} → E2I3={wr_e:.2f}   타이밍오차합: I3={te_i3} → E2I3={te}주   재분배 {sum(res[s]['adult_down'] for s in SEAS)}/3")
    print("="*88)
    (ED/"erlang_E_fit.json").write_text(json.dumps(dict(meta=dict(baseline=0.6,E_stages=2,I_stages=3),results=res,width_ratio_I3=float(wr_i3),width_ratio_E2I3=float(wr_e)),indent=2,default=float))
    print(f"[json] {ED/'erlang_E_fit.json'}")
    # figure: single vs I3 vs E2I3 vs data
    old_full=json.load(open(ED/"final_fit.json"))["per_season"]
    weeks=np.arange(52); fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for k,s in enumerate(SEAS):
        i=IDX[k]; ax=axes[k]; o=C["full_obs"][s].sum(1)
        ps=F.pred_h(C,F.run_inc(C,i,old_full[s]["R0"],old_full[s]["pi"])).sum(1)  # single I (note: uses gammaNEW? final uses GAMMA15 same)
        ax.plot(weeks,o,"o",color=GRAY,ms=3,alpha=0.65,label="데이터")
        ax.plot(weeks,ps,"-",color="#B23A48",lw=1.2,alpha=0.6,label="단일 I")
        ax.plot(weeks,preds[s].sum(1),"-",color=PUR,lw=2,label="E₂I₃")
        ax.set_title(f"{s}",fontsize=11,fontweight="bold"); ax.set_xlabel("주차"); ax.grid(alpha=0.25)
        ax.text(0.03,0.82,f"wid I3{old[s]['erlang']['width_ratio']:.2f}\n→E2I3{res[s]['width_ratio']:.2f}\npk{res[s]['mdl_pk']}/{res[s]['obs_pk']}",transform=ax.transAxes,fontsize=8,color="#333")
        if k==0: ax.legend(fontsize=8); ax.set_ylabel("주간 진료에피소드")
    fig.suptitle("E₂I₃ vs 단일 I — 유행 폭·타이밍 (3시즌, baseline 0.6)",fontsize=13,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIGDIR/"viz_fit_erlangE_total.png",bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {FIGDIR/'viz_fit_erlangE_total.png'}")

if __name__=="__main__": main()
