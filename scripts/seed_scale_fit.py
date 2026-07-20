"""seed_scale 도입 — 초기 과소보고 보정계수. 공통 scale + 시즌별 π/R0 동시 fit.

근거: seed = HIRA 첫3주/γ_report. 유행 초기 과소보고(낮은 care-seeking·검사율)로
실제 초기감염 과소포착 → seed_scale (초기 포착률 1/scale) 보정. 3시즌 공통 1개.

Erlang(3), baseline 0.6, 확정 파라미터(φ 선형, γ 12-17=0.18, κ 3-way). 연령분포·
E:I·Erlang분배는 데이터 유지, 전체 크기만 scale. 초기상태를 jax로 구성(미분가능),
jax_model 무수정. Output: outputs/eda/seed_scale_fit.json + viz_fit_seedscale_total.png
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
from kt_epimodel_hira.calibration.simple_model import estimate_initial_infected_from_hira, R0_IMMUNITY_PROFILE
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS, simulation_to_hira_by_age_jax, nb_nll_jax
from kt_epimodel_hira.jax_model.numpyro_model import derive_beta_from_R0_simplex
from kt_epimodel_hira.jax_model.erlang import simulate_jax_erlang, daily_new_infection_by_age_erlang
import final_pipeline_confirmed as F

ED = Path(__file__).resolve().parent.parent/"outputs"/"eda"
FIGDIR = Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS = ["2016-2017","2017-2018","2019-2020"]; N=len(SEAS)
IDX = [F.SEASONS.index(s) for s in SEAS]
GAMMA_NEW = np.array([0.40,0.40,0.25,0.18]+[0.18]*9+[0.25,0.25])
IMM = np.asarray(R0_IMMUNITY_PROFILE); SCALE_B=(1.0,4.0); GRAY="#666"; MR="#B23A48"


def build_init7(pop_flat, seed, scale):
    I = seed*scale; E = I*0.5; R = IMM*pop_flat; S = pop_flat - I - E - R
    third = I/3.0
    return jnp.stack([S, jnp.zeros_like(S), E, third, third, third, R], axis=0)[:, :, None]


def run(C, init7, R0, pi, phi, p_work=0.6, work_win=(-1e9,1e9), work_base=1.0):
    beta = derive_beta_from_R0_simplex(C["ngm"], R0, pi, phi)
    kw=dict(C["shared"]); kw["beta_h"],kw["beta_w"],kw["beta_s"],kw["beta_o"]=beta[0],beta[1],beta[2],beta[3]
    kw["phi_susc"]=phi; kw["p_school"]=1.0; kw["p_work"]=p_work
    kw["policy_work_start_day"],kw["policy_work_end_day"]=work_win; kw["policy_work_baseline"]=work_base
    st=simulate_jax_erlang(init7, **kw, discretize_time=False)
    return daily_new_infection_by_age_erlang(st)


def main():
    print("="*90); print("SEED_SCALE fit (공통) + Erlang(3) + 확정파라미터, baseline 0.6"); print("="*90)
    t0=time.perf_counter(); C=F.build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
    pop_flat = np.asarray(C["shared"]["pop_15"]); pop_flat = pop_flat.sum(1) if pop_flat.ndim==2 else pop_flat
    seeds = {s: np.asarray(estimate_initial_infected_from_hira(s, pop_flat, sido_codes=list(SUDOGWON_SIDO_CODES),
             gamma_15_assumed=GAMMA_NEW)) for s in SEAS}
    obsj=[jnp.asarray(C["obs"][i]) for i in IDX]; wj=[jnp.asarray(C["w"][i]) for i in IDX]
    phi=jnp.asarray(F.PHI); pf=jnp.asarray(pop_flat)
    seedj=[jnp.asarray(seeds[s]) for s in SEAS]

    def unpack(x):
        scale=x[0]; lr0=x[1:1+N]; lpi=x[1+N:1+N+4*N].reshape(N,4); pnb=x[1+N+4*N]
        return scale,lr0,lpi,pnb
    def loss(x):
        scale,lr0,lpi,pnb=unpack(x); tot=0.0
        for k in range(N):
            R0=jnp.exp(lr0[k]); pi=jax.nn.softmax(lpi[k])
            init7=build_init7(pf, seedj[k], scale)
            inc=run(C, init7, R0, pi, phi)
            pred=simulation_to_hira_by_age_jax(inc, jnp.asarray(GAMMA_NEW), n_weeks=C["nw"])
            centered=lpi[k]-jnp.mean(lpi[k])
            tot=tot+nb_nll_jax(obsj[k],pred,wj[k],concentration=pnb,min_rate=0.01) \
                + 0.5*jnp.sum((centered-jnp.asarray(F.LOGIT_REF))**2/jnp.asarray(F.SIGMA_PIN)**2)
        return tot
    lj=jax.jit(loss); gj=jax.jit(jax.grad(loss))
    def fg(xn):
        x=jnp.asarray(xn); v=float(lj(x)); g=np.array(gj(x))
        if not np.isfinite(v): v=1e15; g=np.where(np.isfinite(g),g,0.0)
        return v,g
    rng=np.random.default_rng(71)
    bounds=[SCALE_B]+[F.LOG_R0_B]*N+[(-10,10)]*(4*N)+[F.PHI_NB_B]
    best=None; scales=[]
    for it in range(10):
        x0=np.concatenate([[rng.uniform(1.2,3.0)], np.log(rng.uniform(1.8,2.4,N)),
                           np.tile(F.LOGIT_REF,N)+rng.normal(0,0.5,4*N), [10.0]])
        try:
            r=minimize(fg,x0,jac=True,method="L-BFGS-B",bounds=bounds,options=dict(maxiter=600,ftol=1e-9,gtol=1e-6))
        except Exception: continue
        scales.append(float(r.x[0]))
        if best is None or r.fun<best.fun: best=r
    scale,lr0,lpi,pnb=unpack(best.x)
    scale=float(scale); scale_std=float(np.std(scales))
    print(f"\n★ seed_scale = {scale:.3f} (multistart std {scale_std:.3f}, 범위 {SCALE_B})  → 초기 포착률 ~1/{scale:.1f}={1/scale:.2f}")
    rail = "⚠ 상한 railing(4.0) → 공통 부족, 시즌별 필요" if abs(scale-4.0)<0.05 else ("⚠ 하한(1.0)" if abs(scale-1.0)<0.02 else "범위 내")
    print(f"  {rail}")

    res={}; preds={}
    old=json.load(open(ED/"seed_gamma_fix.json"))["results"]
    print(f"\n  {'season':>10} {'R0':>6} | peak mdl/obs(구seed→scale) | width_ratio | om_total | 성인↓")
    for k,(s,i) in enumerate(zip(SEAS,IDX)):
        R0=float(np.exp(lr0[k])); pi=np.asarray(jax.nn.softmax(jnp.asarray(lpi[k])))
        init7=build_init7(pf, seedj[k], scale)
        pred=np.asarray(simulation_to_hira_by_age_jax(run(C,init7,jnp.asarray(R0),jnp.asarray(pi),phi), jnp.asarray(GAMMA_NEW), n_weeks=52))
        preds[s]=pred
        obs=np.asarray(C["obs"][i]); w=np.asarray(C["w"][i]); mask=w.sum(1)>0; wk=np.where(mask)[0]
        o=obs[mask].sum(1); m=pred[mask].sum(1)
        opk=int(wk[np.argmax(o)]); mpk=int(wk[np.argmax(m)])
        owid=o.sum()/max(o.max(),1); mwid=m.sum()/max(m.max(),1)
        om={ag:float(obs[mask,a].sum()/max(pred[mask,a].sum(),1.0)) for a,ag in enumerate(HIRA_AGE_GROUPS)}
        omt=float(o.sum()/max(m.sum(),1.0))
        # redistribution
        base=C["H"]@np.asarray(run(C,init7,jnp.asarray(R0),jnp.asarray(pi),phi,p_work=0.6)).sum(0)
        inc2=run(C,init7,jnp.asarray(R0),jnp.asarray(pi),phi,p_work=0.4)
        d=(C["H"]@np.asarray(inc2).sum(0)-base)/C["pop6"]
        adown=all(100*d[HIRA_AGE_GROUPS.index(a)]<0 for a in ["18-44","45-64"])
        res[s]=dict(R0=R0,pi=pi.tolist(),mdl_pk=mpk,obs_pk=opk,width_ratio=float(owid/mwid),om_total=omt,om_age=om,adult_down=bool(adown))
        print(f"  {s:>10} {R0:>6.3f} | {old[s]['mdl_pk']}/{old[s]['obs_pk']} → {mpk}/{opk} | {res[s]['width_ratio']:.2f} | {omt:.2f} | {adown}")

    te=sum(abs(res[s]['mdl_pk']-res[s]['obs_pk']) for s in SEAS)
    te_old=sum(abs(old[s]['mdl_pk']-old[s]['obs_pk']) for s in SEAS)
    print(f"\n★ 타이밍오차합: γ-fix(scale=1)={te_old}주 → seed_scale={te}주")
    print(f"★ 12-17 obs/model mean = {np.mean([res[s]['om_age']['12-17'] for s in SEAS]):.2f}")
    print(f"★ width_ratio mean = {np.mean([res[s]['width_ratio'] for s in SEAS]):.2f} (seed_scale은 폭 무관 예상)")
    print("="*90)
    (ED/"seed_scale_fit.json").write_text(json.dumps(dict(
        meta=dict(baseline=0.6,stages=3,gamma=GAMMA_NEW.tolist(),scale_bounds=SCALE_B,seed_source="HIRA first3wk/confirmedγ"),
        seed_scale=scale, seed_scale_std=scale_std, railing=rail, results=res,
        timing_err=te, timing_err_gammafix=te_old), indent=2, default=float))
    print(f"[json] {ED/'seed_scale_fit.json'}")

    weeks=np.arange(52)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for k,s in enumerate(SEAS):
        ax=axes[k]; o=C["full_obs"][s].sum(1)
        ax.plot(weeks,o,"o",color=GRAY,ms=3.5,alpha=0.7,label="데이터")
        ax.plot(weeks,preds[s].sum(1),"-",color=MR,lw=2,label="모델(Erlang+seed_scale)")
        ax.axvline(res[s]['obs_pk'],color="#888",ls=":",lw=1); ax.axvline(res[s]['mdl_pk'],color=MR,ls=":",lw=1)
        ax.set_title(f"{s}  R0={res[s]['R0']:.2f}",fontsize=11,fontweight="bold"); ax.set_xlabel("주차"); ax.grid(alpha=0.25)
        ax.text(0.03,0.85,f"peak mdl{res[s]['mdl_pk']}/obs{res[s]['obs_pk']}\nom={res[s]['om_total']:.2f}",transform=ax.transAxes,fontsize=8,color="#333")
        if k==0: ax.legend(fontsize=8); ax.set_ylabel("주간 진료에피소드")
    fig.suptitle(f"Erlang(3) + seed_scale={scale:.2f} — 3시즌 타이밍 (baseline 0.6, 확정파라미터)",fontsize=13,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIGDIR/"viz_fit_seedscale_total.png",bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {FIGDIR/'viz_fit_seedscale_total.png'}")

if __name__=="__main__": main()
