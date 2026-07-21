"""viz_holiday_reversal_6seasons.png 만 재생성 (문구 수정본). 6시즌 fit + 방학 holiday만.
데이터도 outputs/eda/regen_holiday.json 에 저장 → 향후 문구 수정은 이 JSON에서 즉시 재생성 가능."""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
import final_pipeline_confirmed as FP
ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"; FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS=FP.SEASONS; IDX=list(range(len(SEAS))) if False else [FP.MJ.SEASONS.index(s) for s in SEAS]
PSHOW=0.2; CHILD=["0-5","6-11","12-17"]; BLUE="#2166AC"; RED="#B23A48"; JSON=ED/"regen_holiday.json"

def compute():
    t0=time.perf_counter(); C=FP.build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
    hol={}
    for i,s in enumerate(SEAS):
        f=FP.fit_season_pi(C,i); R0,pi=f["R0"],f["pi"]
        base=FP.attack6(C,FP.run_inc(C,i,R0,pi,p_work=FP.BASE_PWORK))
        def da(work_win):
            inc=FP.run_inc(C,i,R0,pi,p_work=PSHOW,work_win=work_win,work_base=FP.BASE_PWORK)
            d=(FP.attack6(C,inc)-base)/C["pop6"]; return {ag:float(100*d[a]) for a,ag in enumerate(HIRA_AGE_GROUPS)}
        dt=da(FP.TERM_WIN); dv=da(FP.VAC_WIN)
        hol[s]=dict(term=dict(all=dt,child_sum=sum(dt[c] for c in CHILD)),
                    vacation=dict(all=dv,child_sum=sum(dv[c] for c in CHILD)))
        print(f"  {s}: term_child={hol[s]['term']['child_sum']:+.3f} vac_child={hol[s]['vacation']['child_sum']:+.3f}")
    JSON.write_text(json.dumps(dict(meta=dict(PSHOW=PSHOW,baseline=FP.BASE_PWORK),results=hol),indent=2,default=float))
    return hol

def draw(hol):
    labels=[f"{s[2:4]}-{s[7:9]}" for s in SEAS]
    term=np.array([hol[s]["term"]["child_sum"] for s in SEAS]); vac=np.array([hol[s]["vacation"]["child_sum"] for s in SEAS])
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8.2)); x=np.arange(len(SEAS)); bw=0.38
    # ── 문구 수정본 ──
    ax1.bar(x-bw/2,term,bw,color=BLUE,label="학기",edgecolor="k",lw=0.5)          # legend: 학기
    ax1.bar(x+bw/2,vac,bw,color=RED,label="방학",edgecolor="k",lw=0.5)            # legend: 방학
    ax1.axhline(0,color="k",lw=1.8)
    for xi,(tv,vv) in enumerate(zip(term,vac)):
        ax1.text(xi-bw/2,tv-0.006,f"{tv:+.2f}",ha="center",va="top",fontsize=7.5,color=BLUE)
        ax1.text(xi+bw/2,vv+0.006,f"{vv:+.2f}",ha="center",va="bottom",fontsize=7.5,color=RED)
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylabel("아동(0–17) Δattack 합 (%pt)")
    ax1.set_title(f"병가 아동영향: 학기중(파랑) vs 방학중(빨강) 개입 p_work={PSHOW}",fontsize=11.5,fontweight="bold")
    ax1.legend(loc="upper right"); ax1.grid(axis="y",alpha=0.3); ax1.margins(y=0.18)
    age_c={"0-5":"#9ecae1","6-11":"#ef6548","12-17":"#b30000"}; bw2=0.26
    for j,ag in enumerate(CHILD):
        vals=np.array([hol[s]["vacation"]["all"][ag] for s in SEAS]); ax2.bar(x+(j-1)*bw2,vals,bw2,color=age_c[ag],label=f"{ag}세",edgecolor="k",lw=0.4)
    ax2.axhline(0,color="k",lw=1.5); ax2.set_xticks(x); ax2.set_xticklabels(labels); ax2.set_ylabel("방학중 Δattack (%pt)")
    ax2.set_title("방학중 연령별 변화",fontsize=11.5,fontweight="bold")
    ax2.legend(loc="upper left",ncol=3); ax2.grid(axis="y",alpha=0.3); ax2.margins(y=0.15)
    fig.suptitle(f"학기중과 방학 비교 (p_work={PSHOW})",fontsize=13,fontweight="bold",y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(FIG/"viz_holiday_reversal_6seasons.png",bbox_inches="tight"); plt.close(fig)
    print(f"[fig] viz_holiday_reversal_6seasons.png")

if __name__=="__main__":
    hol=json.load(open(JSON))["results"] if (JSON.exists() and os.environ.get("REDRAW")) else compute()
    draw(hol)
