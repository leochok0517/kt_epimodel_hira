"""발표 그림만 재생 (재계산 없음): JSON 결과 로드 + fit곡선은 forward 시뮬(3개)만.
수정: (1) byage y축 통일, (2) μ=1.0 단일, (3) 그림3 좌 baseline 정렬."""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
import final_presentation_pipeline as FP

ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"; FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS=FP.SEAS; IDX=FP.IDX; AGE_C=FP.AGE_C; GRAY="#666"; MR="#B23A48"
fitj=json.load(open(ED/"final_fit_confirmed.json"))["per_season"]
svj=json.load(open(ED/"final_school_vs_sick.json"))["results"]
pij=json.load(open(ED/"final_policy_intensity.json"))["results"]

t0=time.perf_counter(); C=FP.build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
weeks=np.arange(52)
preds={s: FP.pred_h(C, FP.run(C,i,fitj[s]["R0"],fitj[s]["pi"])) for s,i in zip(SEAS,IDX)}

# ── FIG2: byage, y축 통일 ──
gmax=max(max(C["full_obs"][s][:,c].max(), preds[s][:,c].max()) for s in SEAS for c in range(6))
YMAX=float(np.ceil(gmax/10000)*10000)   # 통일 상한 (전 패널 max 수용, 클리핑 방지)
fig,ax=plt.subplots(3,6,figsize=(16,7.5),sharex=True,sharey=True)
for r,s in enumerate(SEAS):
    for c,ag in enumerate(HIRA_AGE_GROUPS):
        a=ax[r,c]; a.plot(weeks,C["full_obs"][s][:,c],"o",color=GRAY,ms=2,alpha=0.6); a.plot(weeks,preds[s][:,c],"-",color=AGE_C[c],lw=1.5)
        a.set_ylim(0,YMAX); a.grid(alpha=0.2); a.text(0.04,0.82,f"{fitj[s]['obs_model'][ag]:.2f}",transform=a.transAxes,fontsize=7,color="#333")
        if r==0: a.set_title(f"{ag}세",fontsize=9,fontweight="bold")
        if c==0: a.set_ylabel(s,fontsize=8,fontweight="bold")
        a.tick_params(labelsize=6)
fig.suptitle(f"3시즌 연령별 fitting — 확정 파라미터 (y축 통일 0~{YMAX/1000:.0f}k, 셀=obs/model)",fontsize=12.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"pres_fit_byage.png",bbox_inches="tight"); plt.close(fig); print(f"[fig2] pres_fit_byage.png (ylim 0~{YMAX:.0f})")

# ── FIG3: school vs sick, μ=1.0, baseline 정렬 (각자 baseline부터 x=p감소량) ──
P=[1.0,0.8,0.6,0.4,0.2]
sick_x=[0.6-p for p in P if p<=0.6]                     # baseline 0.6: p={0.6,0.4,0.2}→x=0,0.2,0.4
sick_y=[np.mean([svj[s]["sick"][f"1.0_{p}"]["av"] for s in SEAS]) for p in P if p<=0.6]
sch_x=[1.0-p for p in P]                                 # baseline 1.0
sch_y=[np.mean([svj[s]["school"][f"1.0_{p}"]["av"] for s in SEAS]) for p in P]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
a1.plot(sick_x,sick_y,"o-",color="#2166AC",lw=2,ms=7,label="병가 (baseline p_work=0.6)")
a1.plot(sch_x,sch_y,"s-",color="#B2182B",lw=2,ms=7,label="학교결석 (baseline p_school=1.0)")
a1.axhline(0,color="k",lw=0.8,alpha=0.5)
sk4=np.mean([svj[s]["sick"]["1.0_0.4"]["av"] for s in SEAS]); sc4=np.mean([svj[s]["school"]["1.0_0.4"]["av"] for s in SEAS])
a1.annotate(f"×{sc4/sk4:.1f}배",xy=(0.4,sc4),xytext=(0.28,sc4-2),fontsize=11,fontweight="bold",color="#B2182B")
a1.set_xlabel("p 감소량 (자기 baseline 대비)"); a1.set_ylabel("averted % (3시즌 평균)")
a1.set_title("학교 vs 병가 — averted (μ=1, term창)",fontsize=11,fontweight="bold"); a1.legend(fontsize=9); a1.grid(alpha=0.3)
da_sk=np.mean([[svj[s]["sick"]["1.0_0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
da_sc=np.mean([[svj[s]["school"]["1.0_0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
x=np.arange(6); bw=0.38
a2.bar(x-bw/2,da_sk,bw,color="#2166AC",label="병가",edgecolor="k",lw=0.4); a2.bar(x+bw/2,da_sc,bw,color="#B2182B",label="학교결석",edgecolor="k",lw=0.4)
a2.axhline(0,color="k",lw=1); a2.set_xticks(x); a2.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS],rotation=30,fontsize=8)
a2.set_ylabel("Δattack (%pt)"); a2.set_title("연령별 직격 (p=0.4, μ=1): 병가→성인, 학교→학령기",fontsize=11,fontweight="bold"); a2.legend(fontsize=9); a2.grid(axis="y",alpha=0.3)
fig.suptitle("학교결석 vs 병가 — 정책 효과·연령 직격 (확정 파라미터, μ=1.0)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_school_vs_sick.png",bbox_inches="tight"); plt.close(fig); print("[fig3] pres_school_vs_sick.png")

# ── FIG4: policy intensity, μ=1.0 (이미 μ=1) ──
fig,ax=plt.subplots(1,1,figsize=(10,5.5)); x=np.arange(6); PL=[0.4,0.2,0.0]; cols={0.4:"#fdae61",0.2:"#f46d43",0.0:"#a50026"}; bw=0.26
for j,p in enumerate(PL):
    vals=np.mean([[pij[s][str(p)]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
    ax.bar(x+(j-1)*bw,vals,bw,color=cols[p],label=f"p_work={p} (강도{int((0.6-p)/0.6*100)}%)",edgecolor="k",lw=0.4)
ax.axhline(0,color="k",lw=1.2); ax.set_xticks(x); ax.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS]); ax.set_ylabel("Δattack (%pt)")
ax.set_title("병가 강도별 연령 Δattack (baseline 0.6 대비, 3시즌 평균, 전 기간, μ=1.0)\n성인 최대 감소 · 전 연령 이득(재분배 없음)",fontsize=12,fontweight="bold")
ax.legend(); ax.grid(axis="y",alpha=0.3)
fig.tight_layout(); fig.savefig(FIG/"pres_policy_intensity.png",bbox_inches="tight"); plt.close(fig); print("[fig4] pres_policy_intensity.png")
print("DONE")
