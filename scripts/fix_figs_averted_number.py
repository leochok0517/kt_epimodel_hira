"""그림 수정: (1) byage ylim 90k, (2) averted number(명) 그림 추가. 재계산 없음(JSON+fwd sim)."""
import os, json, time
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np, jax; jax.config.update("jax_enable_x64",True); jax.devices()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
import final_symmetric_baseline as SB
ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"; FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
SEAS=SB.SEAS; IDX=SB.IDX
d=json.load(open(ED/"symmetric_baseline.json")); fit=d["fit"]; sv=d["school_vs_sick"]
t0=time.perf_counter(); C=SB.build(); print(f"[setup] {time.perf_counter()-t0:.1f}s")
pop6=np.asarray(C["pop6"]); weeks=np.arange(52)
preds={s:SB.pred_h(C,SB.run(C,i,fit[s]["R0"],fit[s]["pi"])) for s,i in zip(SEAS,IDX)}

# (1) byage ylim 90k
gmax=max(max(C["full_obs"][s][:,c].max(),preds[s][:,c].max()) for s in SEAS for c in range(6))
YMAX=90000.0
fig,ax=plt.subplots(3,6,figsize=(16,7.5),sharex=True,sharey=True)
for r,s in enumerate(SEAS):
    for c,ag in enumerate(HIRA_AGE_GROUPS):
        a=ax[r,c]; a.plot(weeks,C["full_obs"][s][:,c],"o",color="#666",ms=2,alpha=0.6); a.plot(weeks,preds[s][:,c],"-",color=SB.AGE_C[c],lw=1.5)
        a.set_ylim(0,YMAX); a.grid(alpha=0.2); a.text(0.04,0.82,f"{fit[s]['obs_model'][ag]:.2f}",transform=a.transAxes,fontsize=7,color="#333")
        if r==0: a.set_title(f"{ag}세",fontsize=9,fontweight="bold")
        if c==0: a.set_ylabel(s,fontsize=8,fontweight="bold")
        a.tick_params(labelsize=6)
fig.suptitle(f"3시즌 연령별 fit — 대칭 baseline (y통일 0~90k, 셀=obs/model)  [global max={gmax/1000:.0f}k]",fontsize=12.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"pres_fit_byage.png",bbox_inches="tight"); plt.close(fig); print(f"[byage] ylim 90k (max={gmax:.0f})")

# (2) averted NUMBER: Δattack(%pt)/100 × pop = 감소 인원(명). p=0.4, 병가 vs 학교
da_sk=np.mean([[sv[s]["sick"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)   # %pt
da_sc=np.mean([[sv[s]["school"]["0.4"]["da"][ag] for ag in HIRA_AGE_GROUPS] for s in SEAS],0)
num_sk=-da_sk/100.0*pop6   # 감소 인원(양수=감소)
num_sc=-da_sc/100.0*pop6
fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5)); x=np.arange(6); bw=0.38
# left: rate (%pt)
a1.bar(x-bw/2,-da_sk,bw,color="#2166AC",label="병가",edgecolor="k",lw=0.4); a1.bar(x+bw/2,-da_sc,bw,color="#B2182B",label="학교결석",edgecolor="k",lw=0.4)
a1.set_xticks(x); a1.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS],rotation=30,fontsize=8); a1.set_ylabel("attack 감소 (%pt)")
a1.set_title("연령별 감소 — rate (%pt)",fontsize=11,fontweight="bold"); a1.legend(fontsize=9); a1.grid(axis="y",alpha=0.3)
# right: number (명)
a2.bar(x-bw/2,num_sk,bw,color="#2166AC",label="병가",edgecolor="k",lw=0.4); a2.bar(x+bw/2,num_sc,bw,color="#B2182B",label="학교결석",edgecolor="k",lw=0.4)
a2.set_xticks(x); a2.set_xticklabels([a+"세" for a in HIRA_AGE_GROUPS],rotation=30,fontsize=8); a2.set_ylabel("감소 감염 수 (명)")
a2.set_title("연령별 감소 — number (명 = rate × 인구)",fontsize=11,fontweight="bold"); a2.legend(fontsize=9); a2.grid(axis="y",alpha=0.3)
for xi in range(6):
    a2.text(xi-bw/2,num_sk[xi],f"{num_sk[xi]/1000:.0f}k",ha="center",va="bottom",fontsize=6.5,color="#2166AC")
    a2.text(xi+bw/2,num_sc[xi],f"{num_sc[xi]/1000:.0f}k",ha="center",va="bottom",fontsize=6.5,color="#B2182B")
fig.suptitle("병가 vs 학교결석 — attack 감소 rate(%pt) vs number(명), p=0.4 (대칭 baseline)\n★ rate는 학령기 크나, number는 인구 큰 성인(18-44)이 최대",fontsize=12.5,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(FIG/"pres_averted_number.png",bbox_inches="tight"); plt.close(fig)
print(f"[averted_number] pop6={[int(x) for x in pop6]}")
print("  병가 감소수(명): "+" ".join(f"{ag}:{num_sk[a]:.0f}" for a,ag in enumerate(HIRA_AGE_GROUPS)))
print("  학교 감소수(명): "+" ".join(f"{ag}:{num_sc[a]:.0f}" for a,ag in enumerate(HIRA_AGE_GROUPS)))
print("DONE")
