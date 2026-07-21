"""pres_school_vs_sick(.png) 수정본 + pres_school_vs_sick_number.png 신규.
저장 데이터(v3_school_vs_sick.json + v3_rate_vs_number.json)로만 재작도 — 계산 없음.
그림A 오른쪽: averted rate (양수, 위로). 그림B 오른쪽: averted number(명).
확정 파라미터(3시즌, p=0.4, μ=1). model/·.bak 무수정."""
import os, json
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS

ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"
FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
BLUE="#2166AC"; RED="#B2182B"; BASE=0.6
sv=json.load(open(ED/"v3_school_vs_sick.json")); rn=json.load(open(ED/"v3_rate_vs_number.json"))
P=sv["meta"]["P"]; SEAS=list(sv["results"].keys())
xs=[BASE-p for p in P]
sick_curve=[float(np.mean([sv["results"][s]["sick"][str(p)]["av"] for s in SEAS])) for p in P]
sch_curve =[float(np.mean([sv["results"][s]["school"][str(p)]["av"] for s in SEAS])) for p in P]
av_rate_sk=-np.asarray(rn["rate_sick"]); av_rate_sc=-np.asarray(rn["rate_school"])   # 감소를 양수로
num_sk=np.asarray(rn["num_sick"]); num_sc=np.asarray(rn["num_school"])               # 이미 양수(명)
AGL=[a+"세" for a in HIRA_AGE_GROUPS]; x=np.arange(6); bw=0.38

def left(a1):
    a1.plot(xs,sick_curve,"o-",color=BLUE,lw=2,ms=7,label="병가")
    a1.plot(xs,sch_curve,"s-",color=RED,lw=2,ms=7,label="학교결석")
    a1.axhline(0,color="k",lw=0.8,alpha=0.5)
    a1.set_xlabel("p 감소량 (공통 baseline 0.6)"); a1.set_ylabel("averted % (3시즌평균)")
    a1.set_title("학교 vs 병가 averted",fontsize=11,fontweight="bold")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

def right_bars(a2,ysk,ysc,ylab,title):
    a2.bar(x-bw/2,ysk,bw,color=BLUE,label="병가",edgecolor="k",lw=0.4)
    a2.bar(x+bw/2,ysc,bw,color=RED,label="학교결석",edgecolor="k",lw=0.4)
    a2.axhline(0,color="k",lw=1); a2.set_xticks(x); a2.set_xticklabels(AGL,rotation=0,fontsize=8)
    a2.set_ylabel(ylab); a2.set_title(title,fontsize=11,fontweight="bold")
    a2.legend(fontsize=9); a2.grid(axis="y",alpha=0.3); a2.margins(y=0.12)

# ── 그림 A: averted rate (%pt) ──
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
left(a1)
right_bars(a2,av_rate_sk,av_rate_sc,"감염 감소 (%pt)","연령 영향: 병가→성인, 학교→학생")
fig.suptitle("학교결석 vs 병가",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_school_vs_sick.png",bbox_inches="tight"); plt.close(fig)
print(f"[figA] pres_school_vs_sick.png  rate 최대: 병가={HIRA_AGE_GROUPS[av_rate_sk.argmax()]} 학교={HIRA_AGE_GROUPS[av_rate_sc.argmax()]}")

# ── 그림 B: averted number (명) ──
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
left(a1)
right_bars(a2,num_sk,num_sc,"averted number (명)","연령 영향 (감염 수): 병가→성인, 학교→성인")
fig.suptitle("학교결석 vs 병가",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"pres_school_vs_sick_number.png",bbox_inches="tight"); plt.close(fig)
print(f"[figB] pres_school_vs_sick_number.png  number 최대: 병가={HIRA_AGE_GROUPS[num_sk.argmax()]}({num_sk.max():.0f}명) 학교={HIRA_AGE_GROUPS[num_sc.argmax()]}({num_sc.max():.0f}명)")
