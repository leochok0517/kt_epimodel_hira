"""감염기간 분포 개념도: 지수(단일 I) vs Erlang(3). 발표용.
좌: 감염기간 확률밀도 (둘 다 평균 4일, 모양 차이). 우: 유행곡선 폭 연결(개념).
지수=빨강, Erlang=녹색 (Erlang fit 그림과 색 통일). Output: figures/viz_infectious_period.png"""
import os
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
from scipy.stats import expon, erlang
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":10})
FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"; FIG.mkdir(parents=True,exist_ok=True)
EXP="#B23A48"; ERL="#1a9850"
MEAN=4.0; GAMMA=1/MEAN   # 평균 감염기간 4일
t=np.linspace(0,15,600)
# 지수(단일 I): rate γ, 평균 1/γ=4
pdf_exp=expon.pdf(t,scale=MEAN)
# Erlang(3): shape k=3, 각 단계 rate 3γ → scale=1/(3γ)=MEAN/3, 평균 k*scale=4
pdf_erl=erlang.pdf(t,3,scale=MEAN/3)

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
# 좌: 밀도
a1.plot(t,pdf_exp,"-",color=EXP,lw=2.5,label="지수 (단일 I)")
a1.plot(t,pdf_erl,"-",color=ERL,lw=2.5,label="Erlang(3)")
a1.fill_between(t,pdf_exp,alpha=0.08,color=EXP); a1.fill_between(t,pdf_erl,alpha=0.10,color=ERL)
a1.axvline(MEAN,color="k",ls="--",lw=1.3); a1.text(MEAN+0.2,0.235,"평균 4일",fontsize=10,fontweight="bold")
a1.set_xlabel("감염 후 경과일"); a1.set_ylabel("확률밀도")
a1.set_title("감염기간 분포",fontsize=13,fontweight="bold")
a1.legend(fontsize=11,loc="upper right"); a1.grid(alpha=0.25); a1.set_xlim(0,15); a1.set_ylim(bottom=0)

# 우: 개념 유행곡선
x=np.linspace(0,52,400); mu=18
def bell(sd,amp): return amp*np.exp(-0.5*((x-mu)/sd)**2)
a2.plot(x,bell(5.0,1.0),"-",color=EXP,lw=2.5,label="지수")
a2.plot(x,bell(4.0,1.18),"-",color=ERL,lw=2.5,label="Erlang(3)")
a2.set_xlabel("주차"); a2.set_title("유행 곡선",fontsize=13,fontweight="bold")
a2.legend(fontsize=11); a2.grid(alpha=0.25); a2.set_yticks([])

fig.suptitle("지수 vs Erlang(3) — 같은 평균(4일), 다른 분포",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"viz_infectious_period.png",bbox_inches="tight"); plt.close(fig)
# 수치 확인
print(f"지수: 평균={expon.mean(scale=MEAN):.1f}일 분산={expon.var(scale=MEAN):.1f}  P(>10일)={expon.sf(10,scale=MEAN):.3f}")
print(f"Erlang3: 평균={erlang.mean(3,scale=MEAN/3):.1f}일 분산={erlang.var(3,scale=MEAN/3):.1f}  P(>10일)={erlang.sf(10,3,scale=MEAN/3):.3f}")
print(f"[fig] {FIG/'viz_infectious_period.png'}")
