"""viz_fit_erlang_byage.png 재작도 — Erlang만, y축 10만 고정.
재적합(fitting) 없음: erlang_fit.json 의 확정 R0/pi 로 forward 예측만 (3시즌).
예측곡선을 erlang_byage_preds.json 에 저장 → 이후 문구/축 수정은 REDRAW=1 로 즉시(계산 없음).
단일 I 제거. 색=연령별(pres_fit_byage 통일). model/·.bak 무수정."""
import os, json
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS

ED=Path(__file__).resolve().parent.parent/"outputs"/"eda"
FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"
FB=FIG/"viz_fit_erlang_byage.png"; PJ=ED/"erlang_byage_preds.json"
SEAS=["2016-2017","2017-2018","2019-2020"]
AGE_C=["#4575b4","#74add1","#fdae61","#f46d43","#d73027","#7b3294"]; GRAY="#666666"; YMAX=100000.0

fit=json.load(open(ED/"erlang_fit.json"))

def compute():
    import jax; jax.config.update("jax_enable_x64",True); jax.devices()
    import erlang_fit as E   # forward pred helpers + build
    C=E.F.build(); preds={}
    for s in SEAS:
        i=E.F.SEASONS.index(s); r=fit["results"][s]["erlang"]
        preds[s]=E.pred_erlang(C,i,r["R0"],r["pi"]).tolist()
    obs={s:np.asarray(C["full_obs"][s]).tolist() for s in SEAS}
    PJ.write_text(json.dumps(dict(preds=preds,obs=obs),default=float))
    return preds,obs

def draw(preds,obs):
    weeks=np.arange(52)
    fig,axes=plt.subplots(3,6,figsize=(16,7.5),sharex=True,sharey=True)
    for r,s in enumerate(SEAS):
        pe=np.asarray(preds[s]); ob=np.asarray(obs[s])
        for c,ag in enumerate(HIRA_AGE_GROUPS):
            ax=axes[r,c]
            ax.plot(weeks,ob[:,c],"o",color=GRAY,ms=2,alpha=0.6)
            ax.plot(weeks,pe[:,c],"-",color=AGE_C[c],lw=1.6)
            ax.set_ylim(0,YMAX); ax.grid(alpha=0.2)
            ax.text(0.04,0.8,f"{fit['results'][s]['erlang']['om_age'][ag]:.2f}",transform=ax.transAxes,fontsize=7,color="#333")
            if r==0: ax.set_title(f"{ag}세",fontsize=9,fontweight="bold")
            if c==0: ax.set_ylabel(f"{s}",fontsize=8,fontweight="bold")
            ax.tick_params(labelsize=6)
    fig.suptitle("3시즌 × 연령별 fit (Erlang I₃)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FB,bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {FB}")

if __name__=="__main__":
    if PJ.exists() and os.environ.get("REDRAW"):
        d=json.load(open(PJ)); preds,obs=d["preds"],d["obs"]
    else:
        preds,obs=compute()
    draw(preds,obs)
