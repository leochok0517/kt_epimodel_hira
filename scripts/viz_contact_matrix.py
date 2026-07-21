"""학기/방학 접촉행렬 히트맵 (발표용). 3패널: 학기 전체 / 방학 전체 / 차분.

★ 축 관례 (로더 검증 완료):
  load_contact_matrices 는 모델 컨벤션 [contact, participant] 로 반환(transpose=True).
  즉 반환행렬 C[i,j]: 행 i = contact(접촉 상대), 열 j = participant(응답자).
  → 태스크 요구(x=participant, y=contacted)와 이미 일치 → transpose 불필요.
  검증: C_work 아동 열(participant)=0 (아동은 직장 접촉 미보고).
전체행렬 = 4채널(home+work+school+other) 합. 값=1인1일당 접촉수.
Output: figures/viz_contact_matrix_term_vacation.png
"""
import os
os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({"figure.dpi":110,"savefig.dpi":150,"axes.unicode_minus":False,"font.family":"AppleGothic","font.size":9})
from kt_data.data.load_contact import load_contact_matrices

D=Path("../kt_data/data/external/contact_matrices")
FIG=Path(__file__).resolve().parent.parent/"presentations"/"figures"; FIG.mkdir(parents=True,exist_ok=True)
LAB=["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70+"]
CH=("C_home","C_school","C_work","C_other")

def total(path):
    m=load_contact_matrices(path=path)   # [contact, participant]
    return sum(np.asarray(m[c]) for c in CH)  # (15,15)

term=total(D/"empirical_matrices_15.npz")       # 학기 (현 base)
vac=total(D/"empirical_matrices_15_vacation.npz")# 방학
diff=term-vac                                     # 학기−방학 (양수=방학에 감소)
print(f"학기 합={term.sum():.1f} 방학 합={vac.sum():.1f}  차분 합={diff.sum():.1f}")
# 대칭성 (raw 비대칭 확인)
print(f"학기 max|C-C.T|={np.abs(term-term.T).max():.3f} (비대칭 raw)")

vmax=max(term.max(),vac.max())
fig,axes=plt.subplots(1,2,figsize=(13,6))
def draw(ax,M,title,cmap,vmin=None,vmax=None):
    im=ax.imshow(M,origin="lower",cmap=cmap,vmin=vmin,vmax=vmax,aspect="equal")
    ax.set_xticks(range(15)); ax.set_xticklabels(LAB,rotation=90,fontsize=7)
    ax.set_yticks(range(15)); ax.set_yticklabels(LAB,fontsize=7)
    ax.set_xlabel("응답자 연령 (participant)",fontsize=9); ax.set_ylabel("접촉 상대 연령 (contacted)",fontsize=9)
    ax.set_title(title,fontsize=12,fontweight="bold")
    return im
im1=draw(axes[0],term,"학기 (term)","YlOrRd",vmin=0,vmax=vmax)
im2=draw(axes[1],vac,"방학 (vacation)","YlOrRd",vmin=0,vmax=vmax)
cb1=fig.colorbar(im2,ax=axes,fraction=0.023,pad=0.02); cb1.set_label("일 평균 접촉 수",fontsize=9)
fig.suptitle("학기 vs 방학 접촉행렬",fontsize=14,fontweight="bold")
fig.savefig(FIG/"viz_contact_matrix_term_vacation.png",bbox_inches="tight"); plt.close(fig)
print(f"[fig] {FIG/'viz_contact_matrix_term_vacation.png'}")
# 방학 school 감소 확인: 학령기(5-19=idx1-3) 영역 차분
sa=diff[1:4,1:4]
print(f"학령기(5-19) 블록 차분 평균={sa.mean():.3f} (양수=방학에 감소)  최대감소셀={diff.max():.3f}")
