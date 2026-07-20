"""Paper figures — holiday sign-reversal of sick-leave child impact.

Fig 1: 6-season reversal (json outputs/eda/holiday_reversal_multiseason.json).
Fig 2: weekly sliding-window averted heatmap (2019-2020) — extra sims here.

Style: AppleGothic (Korean), axes.unicode_minus=False, dpi 150. Diverging
colors: benefit/negative = blue #2166AC, backfire/positive = red #B23A48.
Saved to presentations/figures/.
"""
from __future__ import annotations
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"; os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.devices()
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.unicode_minus": False,
    "font.family": "AppleGothic", "font.size": 10,
})

import sens_workshare_kappa_v2 as S
import policy_compare_school_vs_sickleave as PC
import holiday_reversal_multiseason as HR
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS

REPO_ROOT = Path(__file__).resolve().parent.parent
JSON1 = REPO_ROOT / "outputs" / "eda" / "holiday_reversal_multiseason.json"
FIGDIR = REPO_ROOT / "presentations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
FIG1 = FIGDIR / "viz_holiday_reversal_6seasons.png"
FIG2 = FIGDIR / "viz_weekly_averted_heatmap_1920.png"

BLUE = "#2166AC"; RED = "#B23A48"
CHILD = ["0-5", "6-11", "12-17"]
DIVERGING = LinearSegmentedColormap.from_list("bwr_paper", [BLUE, "#ffffff", RED])


# ═══════════════════════════ FIG 1 ═══════════════════════════
def fig1():
    d = json.load(open(JSON1))
    seasons = d["meta"]["seasons"]; res = d["results"]
    labels = [f"{s[2:4]}–{s[7:9]}" for s in seasons]  # 2015-2016 → 15–16
    term = np.array([res[s]["term"]["child_sum"] for s in seasons])
    vac = np.array([res[s]["vacation"]["child_sum"] for s in seasons])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.2))
    x = np.arange(len(seasons)); bw = 0.38

    # --- top: term vs vacation child_sum ---
    ax1.bar(x - bw/2, term, bw, color=BLUE, label="학기중 창 (term)", edgecolor="k", linewidth=0.5)
    ax1.bar(x + bw/2, vac, bw, color=RED, label="방학중 창 (vacation)", edgecolor="k", linewidth=0.5)
    ax1.axhline(0, color="k", lw=1.8, zorder=3)
    for xi, (tv, vv) in enumerate(zip(term, vac)):
        ax1.text(xi - bw/2, tv - 0.06, f"{tv:+.2f}", ha="center", va="top", fontsize=8, color=BLUE)
        ax1.text(xi + bw/2, vv + 0.04, f"{vv:+.2f}", ha="center", va="bottom", fontsize=8, color=RED)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("아동(0–17) Δattack 합 (%pt)")
    ax1.set_title("병가의 아동 영향: 학기중 이득(파랑) → 방학중 역효과(빨강)  —  6/6 시즌",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)
    ax1.margins(y=0.18)
    ax1.text(0.01, 0.04, "각 시즌 파랑(아래) → 빨강(위) 반복 = 부호반전", transform=ax1.transAxes,
             fontsize=9, style="italic", color="#444")

    # --- bottom: 3 child ages, vacation window only ---
    age_c = {"0-5": "#9ecae1", "6-11": "#ef6548", "12-17": "#b30000"}
    bw2 = 0.26
    for j, ag in enumerate(CHILD):
        vals = np.array([res[s]["vacation"]["child_d_attack"][ag] for s in seasons])
        ax2.bar(x + (j-1)*bw2, vals, bw2, color=age_c[ag], label=f"{ag}세", edgecolor="k", linewidth=0.4)
    ax2.axhline(0, color="k", lw=1.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("방학중 Δattack (%pt)")
    ax2.set_title("방학중 병가: 학령기(6–17세)가 유아(0–5세)보다 크게 악화  —  spillover 학령기 집중",
                  fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", ncol=3, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3); ax2.margins(y=0.15)

    fig.suptitle("방학 부호반전: 병가는 학기중 아동을 보호하나 방학중엔 아동에 부하 전가 (다년 검증)",
                 fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG1, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] {FIG1}")
    return term, vac


# ═══════════════════════════ FIG 2 ═══════════════════════════
def fig2():
    """Sliding 1-week window: apply sick-leave in week w only, measure Δattack per age."""
    setup = PC.build_setup()   # 2019-2020, π fixed, R0=2.109, β_4 fixed
    kap = 1.0 * HR.KAPPA_BASE
    base_inf6 = PC.attack6(PC.run(setup, kap, 1.0, 1.0, PC.WHOLE_WIN, PC.WHOLE_WIN), setup)
    pop6 = setup["pop_6"]

    WEEKS = list(range(5, 27))     # epi weeks covering rise, peak, break, tail
    n_age = len(HIRA_AGE_GROUPS)
    M = np.zeros((n_age, len(WEEKS)))    # Δattack %pt
    ramp = 1.0                            # sharp 1-week window
    t0 = time.perf_counter()
    for wi, w in enumerate(WEEKS):
        d0 = w * 7.0; d1 = d0 + 7.0
        inc = PC.run(setup, kap, 1.0, 0.4, PC.WHOLE_WIN, (d0, d1))   # sick-leave in week w only
        inf6 = PC.attack6(inc, setup)
        M[:, wi] = 100.0 * (inf6 - base_inf6) / pop6                  # + = more infection (backfire)
    print(f"[fig2] sliding sims {time.perf_counter()-t0:.1f}s")

    vmax = np.abs(M).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, norm=norm, origin="upper",
                   extent=[WEEKS[0]-0.5, WEEKS[-1]+0.5, n_age-0.5, -0.5])
    ax.set_yticks(range(n_age)); ax.set_yticklabels([f"{a}세" for a in HIRA_AGE_GROUPS])
    ax.set_xticks(WEEKS); ax.set_xticklabels(WEEKS, fontsize=8)
    ax.set_xlabel("역학 주차 (epidemiological week, day = week×7)")
    # break start day113 → week ~16.1; break span 113-183 → week 16.1-26.1
    wk_break_start = 113/7.0; wk_break_end = 183/7.0
    ax.axvspan(wk_break_start, wk_break_end, color="0.5", alpha=0.13, zorder=0)
    ax.axvline(wk_break_start, color="k", ls="--", lw=1.8)
    # vertical label along the break line (inside plot, avoids title collision)
    ax.text(wk_break_start+0.18, n_age/2.0, "방학 시작 (day113)", rotation=90,
            va="center", ha="left", fontsize=9, fontweight="bold", color="k")
    ax.text((wk_break_start+wk_break_end)/2, n_age-0.62, "방학 구간", ha="center", fontsize=9.5,
            color="0.25", style="italic", fontweight="bold")
    ax.text(24.6, 0.35, "유행 종료", fontsize=8.5, color="0.45", style="italic", va="center")

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label("Δattack (%pt)   파랑=이득(감소) · 빨강=역효과(증가)", fontsize=9)
    ax.set_title("주차별 병가 효과 (2019–2020): 방학 진입 시 학령기 역효과로 전환",
                 fontsize=12.5, fontweight="bold", pad=30)
    ax.text(0.5, 1.055, "각 주에 1주간 병가(p_work=0.4) 적용 시 연령별 Δattack — 슬라이딩 윈도우",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#444")
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIG2, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] {FIG2}")
    # verdict check: school-age rows flip blue→red at break
    sa = HIRA_AGE_GROUPS.index("6-11")
    pre = M[sa, np.array(WEEKS) < wk_break_start].mean()
    post = M[sa, (np.array(WEEKS) >= wk_break_start) & (np.array(WEEKS) <= wk_break_end)].mean()
    print(f"  6-11세: 방학전 평균 Δ={pre:+.3f} (이득) → 방학중 평균 Δ={post:+.3f} (역효과)  전환={'Y' if pre<0<post else 'N'}")


def main():
    print("=" * 80); print("VIZ — holiday reversal (fig1 + fig2)"); print("=" * 80)
    fig1()
    fig2()
    print("=" * 80)


if __name__ == "__main__":
    main()
