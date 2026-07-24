"""논문 그림 공통 스타일 모듈.

모든 논문 그림 스크립트가 첫 줄에서 import.  rcParams·팔레트·헬퍼를 한 곳에.
"""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── 폴더 ──
REPO = Path(__file__).resolve().parent.parent
PDF_DIR = REPO / "figures" / "paper" / "pdf"
PNG_DIR = REPO / "figures" / "paper" / "png"
PDF_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)

# ── rcParams (전 그림 통일) ──
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,             # 폰트 임베딩 (제출 필수)
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.facecolor": "white",
    "axes.grid": False,
})

# ── 색 매핑 (Okabe–Ito, 색맹 안전) ──
COL_SICK = "#E69F00"        # 병가 = 주황
COL_SCHOOL = "#0072B2"      # 학교결석 = 파랑
COL_SEASON = {
    "2016-2017": "#009E73",
    "2017-2018": "#CC79A7",
    "2019-2020": "#56B4E9",
}
COL_ZERO = "#888888"

# ── 상수 ──
AGES = ["0-5", "6-11", "12-17", "18-44", "45-64", "65+"]
SEASONS = ["2016-2017", "2017-2018", "2019-2020"]
CHANNELS = ["home", "work", "school", "other"]

# ── 그림 폭 (inches, 텍스트폭 6.5 in 기준) ──
W_SINGLE = 3.3     # 단일 패널
W_DOUBLE = 6.5     # 2 패널 or 전폭


def savefig(fig, name):
    """벡터 PDF + 300 dpi PNG 이중 저장, 동일 base name."""
    pdf = PDF_DIR / f"{name}.pdf"
    png = PNG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(pdf), str(png)


def is_sig(q05, q95):
    """0 ∉ CI 이면 유의 (양수·음수 부호 동일)."""
    return q05 * q95 > 0


def marker_style(sig, color):
    """유의: 채운 마커.  비유의: 속 빈 마커 (흰 채움 + 테두리색)."""
    if sig:
        return dict(marker="o", markerfacecolor=color,
                    markeredgecolor=color, markeredgewidth=1.0, markersize=6)
    return dict(marker="o", markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.2, markersize=6)


def panel_label(ax, text, x=-0.10, y=1.02):
    """다중 패널 좌상단 볼드 A/B/C 라벨."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def zero_line(ax, orient="h", **kw):
    """0 기준선.  orient='h' 가로, 'v' 세로."""
    kw = dict(color=COL_ZERO, lw=0.8, ls="--", zorder=0, **kw)
    if orient == "h":
        ax.axhline(0, **kw)
    else:
        ax.axvline(0, **kw)
