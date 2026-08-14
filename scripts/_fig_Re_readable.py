"""R_e(t) 가독성 그림 (reproduction_numbers[_SEASON].json 사용, 재계산 없음).
위: 3 시나리오 R_e(t) (start/peak/end 표시) / 아래: baseline 채널 성분.
시즌별 JSON 이 있으면 각각, 없으면 기본(reproduction_numbers.json) 하나."""
import os, json, sys, glob
from pathlib import Path
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_seasonpop"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from figstyle import savefig, COL_SICK, COL_SCHOOL, COL_ZERO, W_DOUBLE

FITDATA = json.load(open(REPO / "outputs/eda/fit_total_curves.json"))  # observed + model fit
CH_COL = {"home": "#555555", "work": "#55A868", "school": COL_SCHOOL, "other": "#8172B3"}
SC = {"baseline": (COL_ZERO, "Baseline"),
      "sick": (COL_SICK, "Sick leave"),
      "school": (COL_SCHOOL, "School absence")}
# KDCA influenza epidemic advisory ISSUED (ILI > threshold) → season-day (0 = Sep 1)
ADVISORY_DAY = {"2016-2017": 98, "2017-2018": 91, "2019-2020": 75}
ADVISORY_LBL = {"2016-2017": "ILI advisory (Dec 8)",
                "2017-2018": "ILI advisory (Dec 1)",
                "2019-2020": "ILI advisory (Nov 15)"}
# advisory LIFTED (ILI < threshold 3 consecutive weeks) → season-day
LIFTED_DAY = {"2016-2017": 274, "2017-2018": 266, "2019-2020": 208}
LIFTED_LBL = {"2016-2017": "lifted (Jun 2)", "2017-2018": "lifted (May 25)",
              "2019-2020": "lifted (Mar 27)"}
# observed NHIS incidence peak (from fit_total_curves.json) → season-day
DATAPEAK_DAY = {"2016-2017": 105, "2017-2018": 119, "2019-2020": 119}


def make_fig(json_path, out_name):
    D = json.load(open(json_path))
    t = np.array(D["Re"]["t_days"]); Re = D["Re"]["Re"]; Rec = D["Re"]["Re_channel"]
    peakday = D["Re"]["peak_day"]; peakRe = D["Re"]["peak_Re"]
    below1 = D["Re"]["first_below1_day"]; above1 = D["Re"]["first_above1_day"]
    meta = D.get("meta", {})
    season = meta.get("season", D.get("static", {}).get("season", "season"))
    BREAK = tuple(meta.get("break_window", (113, 183)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_DOUBLE, 5.6),
                                    constrained_layout=True, sharex=True)
    # ---- Panel A ----
    ax1.axhline(1.0, color="black", lw=0.9, ls=":", zorder=3)
    ax1.axvspan(*BREAK, color="#4C72B0", alpha=0.07, lw=0)
    # right axis: observed NHIS incidence + model fit (behind R_e curves)
    fc = FITDATA.get(season)
    ax1r = ax1.twinx()
    if fc is not None:
        wk = np.arange(len(fc["mean"])) * 7.0
        ym = np.asarray(fc["mean"], float)
        # weekly model incidence -> smooth daily curve (monotone cubic) for visual
        # consistency with the daily R_e(t) curves
        try:
            from scipy.interpolate import PchipInterpolator
            xd = t[(t >= wk[0]) & (t <= wk[-1])]
            yd = PchipInterpolator(wk, ym)(xd)
        except Exception:
            xd, yd = wk, ym
        ax1r.plot(xd, yd, color="#B0B0B0", lw=1.6, zorder=1, label="Model incidence")
        ax1r.set_ylim(0, ym.max() * 1.1)
    ax1r.set_ylabel("Weekly incidence (NHIS)", color="#7A7A7A")
    ax1r.tick_params(axis="y", colors="#7A7A7A")
    # R_e curves on top
    for sc, (col, lab) in SC.items():
        m = np.array(Re[sc]["mean"]); lo = np.array(Re[sc]["q05"]); hi = np.array(Re[sc]["q95"])
        ax1.fill_between(t, lo, hi, color=col, alpha=0.15, lw=0)
        ax1.plot(t, m, color=col, lw=2.0, label=lab)
    ax1.set_zorder(ax1r.get_zorder() + 1); ax1.patch.set_visible(False)
    # epidemic advisory issued / lifted — both dashed vertical lines
    ymax = max(max(Re[sc]["q95"]) for sc in SC)   # highest CI across scenarios
    ylim_top = ymax * 1.10
    ytop = ylim_top * 0.97          # winter-break label (top)
    ylow = ylim_top * 0.42          # advisory labels (lower, clear of the R_e=1 line)
    # policy application window (what actually drives the scenario divergence).
    # Place the onset line at the last grid point before curves separate, so it
    # aligns with the visible divergence on any grid (weekly ~day63; daily ~day70).
    _dref = np.abs(np.array(Re["baseline"]["mean"]) - np.array(Re["school"]["mean"]))
    _di = int(np.argmax(_dref > 1e-4)) if bool((_dref > 1e-4).any()) else 0
    pol_start = float(t[max(_di - 1, 0)])
    pol_end = BREAK[1]
    pcol = "#333333"
    ax1.axvline(pol_start, color=pcol, lw=1.1, ls="--", zorder=4)
    ax1.axvline(pol_end, color=pcol, lw=1.1, ls="--", zorder=4)
    ax1.annotate("policy applied", (pol_start, ylow), textcoords="offset points",
                 xytext=(-3, 0), ha="right", va="center", fontsize=6.8,
                 color=pcol, fontweight="bold")
    ax1.annotate("policy ends", (pol_end, ylow), textcoords="offset points",
                 xytext=(4, 0), ha="left", va="center", fontsize=6.8,
                 color=pcol, fontweight="bold")
    ax1.text(np.mean(BREAK), ytop, "winter break", ha="center", va="top",
             fontsize=7.5, color=COL_ZERO)
    ax1.set_ylabel(r"Effective $R_e(t)$")
    ax1.set_ylim(0, ylim_top)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1r.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax1.set_title(rf"$R_e(t)$ and observed incidence by scenario ({season})",
                  fontsize=10, fontweight="bold")
    # ---- Panel B ----
    base = Rec["baseline"]
    for ch in ["home", "school", "other", "work"]:
        ax2.plot(t, np.array(base[ch]["mean"]), color=CH_COL[ch], lw=1.8, label=ch)
    ax2.axvspan(*BREAK, color="#4C72B0", alpha=0.07, lw=0)
    ax2.set_xlabel("Season day (0 = Sep 1)")
    ax2.set_ylabel(r"Channel $R_e^{c}(t)$")
    ax2.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    ax2.set_title("Baseline channel components --- the school term collapses in the winter break",
                  fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 290)
    savefig(fig, out_name)
    print(f"saved {out_name}  ({season}: policy window {pol_start:.0f}-{pol_end:.0f})")


if __name__ == "__main__":
    per_season = sorted(glob.glob(str(REPO / "outputs/eda/reproduction_numbers_*.json")))
    if per_season:
        for jp in per_season:
            tag = Path(jp).stem.replace("reproduction_numbers_", "")
            make_fig(jp, f"repro_Re_readable_{tag}")
    else:
        make_fig(REPO / "outputs/eda/reproduction_numbers.json", "repro_Re_readable")
