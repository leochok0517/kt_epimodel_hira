"""Regenerate fit_total (Figure 3) in black/grey only, from the cached
posterior-mean/observed curves — no re-fit, no re-run of posterior
predictive. Numbers are identical to the existing figure; only line and
title colour change (season already distinguished by panel position and
title, colour carried no information — corresponding-author request).

Reads:  outputs/eda/fit_total_curves.json (written by paper_figures_extra.fig_F7)
Writes: figures/paper{FIG_OUTDIR_SUFFIX}/{pdf,png}/fit_total.{pdf,png}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import REPO
from paper_figures_extra import _fig_F7_from_curves

CACHE = REPO / "outputs" / "eda" / "fit_total_curves.json"


def main():
    cache = json.loads(CACHE.read_text())
    pdf, png = _fig_F7_from_curves(cache)
    print(f"  saved {pdf}\n  saved {png}")


if __name__ == "__main__":
    main()
