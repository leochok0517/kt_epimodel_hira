"""Per-season post-peak decline of the main four-season fit (posterior mean curve).

Reads outputs/fluA/fit_total_curves_fluA4.json (500-draw posterior mean and
observed weekly totals) and writes outputs/fluA/fit_decline_summary_fluA4.json.
Forward quantities only, no fitting. Used for the supplement S5 per-season table
and the main-text Limitations paragraph (2026-10-16).

decline_kwk = value k weeks after the peak / value at the peak, each series at
its own peak week within the fitted window. *_at_obs_peak uses the observed peak
week for the model curve too.
"""
import json
from datetime import date
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = json.load(open(REPO / "outputs/fluA/fit_total_curves_fluA4.json"))
out = {}
for s, v in D.items():
    nfit = v["fitted_weeks"]
    obs = np.asarray(v["obs"], float); mod = np.asarray(v["mean"], float)
    po = int(np.argmax(obs[:nfit])); pm = int(np.argmax(mod[:nfit]))
    y0 = int(s[:4])
    iso = lambda i: (date.fromisocalendar(y0, 36, 1).toordinal() + 7 * i)
    iso_wk = lambda i: date.fromordinal(iso(i)).isocalendar()[1]
    out[s] = dict(
        peak_idx_obs=po, peak_idx_model=pm,
        peak_iso_week_obs=iso_wk(po), peak_iso_week_model=iso_wk(pm),
        decline_2wk_obs=float(obs[po + 2] / obs[po]),
        decline_2wk_model=float(mod[pm + 2] / mod[pm]),
        decline_2wk_model_at_obs_peak=float(mod[po + 2] / mod[po]),
        decline_4wk_obs=float(obs[po + 4] / obs[po]),
        decline_4wk_model=float(mod[pm + 4] / mod[pm]),
    )
json.dump(dict(source="outputs/fluA/fit_total_curves_fluA4.json (mean, obs)",
               results=out), open(REPO / "outputs/fluA/fit_decline_summary_fluA4.json", "w"), indent=1)
for s, r in out.items():
    print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})
