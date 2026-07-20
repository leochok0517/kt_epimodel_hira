"""Seasonality peak_day sweep — 3-season fit, find timing-optimal peak.

Model epidemic peaks ~1-2 wk later than data. Sweep seasonality peak_day (cosine
forcing peak, currently 105 = mid-Dec); earlier peak_day → earlier epidemic.
amp=0.9 fixed. Confirmed params otherwise fixed (baseline 0.6, φ linear,
γ 12-17=0.18, κ 3-way, Step A+B). Seasons 16-17,17-18,19-20. first_peak_only.

Per peak_day: refit per-season π+R0, measure peak-timing error, obs/model
(total + per-age), width_ratio (obs_width/model_width). Point estimate.

Output: outputs/eda/peak_sweep.json + console.
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
from kt_epimodel_hira.jax_model.loss_jax import HIRA_AGE_GROUPS
import final_pipeline_confirmed as F

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "outputs" / "eda" / "peak_sweep.json"

SEASONS3 = ["2016-2017", "2017-2018", "2019-2020"]
IDX = [F.SEASONS.index(s) for s in SEASONS3]
PEAK_DAYS = [90, 95, 100, 105, 110, 115]


def metrics(C, i, R0, pi):
    """mask-window peak week/width for obs & model."""
    pred = F.pred_h(C, F.run_inc(C, i, R0, pi))     # (52,6)
    obs = np.asarray(C["obs"][i]); w = np.asarray(C["w"][i]); mask = w.sum(1) > 0
    o = obs[mask].sum(1); m = pred[mask].sum(1)
    wk = np.where(mask)[0]
    opk = int(wk[np.argmax(o)]); mpk = int(wk[np.argmax(m)])
    owid = o.sum()/max(o.max(), 1); mwid = m.sum()/max(m.max(), 1)
    om_age = {ag: float(obs[mask, a].sum()/max(pred[mask, a].sum(), 1.0)) for a, ag in enumerate(HIRA_AGE_GROUPS)}
    om_tot = float(o.sum()/max(m.sum(), 1.0))
    return dict(obs_pk=opk, mdl_pk=mpk, timing_err=abs(mpk-opk),
                width_ratio=float(owid/mwid), om_total=om_tot, om_age=om_age)


def main():
    print("=" * 96)
    print("SEASONALITY peak_day SWEEP — 3 seasons (16-17,17-18,19-20), amp=0.9 fixed")
    print(f"  peak_day ∈ {PEAK_DAYS} (현재 105)   확정: baseline 0.6, φ선형, γ 12-17=0.18")
    print("=" * 96)
    t0 = time.perf_counter(); C = F.build(); print(f"[setup] {time.perf_counter()-t0:.1f}s\n")

    results = {}
    for pd in PEAK_DAYS:
        tp = time.perf_counter()
        C["shared"]["seasonality_peak_day"] = float(pd)
        rec = {}
        for s, i in zip(SEASONS3, IDX):
            f = F.fit_season_pi(C, i)
            mt = metrics(C, i, f["R0"], f["pi"])
            rec[s] = dict(R0=f["R0"], nll=f["nll"], **mt)
        results[pd] = rec
        te = sum(rec[s]["timing_err"] for s in SEASONS3)
        omt = [rec[s]["om_total"] for s in SEASONS3]
        wr = [rec[s]["width_ratio"] for s in SEASONS3]
        nll = sum(rec[s]["nll"] for s in SEASONS3)
        print(f"  peak_day={pd}: 타이밍오차합={te}주  om_total={[round(x,2) for x in omt]}  "
              f"width_ratio={[round(x,2) for x in wr]}  ΣNLL={nll:.1f}  ({time.perf_counter()-tp:.1f}s)")

    # detail table
    print("\n=== 상세: peak_day × 시즌 (mdl_pk/obs_pk, om_total, om[45-64], om[65+], width_ratio) ===")
    print(f"  {'pd':>4} " + " | ".join(f"{s[2:7]}:pk om wid" for s in SEASONS3))
    for pd in PEAK_DAYS:
        cells = []
        for s in SEASONS3:
            r = results[pd][s]
            cells.append(f"{r['mdl_pk']}/{r['obs_pk']} {r['om_total']:.2f} {r['width_ratio']:.2f}")
        print(f"  {pd:>4} " + " | ".join(cells))

    # verdicts
    te_by = {pd: sum(results[pd][s]["timing_err"] for s in SEASONS3) for pd in PEAK_DAYS}
    best_timing = min(te_by, key=te_by.get)
    # obs/model closeness to 1 (geometric mean distance)
    om_dist = {pd: float(np.mean([abs(np.log(results[pd][s]["om_total"])) for s in SEASONS3])) for pd in PEAK_DAYS}
    best_om = min(om_dist, key=om_dist.get)
    wr_by = {pd: float(np.mean([results[pd][s]["width_ratio"] for s in SEASONS3])) for pd in PEAK_DAYS}
    print("\n=== 판정 ===")
    print(f"  타이밍오차 최소 peak_day = {best_timing} (오차합 {te_by[best_timing]}주)")
    print(f"  obs/model 1근접 peak_day = {best_om} (평균 |log om| {om_dist[best_om]:.3f})")
    print(f"  width_ratio 평균: " + " ".join(f"{pd}:{wr_by[pd]:.2f}" for pd in PEAK_DAYS))
    wr_range = max(wr_by.values()) - min(wr_by.values())
    print(f"  width_ratio 변화폭(peak_day 전체) = {wr_range:.2f}  → 폭은 peak_day에 {'민감' if wr_range>0.1 else '거의 무관'}")
    print("=" * 96)

    OUT.write_text(json.dumps(dict(
        meta=dict(seasons=SEASONS3, peak_days=PEAK_DAYS, amp=0.9, baseline=0.6,
                  fixed="phi linear, gamma 12-17=0.18, kappa 3-way"),
        results=results, best_timing_peak_day=best_timing, best_obsmodel_peak_day=best_om,
        width_ratio_by_peakday=wr_by, timing_err_by_peakday=te_by), indent=2, default=float))
    print(f"[json] {OUT}")


if __name__ == "__main__":
    main()
