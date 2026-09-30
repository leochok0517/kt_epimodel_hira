"""Sens 5 ported to fluA4 -- symmetric/asymmetric baseline retention p.

Same six (p_work, p_school) baseline combinations as ``sens5_baseline_p.py``,
on the fluA4 setup, with pi_work pinned at the fluA4 posterior median (0.3143)
rather than v5's 0.29.

Two things inherited from the v5 script that its own docstring gets wrong, and
which are preserved here so the two remain comparable:

* the docstring says the model is refitted at each baseline combination; the
  code does not. It point-fits once at the default baseline (p = 0.6) and then
  varies the baseline only inside ``evaluate_full_stratified``. The inline
  comment in the v5 source says as much. This port keeps that behaviour and
  records it in ``meta``.
* because the fit does not depend on the combination, v5 repeats an identical
  fit six times per season. This port fits once per season and reuses it —
  ``fit_pi_pin`` is deterministic in its seed, so the numbers are unchanged and
  the run is six times cheaper.

``fit_pi_pin`` and ``evaluate_full_stratified`` are imported unmodified from
``sens_common``; only the setup and the pin value differ.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from sens_common import (  # noqa: E402
    evaluate_full_stratified, fit_pi_pin, key_str, load_partial,
    append_partial, ntfy,
)
import fluA_setup as FS  # noqa: E402

OUTD = REPO / "outputs" / "fluA"
OUTD.mkdir(parents=True, exist_ok=True)
NAME = "sens_baseline_p_fluA4"
PARTIAL = OUTD / f"{NAME}.partial.jsonl"
FINAL = OUTD / f"{NAME}.json"

SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
IDX = list(range(len(SEASONS)))   # positional: fluA_setup builds obs per request

BASE_COMBOS = [
    (0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.75, 0.75),   # symmetric
    (0.6, 0.7), (0.7, 0.6),                               # asymmetric
]
#: fluA4 posterior median of pi_work (per_season_pi_fluA4.json shared.pi.work.median),
#: replacing v5's 0.29.
PI_WORK_PIN = 0.31434803969715475
POL = 0.4

CHILD_AGE = ["0-5", "6-11", "12-17"]


def build_setup_by_season():
    C_all = FS.build_fluA_setup(SEASONS, verbose=False)
    return {s: dict(shared=C_all["shared_by_s"][s], ngm3=C_all["ngm3_by_s"][s],
                    st={s: C_all["st_by_s"][s]}, obs=C_all["obs"], w=C_all["w"],
                    nw=C_all["nw"], H=C_all["H"], pop6=C_all["pop6_by_s"][s])
            for s in SEASONS}


def main():
    t0 = time.time()
    done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows;  pi_work pinned at {PI_WORK_PIN:.4f}")

    C_by_s = build_setup_by_season()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s", flush=True)
    ntfy(f"sens5_fluA4 baseline p start ({len(BASE_COMBOS)} combos x {len(SEASONS)} seasons)")

    # One fit per season: the fit is at the default baseline and does not depend
    # on the combination being evaluated (see module docstring).
    fits = {}
    for s, i in zip(SEASONS, IDX):
        t_f = time.time()
        fits[s] = fit_pi_pin(C_by_s[s], s, i, PI_WORK_PIN)
        print(f"  [fit] {s}: R_peak={fits[s]['R0']:.3f} "
              f"pi={[round(x, 4) for x in fits[s]['pi']]} "
              f"({time.time()-t_f:.1f}s)", flush=True)

    total = len(BASE_COMBOS) * len(SEASONS)
    n = 0
    for pw_b, ps_b in BASE_COMBOS:
        for s in SEASONS:
            n += 1
            k = key_str(p_work_base=pw_b, p_school_base=ps_b, season=s)
            if k in done:
                continue
            t_c = time.time()
            fit = fits[s]
            res = evaluate_full_stratified(
                C_by_s[s], s, fit["R0"], fit["pi"],
                p_school_pol=POL, p_work_pol=POL,
                p_school_base=ps_b, p_work_base=pw_b)
            ratio = (res["averted_school_term"] / res["averted_sick_term"]
                     if abs(res["averted_sick_term"]) > 1e-6 else float("nan"))
            row = dict(_key=k, p_work_base=pw_b, p_school_base=ps_b, season=s,
                        R_peak=fit["R0"], pi=fit["pi"],
                        ratio_school_over_sick=ratio,
                        **{k2: v for k2, v in res.items()},
                        wall=time.time() - t_c)
            append_partial(PARTIAL, row)
            done[k] = row
            print(f"  [{n}/{total}] p=({pw_b},{ps_b}) {s} "
                  f"sick_t={res['averted_sick_term']:+.2f}% "
                  f"school_t={res['averted_school_term']:+.2f}% ratio={ratio:.1f}x",
                  flush=True)

    rows = list(done.values())

    # sign audit over the baseline grid, per (season, window, age)
    ages = list(FS.HIRA_AGE_GROUPS)
    by_sc = {(r["season"], r["p_work_base"], r["p_school_base"]): r for r in rows}
    reversals = []
    for s in SEASONS:
        for wn in ("term", "vac"):
            for ag in ages:
                signs = []
                for pw_b, ps_b in BASE_COMBOS:
                    r = by_sc.get((s, pw_b, ps_b))
                    if r is None:
                        continue
                    v = r[f"d_attack_sick_by_age_{wn}"][ag]
                    signs.append(0 if v == 0 else (1 if v > 0 else -1))
                nz = [x for x in signs if x != 0]
                if nz and len(set(nz)) > 1:
                    reversals.append(dict(season=s, window=wn, age=ag,
                                          combos=[list(c) for c in BASE_COMBOS],
                                          signs=signs))

    FINAL.write_text(json.dumps(dict(
        meta=dict(base_combos=[list(c) for c in BASE_COMBOS], seasons=SEASONS,
                  pi_work_pin=PI_WORK_PIN, p_policy=POL, n_rows=len(rows),
                  fit_note=("point-fit once per season at the default baseline "
                            "p=0.6; only the evaluation baseline varies. This "
                            "matches sens5_baseline_p.py's code (not its "
                            "docstring, which claims a refit per combination)."),
                  pin_note="v5 pinned pi_work at 0.29; this uses the fluA4 posterior median",
                  v5_counterpart="outputs/eda/sens_baseline_p_seasonpop.json"),
        rows=rows, sign_reversals=reversals, n_sign_reversals=len(reversals),
        wall_sec=time.time() - t0,
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    ), indent=2, default=float))
    print(f"\n[sign reversals] {len(reversals)}")
    for r in reversals:
        print(f"    {r['season']} {r['window']} {r['age']}: {r['signs']}")
    print(f"[{NAME}] final saved  wall={time.time()-t0:.1f}s -> {FINAL}")
    ntfy(f"sens5_fluA4 baseline p done ({len(rows)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ntfy(f"sens5_fluA4 failed: {type(e).__name__}")
        raise
