"""Sens 1: π_work × κ 2D map.  체크포인트·재개 지원.

Grid: π_work × κ.  각 combo 에서 3시즌 shared π-pin point-fit → 층화 forward.
저장: jsonl (per season × combo) + 최종 JSON aggregate.
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (
    ED, SEASONS, IDX, KAP_DEF, HIRA_AGE_GROUPS,
    build_setup, fit_pi_pin, evaluate_full_stratified,
    load_partial, append_partial, key_str, ntfy,
)

NAME = "sens_piwork_kappa_v4"
PARTIAL = ED / f"{NAME}.partial.jsonl"
FINAL = ED / f"{NAME}.json"

# Grid (spec)
PI_WORK = [0.03, 0.05, 0.06, 0.09, 0.12, 0.15, 0.17, 0.21, 0.24, 0.27, 0.29, 0.36]
KAPPAS = [0.2, 0.34, 0.4, 0.5, 0.7, 1.0]


def kap_vec(k_scalar):
    """κ scalar → 학생 0.85·k / 성인 k / 70+ 0 (η-free 비율 유지, 학생/성인 비 0.85)."""
    ratio = 0.34 / 0.40   # 0.85
    return np.array([k_scalar*ratio]*4 + [k_scalar]*10 + [0.0])


def main():
    t0 = time.time()
    done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows in partial")

    C = build_setup()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s")
    ntfy(f"sens1 π_work×κ 시작 (grid={len(PI_WORK)}×{len(KAPPAS)}={len(PI_WORK)*len(KAPPAS)} combos)")

    total = len(PI_WORK) * len(KAPPAS)
    n = 0
    for pw in PI_WORK:
        for kap_s in KAPPAS:
            n += 1
            for s, i in zip(SEASONS, IDX):
                k = key_str(pi_work=pw, kappa=kap_s, season=s)
                if k in done:
                    continue
                t_c = time.time()
                # v4 default: work strong pin σ=0.01 이지만 여기선 π_work를
                # 사전값(pi_work_pin)으로 강제하기 위해 sigma_pin[work]=0.01 유지
                fit = fit_pi_pin(C, s, i, pw, kap=kap_vec(kap_s))
                res = evaluate_full_stratified(
                    C, s, fit["R0"], fit["pi"], kap=kap_vec(kap_s))
                row = dict(_key=k, pi_work=pw, kappa=kap_s, season=s,
                            R0=fit["R0"], pi=fit["pi"], beta_4=fit["beta_4"],
                            **{k2: v for k2, v in res.items()},
                            wall=time.time()-t_c)
                append_partial(PARTIAL, row)
                done[k] = row
                if n % 6 == 0 or n == total:
                    elapsed = time.time() - t0
                    print(f"  [{n}/{total} · {s}] pw={pw:.2f} κ={kap_s:.2f} "
                          f"R0={fit['R0']:.3f} sick_t={res['averted_sick_term']:+.2f}% "
                          f"| elapsed {elapsed/60:.1f}min", flush=True)

    # Final JSON: rows list
    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(pi_work_grid=PI_WORK, kappa_grid=KAPPAS,
                    seasons=SEASONS, n_combos=len(rows)),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved: {FINAL}  ({len(rows)} rows)  wall={time.time()-t0:.1f}s")
    ntfy(f"sens1 π_work×κ 완료 ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ntfy(f"sens1 실패: {type(e).__name__}")
        raise
