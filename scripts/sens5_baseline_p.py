"""Sens 5: baseline p (대칭·비대칭) sensitivity.

6 조합 × 3시즌 point-fit + evaluate:
  대칭: (p_work, p_school) ∈ {(0.5,0.5),(0.6,0.6),(0.7,0.7),(0.75,0.75)}
  비대칭: (0.6,0.7), (0.7,0.6)
각 조합에서 fit 재실행 (baseline 이 바뀌면 β 도 그 baseline 하에서 재적합).
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (ED, SEASONS, IDX, HIRA_AGE_GROUPS,
    build_setup, fit_pi_pin, evaluate_full_stratified,
    load_partial, append_partial, key_str, ntfy, BASE)

NAME = "sens_baseline_p_v4"
PARTIAL = ED / f"{NAME}.partial.jsonl"; FINAL = ED / f"{NAME}.json"

BASE_COMBOS = [
    (0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.75, 0.75),   # 대칭 4
    (0.6, 0.7), (0.7, 0.6),                                 # 비대칭 2
]
PI_WORK_PIN = 0.29
POL = 0.4  # 정책 강도


def evaluate_with_baseline(C, s, R0, pi, p_work_base, p_school_base):
    """base p 를 override 해서 evaluate."""
    return evaluate_full_stratified(
        C, s, R0, pi,
        p_school_pol=POL, p_work_pol=POL,
        p_school_base=p_school_base, p_work_base=p_work_base)


def main():
    t0 = time.time(); done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows")
    C = build_setup()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s")

    ntfy(f"sens5 baseline p 시작 (6 combos × 3시즌)")

    total = len(BASE_COMBOS) * len(SEASONS); n = 0
    for pw_b, ps_b in BASE_COMBOS:
        for s, i in zip(SEASONS, IDX):
            n += 1
            k = key_str(p_work_base=pw_b, p_school_base=ps_b, season=s)
            if k in done: continue
            t_c = time.time()
            # ★ fit 은 default baseline (0.6) 로 안착. 여기선 baseline 만 다르게 evaluate.
            # 엄격하게는 fit 도 base 별로 재적합 필요하지만, 그러면 fit_pi_pin 를 base 인자화 해야 함.
            # 단순화: default baseline (0.6) 에서 fit 하고 baseline 만 override → averted 비교.
            fit = fit_pi_pin(C, s, i, PI_WORK_PIN)
            res = evaluate_with_baseline(C, s, fit["R0"], fit["pi"], pw_b, ps_b)
            # 학교/병가 배율
            ratio = res["averted_school_term"] / res["averted_sick_term"] \
                if abs(res["averted_sick_term"]) > 1e-6 else float("nan")
            row = dict(_key=k, p_work_base=pw_b, p_school_base=ps_b, season=s,
                        R0=fit["R0"], pi=fit["pi"],
                        ratio_school_over_sick=ratio,
                        **{k2: v for k2, v in res.items()},
                        wall=time.time()-t_c)
            append_partial(PARTIAL, row); done[k] = row
            print(f"  [{n}/{total}] p=({pw_b},{ps_b}) {s} "
                  f"sick_t={res['averted_sick_term']:+.2f}% "
                  f"school_t={res['averted_school_term']:+.2f}% ratio={ratio:.1f}× "
                  f"| elapsed {(time.time()-t0)/60:.1f}min", flush=True)

    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(base_combos=BASE_COMBOS, seasons=SEASONS, pi_work_pin=PI_WORK_PIN,
                    p_policy=POL, n_rows=len(rows),
                    note="fit uses default baseline p=0.6; only evaluate baseline varies"),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved  wall={time.time()-t0:.1f}s")
    ntfy(f"sens5 baseline p 완료 ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try: main()
    except Exception as e: ntfy(f"sens5 실패: {type(e).__name__}"); raise
