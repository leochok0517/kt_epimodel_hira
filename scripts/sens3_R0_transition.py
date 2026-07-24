"""Sens 3: R(0) 성인 초기면역 전환점 탐색.

Grid: R(0)_45-64 ∈ [0.45, 0.65] × R(0)_20-44 ∈ [0.30, 0.45].  5×5 = 25 combos.
각 combo 에서 IMM 배열 재구성 → build_setup 재호출 → point-fit → 층화 forward.
방학 창 아동 (0-17) Δattack 부호 반전 경계 초점.
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (
    ED, SEASONS, IDX, HIRA_AGE_GROUPS,
    build_setup, fit_pi_pin, evaluate_full_stratified,
    load_partial, append_partial, key_str, ntfy,
)

NAME = "sens_R0_transition_v4"
PARTIAL = ED / f"{NAME}.partial.jsonl"
FINAL = ED / f"{NAME}.json"

# Grid
IMM_45_64 = [0.45, 0.50, 0.55, 0.60, 0.65]      # v4 default 0.60
IMM_20_44 = [0.30, 0.34, 0.38, 0.42, 0.45]      # v4 default 0.40

# π_work fixed at NUTS posterior mean ≈ 0.29
PI_WORK_PIN = 0.29
CHILD_AGES = ["0-5","6-11","12-17"]


def build_imm(imm_20_44, imm_45_64):
    """15-vector R(0). 0-19=0.10, 20-44=imm_20_44, 45-64=imm_45_64, 65+=0.65."""
    return np.array([0.10]*4 + [imm_20_44]*5 + [imm_45_64]*4 + [0.65]*2)


def main():
    t0 = time.time()
    done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows")

    ntfy(f"sens3 R(0) 전환점 시작 (5×5 × 3시즌 = 75 fit)")

    total = len(IMM_20_44) * len(IMM_45_64) * len(SEASONS)
    n = 0
    for imm2 in IMM_20_44:
        for imm4 in IMM_45_64:
            imm_vec = build_imm(imm2, imm4)
            C = build_setup(imm=imm_vec)   # setup 재빌드 (state0, ngm 재계산)
            for s, i in zip(SEASONS, IDX):
                n += 1
                k = key_str(imm_20_44=imm2, imm_45_64=imm4, season=s)
                if k in done:
                    continue
                t_c = time.time()
                fit = fit_pi_pin(C, s, i, PI_WORK_PIN)
                res = evaluate_full_stratified(C, s, fit["R0"], fit["pi"])
                # child Δattack (0-17) 방학창
                d_child_vac_sick = sum(
                    res["d_attack_sick_by_age_vac"][a] for a in CHILD_AGES)
                d_child_term_sick = sum(
                    res["d_attack_sick_by_age_term"][a] for a in CHILD_AGES)
                row = dict(_key=k, imm_20_44=imm2, imm_45_64=imm4,
                            imm_vec=imm_vec.tolist(),
                            season=s, R0=fit["R0"], pi=fit["pi"],
                            beta_4=fit["beta_4"],
                            child_delta_vac_sick=d_child_vac_sick,
                            child_delta_term_sick=d_child_term_sick,
                            **{k2: v for k2, v in res.items()},
                            wall=time.time()-t_c)
                append_partial(PARTIAL, row)
                done[k] = row
                if n % 3 == 0 or n == total:
                    print(f"  [{n}/{total}] imm=[20-44:{imm2}, 45-64:{imm4}] {s} "
                          f"R0={fit['R0']:.3f} vac_child={d_child_vac_sick:+.3f} "
                          f"| elapsed {(time.time()-t0)/60:.1f}min", flush=True)

    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(imm_20_44_grid=IMM_20_44, imm_45_64_grid=IMM_45_64,
                    seasons=SEASONS, pi_work_pin=PI_WORK_PIN, n_rows=len(rows)),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved  wall={time.time()-t0:.1f}s")
    ntfy(f"sens3 R(0) 전환점 완료 ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ntfy(f"sens3 실패: {type(e).__name__}")
        raise
