"""Sens 4: w (presymptomatic) sweep.  Forward-only + posterior N=50 × 3 시즌.

w ∈ [0.05, 0.45] 15점.  Nature Hlth A (9.6% [5.9,14.7]) + Donnelly (15-25%) 포괄.
정책 averted (term/vac) + 성인 vs 아동 Δ + flip threshold 탐색.
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (ED, SEASONS, IDX, HIRA_AGE_GROUPS,
    build_setup, evaluate_full_stratified, PI_POST, LOG_R0_POST,
    load_partial, append_partial, key_str, ntfy)

NAME = "sens_w_sweep_v4"
PARTIAL = ED / f"{NAME}.partial.jsonl"; FINAL = ED / f"{NAME}.json"

W_GRID = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.22, 0.25, 0.28, 0.31,
           0.34, 0.37, 0.40, 0.43, 0.45]
N_POST = 50; SEED = 0


def main():
    t0 = time.time(); done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows")
    C = build_setup()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(PI_POST.shape[0], size=N_POST, replace=False)
    pi_s = PI_POST[idx]; log_R0_s = LOG_R0_POST[idx]

    ntfy(f"sens4 w sweep 시작 ({len(W_GRID)} w × 3시즌 × N={N_POST})")

    total = len(W_GRID) * len(SEASONS) * N_POST
    n = 0
    for w_val in W_GRID:
        for j, s in enumerate(SEASONS):
            for k_i in range(N_POST):
                n += 1
                k = key_str(w=w_val, season=s, post=k_i)
                if k in done: continue
                R0 = float(np.exp(log_R0_s[k_i, j])); pi = pi_s[k_i].tolist()
                t_c = time.time()
                res = evaluate_full_stratified(C, s, R0, pi, w=w_val)
                row = dict(_key=k, w=w_val, season=s, post=k_i,
                            R0=R0, pi=pi,
                            **{k2: v for k2, v in res.items()},
                            wall=time.time()-t_c)
                append_partial(PARTIAL, row); done[k] = row
                if n % 45 == 0 or n == total:
                    print(f"  [{n}/{total}] w={w_val} {s} post={k_i} "
                          f"sick_t={res['averted_sick_term']:+.2f}% "
                          f"| elapsed {(time.time()-t0)/60:.1f}min", flush=True)

    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(w_grid=W_GRID, seasons=SEASONS, n_post=N_POST, n_rows=len(rows)),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved  wall={time.time()-t0:.1f}s")
    ntfy(f"sens4 w sweep 완료 ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try: main()
    except Exception as e: ntfy(f"sens4 실패: {type(e).__name__}"); raise
