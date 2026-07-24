"""Sens 2: κ 상한 확장 (Ferguson 2006 검증).

Grid κ ∈ {0.34, 0.40(기준), 0.5, 0.7, 1.0} × 3 시즌 × posterior N=50.
π·R0 는 v4 NUTS posterior 재사용 (β 재유도, fit 없음).
층화: term/vac × age × season.
"""
from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sens_common import (
    ED, SEASONS, IDX, HIRA_AGE_GROUPS,
    build_setup, evaluate_full_stratified, PI_POST, LOG_R0_POST,
    load_partial, append_partial, key_str, ntfy,
)

NAME = "sens_kappa_upper_v4"
PARTIAL = ED / f"{NAME}.partial.jsonl"
FINAL = ED / f"{NAME}.json"

KAPPAS = [0.34, 0.40, 0.5, 0.7, 1.0]
N_POST = 50
SEED = 0


def kap_vec(k):
    ratio = 0.34 / 0.40
    return np.array([k*ratio]*4 + [k]*10 + [0.0])


def main():
    t0 = time.time()
    done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows")
    C = build_setup()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(PI_POST.shape[0], size=N_POST, replace=False)
    pi_s = PI_POST[idx]; log_R0_s = LOG_R0_POST[idx]

    ntfy(f"sens2 κ 상한 시작 (grid {len(KAPPAS)} × {len(SEASONS)} × N={N_POST})")

    total = len(KAPPAS) * len(SEASONS) * N_POST
    n = 0
    for kap_s in KAPPAS:
        for j, s in enumerate(SEASONS):
            for k_i in range(N_POST):
                n += 1
                k = key_str(kappa=kap_s, season=s, post=k_i)
                if k in done:
                    continue
                R0 = float(np.exp(log_R0_s[k_i, j]))
                pi = pi_s[k_i].tolist()
                t_c = time.time()
                res = evaluate_full_stratified(
                    C, s, R0, pi, kap=kap_vec(kap_s))
                row = dict(_key=k, kappa=kap_s, season=s, post=k_i,
                            R0=R0, pi=pi,
                            **{k2: v for k2, v in res.items()},
                            wall=time.time()-t_c)
                append_partial(PARTIAL, row)
                done[k] = row
                if n % 30 == 0 or n == total:
                    elapsed = time.time() - t0
                    print(f"  [{n}/{total}] κ={kap_s} {s} post={k_i} "
                          f"sick_t={res['averted_sick_term']:+.2f}% "
                          f"| elapsed {elapsed/60:.1f}min", flush=True)

    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(kappa_grid=KAPPAS, seasons=SEASONS, n_post=N_POST,
                    n_rows=len(rows)),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved  wall={time.time()-t0:.1f}s")
    ntfy(f"sens2 κ 상한 완료 ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ntfy(f"sens2 실패: {type(e).__name__}")
        raise
