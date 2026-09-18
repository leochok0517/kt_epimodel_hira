"""Sens 2 (fluA4): kappa upper-range sweep with credible intervals.

Copy of sens2_kappa_upper.py with the posterior/setup swapped to fluA4's.
outputs/fluA/sens_kappa_fluA4.json (from prompts/09) only has the child
vacation-window Delta-attack sign, no CI on total averted infection -- this
fills that gap with the full evaluate_full_stratified output (matching
sens2's schema exactly, so sens_S2_kappa_upper_CI's plotting code needs only
season-list and file-path changes).

Grid kappa in {0.34, 0.40, 0.50, 0.70, 1.00} (matches v5's sens2, not sens1's
6-point grid) x 4 seasons x N_POST=50 forward sims (no re-fit -- reuses
fluA4 posterior draws directly).
"""
from __future__ import annotations
import os, time, json, sys
from pathlib import Path
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sens_common import (
    evaluate_full_stratified, load_partial, append_partial, key_str, ntfy,
)
import fluA_setup as FS

REPO = Path(__file__).resolve().parent.parent
FLUA = REPO / "outputs" / "fluA"
NAME = "sens_kappa_upper_fluA4"
PARTIAL = FLUA / f"{NAME}.partial.jsonl"
FINAL = FLUA / f"{NAME}.json"
MERGED = REPO / "outputs" / "eda" / "nuts_fluA4_raw.npz"

SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
KAPPAS = [0.34, 0.40, 0.5, 0.7, 1.0]
N_POST = 50
SEED = 0


def kap_vec(k):
    ratio = 0.34 / 0.40
    return np.array([k * ratio] * 4 + [k] * 10 + [0.0])


def build_setup_by_season():
    C_all = FS.build_fluA_setup(SEASONS, verbose=True)
    C_by_s = {}
    for s in SEASONS:
        C_by_s[s] = dict(
            shared=C_all["shared_by_s"][s], ngm3=C_all["ngm3_by_s"][s],
            st={s: C_all["st_by_s"][s]}, pop6=C_all["pop6_by_s"][s],
            H=C_all["H"],
        )
    return C_by_s


def main():
    t0 = time.time()
    done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows")

    C_by_s = build_setup_by_season()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s")

    d = np.load(MERGED, allow_pickle=True)
    pi_all = np.asarray(d["pi"]); log_R0_all = np.asarray(d["log_R0"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pi_all.shape[0], size=N_POST, replace=False)
    pi_s = pi_all[idx]; log_R0_s = log_R0_all[idx]

    ntfy(f"sens2_fluA4 kappa upper start (grid {len(KAPPAS)} x {len(SEASONS)} x N={N_POST})")

    total = len(KAPPAS) * len(SEASONS) * N_POST
    n = 0
    for kap_s in KAPPAS:
        for j, s in enumerate(SEASONS):
            C = C_by_s[s]
            for k_i in range(N_POST):
                n += 1
                k = key_str(kappa=kap_s, season=s, post=k_i)
                if k in done:
                    continue
                R0 = float(np.exp(log_R0_s[k_i, j]))
                pi = pi_s[k_i].tolist()
                t_c = time.time()
                res = evaluate_full_stratified(C, s, R0, pi, kap=kap_vec(kap_s))
                row = dict(_key=k, kappa=kap_s, season=s, post=k_i,
                            R0=R0, pi=pi,
                            **{k2: v for k2, v in res.items()},
                            wall=time.time() - t_c)
                append_partial(PARTIAL, row)
                done[k] = row
                if n % 30 == 0 or n == total:
                    elapsed = time.time() - t0
                    print(f"  [{n}/{total}] kappa={kap_s} {s} post={k_i} "
                          f"sick_t={res['averted_sick_term']:+.2f}% "
                          f"| elapsed {elapsed/60:.1f}min", flush=True)

    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(kappa_grid=KAPPAS, seasons=SEASONS, n_post=N_POST,
                    n_rows=len(rows)),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved  wall={time.time()-t0:.1f}s")
    ntfy(f"sens2_fluA4 kappa upper done ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ntfy(f"sens2_fluA4 failed: {type(e).__name__}")
        raise
