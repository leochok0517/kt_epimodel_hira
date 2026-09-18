"""Sens 3 (fluA4): adult-immunity x R0 transition grid for the winter-break
child-ward sign.

Copy of sens3_R0_transition.py with the setup swapped to fluA_setup, and the
pi_work pin set to the fluA4 posterior mean (0.315) rather than v5's 0.29.
Same 5x5 immunity grid, same point-fit-per-grid-point design (each grid
point rebuilds the full setup, matching v5's own -- imm only changes the NGM
normalization and initial state, but mirroring v5's structure exactly keeps
this a pure target/season/population swap, per prompts/99_conventions.md).

25 grid points x 4 seasons = 100 fit_pi_pin point-fits (v5 had 75, 3 seasons).
"""
from __future__ import annotations
import os, time, json, sys
from pathlib import Path
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sens_common import (
    fit_pi_pin, evaluate_full_stratified, load_partial, append_partial,
    key_str, ntfy,
)
import fluA_setup as FS

REPO = Path(__file__).resolve().parent.parent
FLUA = REPO / "outputs" / "fluA"
NAME = "sens_r0_immunity_fluA4"
PARTIAL = FLUA / f"{NAME}.partial.jsonl"
FINAL = FLUA / f"{NAME}.json"

SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
IDX = list(range(len(SEASONS)))   # positional (fluA_setup builds obs per-request, see sens1_piwork_kappa_fluA4.py)

IMM_45_64 = [0.45, 0.50, 0.55, 0.60, 0.65]
IMM_20_44 = [0.30, 0.34, 0.38, 0.42, 0.45]

#: pi_work pinned at the fluA4 posterior mean (compare_v5_fluA4.json /
#: policy_posterior_fluA4.json: pi_work = 0.315), not v5's 0.29.
PI_WORK_PIN = 0.315
CHILD_AGES = ["0-5", "6-11", "12-17"]


def build_imm(imm_20_44, imm_45_64):
    return np.array([0.10] * 4 + [imm_20_44] * 5 + [imm_45_64] * 4 + [0.65] * 2)


def build_setup_by_season(imm_vec):
    C_all = FS.build_fluA_setup(SEASONS, imm=imm_vec, verbose=False)
    C_by_s = {}
    for s in SEASONS:
        C_by_s[s] = dict(
            shared=C_all["shared_by_s"][s], ngm3=C_all["ngm3_by_s"][s],
            st={s: C_all["st_by_s"][s]}, obs=C_all["obs"], w=C_all["w"],
            nw=C_all["nw"], H=C_all["H"], pop6=C_all["pop6_by_s"][s],
        )
    return C_by_s


def main():
    t0 = time.time()
    done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows")

    ntfy(f"sens3_fluA4 R0 transition start (5x5 x {len(SEASONS)}seasons = "
         f"{5*5*len(SEASONS)} fits)")

    total = len(IMM_20_44) * len(IMM_45_64) * len(SEASONS)
    n = 0
    for imm2 in IMM_20_44:
        for imm4 in IMM_45_64:
            imm_vec = build_imm(imm2, imm4)
            C_by_s = build_setup_by_season(imm_vec)
            for s, i in zip(SEASONS, IDX):
                n += 1
                k = key_str(imm_20_44=imm2, imm_45_64=imm4, season=s)
                if k in done:
                    continue
                C = C_by_s[s]
                t_c = time.time()
                fit = fit_pi_pin(C, s, i, PI_WORK_PIN)
                res = evaluate_full_stratified(C, s, fit["R0"], fit["pi"])
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
                            wall=time.time() - t_c)
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
    ntfy(f"sens3_fluA4 R0 transition done ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ntfy(f"sens3_fluA4 failed: {type(e).__name__}")
        raise
