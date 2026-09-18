"""T2: π_work × κ 2D sensitivity map for fluA4 (4-season influenza-A refit).

Same grid and point-fit machinery as ``sens1_piwork_kappa.py`` (v5), with the
setup replaced by ``fluA_setup.build_fluA_setup`` — season-specific
population, influenza-A target, 4 seasons — per CLAUDE.md's fluA convention.
``fit_pi_pin`` / ``evaluate_full_stratified`` / the checkpoint helpers are
reused unmodified from ``sens_common``.

Note on the index scheme: ``sens_common``'s own ``build_setup()`` sources
obs/w from the module's default (season_pop_setup) build, whose obs array is
indexed by ``final_pipeline_confirmed``'s global season list — hence
``IDX = [F.SEASONS.index(s) for s in SEASONS]`` in the original script.
``fluA_setup.build_fluA_setup`` instead builds ``obs``/``w`` positionally for
exactly the requested season list, so this script uses a local positional
index instead of ``F.SEASONS.index``. Everything else about the grid and the
point-fit is unchanged.

Checkpointed (jsonl) and resumable, same as v5.
"""
from __future__ import annotations
import os, time, json, sys
from pathlib import Path
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sens_common import (
    KAP_DEF, fit_pi_pin, evaluate_full_stratified,
    load_partial, append_partial, key_str, ntfy,
)
import fluA_setup as FS

REPO = Path(__file__).resolve().parent.parent
FLUA = REPO / "outputs" / "fluA"
NAME = "sens_piwork_kappa_fluA4"
PARTIAL = FLUA / f"{NAME}.partial.jsonl"
FINAL = FLUA / f"{NAME}.json"

SEASONS = ["2016-2017", "2017-2018", "2018-2019", "2019-2020"]
IDX = list(range(len(SEASONS)))   # positional (see module docstring)

# Grid — identical to v5's sens1_piwork_kappa.py, so the comparison is
# apples to apples (only the target/seasons/population changed).
PI_WORK = [0.03, 0.05, 0.06, 0.09, 0.12, 0.15, 0.17, 0.21, 0.24, 0.27, 0.29, 0.36]
KAPPAS = [0.2, 0.34, 0.4, 0.5, 0.7, 1.0]


def kap_vec(k_scalar):
    """κ scalar → student 0.85·k / adult k / 70+ 0 (matches v5's ratio)."""
    ratio = 0.34 / 0.40
    return np.array([k_scalar*ratio]*4 + [k_scalar]*10 + [0.0])


def build_setup_by_season():
    """Per-season view dicts with the keys fit_pi_pin/evaluate_full_stratified
    expect (``shared``, ``ngm3``, ``st``, ``obs``, ``w``, ``nw``, ``H``,
    ``pop6``), sourced from one ``build_fluA_setup`` call."""
    C_all = FS.build_fluA_setup(SEASONS, verbose=True)
    C_by_s = {}
    for s in SEASONS:
        C_by_s[s] = dict(
            shared=C_all["shared_by_s"][s],
            ngm3=C_all["ngm3_by_s"][s],
            st={s: C_all["st_by_s"][s]},
            obs=C_all["obs"], w=C_all["w"], nw=C_all["nw"],
            H=C_all["H"], pop6=C_all["pop6_by_s"][s],
        )
    return C_by_s


def main():
    t0 = time.time()
    done = load_partial(PARTIAL)
    print(f"[{NAME}] existing {len(done)} rows in partial")

    C_by_s = build_setup_by_season()
    print(f"[{NAME}] setup {time.time()-t0:.1f}s")
    ntfy(f"sens1_fluA4 π_work×κ 시작 (grid={len(PI_WORK)}×{len(KAPPAS)}×{len(SEASONS)}"
         f"={len(PI_WORK)*len(KAPPAS)*len(SEASONS)} combos)")

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
                C = C_by_s[s]
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

    rows = list(done.values())
    FINAL.write_text(json.dumps(dict(
        meta=dict(pi_work_grid=PI_WORK, kappa_grid=KAPPAS,
                    seasons=SEASONS, n_combos=len(rows)),
        rows=rows), indent=2, default=float))
    print(f"[{NAME}] final saved: {FINAL}  ({len(rows)} rows)  wall={time.time()-t0:.1f}s")
    ntfy(f"sens1_fluA4 π_work×κ 완료 ({len(rows)} rows, {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        ntfy(f"sens1_fluA4 실패: {type(e).__name__}")
        raise
