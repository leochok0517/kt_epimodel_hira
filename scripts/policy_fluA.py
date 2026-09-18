"""Policy posterior forward for the influenza-A refit.

Rather than copying ``policy_seasonpop.py`` (300+ lines of scenario logic that
the manuscript's numbers depend on), this runs that module unchanged with three
things swapped underneath it:

* ``sens_common.SEASONS``            -> the influenza-A season set
* ``season_pop_setup.build_seasonwise_setup`` -> ``fluA_setup.build_fluA_setup``
* the posterior npz, output json and resume file -> ``*_fluA*`` paths

The swaps happen **before** ``policy_seasonpop`` is imported, because that
module binds ``SEASONS`` and ``build_seasonwise_setup`` at import time. The
scenario code, the window definitions, the summary statistics and the resume
logic are therefore byte-identical to the published run, and any difference in
the output comes from the posterior and the target alone.

``build_fluA_setup`` already returns the keys ``policy_seasonpop`` consumes
(``shared_by_s``, ``ngm3_by_s``, ``st_by_s``, ``H``, ``pop6_by_s``), so the
adapter only forwards arguments.

Usage:
    python scripts/policy_fluA.py [--npz PATH] [--n-post 500] [--seasons ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str,
                    default=str(REPO / "outputs" / "eda" / "nuts_fluA_raw.npz"))
    ap.add_argument("--n-post", type=int, default=500)
    ap.add_argument("--seasons", nargs="*", default=None)
    ap.add_argument("--tag", type=str, default="fluA")
    args = ap.parse_args()

    import fluA_setup as FS

    seasons = args.seasons if args.seasons else list(FS.SEASONS_DEFAULT)

    # ---- swap 1: the season list every downstream module reads -------------
    import sens_common
    sens_common.SEASONS = seasons

    # ---- swap 2: the setup builder ----------------------------------------
    import season_pop_setup

    def _build_fluA(imm=None, gamma_15=None, use_season_pop: bool = True):
        return FS.build_fluA_setup(
            seasons=seasons, imm=imm, gamma_15=gamma_15,
            use_season_pop=use_season_pop,
        )

    season_pop_setup.build_seasonwise_setup = _build_fluA

    # ---- now import the unmodified policy module --------------------------
    import policy_seasonpop as P

    npz = Path(args.npz)
    if not npz.exists():
        raise FileNotFoundError(
            f"posterior npz not found: {npz} - run scripts/nuts_fluA_run.py first"
        )

    # ---- swap 3: paths and sample size ------------------------------------
    P.SEASONS = seasons
    P.MERGED_NPZ = npz
    P.OUT_JSON = REPO / "outputs" / "fluA" / f"policy_posterior_{args.tag}.json"
    P.PARTIAL = REPO / "outputs" / "fluA" / f"policy_posterior_{args.tag}_partial.jsonl"
    P.N_POST = args.n_post
    P.OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("policy_fluA — policy_seasonpop with the influenza-A posterior")
    print(f"  seasons : {seasons}")
    print(f"  npz     : {npz}")
    print(f"  N_POST  : {P.N_POST}")
    print(f"  out     : {P.OUT_JSON}")
    print(f"  resume  : {P.PARTIAL} (delete it to start clean)")
    print("=" * 90, flush=True)

    P.main()


if __name__ == "__main__":
    main()
