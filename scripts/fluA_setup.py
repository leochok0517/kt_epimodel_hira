"""Season-wise setup for the influenza-A refit.

Companion to ``season_pop_setup.py``, which is left untouched so the published
v5/v7 fit stays reproducible. Two things change here:

1. **Target.** Weekly claim counts are rescaled to influenza A by
   ``kt_epimodel_hira.calibration.fluA_target``. The initial infected seed is
   rescaled with the same share, so seed and target refer to one pathogen.
2. **Season list.** The season set is an argument rather than a module
   constant, and per-season populations come from
   ``data/population/pop_all_seasons_fluA.csv`` (built by
   ``scripts/build_pop_seasons_fluA.py``), which covers every season the KOSIS
   export can supply instead of only the three used in v5.

Everything else - contact matrices, mobility, employment, vaccination,
seasonality, the fixed phi / gamma / kappa / immunity vectors, and the
presymptomatic weight - is taken from ``sens_common.build_setup`` and
``nuts_seasonpop``, unchanged and unre-derived, so that a difference between
this fit and v5 can only come from the target, the season set, or the
population.

Usage::

    from fluA_setup import build_fluA_setup, SEASONS_DEFAULT
    C = build_fluA_setup(SEASONS_DEFAULT)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kt_data import SUDOGWON_SIDO_CODES  # noqa: E402
from kt_epimodel_hira.calibration.fluA_target import (  # noqa: E402
    estimate_initial_infected_from_fluA,
    load_fluA_target_by_age,
)
from kt_epimodel_hira.calibration.hira_target import (  # noqa: E402
    HIRA_AGE_GROUPS,
    HIRA_GROUP_TO_NIMS_WEIGHTED,
)
from kt_epimodel_hira.calibration.simple_model import (  # noqa: E402
    _build_initial_state_with_age_seed,
)
from kt_epimodel_hira.jax_model.numpyro_model import make_ngm_eigvalue_fn  # noqa: E402

import nuts_seasonpop as NS  # noqa: E402
from sens_common import build_setup as build_setup_default  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
POP_CSV = REPO / "data" / "population" / "pop_all_seasons_fluA.csv"

#: Seasons that pass the influenza-A single-wave screen *and* have a
#: population. Five consecutive pre-COVID seasons.
SEASONS_DEFAULT: list[str] = [
    "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
]

#: Seasons that pass the single-wave screen but have no population available.
#: The KOSIS 행정구역(읍면동)별 5세별 주민등록인구 table itself starts at 2015 —
#: the series was restructured then, so re-exporting cannot reach further back
#: (checked 2026-08-28). Adding these two means sourcing the population from a
#: different table, e.g. the 시도/시군구 age breakdown, which covers earlier
#: years and suffices here because only Seoul, Incheon and Gyeonggi are summed.
SEASONS_PENDING_POP: list[str] = ["2013-2014", "2014-2015"]

FIRST_PEAK_END_WEEK: int = 26


def load_pop_by_season_fluA() -> dict[str, np.ndarray]:
    """``{'2015-2016': (15,), ...}`` from pop_all_seasons_fluA.csv."""
    if not POP_CSV.exists():
        raise FileNotFoundError(
            f"{POP_CSV} not found - run scripts/build_pop_seasons_fluA.py first"
        )
    out: dict[str, np.ndarray] = {}
    with open(POP_CSV) as f:
        for row in csv.DictReader(f):
            short = row["season"]                       # e.g. 2015_16
            y0 = short.split("_")[0]
            season = f"{y0}-{int(y0) + 1}"
            out.setdefault(season, np.zeros(15, dtype=np.float64))
            out[season][int(row["band_index"])] = float(row["population"])
    return out


def build_hira_matrix() -> np.ndarray:
    """(6, 15) HIRA-group to NIMS-band weight matrix."""
    H = np.zeros((6, 15))
    for i, ag in enumerate(HIRA_AGE_GROUPS):
        for j, w in HIRA_GROUP_TO_NIMS_WEIGHTED[ag].items():
            H[i, j] = w
    return H


def build_fluA_setup(
    seasons: list[str] | None = None,
    imm: np.ndarray | None = None,
    gamma_15: np.ndarray | None = None,
    use_season_pop: bool = True,
    first_peak_end_week: int = FIRST_PEAK_END_WEEK,
    verbose: bool = True,
) -> dict:
    """Build the per-season inputs for an influenza-A joint fit.

    Args:
        seasons: season labels; defaults to :data:`SEASONS_DEFAULT`.
        imm / gamma_15: override the fixed vectors; default to the values
            ``nuts_seasonpop`` pins.
        use_season_pop: False reuses one reference population for every
            season (regression path, matching ``season_pop_setup``).
        first_peak_end_week: weeks at or beyond this index get zero weight.

    Returns:
        ``{'seasons', 'obs', 'w', 'nw', 'ngm3_by_s', 'st_by_s',
        'shared_by_s', 'pop_15_by_s', 'pop6_by_s', 'H', 'obs_all',
        'a_share_by_s', 'coverage_by_s', 'default_setup'}``.
        ``obs`` / ``w`` are lists indexed like ``seasons`` - unlike the v5
        setup they are not indexed through a module-level season constant.
        ``obs_all`` holds the uncorrected A+B counts for diagnostics.
    """
    seasons = list(seasons) if seasons is not None else list(SEASONS_DEFAULT)
    imm_use = np.asarray(imm) if imm is not None else np.asarray(NS.IMM)
    gamma_use = np.asarray(gamma_15) if gamma_15 is not None else np.asarray(NS.GAMMA)

    # Shared physics — identical to the published pipeline.
    default = build_setup_default(imm=imm_use)
    M = default["shared"]
    rho_arr = np.asarray(M["rho"])
    C_mats = dict(
        C_home=np.asarray(M["C_home"]), C_work=np.asarray(M["C_work"]),
        C_school=np.asarray(M["C_school"]), C_other=np.asarray(M["C_other"]),
    )
    seasonal_factor = 1.0 + float(NS.F.S.AMP)
    gamma_scalar = float(M["gamma"])

    if use_season_pop:
        pop_all = load_pop_by_season_fluA()
        missing = [s for s in seasons if s not in pop_all]
        if missing:
            raise KeyError(
                f"no population for {missing} in {POP_CSV.name}. "
                f"Re-export the KOSIS table with those June columns and re-run "
                f"scripts/build_pop_seasons_fluA.py."
            )
        pop_by_s = {s: pop_all[s] for s in seasons}
    else:
        flat = np.asarray(M["pop_15"])
        flat = flat.sum(1) if flat.ndim == 2 else flat.flatten()
        pop_by_s = {s: flat.copy() for s in seasons}

    ngm3_by_s, st_by_s, shared_by_s = {}, {}, {}
    obs_list, w_list, obs_all_list = [], [], []
    a_share_by_s, coverage_by_s = {}, {}
    nw = None

    for s in seasons:
        pop_flat = pop_by_s[s]
        pop_2d = pop_flat.reshape(15, 1)

        ngm3_by_s[s] = make_ngm_eigvalue_fn(
            pop_15=pop_2d, rho=rho_arr, **C_mats,
            R0_immunity=imm_use, gamma=gamma_scalar,
            seasonal_factor=seasonal_factor,
        )

        tgt = load_fluA_target_by_age(
            s, sido_codes=list(SUDOGWON_SIDO_CODES),
            first_peak_only=True, first_peak_end_week=first_peak_end_week,
        )
        nw_s = len(tgt["hira_counts"][HIRA_AGE_GROUPS[0]])
        if nw_s != tgt["n_weeks"]:
            raise ValueError(
                f"season {s}: {nw_s} weeks of claims but n_weeks="
                f"{tgt['n_weeks']} - the season grid is incomplete, which the "
                f"downstream fixed-length loss cannot represent"
            )
        nw = nw_s if nw is None else nw
        if nw_s != nw:
            raise ValueError(f"season {s}: n_weeks {nw_s} != {nw}")

        obs = np.zeros((nw, 6))
        wgt = np.zeros((nw, 6))
        obs_all = np.zeros((nw, 6))
        for i, ag in enumerate(HIRA_AGE_GROUPS):
            obs[:, i] = tgt["hira_counts"][ag]
            wgt[:, i] = tgt["weights"][ag]
            obs_all[:, i] = tgt["hira_counts_all"][ag]
        obs_list.append(jnp.asarray(obs))
        w_list.append(jnp.asarray(wgt))
        obs_all_list.append(obs_all)
        a_share_by_s[s] = tgt["a_share"]
        coverage_by_s[s] = tgt["a_share_coverage"]

        seed = estimate_initial_infected_from_fluA(
            s, pop_flat, sido_codes=list(SUDOGWON_SIDO_CODES),
            gamma_15_assumed=gamma_use,
        )
        st_by_s[s] = jnp.asarray(_build_initial_state_with_age_seed(
            pop_2d, seed, seed_e_factor=0.5, initial_immunity=imm_use,
            initial_vaccinated_fraction=0.0,
        ))

        sh = dict(M)
        sh["pop_15"] = jnp.asarray(pop_2d)
        shared_by_s[s] = sh

        if verbose:
            keep = wgt[:, 0] > 0
            print(
                f"  {s}: pop={pop_flat.sum():>12,.0f}  "
                f"A-share cov={coverage_by_s[s]:>5.1%}  "
                f"obs_A/obs_all={obs[keep].sum() / max(obs_all[keep].sum(), 1):.3f}  "
                f"peak_week={int(np.argmax(obs.sum(1)))}",
                flush=True,
            )

    H = build_hira_matrix()
    return dict(
        seasons=seasons,
        obs=obs_list, w=w_list, nw=nw,
        obs_all=obs_all_list,
        a_share_by_s=a_share_by_s, coverage_by_s=coverage_by_s,
        ngm3_by_s=ngm3_by_s, st_by_s=st_by_s, shared_by_s=shared_by_s,
        pop_15_by_s=pop_by_s,
        pop6_by_s={s: H @ pop_by_s[s] for s in seasons},
        H=H, default_setup=default,
    )


if __name__ == "__main__":
    print(f"seasons: {SEASONS_DEFAULT}")
    print(f"pending population: {SEASONS_PENDING_POP}")
    C = build_fluA_setup()
    print(f"\nn_weeks={C['nw']}  seasons={len(C['seasons'])}")
