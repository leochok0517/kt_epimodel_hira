"""Influenza-A-only calibration target.

The NHIS / HIRA claim counts used by :mod:`hira_target` are coded J09-J11 and
therefore mix influenza A and B. This module rescales those weekly counts by
the influenza-A share of positive sentinel specimens, so the calibration
target is the influenza-A epidemic alone:

.. code::

    obs_A[t, a] = obs_all[t, a] * a_share[t]

Rationale
---------
The model is a single-pathogen SVEIR with an influenza-A presymptomatic
weight. Fitting it to A+B claims makes it absorb a B wave that has different
timing and, per current evidence, no clear presymptomatic transmission. The
A share is strongly season-dependent (about 0.5 of the 2017-18 wave against
about 1.0 in 2016-17 and 2019-20), so the correction is not a common scale
factor that the reporting fraction could absorb - it changes curve shape both
within and across seasons.

Assumptions this introduces (state them in the manuscript)
----------------------------------------------------------
1. **National, not regional.** The sentinel positivity stream is nationwide;
   the claims are the Seoul Capital Area. The A:B mix is assumed equal.
2. **Not age-stratified.** The same weekly share multiplies all six age
   groups, so the correction cannot express an age-varying A:B mix.
3. **Specimen week vs onset week.** Positivity is dated by the surveillance
   week of the specimen, claims by the treatment date already aligned to
   onset; the two are assumed to refer to the same week (they overlap on 6 of
   7 days after the Sunday-to-Monday shift performed in the loader).
4. **Type-independent reporting.** A and B cases are assumed to generate
   claims at the same rate, so ``gamma_report`` keeps its meaning.

Missing weeks are interpolated **within the fitted window only**, and the
resulting coverage is returned so a season with too little support can be
rejected rather than silently imputed. See ``kt_data.data.load_flu_positivity``
for the smoothing of the share itself.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from kt_data.data.load_flu_positivity import flu_a_share_weekly
from kt_epimodel_hira.calibration.hira_target import (
    HIRA_GROUP_TO_NIMS_WEIGHTED,
    load_hira_target_by_age,
)

#: Default centred smoothing width for the A share, in weeks.
SMOOTH_WEEKS: int = 3

#: Minimum share of the fitted epidemic *mass* that must carry an observed
#: A share (as opposed to an interpolated one) for a season to be usable.
#: Coverage is mass-weighted on purpose: the weeks the positivity source most
#: often omits are the near-zero shoulder weeks, where an interpolated share
#: multiplies almost no claims and cannot move the likelihood. Counting weeks
#: instead would reject seasons whose epidemic wave is fully observed.
MIN_OBSERVED_COVERAGE: float = 0.80


def season_week_dates(season: str, n_weeks: int) -> list[_dt.date]:
    """ISO-week Mondays for ``week_in_season`` 0 .. n_weeks-1.

    Week 0 is the Monday of ISO week 36 of the season's start year, matching
    ``kt_data.data.load_hira.extract_hira_season``.
    """
    start_year = int(season.split("-")[0])
    start = _dt.date.fromisocalendar(start_year, 36, 1)
    return [start + _dt.timedelta(days=7 * i) for i in range(n_weeks)]


def a_share_for_season(
    season: str,
    n_weeks: int,
    weights: np.ndarray | None = None,
    magnitude: np.ndarray | None = None,
    smooth_weeks: int = SMOOTH_WEEKS,
    min_observed_coverage: float = MIN_OBSERVED_COVERAGE,
) -> dict:
    """Weekly influenza-A share aligned to a season's week grid.

    Args:
        season: e.g. ``'2017-2018'``.
        n_weeks: length of the season grid.
        weights: optional ``(n_weeks,)`` NLL weights. Coverage is judged on
            the weeks that actually enter the likelihood, so trailing weeks
            zeroed out by ``first_peak_only`` do not count against a season.
        magnitude: optional ``(n_weeks,)`` claim counts. When given, coverage
            is weighted by ``weights * magnitude`` rather than by ``weights``
            alone, so that a missing share on a near-zero shoulder week does
            not count the same as a missing share at the peak.
        smooth_weeks: centred smoothing width passed to the loader.
        min_observed_coverage: minimum coverage for the season to be accepted.

    Returns:
        ``{'a_share', 'observed', 'coverage', 'coverage_weeks',
        'n_interpolated', 'dates'}``. ``a_share`` is gap-filled by linear
        interpolation in week index and held constant beyond the observed
        range; ``observed`` marks the weeks that needed no filling.
        ``coverage`` is the mass-weighted figure that gates acceptance and
        ``coverage_weeks`` the plain week count, reported for transparency.

    Raises:
        ValueError: if fewer than two weeks of the season carry an observed
            share, or if coverage over the weighted window is below
            ``min_observed_coverage``.
    """
    dates = season_week_dates(season, n_weeks)
    tbl = flu_a_share_weekly(smooth_weeks=smooth_weeks)
    lut = dict(zip(tbl["iso_week_start"].to_list(), tbl["a_share"].to_list()))

    raw = np.array(
        [lut.get(d, np.nan) if lut.get(d) is not None else np.nan for d in dates],
        dtype=np.float64,
    )
    observed = ~np.isnan(raw)
    if observed.sum() < 2:
        raise ValueError(
            f"season {season!r}: only {int(observed.sum())} week(s) of influenza-A "
            f"share available - cannot build an A-only target"
        )

    idx = np.arange(n_weeks, dtype=np.float64)
    a_share = np.interp(idx, idx[observed], raw[observed])

    w = np.ones(n_weeks) if weights is None else np.asarray(weights, dtype=np.float64)
    m = (
        np.ones(n_weeks)
        if magnitude is None
        else np.maximum(np.asarray(magnitude, dtype=np.float64), 0.0)
    )
    wsum = float(w.sum())
    coverage_weeks = float((w * observed).sum() / wsum) if wsum > 0 else 0.0
    msum = float((w * m).sum())
    coverage = float((w * m * observed).sum() / msum) if msum > 0 else coverage_weeks
    if coverage < min_observed_coverage:
        raise ValueError(
            f"season {season!r}: influenza-A share observed for only "
            f"{coverage:.0%} of the fitted epidemic mass "
            f"({coverage_weeks:.0%} of weeks; need "
            f"{min_observed_coverage:.0%}) - exclude the season or widen "
            f"the positivity source"
        )

    return {
        "a_share": a_share,
        "observed": observed,
        "coverage": coverage,
        "coverage_weeks": coverage_weeks,
        "n_interpolated": int((~observed).sum()),
        "dates": dates,
    }


def load_fluA_target_by_age(
    season: str,
    sido_codes: list[int] | None = None,
    setting: str = "outpatient_inpatient",
    first_peak_only: bool = True,
    first_peak_end_week: int = 26,
    smooth_weeks: int = SMOOTH_WEEKS,
    min_observed_coverage: float = MIN_OBSERVED_COVERAGE,
) -> dict:
    """A-only version of :func:`hira_target.load_hira_target_by_age`.

    Returns the same dict, with ``hira_counts`` rescaled to influenza A, plus:

    ``a_share``
        ``(n_weeks,)`` multiplier actually applied.
    ``a_share_observed``
        ``(n_weeks,)`` bool, False where the share was interpolated.
    ``a_share_coverage``
        share of the fitted epidemic mass with an observed A share.
    ``a_share_coverage_weeks``
        the same as a plain fraction of weighted weeks.
    ``hira_counts_all``
        the uncorrected A+B counts, kept for diagnostics and figures.

    The rescaled counts are **not integers**. ``nb_nll_jax`` evaluates the
    negative-binomial log-pmf through ``lgamma`` and so extends continuously;
    the likelihood is therefore a continuous NB rather than a count NB. That
    is intentional: rounding would discard the correction on small age-week
    cells, where it matters most.
    """
    tgt = load_hira_target_by_age(
        season,
        sido_codes=sido_codes,
        setting=setting,
        first_peak_only=first_peak_only,
        first_peak_end_week=first_peak_end_week,
    )
    any_ag = next(iter(tgt["hira_counts"]))
    n_weeks = len(tgt["hira_counts"][any_ag])

    all_ages = np.sum([v for v in tgt["hira_counts"].values()], axis=0)
    sh = a_share_for_season(
        season,
        n_weeks=n_weeks,
        weights=tgt["weights"][any_ag],
        magnitude=all_ages,
        smooth_weeks=smooth_weeks,
        min_observed_coverage=min_observed_coverage,
    )
    a_share = sh["a_share"]

    tgt["hira_counts_all"] = {k: v.copy() for k, v in tgt["hira_counts"].items()}
    tgt["hira_counts"] = {
        ag: v * a_share for ag, v in tgt["hira_counts"].items()
    }
    tgt["a_share"] = a_share
    tgt["a_share_observed"] = sh["observed"]
    tgt["a_share_coverage"] = sh["coverage"]
    tgt["a_share_coverage_weeks"] = sh["coverage_weeks"]
    tgt["virus_type"] = "A"
    return tgt


def estimate_initial_infected_from_fluA(
    season: str,
    pop_15: np.ndarray,
    sido_codes: list[int] | None = None,
    setting: str = "outpatient_inpatient",
    week_zero_average_n: int = 3,
    gamma_15_assumed: np.ndarray | None = None,
    smooth_weeks: int = SMOOTH_WEEKS,
) -> np.ndarray:
    """Initial infected seed from the A-corrected early-season baseline.

    Mirrors ``simple_model.estimate_initial_infected_from_hira`` -
    ``I = baseline_count / gamma_weighted``, distributed within each HIRA group
    by population - but takes the baseline from the A-corrected counts, so the
    seed is consistent with the target the model is fitted to.

    Early-season weeks are exactly where the A share is least well observed
    (low specimen positivity, coarse rounding). The share used here is
    therefore the one from :func:`a_share_for_season`, which is smoothed and
    interpolated on the same grid as the target, and coverage is not enforced:
    the seed is a starting value, not a likelihood term.
    """
    from kt_epimodel_hira.model.parameters import CalibrationParameters

    gamma_15 = (
        np.asarray(gamma_15_assumed, dtype=np.float64)
        if gamma_15_assumed is not None
        else CalibrationParameters().gamma_15
    )
    if (gamma_15 <= 0).any():
        raise ValueError(f"gamma values must be > 0, got min={gamma_15.min()}")

    pop_arr = np.asarray(pop_15, dtype=np.float64)
    if pop_arr.ndim == 2:
        pop_total_15 = (
            pop_arr.sum(axis=1) if pop_arr.shape[0] == 15 else pop_arr.flatten()
        )
    else:
        pop_total_15 = pop_arr
    if pop_total_15.shape != (15,):
        raise ValueError(f"pop_15 must reduce to (15,), got {pop_total_15.shape}")

    tgt = load_hira_target_by_age(
        season, sido_codes=sido_codes, setting=setting, first_peak_only=False
    )
    any_ag = next(iter(tgt["hira_counts"]))
    n_weeks = len(tgt["hira_counts"][any_ag])
    a_share = a_share_for_season(
        season, n_weeks=n_weeks, smooth_weeks=smooth_weeks,
        min_observed_coverage=0.0,
    )["a_share"]

    n0 = int(week_zero_average_n)
    baseline = {
        ag: float(np.nanmean((v * a_share)[:n0]))
        for ag, v in tgt["hira_counts"].items()
    }

    I_initial_15 = np.zeros(15, dtype=np.float64)
    for ag, weights in HIRA_GROUP_TO_NIMS_WEIGHTED.items():
        wpop = sum(w * float(pop_total_15[i]) for i, w in weights.items())
        if wpop < 1e-10:
            continue
        gamma_avg = max(
            sum(
                w * float(pop_total_15[i]) * float(gamma_15[i])
                for i, w in weights.items()
            )
            / wpop,
            1e-10,
        )
        I_group = baseline.get(ag, 0.0) / gamma_avg
        for i, w in weights.items():
            I_initial_15[i] += I_group * w * float(pop_total_15[i]) / wpop
    return I_initial_15
