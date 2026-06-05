"""gamma (reporting/detection rate) external-input registry.

Background
----------
γ is not identifiable from data alone — the model produces observations as
predicted = γ × incidence(β, φ, ...). The 3-way multiplicative coupling
(β × φ × γ) creates a posterior ridge along which γ → 0 (compensated by β/φ).
This is documented in:
  - docs/PRIOR_SPECIFICATION.md Appendix A (NUTS diagnosis)
  - docs/GAMMA_STRATEGY.md (final decision)

Conclusion: γ is set externally, not fit. PSA handles uncertainty.

How to use
----------
- Production code calls ``get_active_gamma()`` for a (15,) array, never
  hard-codes γ values.
- To switch γ source (e.g. when Korean data arrives): add an entry to
  ``GAMMA_REGISTRY`` and change ``ACTIVE_GAMMA`` to its key. No other
  code changes.
- Adapters (``gamma_adapters`` below) convert raw data to GammaSource
  entries. Each adapter is a stub until that data category is acquired.

Layout
------
NIMS 15-age groups -> (child / adult / elder) by index range matching
``model/parameters.py::GAMMA_AGE_GROUPS``:
  child: NIMS 0-3   (0-19y)
  adult: NIMS 4-12  (20-64y)
  elder: NIMS 13-14 (65+y)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

CHILD_IDX = list(range(0, 4))    # NIMS 0-3 (0-19y)
ADULT_IDX = list(range(4, 13))   # NIMS 4-12 (20-64y)
ELDER_IDX = list(range(13, 15))  # NIMS 13-14 (65+y)


@dataclass(frozen=True)
class GammaSource:
    """Single external γ specification.

    Frozen to prevent accidental mutation. To override, register a new
    entry under a distinct ``key`` and switch ``ACTIVE_GAMMA``.
    """
    key: str
    child: float
    adult: float
    elder: float
    source: str              # bibliographic citation
    region: str              # e.g. "US", "Korea"
    season: str              # e.g. "2019-2020" or "generic"
    note: str = ""
    psa_sd: tuple = (0.07, 0.05, 0.07)  # PSA std-dev (child, adult, elder)

    def to_15(self) -> np.ndarray:
        """Expand to NIMS 15-age vector."""
        v = np.empty(15, dtype=np.float64)
        for i in CHILD_IDX: v[i] = self.child
        for i in ADULT_IDX: v[i] = self.adult
        for i in ELDER_IDX: v[i] = self.elder
        return v


# ===== Registered γ sources =====
GAMMA_REGISTRY: dict[str, GammaSource] = {
    "cdc_reed2015": GammaSource(
        key="cdc_reed2015",
        child=0.40, adult=0.18, elder=0.25,
        source="CDC Reed et al., PLOS One 2015 — symptomatic multiplier inversion",
        region="US",
        season="generic",
        note=(
            "default. US-based, not Korea-direct. See "
            "docs/AGE_DEPENDENT_GAMMA.md §4 for derivation. PSA (Stage 5) "
            "shakes this with psa_sd."
        ),
        psa_sd=(0.07, 0.05, 0.07),
    ),
    # ---------------------------------------------------------------
    # Korean data slots (uncomment + fill when data arrives via adapters)
    # ---------------------------------------------------------------
    # "korea_senior_TBD": GammaSource(
    #     key="korea_senior_TBD",
    #     child=..., adult=..., elder=...,
    #     source="선배 연구원 데이터 (출처 미정)",
    #     region="Korea",
    #     season="...",
    #     note="adapter 로 변환됨 — see gamma_adapters",
    # ),
    # "korea_himm_2013_adults": GammaSource(...)
}

# === Switch γ source here ===
ACTIVE_GAMMA: str = "cdc_reed2015"


def get_active_gamma() -> np.ndarray:
    """NIMS (15,) array of active γ per age. Sole entry point for fitters."""
    return GAMMA_REGISTRY[ACTIVE_GAMMA].to_15()


def get_active_source() -> GammaSource:
    """Active GammaSource metadata (for logging / provenance)."""
    return GAMMA_REGISTRY[ACTIVE_GAMMA]


def list_sources() -> list[str]:
    return list(GAMMA_REGISTRY.keys())


def gamma_psa_samples(
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw PSA samples of γ around the active source.

    Returns: (n_samples, 15) array. Each row is a (15,) γ vector.
    Truncated Normal on [0.01, 0.99] per (child/adult/elder) group.
    """
    if rng is None:
        rng = np.random.default_rng()
    src = get_active_source()
    sd_c, sd_a, sd_e = src.psa_sd

    def _draw(mu, sd, n):
        # Truncated Normal in [0.01, 0.99] via rejection (sd small -> few rejects)
        out = np.empty(n)
        i = 0
        while i < n:
            cand = rng.normal(mu, sd, n - i)
            ok = (cand >= 0.01) & (cand <= 0.99)
            taken = cand[ok]
            out[i:i+len(taken)] = taken
            i += len(taken)
        return out

    cs = _draw(src.child, sd_c, n_samples)
    as_ = _draw(src.adult, sd_a, n_samples)
    es = _draw(src.elder, sd_e, n_samples)

    gammas = np.empty((n_samples, 15), dtype=np.float64)
    for j in CHILD_IDX: gammas[:, j] = cs
    for j in ADULT_IDX: gammas[:, j] = as_
    for j in ELDER_IDX: gammas[:, j] = es
    return gammas


# ===== Data -> γ adapters (stubs; fill on data arrival) =====

def gamma_from_serosurvey(
    hira_rate_by_age: Sequence[float],
    sero_infection_rate_by_age: Sequence[float],
) -> dict[str, float]:
    """Paired serosurvey -> γ = HIRA claims rate / true infection rate.

    Args:
        hira_rate_by_age: HIRA episodes / pop, per (child, adult, elder).
        sero_infection_rate_by_age: seropositive incidence per same groups.

    Returns:
        {"child": γ_c, "adult": γ_a, "elder": γ_e}
    """
    raise NotImplementedError(
        "Serosurvey data not yet available. Once data arrives, "
        "implement and register a new GammaSource in GAMMA_REGISTRY."
    )


def gamma_from_cohort_attack_rate(
    hira_rate_by_age: Sequence[float],
    cohort_attack_rate_by_age: Sequence[float],
) -> dict[str, float]:
    """Cohort attack rate -> γ. Same logic as serosurvey."""
    raise NotImplementedError(
        "Cohort attack-rate data not yet available."
    )


def gamma_from_test_positivity(
    hira_claims: Sequence[float],
    test_confirmed: Sequence[float],
    test_total: Sequence[float],
) -> dict[str, float]:
    """Test-positivity-based correction -> γ (partial; lab-confirmed leg only)."""
    raise NotImplementedError(
        "Test-positivity data not yet available."
    )


def gamma_direct(child: float, adult: float, elder: float) -> dict[str, float]:
    """When the upstream researcher has already computed γ in the target form."""
    return {"child": child, "adult": adult, "elder": elder}
