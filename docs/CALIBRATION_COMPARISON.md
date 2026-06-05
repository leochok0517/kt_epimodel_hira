# Calibration Methodology Comparison: kt_epimodel (ILI) vs kt_epimodel_hira

Comparative analysis of calibration approach in the two sister projects.
Read-only analysis based on source code + fit JSON results.

## 1. Summary

### Common ground
- **Same model core**: SVEIR 5-compartment, NIMS 15 age groups, 4-channel FOI
  (home/work/school/other), Gaussian (now cosine in HIRA) seasonality.
- **Same vector layout for beta and phi**: 4 beta channels + 14 phi (phi_5=1.0 reference).
- **Same optimization framework**: scipy.optimize.minimize, NM and L-BFGS-B available.
- **Same loss family**: Poisson NLL with `min_rate` floor, `first_peak_only` weighting.

### Key differences

| Dimension | ILI | HIRA |
|---|---|---|
| Target unit | per-1000 rate | absolute count |
| Vector dim | 23 (fits seasonality) | 21 single / 33 multi-season joint |
| Seasonality | 4 params fit | cosine 2 params **fixed** |
| gamma_report | single scalar | child/adult/elder (3 values) |
| min_rate | 0.1 | 0.01 |
| Multi-season | not implemented | joint fit prototype done |
| Smoothing | none | phi adjacent λ-penalty (v2, in progress) |

### ILI fit re-evaluation (critical finding)
ILI fits **did move** (6,144 evals, NLL +5168 → -7815) — not stuck in 24-eval
corner as previously suspected. BUT they converge to **wildly different parameter
sets across runs** at similar NLL values — clear identifiability problem.
Two of the saved JSON files show:

- Run A (main): beta ≈ 1.0, sigma=15.1 (**at lower bound**), peak_day=149.99 (**at upper bound**)
- Run B (recent): beta ≈ 0.01-0.17 (~10x lower), gamma=1.0 (**at upper bound**), sigma=47.6

Same NLL within 4%, completely different parameters. This is precisely the
identifiability pathology HIRA later diagnosed and addressed by **fixing seasonality**.

---

## 2. Loss function comparison

[ILI loss.py:118-145](~/Documents/python/NIMS/kt_epimodel/src/kt_epimodel/calibration/loss.py),
[HIRA loss.py:91-130](~/Documents/python/NIMS/kt_epimodel_hira/src/kt_epimodel_hira/calibration/loss.py)

| Item | ILI | HIRA |
|---|---|---|
| Likelihood | Poisson NLL (per-1000 scale) | Poisson NLL (count scale) |
| min_rate default | **0.1** (per-1000 unit) | **0.01** (count unit) |
| Why different | ILI rate is bounded by population; min 0.1/1000 is reasonable | HIRA counts can be small in baseline weeks; 0.01 floor avoids log(0) without spurious NLL terms |
| Weights | `first_peak_only=True` zeros weeks ≥ 26 | Same |
| simulation_to_X | `simulation_to_ili(daily_inc, pop_total, gamma_report)` | `simulation_to_hira_by_age(daily_inc_by_age, gamma_15)` — per-age γ applied at NIMS level |
| gamma application | Single scalar multiplies all ages | (15,) array, age-dependent |
| Smoothing | none | v2 prototype: `λ × Σ(phi[i+1] - phi[i])²` |
| Multi-season aggregate | not implemented | Implemented (sum of 4 season NLLs) |

**Why per-age gamma is structural difference**: ILI's rate denominator (outpatient
population) already entangles reporting fraction with denominator definition.
HIRA's pure count target makes gamma a clean reporting fraction, which then
needed to be split by age once EDA showed monotonic 10x ratio child-to-elder.

---

## 3. Parameter vector structure

[ILI param_vector.py:1-30](~/Documents/python/NIMS/kt_epimodel/src/kt_epimodel/calibration/param_vector.py),
[HIRA param_vector.py:1-30](~/Documents/python/NIMS/kt_epimodel_hira/src/kt_epimodel_hira/calibration/param_vector.py)

### ILI (23-dim)
```
[0-3]   beta_h, beta_w, beta_s, beta_o
[4-17]  phi_a (a in 0..14 \ {5})
[18]    gamma_report
[19-22] seasonality_amp, base, sigma, peak_day   ← fit targets
```

### HIRA single-season (21-dim)
```
[0-3]   beta_h, beta_w, beta_s, beta_o
[4-17]  phi_a (14 values)
[18-20] gamma_child, gamma_adult, gamma_elder    ← split from single γ
                                                  (seasonality removed)
```

### HIRA multi-season joint (33-dim)
```
[0-13]   phi_a (14)            ← shared across seasons
[14-16]  gamma_child/adult/elder ← shared
[17-32]  beta_h/w/s/o × 4 seasons (16) ← per-season
```

**Net structural change**:
- ILI fits 4 seasonality params but a single γ
- HIRA fits 3 γs but holds seasonality fixed
- ILI assumes seasonality differs across seasons (within a fit it's one season anyway)
- HIRA assumes seasonality is constant flu biology, γ varies by age but constant across seasons

---

## 4. Seasonality treatment (root identifiability issue)

### ILI: Gaussian, 4 params fit
- `seasonality_mode = "gaussian"`
- amp / base / sigma / peak_day all fit
- Bounds: amp [0,3], base [0,1], sigma [15,80], peak_day [80,150]
- Result: **fit pushes sigma → 15 (lower bound), peak → 150 (upper bound)** in main fit
  - This is a degenerate "extreme narrow peak at season end" solution
  - Same pathology in HIRA v2 prototype before cosine switch

### HIRA: cosine, fixed
- `seasonality_mode = "cosine"` (changed from gaussian)
- `seasonality_amp = 0.7` (fixed default)
- `seasonality_peak_day = 105.0` (fixed default)
- Removed from fit vector entirely
- **Rationale (HIRA only)**: β × seasonal_factor is multiplicatively coupled.
  Optimizer can rescale β and sf inversely with no NLL change. ILI fits this
  same coupling unawares; HIRA explicitly removed the coupling by fixing sf.

**Why HIRA reached this conclusion first**: HIRA's β=0.5 sanity check
(`fit_01_sanity_nm_v2`) produced R0=13.6 (way too high), which led to
explicit NGM-based R0 computation. The R0 calculation made the β × sf
coupling visible. ILI never built that diagnostic.

---

## 5. Optimizer comparison

| Item | ILI | HIRA |
|---|---|---|
| Methods | NM + L-BFGS-B | Same + multi-season joint wrapper |
| L-BFGS-B maxiter default | 2000 (in `optimize_calibration_by_age`) | 2000 / per-call override (1000 in v2) |
| Bounds | beta (0.001, 5.0), phi (0.1, 5.0), gamma (0.01, 1.0) | beta (0.01, 0.30) tighter, gamma per-age (0.05-0.60) |
| Multi-season warm-start | n/a | v1 result → v2 init |
| Wall-time logged | yes (`elapsed_seconds`) | yes + mlflow `wall_time_min` |
| Verbose stdout | print-based | mlflow + per-eval log |

**Bounds tightening matters**: HIRA's beta upper of 0.30 is much tighter than
ILI's 5.0. This is because HIRA's NGM analysis showed β ≈ 0.06 gives R0 ≈ 1.5.
ILI's loose upper allows the degenerate "high beta + extreme seasonality" corner.

---

## 6. ILI fit re-evaluation

### Main saved fit (`2019-2020_by_age_LBFGS.json`)

| Field | Value | Note |
|---|---|---|
| NLL | -7,814.73 | Reported in ProjectConcept as "success" |
| nll_initial | +5,167.88 | Massive improvement |
| n_evaluations | **6,144** | **NOT 24 — fit actually moved a lot** |
| elapsed | 3,591s (60 min) | Real wall time |
| Convergence message | `RELATIVE REDUCTION OF F <= FACTR*EPSMCH` | True convergence |
| beta_h/w/s/o | 1.016 / 0.947 / 0.677 / 0.999 | All in [0.6, 1.0] |
| phi range | 0.58 - 1.17 | Reasonable |
| seasonality_amp | 0.159 | Far below default 1.0 |
| seasonality_base | 0.142 | Near default |
| seasonality_sigma | **15.116** | **At lower bound 15.0** |
| seasonality_peak_day | **149.99** | **At upper bound 150.0** |
| gamma_report | 0.857 | High |

### Alternative fit (`...122618.json`, same season, different starting point)

| Field | Value | Note |
|---|---|---|
| NLL | -7,504.80 | 4% worse than main fit |
| n_evaluations | 2,567 | Different convergence path |
| beta_h/w/s/o | **0.013 / 0.031 / 0.080 / 0.168** | **10-60x lower than main** |
| phi range | 1.12 - 1.37 | Different normalization (gamma_report = 1.0 = upper bound) |
| seasonality_amp | 0.903 | Near default |
| seasonality_sigma | 47.58 | Mid-range |
| seasonality_peak_day | 97.93 | Mid-range |
| gamma_report | **1.0** | **At upper bound** |

### Verdict on ILI fit

**(b) corner solution achieving plausible NLL** — clearly demonstrated by multimodal
results. Two converged fits with similar NLL (-7815 vs -7505, < 4% gap) reach
parameter sets with 10-60x differences in beta and seasonality at opposite ends
of the bounds. ProjectConcept's "NLL -7815 success" record was a snapshot of
one local minimum among many.

The "false peak" issue and "24-eval stuck" diagnosis from earlier conversation
were probably **different runs** with bad starting points. The saved
`2019-2020_by_age_LBFGS.json` is a real 6,144-eval fit but it sits in a corner
where seasonality pushes to bounds.

**HIRA's diagnoses (identifiability, β × sf coupling, age-dependent γ need) all
apply equally to ILI**. ILI just didn't have the diagnostic tools
(NGM-based R0, per-age age bias plots) to surface them.

---

## 7. Data difference effects on calibration

| Difference | Effect on calibration |
|---|---|
| Per-1000 rate (ILI) vs count (HIRA) | ILI's denominator absorbs ambiguity; HIRA's pure count exposes clean reporting fraction |
| ILI's outpatient denominator | gamma_report becomes a mixed scaling term, harder to interpret |
| HIRA's age groups (6 vs ILI 7) | Minor; cross-boundary HIRA groups (18-44, 65+) need careful per-NIMS gamma application |
| HIRA Sudogwon vs ILI national | HIRA is regional; could extend to per-sido fits but currently aggregated |
| HIRA daily granularity | More information than ILI weekly, but currently aggregated to weekly for calibration |
| Multi-season availability | Both have historical data; HIRA first exploited 4-season joint fit to anchor params |

The **denominator clarity** is the deepest practical difference. HIRA's gamma is
a domain-interpretable "fraction of infections reported to NHIS", which made
the age-split natural and prior values (CDC multipliers) immediately applicable.
ILI's gamma confounds reporting fraction × outpatient rate × population
denominator factor, defeating prior-based bounds.

---

## 8. HIRA improvements: applicability to ILI

| HIRA improvement | Applicable to ILI? | Notes |
|---|---|---|
| Cosine seasonality fixed | ✅ Yes, directly | ILI fit's sigma/peak bound hits prove the same identifiability problem exists |
| Age-dependent gamma | ❓ Conceptually yes, but ILI's gamma is already mixed; per-age split risks over-parameterizing | Would need re-interpretation of what gamma means in ILI |
| Multi-season joint | ✅ Yes | ILI's loss already supports it structurally; just wasn't done |
| min_rate=0.01 | ❌ No, ILI scale needs higher floor | Per-1000 rate has different baseline |
| phi smoothing λ | ✅ Yes | Universal regularization tactic |
| NGM-based R0 sanity | ✅ Yes, important diagnostic | Would surface ILI's bound-hit corner immediately |
| Bound tightening (beta < 0.3) | ✅ Yes | But needs ILI-specific NGM calculation since ILI loss has different scale |

**ILI would benefit substantially from at least**: cosine-fixed seasonality,
NGM R0 sanity check, multi-season joint fit, and bound re-tuning based on
proper R0 calculation.

---

## 9. Implications

### Is HIRA methodology a real improvement over ILI?
**Yes, substantially.** ILI's apparent "success" (NLL -7815) is one mode in a
multimodal landscape. HIRA's pipeline:
- exposes identifiability via NGM diagnostics
- removes one coupling (seasonality fix)
- enriches another with structure (age-dependent γ with CDC prior)
- regularizes across seasons (joint fit)
- (in progress) regularizes phi smoothness

These are not HIRA-specific tricks — they are general epi-calibration discipline
that the ILI project would have needed anyway.

### Data choice for paper
- **HIRA over ILI** for the calibration story: count data + clean γ semantics +
  more granular age makes the methodological narrative stronger
- Both as side-by-side validation: if HIRA-calibrated φ and γ also explain ILI
  rates without re-fitting, that's a strong robustness signal

### Maintenance strategy
- **kt_epimodel (ILI)**: keep current code state as reference for ILI-target
  validation. Avoid further development unless a specific ILI-only analysis is
  needed (e.g. national-level rate comparison).
- **kt_epimodel_hira**: continue as the active calibration platform.
  Backport diagnostics (NGM R0, multi-season joint) to ILI if/when needed
  for cross-validation.
- Tests should be kept in sync across both projects but fit results are not
  expected to match (different targets).
