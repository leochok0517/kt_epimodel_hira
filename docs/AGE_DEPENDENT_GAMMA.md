# Age-dependent gamma_report: rationale and assumptions

## 1. Motivation

### EDA observation
00_hira_eda.ipynb normal-season per-100k peak rate ratio:
- 0-5 / 65+ = 10.14
- 6-11 / 65+ = 13.99

### v3 sanity fit age bias
| Age | obs/pred peak ratio | Interpretation |
|---|---|---|
| 0-5 | 0.15 | massive undershoot |
| 6-11 | 0.33 | undershoot |
| 12-17 | 0.66 | mild undershoot |
| 18-44 | 0.98 | ok |
| 45-64 | 1.50 | overshoot |
| 65+ | 1.38 | overshoot |

Monotonic pattern — cannot be absorbed by a single gamma_report.
Structural age-dependent reporting fraction differences exist.

## 2. Medical/behavioral rationale

### Children (high gamma)
- Parents bring them in: low barrier to medical access
- Daycare/school absence requires doctor's note: strong visit motivation
- Complication concerns (otitis media, pneumonia): parents seek care early
- Generally higher healthcare utilization across all conditions

### Adults (low gamma)
- High self-care rate (rest, OTC meds)
- Work absence burden: defer visits
- Mild ILI often results in presenteeism

### Elderly (intermediate gamma)
- Complication concerns (pneumonia, cardiovascular): motivation to visit
- Chronic disease follow-up visits catch concurrent influenza diagnosis
- But mobility limitations for frail elderly: some cases missed
- High vaccination rate (82% coverage for 70+) reduces incidence itself
  — must separate from gamma estimation

## 3. Korean data evidence

### JKMS 2023 (Park et al.) [Ref 1]
- Same dataset (HIRA J09-J11) medically-attended incidence rate
- 2010-2020, 11 seasons analyzed
- Age distribution: <20y accounts for 50%+, >=60y only 8%
- Confirms strong age skew in Korean medically-attended incidence

### Limitations
- No study estimates actual incidence (who did NOT visit) in Korea
- No direct reporting fraction measurement available for Korea
- Must combine our fit results with external multiplier references

## 4. International multiplier comparison

### CDC Reed et al. 2015 (PLOS One) [Ref 2]
- US FluSurv-NET surveillance -> burden estimation standard method
- Age-specific symptomatic multipliers:
  - <18y: 2.1x
  - 18-64y: 3.1x
  - 65+: 5.2x
- Implied reporting fraction (symptomatic -> medically-attended):
  - <18y: ~0.40-0.50 (high)
  - 18-64y: ~0.15-0.25 (low)
  - 65+: ~0.25-0.40 (intermediate)

### Tokars et al. 2018 (Clin Infect Dis) [Ref 3]
- US seasonal influenza symptomatic incidence 2010-2016
- <5y 13.2%, >=65y 3.9% (8x difference)
- Similar order of magnitude to our Korean EDA 10-14x ratio

## 5. Interaction with V (Vaccinated) compartment

This model uses SVEIR structure with explicit vaccination:
- annual_coverage: 0-9y 75%, 10-19y 40%, 20-69y 30%, 70+ 82%
- VE = 0.5
- V -> I at (1-VE) * lambda (reduced)

This means the model **already accounts for** part of why elderly
incidence is lower than children. When estimating gamma_elder:

- Reasons for low elderly data incidence:
  (a) Lower contact rate (reflected in NIMS contact matrix)
  (b) **High vaccination rate -> reduced infection** (V compartment)
  (c) Potentially lower reporting fraction (gamma_elder)

Since (a) and (b) are already in the model, setting gamma_elder
**too low** risks double-counting. Therefore gamma_elder should be
similar to or slightly higher than gamma_adult (consistent with
CDC multiplier estimates).

## 6. Adopted values + uncertainty range

| Group | NIMS idx | Age | Default | Range (sensitivity) | Rationale |
|---|---|---|---|---|---|
| child | 0-3 | 0-19 | 0.40 | 0.30-0.50 | CDC <18 multiplier inverse, child care-seeking |
| adult | 4-12 | 20-64 | 0.18 | 0.12-0.25 | CDC 18-64 multiplier, self-care tendency |
| elder | 13-14 | 65+ | 0.25 | 0.18-0.35 | CDC 65+ multiplier, conservative for V effect |

**Optimizer bounds**: slightly wider than range (0.05-0.60) to allow
fit freedom. If fit result falls outside range, requires review.

## 7. Limitations

- No Korean-specific multiplier measurement -> US CDC values adapted
  (healthcare system differences exist)
- 3-group boundary (17/64) is conventional but arbitrary — medically
  reasonable but cut-off itself is not data-validated
- Within-group homogeneity assumption — 18-44 and 45-64 sharing
  gamma_adult is a simplification
- J11 (suspected code) may include some non-influenza respiratory
  illness reporting

## 8. Future work

- Stage 5: HIRA outpatient data (total claims vs influenza claims ratio)
  + KDCA sentinel surveillance matching for direct Korean multiplier
  estimation
- State as limitation in paper + propose as follow-up

## 9. References

1. Park JY, et al. Incidence, Severity, and Mortality of Influenza During
   2010-2020 in Korea. J Korean Med Sci. 2023;38(8):e58.
2. Reed C, et al. Estimating influenza disease burden from population-based
   surveillance data in the United States. PLOS One. 2015;10(3):e0118369.
3. Tokars JI, et al. Seasonal incidence of symptomatic influenza in the
   United States. Clin Infect Dis. 2018;66(10):1511-1518.
4. Noda T. Incidence Rate of Seasonal Influenza Calculated from Japanese
   Medical Database. MHLW Japan. 2022.
