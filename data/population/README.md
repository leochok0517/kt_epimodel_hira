# Season-specific population (Seoul Capital Area)

## Source
- MOIS resident registration (주민등록인구), KOSIS export
  `행정구역_읍면동_별_5세별_주민등록인구_2011년_...csv` (original in `raw/`, EUC-KR).
- Item: 총인구수 (resident-registration total).
- Regions summed: 서울특별시 + 인천광역시 + 경기도 (Seoul Capital Area).
- Reference: **mid-year (June)** of each season's peak year. (In this KOSIS table the annual
  columns are June snapshots; the latest column is labelled 2026.06.)
  - 2016–17 season → **2016-06**
  - 2017–18 season → **2017-06**
  - 2019–20 season → **2019-06**

## Aggregation to the model's 15 bands
5-year bands 0–4 … 65–69 kept as-is (14 bands); **70+ = sum of 70–74 … 100+**.
Each vector's sum was verified to equal the reported 계 (total) for all three years (exact match).

## Files
- `pop_2016_17.csv`, `pop_2017_18.csv`, `pop_2019_20.csv` — per season: `band_index, age_band, population` (15 rows, index 0–14 matching the model band order).
- `pop_all_seasons.csv` — tidy long form: `season, reference, band_index, age_band, population`.
- `raw/` — original KOSIS CSV (provenance).

## Totals (checksum-verified)
| season | reference | total |
|--|--|--|
| 2016–17 | 2016-06 | 25,590,465 |
| 2017–18 | 2017-06 | 25,679,863 |
| 2019–20 | 2019-06 | 25,925,799 |

Note the aging trend across seasons (0–4: 1,105k→934k; 70+: 1,997k→2,361k), which motivates using
year-specific rather than single-year populations.
