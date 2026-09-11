"""Build season-specific Seoul Capital Area populations for the influenza-A refit.

Generalizes data/population/pop_all_seasons.csv (which covers only 2016-17,
2017-18 and 2019-20) to every season the KOSIS export can supply.

Source / convention are unchanged from data/population/README.md:
  - MOIS resident registration, KOSIS export in data/population/raw/ (EUC-KR),
    item 총인구수, regions 서울 + 인천 + 경기.
  - Reference = the June snapshot of the season's START year
    (season "2016-2017" -> the 2016 column).
  - 15 model bands: 0-4 ... 65-69 as-is, 70+ = sum of 70-74 ... 100+.

Existing rows are reproduced byte-identically; the script asserts this against
pop_all_seasons.csv before writing, so the refit cannot silently move the three
seasons the published fit used.

Output: data/population/pop_all_seasons_fluA.csv (same schema as the original).
Seasons whose start year is absent from the KOSIS export are reported as
missing rather than guessed - re-export the same KOSIS table with the extra
year columns to add them.

Usage:
    python scripts/build_pop_seasons_fluA.py [--seasons 2013-2014 2014-2015 ...]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POPDIR = REPO / "data" / "population"
RAW = next(POPDIR.glob("raw/행정구역*.csv"))
REF_CSV = POPDIR / "pop_all_seasons.csv"
OUT_CSV = POPDIR / "pop_all_seasons_fluA.csv"

SIDO = ["서울특별시", "인천광역시", "경기도"]
BANDS = ["0 - 4세", "5 - 9세", "10 - 14세", "15 - 19세", "20 - 24세",
         "25 - 29세", "30 - 34세", "35 - 39세", "40 - 44세", "45 - 49세",
         "50 - 54세", "55 - 59세", "60 - 64세", "65 - 69세"]
BAND70 = ["70 - 74세", "75 - 79세", "80 - 84세", "85 - 89세", "90 - 94세",
          "95 - 99세", "100+"]
BAND_LABELS = [b.replace(" ", "").replace("세", "") for b in BANDS] + ["70+"]

DEFAULT_SEASONS = [f"{y}-{y+1}" for y in range(2013, 2020)]


def read_kosis() -> tuple[dict[str, dict[str, dict[str, int]]], list[str]]:
    """-> {year_label: {sido: {band_label: value}}}, [year labels]."""
    rows = list(csv.reader(open(RAW, encoding="euc-kr")))
    h_year, h_band = rows[0], rows[1]
    years: list[str] = []
    for c in h_year[2:]:
        if c not in years:
            years.append(c)
    out: dict[str, dict[str, dict[str, int]]] = {y: {} for y in years}
    for r in rows[2:]:
        sido = r[0].strip()
        if sido not in SIDO:
            continue
        for j in range(2, len(r)):
            y, band = h_year[j], h_band[j]
            if band == "계":
                continue
            out[y].setdefault(sido, {})[band] = int(r[j])
    return out, years


def year_key(start_year: int, years: list[str]) -> str | None:
    """KOSIS column label for a season's June reference."""
    for cand in (str(start_year), f"{start_year}.06"):
        if cand in years:
            return cand
    return None


def build_vector(tbl: dict[str, dict[str, int]]) -> list[int]:
    vec = [sum(tbl[s][b] for s in SIDO) for b in BANDS]
    vec.append(sum(tbl[s][b] for s in SIDO for b in BAND70))
    return vec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS)
    args = ap.parse_args()

    data, years = read_kosis()
    print(f"[kosis] {RAW.name}")
    print(f"[kosis] year columns: {years}")

    built: dict[str, list[int]] = {}
    missing: list[str] = []
    for season in args.seasons:
        y0 = int(season.split("-")[0])
        yk = year_key(y0, years)
        if yk is None:
            missing.append(season)
            continue
        built[season] = build_vector(data[yk])

    # regression: the three published seasons must come out unchanged
    ref: dict[str, dict[int, int]] = {}
    for row in csv.DictReader(open(REF_CSV)):
        ref.setdefault(row["season"], {})[int(row["band_index"])] = int(row["population"])
    for season, vec in built.items():
        short = season.replace("-", "_")[:4] + "_" + season.split("-")[1][2:]
        if short in ref:
            got = {i: v for i, v in enumerate(vec)}
            assert got == ref[short], (
                f"{season}: rebuilt population differs from published "
                f"pop_all_seasons.csv - refusing to write"
            )
            print(f"  {season}: matches published values  (total {sum(vec):,})")
        else:
            print(f"  {season}: NEW                        (total {sum(vec):,})")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "reference", "band_index", "age_band", "population"])
        for season in sorted(built):
            y0 = season.split("-")[0]
            short = f"{y0}_{season.split('-')[1][2:]}"
            for i, (lab, v) in enumerate(zip(BAND_LABELS, built[season])):
                w.writerow([short, f"{y0}-06", i, lab, v])
    print(f"\n[written] {OUT_CSV}  ({len(built)} seasons)")
    if missing:
        print(f"[MISSING] {missing}")
        print("  -> re-export the same KOSIS table including those June columns.")


if __name__ == "__main__":
    main()
