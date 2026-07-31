"""시즌별 인구 로더 + season-aware setup builder.

data/population/pop_all_seasons.csv 에서 season → (15,) numpy 배열 로드.
season key 매핑:
  "2016-2017" → "2016_17"
  "2017-2018" → "2017_18"
  "2019-2020" → "2019_20"
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import jax.numpy as jnp

_REPO = Path(__file__).resolve().parent.parent
_POP_CSV = _REPO / "data" / "population" / "pop_all_seasons.csv"

SEASON_MAP = {
    "2016-2017": "2016_17",
    "2017-2018": "2017_18",
    "2019-2020": "2019_20",
}


def load_pop_by_season() -> dict[str, np.ndarray]:
    """Returns dict {season_label: (15,) numpy array}."""
    out = {s: np.zeros(15, dtype=np.float64) for s in SEASON_MAP.keys()}
    with open(_POP_CSV) as f:
        for row in csv.DictReader(f):
            key = row["season"]
            for full, short in SEASON_MAP.items():
                if key == short:
                    out[full][int(row["band_index"])] = float(row["population"])
                    break
    return out


def pop_15_2d_for_season(season: str) -> np.ndarray:
    """Returns (15, 1) 2-D array matching build_aggregated_inputs pop_15 shape."""
    all_pop = load_pop_by_season()
    if season not in all_pop:
        raise KeyError(f"season {season} not in pop CSV; keys={list(all_pop)}")
    return all_pop[season].reshape(15, 1)


if __name__ == "__main__":
    pops = load_pop_by_season()
    print(f"loaded {len(pops)} seasons from {_POP_CSV}")
    for s, arr in pops.items():
        print(f"  {s}: total={arr.sum():,.0f}  bands={arr.shape}")
