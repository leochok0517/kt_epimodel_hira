# Naming & Experiment Conventions

## File naming

```
outputs/calibration/
  {scope}_{season}_{method}_{tag}.json
  {scope}_{season}_{method}_{tag}.png

scope:   sanity | calib | scenario
season:  YYYY-YYYY (e.g. 2019-2020) or multi-season
method:  NM | LBFGS | MCMC
tag:     description (e.g. v3, cosine, baseline, holdout)
```

Examples:
- `sanity_2019-2020_NM_v3.json`
- `calib_2019-2020_LBFGS_cosine.json`
- `calib_2018-2019_LBFGS_holdout.json`
- `scenario_2019-2020_NM_school_closure.json`

## Result JSON metadata

All fit result JSON files include:

```json
{
  "metadata": {
    "git_commit": "abc123",
    "timestamp": "2025-XX-XX HH:MM:SS",
    "data_source": "HIRA",
    "season": "2019-2020",
    "sido_codes": [11, 28, 41],
    "method": "NM",
    "n_iter": 100,
    "wall_time_seconds": 114.3,
    "tag": "v3_cosine_amp07"
  },
  "fixed_params": {
    "seasonality_function": "cosine",
    "seasonality_amp": 0.7,
    "seasonality_base": 1.0,
    "seasonality_peak_day": 105,
    "sigma": 0.5,
    "gamma": 0.25,
    "VE": 0.5,
    "min_rate": 0.01
  },
  "fit_params": {
    "vec_19dim": ["..."],
    "labels": ["beta_h", "beta_w", "...", "gamma_report"]
  },
  "diagnostics": {
    "nll_initial": 305079.0,
    "nll_final": 12345.0,
    "R0_initial": 1.8,
    "R0_final": 1.5,
    "bound_hits": ["..."],
    "floor_hit_pct": 0.0
  }
}
```

## mlflow logging

Tracking URI: `file://.../kt_epimodel_hira/outputs/mlruns` (set in `__init__.py`).

Standard run logging:
```python
with mlflow.start_run(run_name=f"{method}_{season}_{tag}"):
    mlflow.log_params(fixed_params)
    # ... fit ...
    mlflow.log_metric("final_nll", final_nll)
    mlflow.log_metric("R0_final", R0_final)
    mlflow.log_metric("wall_time", wall_time)
    mlflow.log_artifact("outputs/calibration/{filename}.json")
    mlflow.log_artifact("outputs/calibration/{filename}.png")
```

UI: `mlflow ui --backend-store-uri outputs/mlruns` -> http://127.0.0.1:5000

## Parallel execution

BLAS single-thread enforced via `~/.zshrc`:
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

joblib usage:
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=5, backend="loky")(
    delayed(run_fit)(season) for season in seasons
)
```

- `n_jobs=5` default (conservative)
- `backend="loky"` (process-based, BLAS-safe)
- Memory-heavy: `n_jobs=3`

## Git worktree

```
~/Documents/python/NIMS/
  kt_epimodel_hira/              (main)
  kt_epimodel_hira_exp_{tag}/    (experiment worktree)

branch: exp/{tag}
```

After worktree setup: `cd kt_epimodel_hira_exp_{tag} && uv sync`
