"""Calibration optimizer — 한 시즌 beta/phi/gamma_report fit (HIRA 버전).

scipy.optimize.minimize 래퍼:
- Nelder-Mead (gradient-free, robust)
- L-BFGS-B (bounds 지원, faster but local)

결과: CalibrationResult dataclass + JSON save/load.

19-dim vector (seasonality 고정): beta(4) + phi(14) + gamma_report(1).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from kt_epimodel_hira.calibration.hira_target import (
    load_hira_target,
    load_hira_target_by_age,
)
from kt_epimodel_hira.calibration.loss import (
    make_loss_function,
    make_loss_function_by_age,
)
from kt_epimodel_hira.calibration.param_vector import (
    get_bounds_vector,
    initial_guess,
    vector_to_params,
)
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs,
    estimate_initial_infected_from_hira,
)
from kt_epimodel_hira.model.parameters import (
    CalibrationParameters,
    ModelParameters,
)

VALID_METHODS = ("Nelder-Mead", "L-BFGS-B")


@dataclass
class CalibrationResult:
    """Calibration 결과."""
    season: str
    method: str
    success: bool
    nll: float
    nll_initial: float
    calibration: CalibrationParameters
    seasonality_amp: float
    seasonality_base: float
    seasonality_sigma: float
    seasonality_peak_day: float
    seasonality_mode: str
    vector: np.ndarray
    n_evaluations: int
    elapsed_seconds: float
    message: str
    seed_total: float
    initial_immunity: float
    initial_vaccinated_fraction: float
    first_peak_only: bool = False
    first_peak_end_week: int = 26
    use_data_seed: bool = False
    seed_by_age: list[float] | None = None
    gamma_report_assumed: float = 1.0


def _resolve_initial_vec(
    initial_vec: np.ndarray | None,
    initial_from_result: "CalibrationResult | None",
) -> np.ndarray:
    """Initial vec 결정 — 명시 vec > warm-start result > default initial_guess()."""
    if initial_vec is not None and initial_from_result is not None:
        raise ValueError(
            "Pass at most one of initial_vec / initial_from_result, not both."
        )
    if initial_from_result is not None:
        vec = np.asarray(initial_from_result.vector, dtype=np.float64).copy()
        bounds = get_bounds_vector()
        if vec.shape != (len(bounds),):
            raise ValueError(
                f"initial_from_result.vector has shape {vec.shape}; "
                f"expected ({len(bounds)},)"
            )
        for i, (lo, hi) in enumerate(bounds):
            if vec[i] < lo:
                vec[i] = lo
            elif vec[i] > hi:
                vec[i] = hi
        return vec
    if initial_vec is not None:
        return np.asarray(initial_vec, dtype=np.float64)
    return initial_guess()


def optimize_calibration(
    season: str,
    base_params: ModelParameters | None = None,
    sido_codes: list[int] | None = None,
    setting: str = "outpatient_inpatient",
    seed_total: float = 100.0,
    initial_immunity: float = 0.0,
    initial_vaccinated_fraction: float = 0.0,
    method: str = "Nelder-Mead",
    initial_vec: np.ndarray | None = None,
    initial_from_result: "CalibrationResult | None" = None,
    max_iterations: int = 2000,
    verbose: bool = True,
    t_span: tuple[float, float] = (0.0, 364.0),
    first_peak_only: bool = False,
    first_peak_end_week: int = 26,
) -> CalibrationResult:
    """한 시즌 HIRA calibration 수행 (연령 합산 single target)."""
    if method not in VALID_METHODS:
        raise ValueError(f"method must be in {VALID_METHODS}, got {method!r}")
    if base_params is None:
        base_params = ModelParameters()

    target = load_hira_target(
        season,
        sido_codes=sido_codes,
        setting=setting,
        first_peak_only=first_peak_only,
        first_peak_end_week=first_peak_end_week,
    )
    inputs = build_aggregated_inputs()

    loss_fn = make_loss_function(
        target, inputs, base_params,
        seed_total=seed_total,
        initial_immunity=initial_immunity,
        initial_vaccinated_fraction=initial_vaccinated_fraction,
        t_span=t_span,
        verbose=verbose,
    )

    initial_vec = _resolve_initial_vec(initial_vec, initial_from_result)
    nll_initial = float(loss_fn(initial_vec))

    if verbose:
        print(f"=== Optimizing {season} (HIRA) with {method} ===")
        print(f"Initial NLL: {nll_initial:.2f}")

    t0 = time.perf_counter()
    bounds = get_bounds_vector()
    if method == "L-BFGS-B":
        sol = minimize(
            loss_fn, initial_vec,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iterations, "disp": verbose,
                     "ftol": 1e-5, "gtol": 1e-3},
        )
    else:
        sol = minimize(
            loss_fn, initial_vec,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": max_iterations,
                "xatol": 1e-4, "fatol": 1e-2,
                "disp": verbose,
                "adaptive": True,
            },
        )
    elapsed = time.perf_counter() - t0

    fit_params = vector_to_params(sol.x)
    dis = base_params.disease

    if verbose:
        print("\n=== Result (HIRA) ===")
        print(f"  success:      {sol.success}")
        print(f"  NLL:          {nll_initial:.2f} -> {sol.fun:.2f}")
        print(f"  evaluations:  {sol.nfev}")
        print(f"  elapsed:      {elapsed:.1f}s")
        print(
            f"  fitted beta:  h={fit_params.beta_h:.4f} w={fit_params.beta_w:.4f} "
            f"s={fit_params.beta_s:.4f} o={fit_params.beta_o:.4f}"
        )
        print(f"  gamma: child={fit_params.gamma_child:.4f} "
              f"adult={fit_params.gamma_adult:.4f} elder={fit_params.gamma_elder:.4f}")
        print(f"  seasonality:  {dis.seasonality_mode} amp={dis.seasonality_amp} "
              f"peak={dis.seasonality_peak_day} (fixed)")

    return CalibrationResult(
        season=season,
        method=method,
        success=bool(sol.success),
        nll=float(sol.fun),
        nll_initial=nll_initial,
        calibration=fit_params,
        seasonality_amp=dis.seasonality_amp,
        seasonality_base=dis.seasonality_base,
        seasonality_sigma=dis.seasonality_sigma,
        seasonality_peak_day=dis.seasonality_peak_day,
        seasonality_mode=dis.seasonality_mode,
        vector=np.asarray(sol.x, dtype=np.float64),
        n_evaluations=int(sol.nfev),
        elapsed_seconds=elapsed,
        message=str(sol.message),
        seed_total=float(seed_total),
        initial_immunity=float(initial_immunity),
        initial_vaccinated_fraction=float(initial_vaccinated_fraction),
        first_peak_only=bool(first_peak_only),
        first_peak_end_week=int(first_peak_end_week),
    )


def optimize_calibration_by_age(
    season: str,
    base_params: ModelParameters | None = None,
    sido_codes: list[int] | None = None,
    setting: str = "outpatient_inpatient",
    seed_total: float = 100.0,
    use_data_seed: bool = True,
    seed_e_factor: float = 0.5,
    gamma_report_assumed: float = 0.2,
    initial_immunity: float = 0.3,
    initial_vaccinated_fraction: float = 0.0,
    method: str = "Nelder-Mead",
    initial_vec: np.ndarray | None = None,
    initial_from_result: "CalibrationResult | None" = None,
    max_iterations: int = 2000,
    verbose: bool = True,
    t_span: tuple[float, float] = (0.0, 364.0),
    first_peak_only: bool = True,
    first_peak_end_week: int = 26,
) -> CalibrationResult:
    """6 HIRA 연령 그룹 동시 fit."""
    if method not in VALID_METHODS:
        raise ValueError(f"method must be in {VALID_METHODS}, got {method!r}")
    if base_params is None:
        base_params = ModelParameters()

    target_by_age = load_hira_target_by_age(
        season,
        sido_codes=sido_codes,
        setting=setting,
        first_peak_only=first_peak_only,
        first_peak_end_week=first_peak_end_week,
    )
    inputs = build_aggregated_inputs()

    seed_by_age_arr: np.ndarray | None = None
    if use_data_seed:
        seed_by_age_arr = estimate_initial_infected_from_hira(
            season, inputs["pop_15"].flatten(),
            sido_codes=sido_codes, setting=setting,
            gamma_report_assumed=gamma_report_assumed,
        )
        seed_total_effective = float(seed_by_age_arr.sum())
        if verbose:
            print(
                f"Estimated seed by age "
                f"(gamma_report_assumed={gamma_report_assumed}, "
                f"total={seed_total_effective:,.0f}):"
            )
            for a, n in enumerate(seed_by_age_arr):
                print(f"  age {a*5}-{a*5+4 if a<14 else 99}: {n:,.0f}")
    else:
        seed_total_effective = float(seed_total)

    loss_fn = make_loss_function_by_age(
        target_by_age, inputs, base_params,
        seed_total=seed_total_effective,
        seed_by_age=seed_by_age_arr,
        seed_e_factor=seed_e_factor,
        initial_immunity=initial_immunity,
        initial_vaccinated_fraction=initial_vaccinated_fraction,
        t_span=t_span,
        verbose=verbose,
    )

    initial_vec = _resolve_initial_vec(initial_vec, initial_from_result)
    nll_initial = float(loss_fn(initial_vec))

    if verbose:
        print(f"=== Optimizing {season} (HIRA by_age, 6 groups) with {method} ===")
        print(f"Initial NLL: {nll_initial:.2f}")

    t0 = time.perf_counter()
    bounds = get_bounds_vector()
    if method == "L-BFGS-B":
        sol = minimize(
            loss_fn, initial_vec,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max_iterations, "disp": verbose,
                     "ftol": 1e-5, "gtol": 1e-3},
        )
    else:
        sol = minimize(
            loss_fn, initial_vec,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": max_iterations,
                "xatol": 1e-4, "fatol": 1e-2,
                "disp": verbose, "adaptive": True,
            },
        )
    elapsed = time.perf_counter() - t0
    fit_params = vector_to_params(sol.x)
    dis = base_params.disease

    if verbose:
        print("\n=== Result (HIRA by_age) ===")
        print(f"  success:      {sol.success}")
        print(f"  NLL:          {nll_initial:.2f} -> {sol.fun:.2f}")
        print(f"  evaluations:  {sol.nfev}")
        print(f"  elapsed:      {elapsed:.1f}s")
        print(
            f"  beta fit:     h={fit_params.beta_h:.4f} w={fit_params.beta_w:.4f} "
            f"s={fit_params.beta_s:.4f} o={fit_params.beta_o:.4f}"
        )
        print(f"  gamma: child={fit_params.gamma_child:.4f} "
              f"adult={fit_params.gamma_adult:.4f} elder={fit_params.gamma_elder:.4f}")
        print(f"  seasonality:  {dis.seasonality_mode} amp={dis.seasonality_amp} "
              f"peak={dis.seasonality_peak_day} (fixed)")

    return CalibrationResult(
        season=f"{season}_by_age",
        method=method,
        success=bool(sol.success),
        nll=float(sol.fun),
        nll_initial=nll_initial,
        calibration=fit_params,
        seasonality_amp=dis.seasonality_amp,
        seasonality_base=dis.seasonality_base,
        seasonality_sigma=dis.seasonality_sigma,
        seasonality_peak_day=dis.seasonality_peak_day,
        seasonality_mode=dis.seasonality_mode,
        vector=np.asarray(sol.x, dtype=np.float64),
        n_evaluations=int(sol.nfev),
        elapsed_seconds=elapsed,
        message=str(sol.message),
        seed_total=float(seed_total_effective),
        initial_immunity=float(initial_immunity),
        initial_vaccinated_fraction=float(initial_vaccinated_fraction),
        first_peak_only=bool(first_peak_only),
        first_peak_end_week=int(first_peak_end_week),
        use_data_seed=bool(use_data_seed),
        seed_by_age=(seed_by_age_arr.tolist() if seed_by_age_arr is not None else None),
        gamma_report_assumed=float(gamma_report_assumed),
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_result(result: CalibrationResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "season": result.season,
        "method": result.method,
        "success": result.success,
        "nll": result.nll,
        "nll_initial": result.nll_initial,
        "vector": result.vector.tolist(),
        "n_evaluations": result.n_evaluations,
        "elapsed_seconds": result.elapsed_seconds,
        "message": result.message,
        "seed_total": result.seed_total,
        "initial_immunity": result.initial_immunity,
        "initial_vaccinated_fraction": result.initial_vaccinated_fraction,
        "seasonality_amp": result.seasonality_amp,
        "seasonality_base": result.seasonality_base,
        "seasonality_sigma": result.seasonality_sigma,
        "seasonality_peak_day": result.seasonality_peak_day,
        "seasonality_mode": result.seasonality_mode,
        "first_peak_only": result.first_peak_only,
        "first_peak_end_week": result.first_peak_end_week,
        "use_data_seed": result.use_data_seed,
        "seed_by_age": result.seed_by_age,
        "gamma_report_assumed": result.gamma_report_assumed,
        "calibration": {
            "beta_h": result.calibration.beta_h,
            "beta_w": result.calibration.beta_w,
            "beta_s": result.calibration.beta_s,
            "beta_o": result.calibration.beta_o,
            "phi": result.calibration.phi.tolist(),
            "gamma_child": result.calibration.gamma_child,
            "gamma_adult": result.calibration.gamma_adult,
            "gamma_elder": result.calibration.gamma_elder,
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_result(path: Path | str) -> CalibrationResult:
    with open(path) as f:
        data = json.load(f)
    cal_data = data["calibration"]
    cal = CalibrationParameters(
        beta_h=cal_data["beta_h"],
        beta_w=cal_data["beta_w"],
        beta_s=cal_data["beta_s"],
        beta_o=cal_data["beta_o"],
        phi=np.asarray(cal_data["phi"], dtype=np.float64),
        gamma_child=float(cal_data.get("gamma_child", 0.40)),
        gamma_adult=float(cal_data.get("gamma_adult", 0.18)),
        gamma_elder=float(cal_data.get("gamma_elder", 0.25)),
    )
    return CalibrationResult(
        season=data["season"],
        method=data["method"],
        success=data["success"],
        nll=data["nll"],
        nll_initial=data["nll_initial"],
        calibration=cal,
        seasonality_amp=float(data.get("seasonality_amp", 0.7)),
        seasonality_base=float(data.get("seasonality_base", 1.0)),
        seasonality_sigma=float(data.get("seasonality_sigma", 40.0)),
        seasonality_peak_day=float(data.get("seasonality_peak_day", 105.0)),
        seasonality_mode=str(data.get("seasonality_mode", "cosine")),
        vector=np.asarray(data["vector"], dtype=np.float64),
        n_evaluations=data["n_evaluations"],
        elapsed_seconds=data["elapsed_seconds"],
        message=data["message"],
        seed_total=data["seed_total"],
        initial_immunity=data["initial_immunity"],
        initial_vaccinated_fraction=data["initial_vaccinated_fraction"],
        first_peak_only=bool(data.get("first_peak_only", False)),
        first_peak_end_week=int(data.get("first_peak_end_week", 26)),
        use_data_seed=bool(data.get("use_data_seed", False)),
        seed_by_age=(
            list(data["seed_by_age"]) if data.get("seed_by_age") is not None else None
        ),
        gamma_report_assumed=float(data.get("gamma_report_assumed", 1.0)),
    )
