"""Calibration loss function (Poisson NLL) — HIRA.

vec -> simulate -> HIRA count (age-dependent gamma) -> Poisson NLL.
Seasonality fixed in base_params.disease. gamma_15 from CalibrationParameters.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from kt_epimodel_hira.calibration.hira_target import (
    poisson_log_likelihood,
    simulation_to_hira,
    simulation_to_hira_by_age,
)
from kt_epimodel_hira.calibration.param_vector import vector_to_params
from kt_epimodel_hira.calibration.simple_model import simulate_aggregated
from kt_epimodel_hira.model.parameters import ModelParameters


def make_loss_function(
    target: dict[str, Any],
    inputs: dict[str, Any],
    base_params: ModelParameters,
    seed_total: float = 100.0,
    initial_immunity: float = 0.0,
    initial_vaccinated_fraction: float = 0.0,
    t_span: tuple[float, float] = (0.0, 364.0),
    verbose: bool = False,
    log_every: int = 10,
    penalty: float = 1e10,
) -> Callable[[np.ndarray], float]:
    """Optimizer loss (single target, age-summed HIRA)."""
    observed = np.asarray(target["hira_count"], dtype=np.float64)
    is_valid = np.asarray(target["is_valid"], dtype=bool)
    weights = np.asarray(
        target.get("weights", is_valid.astype(np.float64)),
        dtype=np.float64,
    )
    n_weeks = int(target["n_weeks"])

    call_count = [0]

    def loss(vec: np.ndarray) -> float:
        call_count[0] += 1
        try:
            new_cal = vector_to_params(vec)
            new_params = base_params.with_calibration(new_cal)

            result = simulate_aggregated(
                new_params, inputs,
                seed_total=seed_total,
                initial_immunity=initial_immunity,
                initial_vaccinated_fraction=initial_vaccinated_fraction,
                t_span=t_span,
            )
            if not result.success:
                if verbose:
                    print(f"[Eval {call_count[0]}] solver failed: {result.message}")
                return float(penalty)

            daily_inc = result.daily_new_infection_by_age()
            predicted = simulation_to_hira(
                daily_inc, new_cal.gamma_15, n_weeks=n_weeks,
            )
            nll = poisson_log_likelihood(
                observed, predicted, is_valid, weights=weights,
            )

            if verbose and call_count[0] % log_every == 0:
                print(
                    f"[Eval {call_count[0]:>4}] "
                    f"beta=({vec[0]:.3f},{vec[1]:.3f},{vec[2]:.3f},{vec[3]:.3f}) "
                    f"g_c={vec[18]:.3f} g_a={vec[19]:.3f} g_e={vec[20]:.3f} "
                    f"NLL={nll:.2f}"
                )
            return float(nll) if np.isfinite(nll) else float(penalty)

        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"[Eval {call_count[0]}] FAILED: {e}")
            return float(penalty)

    loss.call_count = call_count   # type: ignore[attr-defined]
    return loss


def make_loss_function_by_age(
    target_by_age: dict,
    inputs: dict,
    base_params: ModelParameters,
    seed_total: float = 100.0,
    seed_by_age: np.ndarray | None = None,
    seed_e_factor: float = 0.5,
    initial_immunity: float = 0.0,
    initial_vaccinated_fraction: float = 0.0,
    t_span: tuple[float, float] = (0.0, 364.0),
    verbose: bool = False,
    log_every: int = 10,
    penalty: float = 1e10,
) -> Callable[[np.ndarray], float]:
    """6 HIRA age group simultaneous fit loss."""
    age_groups: list[str] = list(target_by_age["age_groups"])
    n_weeks = int(target_by_age["n_weeks"])

    call_count = [0]

    def loss(vec: np.ndarray) -> float:
        call_count[0] += 1
        try:
            cal_new = vector_to_params(vec)
            new_params = base_params.with_calibration(cal_new)

            result = simulate_aggregated(
                new_params, inputs,
                seed_total=seed_total,
                seed_by_age=seed_by_age,
                seed_e_factor=seed_e_factor,
                initial_immunity=initial_immunity,
                initial_vaccinated_fraction=initial_vaccinated_fraction,
                t_span=t_span,
            )
            if not result.success:
                if verbose:
                    print(f"[Eval {call_count[0]}] solver failed: {result.message}")
                return float(penalty)

            daily_inc_by_age = result.daily_new_infection_by_age()

            predictions = simulation_to_hira_by_age(
                daily_inc_by_age, cal_new.gamma_15, n_weeks=n_weeks,
            )

            total_nll = 0.0
            for ag in age_groups:
                nll = poisson_log_likelihood(
                    target_by_age["hira_counts"][ag],
                    predictions[ag],
                    is_valid=target_by_age["is_valid"][ag],
                    weights=target_by_age["weights"][ag],
                )
                if not np.isfinite(nll):
                    return float(penalty)
                total_nll += nll

            if verbose and call_count[0] % log_every == 0:
                print(
                    f"[Eval {call_count[0]:>4}] "
                    f"beta=({vec[0]:.3f},{vec[1]:.3f},{vec[2]:.3f},{vec[3]:.3f}) "
                    f"g_c={vec[18]:.3f} g_a={vec[19]:.3f} g_e={vec[20]:.3f} "
                    f"NLL={total_nll:.2f}"
                )
            return float(total_nll)

        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"[Eval {call_count[0]}] FAILED: {e}")
            return float(penalty)

    loss.call_count = call_count   # type: ignore[attr-defined]
    return loss
