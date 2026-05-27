"""kt_epimodel_hira calibration — HIRA episode count fitting."""

from kt_epimodel_hira.calibration.hira_target import (
    HIRA_GROUP_TO_NIMS_WEIGHTED,
    N_WEEKS,
    load_hira_target,
    load_hira_target_by_age,
    poisson_log_likelihood,
    season_start_date,
    simulation_to_hira,
    simulation_to_hira_by_age,
)
from kt_epimodel_hira.calibration.loss import (
    make_loss_function,
    make_loss_function_by_age,
)
from kt_epimodel_hira.calibration.optimizer import (
    CalibrationResult,
    load_result,
    optimize_calibration,
    optimize_calibration_by_age,
    save_result,
)
from kt_epimodel_hira.calibration.param_vector import (
    N_VECTOR,
    ParameterBounds,
    get_bounds_vector,
    get_param_names,
    initial_guess,
    params_to_vector,
    vector_to_params,
)
from kt_epimodel_hira.calibration.simple_model import (
    build_aggregated_inputs,
    estimate_initial_infected_from_hira,
    simulate_aggregated,
)

__all__ = [
    "CalibrationResult",
    "HIRA_GROUP_TO_NIMS_WEIGHTED",
    "N_VECTOR",
    "N_WEEKS",
    "ParameterBounds",
    "build_aggregated_inputs",
    "estimate_initial_infected_from_hira",
    "get_bounds_vector",
    "get_param_names",
    "initial_guess",
    "load_hira_target",
    "load_hira_target_by_age",
    "load_result",
    "make_loss_function",
    "make_loss_function_by_age",
    "optimize_calibration",
    "optimize_calibration_by_age",
    "params_to_vector",
    "poisson_log_likelihood",
    "save_result",
    "season_start_date",
    "simulate_aggregated",
    "simulation_to_hira",
    "simulation_to_hira_by_age",
    "vector_to_params",
]
