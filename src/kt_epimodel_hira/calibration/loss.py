"""Calibration loss function (Poisson NLL) — HIRA 버전.

Optimizer (scipy.optimize.minimize) 호환 closure 생성.
vec → simulate → HIRA count 변환 → Poisson NLL.

ILI 버전 (kt_epimodel/calibration/loss.py) 대비 차이:
    - ``simulation_to_ili{_by_age}`` → ``simulation_to_hira{_by_age}``
    - target dict 키 ``ili_rate``/``ili_rates`` → ``hira_count``/``hira_counts``
    - ``pop_total`` / ``pop_15_flat`` 인자 불필요 (카운트 단위라 인구 분모 없음)
    - 그 외 로직 동일 (Poisson NLL 은 단위 무관, daily_new_infection_by_age()
      vax flux 분리도 동일).
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
from kt_epimodel_hira.model.parameters import DiseaseParameters, ModelParameters


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
    """Optimizer 용 loss function 생성 (single — 연령 합산 HIRA).

    Args:
        target: ``load_hira_target()`` 결과 — keys: ``hira_count``, ``is_valid``,
            ``n_weeks``, optional ``weights``.
        inputs: ``build_aggregated_inputs()`` 결과.
        base_params: 정책/백신/disease 등 calibration 외 파라미터.
        seed_total: 시즌 시작 I 총수 (seed_by_age 없을 때만).
        initial_immunity, initial_vaccinated_fraction: 초기 R, V 비율.
        t_span: 시뮬레이션 기간 (default 364).
        verbose: 주기적 로그.
        log_every: 로그 빈도.
        penalty: 시뮬레이션/파라미터 오류 시 반환값.

    Returns:
        loss(vec) → float (Poisson NLL, 작을수록 좋음).
    """
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
            new_cal, new_amp, new_base, new_sigma, new_peak = vector_to_params(vec)
            base_d = base_params.disease
            new_disease = DiseaseParameters(
                sigma=base_d.sigma,
                gamma=base_d.gamma,
                kappa=base_d.kappa,
                seasonality_mode=base_d.seasonality_mode,
                seasonality_amp=new_amp,
                seasonality_base=new_base,
                seasonality_peak_day=new_peak,
                seasonality_period=base_d.seasonality_period,
                seasonality_sigma=new_sigma,
            )
            new_params = (
                base_params.with_calibration(new_cal).with_disease(new_disease)
            )

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

            daily_inc = result.daily_new_infection()
            predicted = simulation_to_hira(
                daily_inc, new_cal.gamma_report, n_weeks=n_weeks,
            )
            nll = poisson_log_likelihood(
                observed, predicted, is_valid, weights=weights,
            )

            if verbose and call_count[0] % log_every == 0:
                print(
                    f"[Eval {call_count[0]:>4}] "
                    f"β=({vec[0]:.3f},{vec[1]:.3f},{vec[2]:.3f},{vec[3]:.3f}) "
                    f"γ_r={vec[18]:.3f} amp={vec[19]:.3f} base={vec[20]:.3f} "
                    f"σ={vec[21]:.1f} pk={vec[22]:.0f}  NLL={nll:.2f}"
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
    """6 HIRA 연령 그룹 동시 fit 용 loss (총 NLL = Σ 그룹별 NLL).

    daily incidence by age 는 ``Δ(E+I+R)`` (vax 흐름 제외).
    """
    age_groups: list[str] = list(target_by_age["age_groups"])
    n_weeks = int(target_by_age["n_weeks"])

    call_count = [0]

    def loss(vec: np.ndarray) -> float:
        call_count[0] += 1
        try:
            cal_new, amp_new, base_new, sigma_new, peak_new = vector_to_params(vec)
            base_d = base_params.disease
            new_disease = DiseaseParameters(
                sigma=base_d.sigma, gamma=base_d.gamma, kappa=base_d.kappa,
                seasonality_mode=base_d.seasonality_mode,
                seasonality_amp=amp_new,
                seasonality_base=base_new,
                seasonality_peak_day=peak_new,
                seasonality_period=base_d.seasonality_period,
                seasonality_sigma=sigma_new,
            )
            new_params = (
                base_params.with_calibration(cal_new).with_disease(new_disease)
            )

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

            # Δ(E + I + R) per age (백신 흐름 S→V 제외)
            daily_inc_by_age = result.daily_new_infection_by_age()  # (n_t-1, 15)

            predictions = simulation_to_hira_by_age(
                daily_inc_by_age, cal_new.gamma_report, n_weeks=n_weeks,
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
                    f"β=({vec[0]:.3f},{vec[1]:.3f},{vec[2]:.3f},{vec[3]:.3f}) "
                    f"γ_r={vec[18]:.3f} amp={vec[19]:.3f} base={vec[20]:.3f} "
                    f"σ={vec[21]:.1f} pk={vec[22]:.0f}  NLL={total_nll:.2f}"
                )
            return float(total_nll)

        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"[Eval {call_count[0]}] FAILED: {e}")
            return float(penalty)

    loss.call_count = call_count   # type: ignore[attr-defined]
    return loss
