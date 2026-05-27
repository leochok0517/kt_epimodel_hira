"""ModelParameters ↔ 1D vector 변환 (optimizer 용).

Vector layout (23 elements):
    [0]  beta_h
    [1]  beta_w
    [2]  beta_s
    [3]  beta_o
    [4..17]  phi_a (a in {0..14} \\ {5})  — phi_5 (25-29) 제외
    [18] gamma_report
    [19] seasonality_amp
    [20] seasonality_base
    [21] seasonality_sigma
    [22] seasonality_peak_day

phi[5] = 1.0 reference.
seasonality_mode, period 는 vector 외 (caller 보존).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kt_epimodel_hira.model.parameters import (
    CalibrationParameters,
    DiseaseParameters,
)

N_AGE: int = 15
REF_AGE_IDX: int = 5
N_VECTOR: int = 23


@dataclass
class ParameterBounds:
    """파라미터별 (lower, upper) 경계.

    Phase A rollback: 이전 tightening 이 새 corner solution (β·φ floor) 을 만들어,
    fit 영역을 다시 넓힌다.
    """
    beta_h: tuple[float, float] = (0.001, 5.0)
    beta_w: tuple[float, float] = (0.001, 5.0)
    beta_s: tuple[float, float] = (0.001, 5.0)
    beta_o: tuple[float, float] = (0.001, 5.0)
    phi: tuple[float, float] = (0.1, 5.0)
    # TODO HIRA-B: gamma_report bound 좁히기 검토 — 권장 (0.05, 0.5).
    # 근거: HIRA 에서는 분모가 인구로 깨끗해져 의미가 "reporting fraction"
    # (감염자 중 진료 청구된 비율) 으로 단순화. ILI 의 (0.01, 1.0) 은 너무 넓음.
    # 상한 0.5: 청구가 실제 감염의 ≥ 50% 라는 가정 비현실 → 차단.
    # 하한 0.05: 그 이하면 다른 시즌으로 sanity check 필요.
    # 1차 fit 후 박힘 여부 보고 결정. 현재는 ILI 와 동일 (0.01, 1.0) 유지.
    gamma_report: tuple[float, float] = (0.01, 1.0)
    seasonality_amp: tuple[float, float] = (0.0, 3.0)
    seasonality_base: tuple[float, float] = (0.0, 1.0)
    seasonality_sigma: tuple[float, float] = (15.0, 80.0)
    seasonality_peak_day: tuple[float, float] = (80.0, 150.0)


def get_param_names() -> list[str]:
    names = ["beta_h", "beta_w", "beta_s", "beta_o"]
    for a in range(N_AGE):
        if a == REF_AGE_IDX:
            continue
        names.append(f"phi_{a}")
    names.append("gamma_report")
    names.append("seasonality_amp")
    names.append("seasonality_base")
    names.append("seasonality_sigma")
    names.append("seasonality_peak_day")
    return names


def params_to_vector(
    calibration: CalibrationParameters,
    disease: DiseaseParameters | None = None,
) -> np.ndarray:
    if disease is None:
        disease = DiseaseParameters()
    vec = np.zeros(N_VECTOR, dtype=np.float64)
    vec[0] = calibration.beta_h
    vec[1] = calibration.beta_w
    vec[2] = calibration.beta_s
    vec[3] = calibration.beta_o
    idx = 4
    for a in range(N_AGE):
        if a == REF_AGE_IDX:
            continue
        vec[idx] = calibration.phi[a]
        idx += 1
    vec[18] = calibration.gamma_report
    vec[19] = disease.seasonality_amp
    vec[20] = disease.seasonality_base
    vec[21] = disease.seasonality_sigma
    vec[22] = disease.seasonality_peak_day
    return vec


def vector_to_params(
    vec: np.ndarray,
) -> tuple[CalibrationParameters, float, float, float, float]:
    """(23,) → (CalibrationParameters, amp, base, sigma, peak_day)."""
    if vec.shape != (N_VECTOR,):
        raise ValueError(f"vec shape must be ({N_VECTOR},), got {vec.shape}")

    phi = np.ones(N_AGE, dtype=np.float64)
    idx = 4
    for a in range(N_AGE):
        if a == REF_AGE_IDX:
            continue
        phi[a] = vec[idx]
        idx += 1

    cal = CalibrationParameters(
        beta_h=float(vec[0]),
        beta_w=float(vec[1]),
        beta_s=float(vec[2]),
        beta_o=float(vec[3]),
        phi=phi,
        gamma_report=float(vec[18]),
    )
    return cal, float(vec[19]), float(vec[20]), float(vec[21]), float(vec[22])


def get_bounds_vector(
    bounds: ParameterBounds | None = None,
) -> list[tuple[float, float]]:
    if bounds is None:
        bounds = ParameterBounds()
    out: list[tuple[float, float]] = [
        bounds.beta_h, bounds.beta_w, bounds.beta_s, bounds.beta_o,
    ]
    for a in range(N_AGE):
        if a == REF_AGE_IDX:
            continue
        out.append(bounds.phi)
    out.append(bounds.gamma_report)
    out.append(bounds.seasonality_amp)
    out.append(bounds.seasonality_base)
    out.append(bounds.seasonality_sigma)
    out.append(bounds.seasonality_peak_day)
    return out


def initial_guess(
    base_calibration: CalibrationParameters | None = None,
    base_disease: DiseaseParameters | None = None,
) -> np.ndarray:
    if base_calibration is None:
        base_calibration = CalibrationParameters()
    if base_disease is None:
        # ILI peak ~ week 16-18 day 112-126. peak_day 110 으로 약간 앞당김.
        base_disease = DiseaseParameters(seasonality_peak_day=110.0)
    return params_to_vector(base_calibration, base_disease)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cal = CalibrationParameters()
    dis = DiseaseParameters()
    vec = params_to_vector(cal, dis)
    print(f"N_VECTOR: {N_VECTOR}, vec shape: {vec.shape}")
    print(f"Names (last 5): {get_param_names()[-5:]}")
    print(f"Vector: {vec}")

    cal2, amp2, base2, sigma2, peak2 = vector_to_params(vec)
    print(f"\nRound-trip phi[5]:  {cal2.phi[5]}")
    print(f"  amp:      {amp2}")
    print(f"  base:     {base2}")
    print(f"  sigma:    {sigma2}")
    print(f"  peak_day: {peak2}")
    print(f"Bounds length: {len(get_bounds_vector())}")
