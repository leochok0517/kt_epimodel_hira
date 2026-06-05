"""CalibrationParameters <-> 1D vector (optimizer).

Vector layout (21 elements):
    [0]  beta_h
    [1]  beta_w
    [2]  beta_s
    [3]  beta_o
    [4..17]  phi_a (a in {0..14} \\ {5})  — phi_5 (25-29) excluded
    [18] gamma_child   (NIMS 0-3, age 0-19)
    [19] gamma_adult   (NIMS 4-12, age 20-64)
    [20] gamma_elder   (NIMS 13-14, age 65+)

phi[5] = 1.0 reference.
Seasonality fixed (cosine, amp=0.7, peak=105).
See docs/AGE_DEPENDENT_GAMMA.md for gamma rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kt_epimodel_hira.model.parameters import CalibrationParameters

N_AGE: int = 15
REF_AGE_IDX: int = 5
N_VECTOR: int = 21


@dataclass
class ParameterBounds:
    beta_h: tuple[float, float] = (0.001, 5.0)
    beta_w: tuple[float, float] = (0.001, 5.0)
    beta_s: tuple[float, float] = (0.001, 5.0)
    beta_o: tuple[float, float] = (0.001, 5.0)
    phi: tuple[float, float] = (0.1, 5.0)
    gamma_child: tuple[float, float] = (0.05, 0.60)
    gamma_adult: tuple[float, float] = (0.05, 0.60)
    gamma_elder: tuple[float, float] = (0.05, 0.60)


def get_param_names() -> list[str]:
    names = ["beta_h", "beta_w", "beta_s", "beta_o"]
    for a in range(N_AGE):
        if a == REF_AGE_IDX:
            continue
        names.append(f"phi_{a}")
    names.extend(["gamma_child", "gamma_adult", "gamma_elder"])
    return names


def params_to_vector(calibration: CalibrationParameters) -> np.ndarray:
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
    vec[18] = calibration.gamma_child
    vec[19] = calibration.gamma_adult
    vec[20] = calibration.gamma_elder
    return vec


def vector_to_params(vec: np.ndarray) -> CalibrationParameters:
    """(21,) -> CalibrationParameters."""
    if vec.shape != (N_VECTOR,):
        raise ValueError(f"vec shape must be ({N_VECTOR},), got {vec.shape}")

    phi = np.ones(N_AGE, dtype=np.float64)
    idx = 4
    for a in range(N_AGE):
        if a == REF_AGE_IDX:
            continue
        phi[a] = vec[idx]
        idx += 1

    return CalibrationParameters(
        beta_h=float(vec[0]),
        beta_w=float(vec[1]),
        beta_s=float(vec[2]),
        beta_o=float(vec[3]),
        phi=phi,
        gamma_child=float(vec[18]),
        gamma_adult=float(vec[19]),
        gamma_elder=float(vec[20]),
    )


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
    out.extend([bounds.gamma_child, bounds.gamma_adult, bounds.gamma_elder])
    return out


def initial_guess(
    base_calibration: CalibrationParameters | None = None,
) -> np.ndarray:
    if base_calibration is None:
        base_calibration = CalibrationParameters()
    return params_to_vector(base_calibration)


if __name__ == "__main__":
    cal = CalibrationParameters()
    vec = params_to_vector(cal)
    print(f"N_VECTOR: {N_VECTOR}, vec shape: {vec.shape}")
    print(f"Names: {get_param_names()}")
    print(f"Vector: {vec}")

    cal2 = vector_to_params(vec)
    print(f"\nRound-trip gamma: child={cal2.gamma_child} adult={cal2.gamma_adult} "
          f"elder={cal2.gamma_elder}")
    print(f"Bounds length: {len(get_bounds_vector())}")
