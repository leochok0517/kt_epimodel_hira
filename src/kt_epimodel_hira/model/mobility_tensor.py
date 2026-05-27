"""KT mobility (1154×1154×7×24) → 채널별 NIMS 15군 M 텐서 (Step C 갱신).

3 종류 M:
- M_home, M_school: identity (자기 행정동에서 노출)
- M_work:  KT 9-17시 평균, 근로자 (NIMS 4-13) 만 mobile, 그 외 정적
- M_other: KT 17-22시 평균, 학생·근로자 (NIMS 2-13) mobile, 0-9·70+ 정적

KT 그룹 → NIMS 15군 매핑 (work / other 별로 다름):
- KT 0 (10-19): work 제외 (학생), other 적용 → NIMS [2, 3]
- KT 1-5 (20-69): 양쪽 적용 → NIMS [4..13]
- KT 6 (70+): 양쪽 제외 (은퇴/정적)

기존 `build_pi_from_kt_array` / `build_pi_tensor` 는 시간 평균 (legacy) — 호환 유지.
"""

from __future__ import annotations

import numpy as np

N_AGE: int = 15

# 채널별 정적 연령 (mobility 미적용 → identity row)
STATIC_WORK: tuple[int, ...] = (0, 1, 2, 3, 14)   # 학생 + 70+
STATIC_OTHER: tuple[int, ...] = (0, 1, 14)         # 0-9 + 70+ (10-19 는 학원/저녁 이동)

# legacy 호환
STATIC_NIMS: tuple[int, ...] = STATIC_WORK

# KT → NIMS 매핑 (work / other 공통 — 5-year 분할은 normalize 후 동일 패턴)
KT_TO_NIMS_WORK: dict[int, tuple[int, int]] = {
    1: (4, 5), 2: (6, 7), 3: (8, 9), 4: (10, 11), 5: (12, 13),
}
KT_TO_NIMS_OTHER: dict[int, tuple[int, int]] = {
    0: (2, 3),                                   # 10-19 (학원/저녁)
    1: (4, 5), 2: (6, 7), 3: (8, 9), 4: (10, 11), 5: (12, 13),
}

# legacy
KT_TO_NIMS_MOBILE: dict[int, tuple[int, int]] = KT_TO_NIMS_WORK


# ---------------------------------------------------------------------------
# Identity helpers (home / school)
# ---------------------------------------------------------------------------

def build_M_identity(n_admdong: int, n_ages: int = N_AGE) -> np.ndarray:
    """(n_ages, n, n) identity stack — 정적 mixing 용."""
    eye = np.eye(n_admdong, dtype=np.float64)
    return np.broadcast_to(eye, (n_ages, n_admdong, n_admdong)).copy()


def build_M_home(n_admdong: int, n_ages: int = N_AGE) -> np.ndarray:
    """Home: 자기 행정동."""
    return build_M_identity(n_admdong, n_ages)


def build_M_school(n_admdong: int, n_ages: int = N_AGE) -> np.ndarray:
    """School: 학생 자기 행정동 (단순화)."""
    return build_M_identity(n_admdong, n_ages)


# ---------------------------------------------------------------------------
# Work / Other (hour-range filtered)
# ---------------------------------------------------------------------------

def _build_M_from_pi_hour_range(
    pi_full: np.ndarray,
    hours: range,
    kt_to_nims: dict[int, tuple[int, int]],
    static_ages: tuple[int, ...],
    correction_factor: float = 1.0 / 0.9,
) -> np.ndarray:
    """KT (n, n, 7, 24) + hour-range → (15, n, n) M 텐서. 행 합 = 1.

    Args:
        pi_full: (n, n, 7, 24) — pi[o, d, k, h] = daytype 일평균 o→d 이동량.
        hours: 사용할 시간대 (range).
        kt_to_nims: {KT_idx: (NIMS_a1, NIMS_a2)}.
        static_ages: 정적 처리할 NIMS 인덱스 — identity row.
        correction_factor: KT 검출률 보정 (normalize 후 효과 무 — API 호환).

    Returns:
        M: (15, n, n) — M[a, o, d] = a 연령 o거주민이 d에 있을 비율, Σ_d = 1.
    """
    if pi_full.ndim != 4 or pi_full.shape[2] != 7:
        raise ValueError(f"pi_full must be (n, n, 7, h), got {pi_full.shape}")
    n_adm = pi_full.shape[0]

    hour_idx = list(hours)
    pi_h = pi_full[:, :, :, hour_idx].astype(np.float64).sum(axis=3) * correction_factor
    # pi_h: (n_o, n_d, 7)

    eye = np.eye(n_adm, dtype=np.float64)
    M = np.zeros((N_AGE, n_adm, n_adm), dtype=np.float64)

    for a in static_ages:
        M[a] = eye

    for k, ages in kt_to_nims.items():
        # ages 모두 static 이면 skip
        ages_mobile = [a for a in ages if a not in static_ages]
        if not ages_mobile:
            continue
        flow = pi_h[:, :, k]                                 # (n_o, n_d)
        row_sum = flow.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            M_k = np.where(row_sum[:, None] > 0, flow / row_sum[:, None], 0.0)
        zero_origins = row_sum == 0
        if zero_origins.any():
            M_k[zero_origins] = eye[zero_origins]
        for a in ages_mobile:
            M[a] = M_k.copy()

    return M


def build_M_from_kt_array(
    pi_full: np.ndarray,
    channel: str,
    correction_factor: float = 1.0 / 0.9,
) -> np.ndarray:
    """KT 텐서로부터 직접 채널별 M 빌드 (테스트/내부 용).

    Args:
        channel: 'work' | 'other'.
    """
    if channel == "work":
        return _build_M_from_pi_hour_range(
            pi_full, hours=range(9, 17),
            kt_to_nims=KT_TO_NIMS_WORK, static_ages=STATIC_WORK,
            correction_factor=correction_factor,
        )
    if channel == "other":
        return _build_M_from_pi_hour_range(
            pi_full, hours=range(17, 22),
            kt_to_nims=KT_TO_NIMS_OTHER, static_ages=STATIC_OTHER,
            correction_factor=correction_factor,
        )
    raise ValueError(f"channel must be 'work' or 'other', got {channel!r}")


def build_M_work(
    yyyymm: str,
    daytype: str = "weekday",
    admdong_codes: list[str] | None = None,
    pop_15: np.ndarray | None = None,
    correction_factor: float = 1.0 / 0.9,
) -> np.ndarray:
    """KT mobility 9-17시 weekday 평균 → work commuting M (15, n, n).

    Args:
        admdong_codes: load_population 의 코드 순서와 일치해야 함.
        pop_15: API 호환용 (현 구현 미사용 — KT 그룹 내 동일 패턴 공유).
    """
    from kt_data.data.load_mobility import load_mobility

    mob = load_mobility(yyyymm, daytype=daytype)
    if admdong_codes is not None and mob["admdong_codes"] != admdong_codes:
        raise ValueError("admdong_codes mismatch")
    return build_M_from_kt_array(mob["pi"], "work", correction_factor)


def build_M_other(
    yyyymm: str,
    daytype: str = "weekday",
    admdong_codes: list[str] | None = None,
    pop_15: np.ndarray | None = None,
    correction_factor: float = 1.0 / 0.9,
) -> np.ndarray:
    """KT mobility 17-22시 weekday 평균 → other commuting M (15, n, n)."""
    from kt_data.data.load_mobility import load_mobility

    mob = load_mobility(yyyymm, daytype=daytype)
    if admdong_codes is not None and mob["admdong_codes"] != admdong_codes:
        raise ValueError("admdong_codes mismatch")
    return build_M_from_kt_array(mob["pi"], "other", correction_factor)


# ---------------------------------------------------------------------------
# Legacy (시간 전체 평균) — 호환 유지
# ---------------------------------------------------------------------------

def build_pi_from_kt_array(
    pi_kt: np.ndarray,
    pop_15: np.ndarray,
    correction_factor: float = 1.0 / 0.9,
) -> np.ndarray:
    """Legacy: 전체 24시간 평균 (build_M_work 와 별개)."""
    if pi_kt.ndim != 4 or pi_kt.shape[2] != 7:
        raise ValueError(f"pi_kt shape must be (n, n, 7, h), got {pi_kt.shape}")
    n_adm = pi_kt.shape[0]
    if pop_15.shape != (N_AGE, n_adm):
        raise ValueError(f"pop_15 shape {pop_15.shape} != ({N_AGE}, {n_adm})")

    flow_kt = pi_kt.astype(np.float64).sum(axis=3) * correction_factor

    pi = np.zeros((N_AGE, n_adm, n_adm), dtype=np.float64)
    eye = np.eye(n_adm, dtype=np.float64)
    for a in STATIC_NIMS:
        pi[a] = eye
    for k, (a1, a2) in KT_TO_NIMS_MOBILE.items():
        flow = flow_kt[:, :, k]
        row_sum = flow.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pi_k = np.where(row_sum[:, None] > 0, flow / row_sum[:, None], 0.0)
        zero_origins = row_sum == 0
        if zero_origins.any():
            pi_k[zero_origins] = eye[zero_origins]
        pi[a1] = pi_k
        pi[a2] = pi_k.copy()
    return pi


def build_pi_tensor(
    yyyymm: str,
    daytype: str,
    admdong_codes: list[str],
    pop_15: np.ndarray,
    correction_factor: float = 1.0 / 0.9,
) -> np.ndarray:
    """Legacy wrapper — 전체 24시간 평균."""
    from kt_data.data.load_mobility import load_mobility

    mob = load_mobility(yyyymm, daytype=daytype)
    if mob["admdong_codes"] != admdong_codes:
        raise ValueError("admdong_codes mismatch")
    return build_pi_from_kt_array(mob["pi"], pop_15, correction_factor)
