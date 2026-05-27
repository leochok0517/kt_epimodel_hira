"""Model parameters (Step A — 4-channel β + κ + ρ + V).

이전 SEIR+J 모델에서 SEIRV (J 제거) 로 이행하기 위한 parameters 전면 개편.
J 격리 → channel별 β·κ·ρ 로 재구성.

6 카테고리:
- DiseaseParameters: σ, γ, κ (가구 노출 factor)
- CalibrationParameters: β_h, β_w, β_s, β_o, φ, γ_report
- PolicyParameters: p_school, p_work
- VaccinationParameters: VE, annual_coverage, Gaussian timing
- EmploymentParameters: ρ (행정동 × 연령 경제활동인구 비율)
- TimeVaryingParameters: daytype 별 채널 활성도

ModelParameters = 위 6개 묶음.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_AGE: int = 15
REF_AGE_IDX: int = 5            # 25-29세 (phi normalization reference)
_SEASON_START_ISO_WEEK: int = 36


# ---------------------------------------------------------------------------
# DiseaseParameters
# ---------------------------------------------------------------------------

_DEFAULT_KAPPA: tuple[float, ...] = (
    0.42, 0.42, 0.42, 0.42,         # 0-4, 5-9, 10-14, 15-19 (학생)
    0.60, 0.60, 0.60, 0.60,         # 20-24 ~ 35-39 (성인)
    0.60, 0.60, 0.60, 0.60,         # 40-44 ~ 55-59
    0.60, 0.60,                      # 60-64, 65-69
    0.0,                              # 70+ (은퇴, 학교/직장 없음)
)


@dataclass(frozen=True)
class DiseaseParameters:
    """인플루엔자 자연사 + 가구 노출 factor κ + 환경 시즌성.

    시즌성: f_env(t) = 1 + amp · cos(2π·(t − peak_day) / period).
    """
    sigma: float = 0.5
    gamma: float = 0.25
    kappa: tuple[float, ...] = _DEFAULT_KAPPA
    # 환경 시즌성 (cosine factor on β)
    seasonality_mode: str = "gaussian"   # 'gaussian' or 'cosine'
    seasonality_amp: float = 1.0        # peak 추가량
    seasonality_base: float = 0.1        # baseline (off-season factor)
    seasonality_peak_day: float = 130.0  # 정점 day — ≈ 1월 중순
    seasonality_period: float = 365.0    # cosine 주기 (gaussian 모드선 무시)
    seasonality_sigma: float = 40.0      # gaussian σ (days)

    def __post_init__(self) -> None:
        for name in ("sigma", "gamma"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be positive finite, got {v}")
        if len(self.kappa) != N_AGE:
            raise ValueError(f"kappa must have {N_AGE} elements, got {len(self.kappa)}")
        if any((k < 0) or (not np.isfinite(k)) for k in self.kappa):
            raise ValueError("kappa must be non-negative finite")
        if self.seasonality_mode not in ("gaussian", "cosine"):
            raise ValueError(
                f"seasonality_mode must be 'gaussian' or 'cosine', "
                f"got {self.seasonality_mode!r}"
            )
        if not (0.0 <= self.seasonality_amp <= 5.0) or not np.isfinite(self.seasonality_amp):
            raise ValueError(
                f"seasonality_amp must be in [0, 5], got {self.seasonality_amp}"
            )
        if not np.isfinite(self.seasonality_base) or self.seasonality_base < 0:
            raise ValueError(
                f"seasonality_base must be nonneg finite, got {self.seasonality_base}"
            )
        if not np.isfinite(self.seasonality_peak_day):
            raise ValueError("seasonality_peak_day must be finite")
        if self.seasonality_period <= 0 or not np.isfinite(self.seasonality_period):
            raise ValueError(
                f"seasonality_period must be positive, got {self.seasonality_period}"
            )
        if self.seasonality_sigma <= 0 or not np.isfinite(self.seasonality_sigma):
            raise ValueError(
                f"seasonality_sigma must be positive, got {self.seasonality_sigma}"
            )

    @property
    def kappa_array(self) -> np.ndarray:
        return np.array(self.kappa, dtype=np.float64)

    @property
    def latent_period(self) -> float:
        return 1.0 / self.sigma

    @property
    def infectious_period(self) -> float:
        return 1.0 / self.gamma

    def seasonal_factor(self, day_in_season: float) -> float:
        """f_env(t) = base + amp · g(t) where g depends on mode.

        - 'gaussian': g = exp(-(t-peak)² / (2σ²))
        - 'cosine':   g = cos(2π·(t-peak)/period)

        Clip to [0, ∞) (β 음수 방지).
        """
        if self.seasonality_mode == "gaussian":
            g = np.exp(
                -((day_in_season - self.seasonality_peak_day) ** 2)
                / (2.0 * self.seasonality_sigma ** 2)
            )
        else:   # cosine
            g = np.cos(
                2.0 * np.pi * (day_in_season - self.seasonality_peak_day)
                / self.seasonality_period
            )
        raw = self.seasonality_base + self.seasonality_amp * g
        return float(max(raw, 0.0))


# ---------------------------------------------------------------------------
# CalibrationParameters
# ---------------------------------------------------------------------------

@dataclass
class CalibrationParameters:
    """4 채널 β + 연령별 susceptibility φ + 보고율 γ_report.

    β_h, β_w, β_s, β_o: home, work, school, other 채널의 transmission rate.
    Calibration 에서 fit 대상.
    """
    beta_h: float = 0.05
    beta_w: float = 0.05
    beta_s: float = 0.05
    beta_o: float = 0.05
    phi: np.ndarray = field(default_factory=lambda: np.ones(N_AGE))
    gamma_report: float = 0.5

    def __post_init__(self) -> None:
        self.phi = np.asarray(self.phi, dtype=np.float64)
        if self.phi.shape != (N_AGE,):
            raise ValueError(f"phi must be shape ({N_AGE},), got {self.phi.shape}")
        if not np.isfinite(self.phi).all() or (self.phi <= 0).any():
            raise ValueError("phi must be all positive finite")
        for name in ("beta_h", "beta_w", "beta_s", "beta_o"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be non-negative finite, got {v}")
        if not (0 < self.gamma_report <= 1):
            raise ValueError(f"gamma_report must be in (0, 1], got {self.gamma_report}")

    @property
    def betas(self) -> dict[str, float]:
        return {
            "home": self.beta_h,
            "work": self.beta_w,
            "school": self.beta_s,
            "other": self.beta_o,
        }

    def with_reference_normalized(self) -> "CalibrationParameters":
        """φ_25-29 = 1 정규화 — 4채널 β 모두 ratio 만큼 흡수."""
        ratio = float(self.phi[REF_AGE_IDX])
        if ratio <= 0:
            raise ValueError("phi[reference] must be positive")
        return CalibrationParameters(
            beta_h=self.beta_h * ratio,
            beta_w=self.beta_w * ratio,
            beta_s=self.beta_s * ratio,
            beta_o=self.beta_o * ratio,
            phi=self.phi / ratio,
            gamma_report=self.gamma_report,
        )


# ---------------------------------------------------------------------------
# PolicyParameters
# ---------------------------------------------------------------------------

@dataclass
class PolicyParameters:
    """정책 변수 — 학교 출석률, 직장 출근률 (스칼라)."""
    p_school: float = 1.0
    p_work: float = 1.0

    def __post_init__(self) -> None:
        for name in ("p_school", "p_work"):
            v = getattr(self, name)
            if not np.isfinite(v) or not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")

    @classmethod
    def baseline(cls) -> "PolicyParameters":
        return cls(p_school=1.0, p_work=1.0)

    @classmethod
    def school_closure(cls, attendance: float = 0.05) -> "PolicyParameters":
        return cls(p_school=attendance, p_work=1.0)

    @classmethod
    def sick_leave_enhanced(cls, work_rate: float = 0.4) -> "PolicyParameters":
        return cls(p_school=1.0, p_work=work_rate)

    @classmethod
    def comprehensive(
        cls, school_attendance: float = 0.05, work_rate: float = 0.4,
    ) -> "PolicyParameters":
        return cls(p_school=school_attendance, p_work=work_rate)


# ---------------------------------------------------------------------------
# VaccinationParameters
# ---------------------------------------------------------------------------

def _default_annual_coverage() -> np.ndarray:
    """한국 인플루엔자 시즌 누적 접종률 (잠정)."""
    return np.array([
        0.75, 0.75,
        0.40, 0.40,
        0.30, 0.30, 0.30, 0.30,
        0.30, 0.30, 0.30, 0.30,
        0.30, 0.30,
        0.82,
    ], dtype=np.float64)


@dataclass
class VaccinationParameters:
    """백신: VE 고정, 시간 의존 접종률 (Gaussian)."""
    VE: float = 0.5
    annual_coverage: np.ndarray = field(default_factory=_default_annual_coverage)
    peak_iso_week: int = 42
    spread_weeks: float = 4.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.VE <= 1.0) or not np.isfinite(self.VE):
            raise ValueError(f"VE must be in [0, 1], got {self.VE}")
        self.annual_coverage = np.asarray(self.annual_coverage, dtype=np.float64)
        if self.annual_coverage.shape != (N_AGE,):
            raise ValueError(
                f"annual_coverage shape {self.annual_coverage.shape} != ({N_AGE},)"
            )
        if not np.isfinite(self.annual_coverage).all():
            raise ValueError("annual_coverage must be finite")
        if ((self.annual_coverage < 0) | (self.annual_coverage > 1)).any():
            raise ValueError("annual_coverage must be in [0, 1]")
        if self.spread_weeks <= 0 or not np.isfinite(self.spread_weeks):
            raise ValueError(f"spread_weeks must be positive, got {self.spread_weeks}")
        if not isinstance(self.peak_iso_week, (int, np.integer)):
            raise ValueError(f"peak_iso_week must be int, got {type(self.peak_iso_week)}")

    def _density(self, day_in_season: float) -> float:
        peak_day = (self.peak_iso_week - _SEASON_START_ISO_WEEK) * 7
        sigma_days = self.spread_weeks * 7
        z = (day_in_season - peak_day) / sigma_days
        return float(np.exp(-0.5 * z * z) / (sigma_days * np.sqrt(2 * np.pi)))

    def daily_rate(self, day_in_season: int, age_idx: int) -> float:
        return float(self.annual_coverage[age_idx]) * self._density(day_in_season)

    def rate_vector(self, day_in_season: int) -> np.ndarray:
        return self.annual_coverage * self._density(day_in_season)


# ---------------------------------------------------------------------------
# EmploymentParameters
# ---------------------------------------------------------------------------

@dataclass
class EmploymentParameters:
    """행정동 × 연령 경제활동인구 비율 ρ."""
    rho: np.ndarray = field(default_factory=lambda: np.zeros((1154, N_AGE)))

    def __post_init__(self) -> None:
        self.rho = np.asarray(self.rho, dtype=np.float64)
        if self.rho.ndim != 2 or self.rho.shape[1] != N_AGE:
            raise ValueError(
                f"rho must be (n_admdong, {N_AGE}), got {self.rho.shape}"
            )
        if not np.isfinite(self.rho).all():
            raise ValueError("rho must be finite")
        if ((self.rho < 0) | (self.rho > 1)).any():
            raise ValueError("rho must be in [0, 1]")

    @classmethod
    def from_kt_data(
        cls, admdong_codes: list[str] | None = None,
    ) -> "EmploymentParameters":
        """kt_data.build_rho_matrix 로 자동 빌드."""
        from kt_data import build_rho_matrix, load_population_15groups

        if admdong_codes is None:
            df = load_population_15groups()
            admdong_codes = (
                df.select("admdong_cd")
                .unique()
                .sort("admdong_cd")
                .get_column("admdong_cd")
                .to_list()
            )
        return cls(rho=build_rho_matrix(admdong_codes))

    @property
    def n_admdong(self) -> int:
        return int(self.rho.shape[0])


# ---------------------------------------------------------------------------
# TimeVaryingParameters
# ---------------------------------------------------------------------------

def _factor_weekday() -> dict[str, float]:
    return {"home": 1.0, "work": 1.0, "school": 1.0, "other": 1.0}


def _factor_weekend() -> dict[str, float]:
    return {"home": 1.3, "work": 0.2, "school": 0.0, "other": 1.2}


def _factor_holiday() -> dict[str, float]:
    return {"home": 1.3, "work": 0.2, "school": 0.0, "other": 1.2}


def _factor_vacation() -> dict[str, float]:
    return {"home": 1.1, "work": 1.0, "school": 0.2, "other": 1.1}


_DAYTYPE_MAP_ATTR: dict[str, str] = {
    "weekday_school": "weekday_factor",
    "vacation_weekday": "vacation_factor",
    "weekend": "weekend_factor",
    "holiday": "holiday_factor",
}


@dataclass
class TimeVaryingParameters:
    """채널 활성도 (계절/주중주말/방학/공휴일)."""
    weekday_factor: dict[str, float] = field(default_factory=_factor_weekday)
    weekend_factor: dict[str, float] = field(default_factory=_factor_weekend)
    holiday_factor: dict[str, float] = field(default_factory=_factor_holiday)
    vacation_factor: dict[str, float] = field(default_factory=_factor_vacation)

    def __post_init__(self) -> None:
        required = {"home", "work", "school", "other"}
        for name in ("weekday_factor", "weekend_factor", "holiday_factor", "vacation_factor"):
            d = getattr(self, name)
            missing = required - set(d.keys())
            if missing:
                raise ValueError(f"{name} missing channels: {missing}")
            for ch, v in d.items():
                if not np.isfinite(v) or v < 0:
                    raise ValueError(f"{name}[{ch}] must be nonneg finite, got {v}")

    def get(self, daytype: str) -> dict[str, float]:
        attr = _DAYTYPE_MAP_ATTR.get(daytype)
        if attr is None:
            raise ValueError(f"Unknown daytype: {daytype!r}")
        return getattr(self, attr)


# ---------------------------------------------------------------------------
# ModelParameters
# ---------------------------------------------------------------------------

@dataclass
class ModelParameters:
    """전체 모델 파라미터 (6 카테고리)."""
    disease: DiseaseParameters = field(default_factory=DiseaseParameters)
    calibration: CalibrationParameters = field(default_factory=CalibrationParameters)
    policy: PolicyParameters = field(default_factory=PolicyParameters.baseline)
    time_varying: TimeVaryingParameters = field(default_factory=TimeVaryingParameters)
    vaccination: VaccinationParameters = field(default_factory=VaccinationParameters)
    employment: EmploymentParameters | None = None

    def with_policy(self, policy: PolicyParameters) -> "ModelParameters":
        return ModelParameters(
            disease=self.disease, calibration=self.calibration, policy=policy,
            time_varying=self.time_varying, vaccination=self.vaccination,
            employment=self.employment,
        )

    def with_calibration(self, calibration: CalibrationParameters) -> "ModelParameters":
        return ModelParameters(
            disease=self.disease, calibration=calibration, policy=self.policy,
            time_varying=self.time_varying, vaccination=self.vaccination,
            employment=self.employment,
        )

    def with_vaccination(self, vaccination: VaccinationParameters) -> "ModelParameters":
        return ModelParameters(
            disease=self.disease, calibration=self.calibration, policy=self.policy,
            time_varying=self.time_varying, vaccination=vaccination,
            employment=self.employment,
        )

    def with_employment(self, employment: EmploymentParameters) -> "ModelParameters":
        return ModelParameters(
            disease=self.disease, calibration=self.calibration, policy=self.policy,
            time_varying=self.time_varying, vaccination=self.vaccination,
            employment=employment,
        )

    def with_disease(self, disease: DiseaseParameters) -> "ModelParameters":
        return ModelParameters(
            disease=disease, calibration=self.calibration, policy=self.policy,
            time_varying=self.time_varying, vaccination=self.vaccination,
            employment=self.employment,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

AGE_LABELS_15 = [
    "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39",
    "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70+",
]


if __name__ == "__main__":
    params = ModelParameters()

    print("=== Default parameters ===")
    print(f"Disease: σ={params.disease.sigma}, γ={params.disease.gamma}")
    ka = params.disease.kappa_array
    print(f"  kappa (student 0-4):  {ka[0]}")
    print(f"  kappa (adult 25-29):  {ka[5]}")
    print(f"  kappa (70+):          {ka[14]}")

    c = params.calibration
    print(f"\nCalibration: β_h={c.beta_h}, β_w={c.beta_w}, "
          f"β_s={c.beta_s}, β_o={c.beta_o}")
    print(f"  γ_report = {c.gamma_report}")

    p = params.policy
    print(f"\nPolicy (baseline): p_school={p.p_school}, p_work={p.p_work}")

    v = params.vaccination
    print(f"\nVaccination: VE={v.VE}, peak ISO week={v.peak_iso_week}, "
          f"spread σ={v.spread_weeks}w")

    print("\n=== Building rho from kt_data ===")
    emp = EmploymentParameters.from_kt_data()
    params = params.with_employment(emp)
    print(f"ρ shape:        {emp.rho.shape}")
    print(f"ρ mean overall: {emp.rho.mean():.3f}")
    print("ρ by age (admdong avg):")
    for a, label in enumerate(AGE_LABELS_15):
        print(f"  [{a:>2}] {label:>5}: {emp.rho[:, a].mean():.3f}")

    print("\n=== Policy scenarios ===")
    for name, pol in [
        ("baseline", PolicyParameters.baseline()),
        ("school_closure", PolicyParameters.school_closure()),
        ("sick_leave", PolicyParameters.sick_leave_enhanced()),
        ("comprehensive", PolicyParameters.comprehensive()),
    ]:
        print(f"  {name:>16}: p_school={pol.p_school}, p_work={pol.p_work}")

    print("\n=== Vaccination rate at peak (day 42) ===")
    rates = params.vaccination.rate_vector(42)
    for a, label in enumerate(AGE_LABELS_15):
        print(f"  [{a:>2}] {label:>5}: {rates[a]:.4f}")
