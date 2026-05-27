# kt_epimodel_hira

**Sister project of [`kt_epimodel`](../kt_epimodel) — uses HIRA episode count data instead of ILI.**

수도권 metapop 인플루엔자 모델 + sick-leave 정책 ICER 분석.
Calibration target 만 ILI (질병청 외래환자 1,000명당 의사환자 분율) 에서
HIRA (국민건강보험공단 인플루엔자 진료에피소드 카운트) 로 교체한 자매 프로젝트.

설계는 [`../kt_epimodel/docs/SKELETON_ANALYSIS.md`](../kt_epimodel/docs/SKELETON_ANALYSIS.md)
섹션 6 권장안을 따른다. 본 저장소에도 동일 사본을 `docs/SKELETON_ANALYSIS.md`
에 둠.

## 의존성
- [`kt_data`](../kt_data/) — KT mobility + NIMS contact + ILI/HIRA 정제 데이터 및 로더 (`uv sources`로 editable 참조)

## 설치
같은 부모 폴더(`~/Documents/python/NIMS/`)에 `kt_data`가 있어야 함.
```bash
uv sync
```

데이터를 다른 위치에 두려면 환경변수 사용:
```bash
export KT_DATA_ROOT=/path/to/kt_data/data
```

HIRA xlsx 는 `kt_data/data/external/hira/` 에 배치 (별도 데이터 동기화 필요).

## 구조
```
src/kt_epimodel_hira/
├── model/         # SEIRV compartment + FOI + dynamics  (kt_epimodel과 동일)
├── simulation/    # ODE solver + runner                 (kt_epimodel과 동일)
├── calibration/   # HIRA calibration  ← ILI 대비 4개 모듈 새로 작성
├── viz/           # 결과 시각화
└── scenarios/     # 정책 시나리오
tests/
notebooks/         # calibration_04_* 시리즈 (HIRA 버전)
docs/
outputs/
```

## kt_epimodel 과의 차이
| 항목 | kt_epimodel (ILI) | kt_epimodel_hira (HIRA) |
|---|---|---|
| Target | 외래환자 1,000명당 분율 | 진료에피소드 카운트 |
| 연령 그룹 | 7 그룹 | 6 그룹 |
| 단위 | per 1,000 outpatients | absolute count |
| `gamma_report` 의미 | 외래 분모 흡수 + scaling | reporting fraction (직접) |
| `gamma_report` bound | (0.01, 1.0) | **(0.05, 0.5)** 권장 (1차 fit 후 결정) |
| `gamma_report_assumed` default | 2.0 (노트북 200.0) | **0.2** (도메인 산수 기반) |
| 출력 파일 suffix | `_LBFGS.json` | `_LBFGS_HIRA.json` |

상세는 [`docs/HIRA_VS_ILI_DIFF.md`](docs/HIRA_VS_ILI_DIFF.md) 참조.

## 사용 예시
```python
from kt_data import (
    load_hira_episodes, aggregate_hira_weekly, extract_hira_season,
    HIRA_AGE_GROUPS, SUDOGWON_SIDO_CODES,
)
from kt_epimodel_hira.calibration import (
    load_hira_target_by_age,
    estimate_initial_infected_from_hira,
    make_loss_function_by_age,
    optimize_calibration_by_age,
)

# 수도권 2019-2020 시즌 6-그룹 target
target = load_hira_target_by_age(
    season_start_year=2019,
    sido_codes=SUDOGWON_SIDO_CODES,
    setting="outpatient_inpatient",
    first_peak_only=True, first_peak_end_week=26,
)

# Fit
result = optimize_calibration_by_age(
    season="2019-2020",
    sido_codes=SUDOGWON_SIDO_CODES,
    gamma_report_assumed=0.2,   # 1차 추정값
    method="L-BFGS-B",
)
```

## 진행 상태
- [x] **Phase 1-7**: 스켈레톤 셋업 (이 commit)
- [ ] **첫 fit**: gamma_report bound / min_rate / 인구 표준화 결정 (6.4-B/D/E)
- [ ] **04_2 LBFGS 노트북 실행**: corner solution 진단 및 확인
- [ ] **다른 시즌 holdout validation**
- [ ] **ILI vs HIRA 비교 분석** (별도 노트북, ad-hoc)
