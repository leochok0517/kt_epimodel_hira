# kt_epimodel_hira — Claude 작업 가이드

## 프로젝트 위치 / 목적

- **경로**: `~/Documents/python/NIMS/kt_epimodel_hira/`
- `kt_epimodel` (ILI 전용) 의 **자매 프로젝트**. 동일한 모델 코어를 공유하되
  calibration target 만 ILI → HIRA (인플루엔자 진료에피소드 카운트) 로 교체.

## kt_epimodel 과의 관계

| 영역 | 공유 여부 | 비고 |
|---|---|---|
| `model/` (SEIRV, FOI, dynamics, mobility, parameters) | **그대로 복제** (코드 동일) | ODE 자체는 fit target 무관 |
| `simulation/` (solver, runner) | **그대로 복제** | |
| `calibration/param_vector.py` (23-D 벡터 layout) | **그대로 복제** | 벡터 layout 동일, bounds 만 조정 |
| `calibration/` 의 나머지 4개 (ili_target/loss/optimizer/simple_model) | **새로 작성** (HIRA 어댑터) | ILI → HIRA 변환만 달라짐 |
| 노트북 | **본체 복제 + 함수 호출 교체** | 시각화 단위 변경 |

**설계 근거**: [`docs/SKELETON_ANALYSIS.md`](docs/SKELETON_ANALYSIS.md) 섹션 6.
이 문서는 `kt_epimodel/docs/SKELETON_ANALYSIS.md` 의 동기화 사본 (출처 명시).

## 의존성

- **kt_data** (editable, `../kt_data`): 정제된 데이터 + 표준 로더
  - `from kt_data.data.load_population import load_population_15groups`
  - `from kt_data.data.load_mobility import load_mobility`
  - `from kt_data.data.load_contact import load_contact_matrices`
  - `from kt_data.data.load_calendar import classify_date`
  - **`from kt_data.data.load_hira import load_hira_episodes,
    aggregate_hira_weekly, extract_hira_season,
    HIRA_AGE_GROUPS, SUDOGWON_SIDO_CODES`**
  - ILI 로더 (`load_ili_seasons`) 는 본 프로젝트에서 사용 안 함

## 모델 명세

`kt_epimodel/CLAUDE.md` 의 모델 명세와 **완전 동일** (SVEIR 5-compartment,
NIMS 15군, 4채널 FOI, Gaussian seasonality, 23-D fit vector, …).
중복 작성하지 않음 — `../kt_epimodel/CLAUDE.md` 참조.

## ILI 대비 차이점 (요약)

상세는 [`docs/HIRA_VS_ILI_DIFF.md`](docs/HIRA_VS_ILI_DIFF.md). 핵심만:

1. **단위**: ILI 는 외래환자 1,000명당 분율, HIRA 는 절대 카운트 (인구 분모 없음).
2. **연령 그룹**: ILI 7군 vs HIRA 6군 (`HIRA_AGE_GROUPS`).
3. **6→NIMS 15 매핑**: `kt_epimodel_hira.calibration.hira_target.HIRA_GROUP_TO_NIMS_WEIGHTED`
   상수. 각 NIMS idx 가중치 합 = 1.0 (단위 테스트 검증).
4. **`gamma_report` 의미**: ILI 에선 외래 분모 흡수 + scaling 항이라 (0.01, 1.0)
   bound 의미가 모호. HIRA 에선 **"reporting fraction"** (감염자 중 진료 청구된
   비율) 으로 직접 해석 가능 → bound 권장 `(0.05, 0.5)`.
5. **`gamma_report_assumed` default**: ILI 노트북 200.0 → HIRA **0.2**
   (도메인 산수: 수도권 2019-2020 시즌 ~82만 / 실제 감염 추정 260-520만
   ≈ 0.16-0.32 → 중간값 0.2).
6. **출력 파일 suffix**: `*_LBFGS.json` → `*_LBFGS_HIRA.json`.

## 1차 fit 후 결정 사항 (TBD)

`SKELETON_ANALYSIS.md` 6.4 의 B/C/D/E. 노트북 `calibration_04_2_lbfgsb.ipynb`
실행 후 결정:

- **B**: `ParameterBounds.gamma_report` 좁힌 `(0.05, 0.5)` 적합 여부
- **C**: `gamma_report_assumed` default 0.2 → fit 결과 보고 조정
- **D**: `poisson_log_likelihood(min_rate=?)` 카운트 스케일 재튜닝
- **E**: 인구 표준화 옵션 필요성 (시도 비교 단계)

각 항목은 코드에 `# TODO HIRA-B/C/D/E:` 코멘트로 위치 표시.

## 코드 스타일

- **Polars** 우선 (Pandas 회피) — `kt_data` 컨벤션과 일관
- **NumPy** 행렬 연산
- Type hint 사용
- 한글 라벨 회피 (그래프, 변수명 모두 영어)
- 테스트: pytest

## 절대 하지 말 것

- `model/`, `simulation/`, `calibration/param_vector.py` 본체 수정 — bug fix 가
  필요하면 **`kt_epimodel` 쪽에 먼저 적용 후 본 프로젝트에 동기화**
- ILI 매핑 / 함수 / 로더 import — 본 프로젝트는 HIRA 전용
- 14군 컨벤션 (kt_epimodel 과 동일하게 15군 확정)

## 자주 헷갈리는 점

- HIRA 연령 그룹은 **6** (ILI 는 7). `HIRA_AGE_GROUPS`.
- 시도 코드: 서울=11, 인천=28, 경기=41. catalog 18개지만 실 데이터 17개.
- 단위는 **카운트** — per-1000 곱셈/나눗셈 코드 절대 금물
- `gamma_report_assumed` 단위: ILI 와 자릿수 다름 (0.2 vs 200)
