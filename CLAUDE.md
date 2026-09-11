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

## Influenza A 전용 재피팅 (2026-08-28 ~, 진행 중)

논문 `~/Documents/cowork/sickleave_InfluenzaA/manuscript/main_v5.tex` 의 후속.
**fit target 을 인플루엔자 A 전용으로 바꾸고 시즌을 확장**하는 작업. v5 재현성은
그대로 보존 — 기존 파일은 한 줄도 고치지 않고 전부 `_fluA` 접미사 신규 파일.

### 왜

J09-J11 청구건은 A + B 혼합이다. 모델은 A 기반 presymptomatic weight 를 쓰는
단일 병원체 SVEIR 이라, A+B 에 맞추면 타이밍이 다르고 presymptomatic 전파
근거도 다른 B 유행을 같이 흡수한다. A 점유율은 시즌마다 크게 달라
(2017-18 약 0.50 vs 2016-17 약 1.00) `gamma_report` 가 흡수할 수 있는 공통
스케일이 아니라 **곡선 모양 자체를 바꾼다**.

부수 효과: main_v5 Limitations 의 "presymptomatic weight 가 A 기반이라 B 에
보수적" 항목이 사라진다.

### 신규 파일

| 파일 | 역할 |
|---|---|
| `kt_data/data/load_flu_positivity.py` | WHO FluNet + KDCA 병합 양성률. `flu_a_share_weekly()` 가 **total 가중** 3주 중심이동평균으로 A/total 평활 |
| `calibration/fluA_target.py` | `load_fluA_target_by_age`, `estimate_initial_infected_from_fluA` |
| `scripts/build_pop_seasons_fluA.py` | 시즌별 인구 확장 → `data/population/pop_all_seasons_fluA.csv` |
| `scripts/fluA_setup.py` | 시즌 리스트를 **인자로** 받는 setup (v5 는 모듈 상수) |
| `scripts/nuts_fluA.py` | 결합 손실 + v7 좌표계 numpyro 모델 |
| `scripts/nuts_fluA_run.py` | 러너. `--smoke` 는 점추정 + 짧은 NUTS |
| `outputs/fluA/` | 진단 그림 · 실행 JSON · README |

데이터: `KT_data/data/external/flu_positivity/flu_positivity_2009w22_2026w25.csv`
(2009 w22 ~ 2026 w25, `total`/`A`/`B` 양성률 %, WHO 우선 병합).

### 고정값은 import, 재작성 금지

`fluA_setup` · `nuts_fluA` 는 `PHI` `GAMMA` `KAP` `BASE` `W` `NGM_F`
`LOGIT_REF` `SIGMA_PIN` 을 전부 `nuts_seasonpop` 에서 import 한다. 다시 적으면
v5 와 조용히 갈라질 수 있다. 접촉행렬 · 이동 · 고용 · 백신 · 계절성도
`sens_common.build_setup` 결과를 그대로 쓴다 → **이 피팅과 v5 의 차이는
target · 시즌 집합 · 인구, 셋뿐이어야 한다.**

### 시즌 집합

A 보정 후 단봉 판정 (`outputs/fluA/fluA_season_diagnostic.png`):

| 시즌 | 보정 후 단봉 | 인구 | 비고 |
|---|---|---|---|
| 2013-14 | O | **없음** | KOSIS 2013.06 열 필요 |
| 2014-15 | O | **없음** | 보정 전 쌍봉, 2차 봉우리가 전부 B |
| 2015-16 | O | O | v5 는 "느린 onset" 으로 제외했던 시즌 |
| 2016-17 | O | O | A 점유 ~1.0, target 거의 불변 |
| 2017-18 | O | O | A 점유 ~0.50, 피크 절반으로 |
| 2018-19 | O | O | 보정 전 쌍봉, 2차 봉우리가 전부 B |
| 2019-20 | O | O | A 점유 ~0.97 |
| 2022-23 | X | O | 봄 재유행이 A 자체 — 계속 제외 |
| 2023-24 | X | O | fit 창 안에 A 봉우리가 여럿 |

`SEASONS_DEFAULT` = 인구가 확보된 5 시즌 (2015-16 ~ 2019-20).
`SEASONS_PENDING_POP` = 2013-14, 2014-15.

### A 보정이 들여오는 가정 (논문에 명시할 것)

1. 양성률은 **전국** 표본감시, 청구건은 수도권 — A:B 구성이 같다고 가정.
2. 양성률은 **연령 미구분** — 같은 주별 배수를 6 연령군에 공통 적용.
3. 검체 주 vs 증상발현 주 — 일요일→월요일 이동 후 7일 중 6일 겹침으로 동일 주 취급.
4. **형별 보고율 동일** — A 와 B 가 같은 비율로 청구된다고 가정해야
   `gamma_report` 의미가 유지된다.

### 커버리지 판정은 유행 규모 가중

`MIN_OBSERVED_COVERAGE` 는 주 개수가 아니라 **가중 유행량** 기준이다. 양성률이
비어 있는 주는 대부분 초가을 0 근처 shoulder 라, 주 개수로 재면 유행 구간이
완전히 관측된 시즌 (2013-15) 이 46-69% 로 탈락한다. 규모 가중으로는 전부 97%+.

### 관측모델 주의

A 보정된 obs 는 **정수가 아니다**. `nb_nll_jax` 는 `lgamma` 로 연속 확장되므로
그대로 돌아가지만, 엄밀히는 count NB 가 아니라 연속 NB 다. 반올림하면 작은
연령×주 셀에서 보정이 통째로 날아가므로 의도적으로 반올림하지 않는다.

### 작업 프롬프트

단계별 Claude Code 프롬프트는 `prompts/` 에 있다 (`prompts/README.md` 가 순서).
공통 주의사항은 `prompts/99_conventions.md`.

### 남은 블로커

1. **2013-14 · 2014-15 는 인구가 없어 제외** (2026-08-28 확인). KOSIS
   `행정구역(읍면동)별 5세별 주민등록인구` 표 **자체가 2015 년부터** 시작한다 —
   그 시점에 집계 방식이 바뀐 것으로 보이며 재-export 로는 더 못 내려간다.
   두 시즌을 살리려면 **다른 표** (시도/시군구 단위 연령별 주민등록인구) 에서
   가져와야 한다. 어차피 서울·인천·경기 3 개만 합산하므로 시도 단위로 충분하고,
   `build_pop_seasons_fluA.py` 의 `read_kosis()` 만 확장하면 된다.
   → **확정 시즌 집합은 5 개: 2015-16 ~ 2019-20.**
2. 본 실행 후 하위 산출물 (정책 시뮬 · 그림 · 원고 수치) 전면 갱신 필요.

### JAX 함정 — loss 를 jit 밖에서 반복 호출하지 말 것

2026-08-28 에 `nuts_fluA_run.py --smoke` 가 182 분 동안 멈추고 메모리가 54GB
까지 늘어났다. 원인은 스모크의 점추정이 `make_joint_loss_fluA` 결과를
**jit 없이** Nelder-Mead 로 수백 번 호출한 것. 매 호출마다 5 시즌 diffrax ODE
가 eager 로 다시 돌면서 메모리가 누적됐다.

- loss 를 여러 번 평가해야 하면 **반드시 `jax.jit`** (또는
  `jax.value_and_grad` + jit). 기존 패턴은
  `final_pipeline_confirmed.fit_season_pi` 참조 — jit + grad + L-BFGS.
- NUTS 경로는 numpyro 가 모델을 한 번 트레이스해 jit 하므로 안전하다.
  이 함정은 **본 피팅이 아니라 보조 스크립트에서** 생긴다.
- 현재 `--smoke` 는 최적화를 하지 않고 **jit 된 NLL 을 R0 세 점에서 평가**만
  한다 (유한성 · 응답성 · 컴파일 확인). 몇 분이면 끝난다.

### 접촉행렬 상호성 보정 — 플래그가 아니라 데이터에 있다

`deploy_reciprocity_matrices.py` (2026-08-24) 가 방식 A 보정본을 kt_data 표준
경로에 **제자리 교체**했다 (`empirical_matrices_15.npz`, `..._vacation.npz`;
원본은 `..._raw_asymmetric.npz` 로 백업). 따라서 `load_contact_matrices()` 를
거치는 모든 경로가 자동으로 보정본을 받는다 — `fluA_setup` 포함, 겨울방학
행렬 (`C_*_vac`) 포함.

- 검증: `python scripts/check_reciprocity_fluA.py` (주말 체인의 첫 게이트).
  상호성 `G_i C_ij = G_j C_ji` 가 ~1e-10 (T 스케일 1e7) 수준으로 성립.
- method A 는 **멱등**이라 `nuts_seasonpop_v7_run.py --corrected 1` 이 배포본
  위에 다시 적용한 것도 no-op 이었다 (이중 보정 아님, ~1e-17).
- 백업본을 되돌리거나 행렬을 재-export 하면 **조용히 미보정 상태가 된다.**
  그래서 긴 피팅 전에 매번 확인한다.

## 절대 하지 말 것

- `model/`, `simulation/`, `calibration/param_vector.py` 본체 수정 — bug fix 가
  필요하면 **`kt_epimodel` 쪽에 먼저 적용 후 본 프로젝트에 동기화**
- ILI 매핑 / 함수 / 로더 import — 본 프로젝트는 HIRA 전용
- 14군 컨벤션 (kt_epimodel 과 동일하게 15군 확정)
- `_fluA` 파일에서 고정 파라미터 (PHI/GAMMA/KAP/BASE/W/LOGIT_REF/SIGMA_PIN) 재정의
  — 반드시 `nuts_seasonpop` 에서 import
- `nuts_seasonpop*.py` · `season_pop_setup.py` · `sens_common.py` 등 v5 경로 수정
  — A 재피팅은 전부 `_fluA` 신규 파일로
- A 보정 obs 를 정수로 반올림

## 자주 헷갈리는 점

- HIRA 연령 그룹은 **6** (ILI 는 7). `HIRA_AGE_GROUPS`.
- 시도 코드: 서울=11, 인천=28, 경기=41. catalog 18개지만 실 데이터 17개.
- 단위는 **카운트** — per-1000 곱셈/나눗셈 코드 절대 금물
- `gamma_report_assumed` 단위: ILI 와 자릿수 다름 (0.2 vs 200)
