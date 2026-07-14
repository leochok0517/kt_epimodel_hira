# 프로젝트 브리프 — kt_epimodel_hira

> Claude 채팅에 전달할 현황 보고서. **2026-06-19 기준.**
> 목적: 새 채팅 세션이 프로젝트 구조·진행 상황·할 일을 빠르게 파악하도록.

---

## 1. 프로젝트 개요

### 1.1 정체

- **이름**: `kt_epimodel_hira`
- **위치**: `~/Documents/python/NIMS/kt_epimodel_hira`
- **자매 프로젝트**: `kt_epimodel` (ILI 전용) — 모델 코어는 공유, calibration target 만 HIRA 로 교체
- **데이터 의존성**: editable install `../kt_data`

### 1.2 연구 목적

수도권 (서울 + 인천 + 경기) 인플루엔자에 대해

1. **Calibration**: HIRA 진료 청구 카운트를 SVEIR + 4 채널 FOI 모델에 fit
2. **식별성 진단**: β · φ · γ 곱셈 ridge 분해 가능성
3. **정책 시뮬레이션**: 감염 학생 결석 / 병가 정책의 averted 감염 수 추정
4. **공간 이질성**: KT mobility 기반 metapop, 거주지 기준 정책 효과
5. **(향후) ICER**: PSA + 외부 단가

### 1.3 데이터

- **HIRA J09-J11 진료에피소드** (2018-2024), 6 연령군 (0-5 / 6-11 / 12-17 / 18-44 / 45-64 / 65+)
- **시도**: 서울 (11) · 인천 (28) · 경기 (41), 4 정상 시즌 (2017-18, 2018-19, 2019-20, 2022-23)
- **KT mobility**: 1154 행정동, 시간대별 노동자 위치 (work 9-17 시, other 17-22 시)
- **NIMS 15군 인구 + 접촉행렬** (5세 단위, 4 채널)

---

## 2. 모델 명세

### 2.1 구조

- **컴파트먼트**: SVEIR (S → V → E → I → R, 백신 V 별도)
- **연령**: NIMS 15군 (0-4, 5-9, …, 70+)
- **FOI**: 4 채널 (가정 / 직장 / 학교 / 기타) 가산
- **계절성**: Gaussian seasonality (amplitude `amp`, peak day)
- **겨울방학**: 학기 1.0 → 방학 0.3, smooth ramp, 감소분 → home spillover (κ 재사용)

### 2.2 파라미터 (23-D fit vector)

- **log R0**: 시즌별 4
- **채널 비율 π**: 시즌별 4 (4 채널 simplex, centered logit)
- **φ_age**: 연령별 감수성 (현재 1.0 고정 — TODO-3 항 참조)
- **γ_report**: 탐지율 (level 0.6 + CDC 연령 상대비, 고정)

### 2.3 핵심 reparam (Reparam A)

```
log R0 (4) + 채널 simplex (4×4) + φ (14)
→ β_h, β_w, β_s, β_o 는 NGM 역산으로 derive (R0, π, φ 로부터)
```

곱셈 ridge 의 primary axis 를 **log R0** 로 회전 — chain swap 방지.

### 2.4 관측 모델

**Negative Binomial-2** (TODO-3 에서 Poisson → NB 교체):
```
obs ~ NegBin(μ = pred, Var = μ + μ²/k)
phi_nb ≈ 1.3 (overdispersion)
```

→ coverage 95.2%, r_hat 1.02, wall 41분 (이전 Poisson 6h25m)

---

## 3. 프로젝트 구조

```
kt_epimodel_hira/
├── CLAUDE.md                          # 작업 가이드 (Claude 용)
├── NEXT_SESSION.md                    # ★ 오래됨 (2026-06-05), 이 PROJECT_BRIEF 가 대체
├── README.md
├── pyproject.toml / uv.lock
│
├── src/kt_epimodel_hira/
│   ├── model/                         # SVEIR ODE, FOI, mobility, parameters (kt_epimodel 동기화)
│   ├── simulation/                    # solver, runner (동기화)
│   ├── calibration/
│   │   ├── param_vector.py            # 23-D 벡터 layout (동기화)
│   │   ├── hira_target.py             # HIRA 6→NIMS 15 매핑, 가중치
│   │   ├── simple_model.py            # build_aggregated_inputs, initial state
│   │   └── loss/optimizer/ili_target  # HIRA 어댑터
│   ├── jax_model/                     # JAX/numpyro/diffrax 본추정 스택
│   │   ├── numpyro_model.py           # hira_model_nb, hira_model_nb_chprior
│   │   ├── solver_jax.py              # simulate_jax, daily_new_infection_by_age_jax
│   │   └── loss_jax.py
│   ├── scenarios/                     # 정책 시뮬, metapop
│   └── viz/
│
├── scripts/                           # 분석·진단 스크립트 (다수)
│   ├── m2_production*.py              # production NUTS (A, B, chprior)
│   ├── m2_fit_vs_data_AB.py           # 슬라이드 19 fit 그림
│   ├── m2_spatial_heterogeneity_AB*.py # 공간 분석
│   ├── diagnose_*.py                  # 비식별·misspec 진단 (channel_prior, winter_break, work_channel ...)
│   └── metapop_step_*.py              # metapop 단계별 검증
│
├── outputs/
│   ├── calibration/
│   │   ├── m2_prod_A_samples.npz      # production NUTS posterior A (NIMS, 9h)
│   │   └── m2_prod_B_samples.npz      # production NUTS posterior B (literature, 9h)
│   └── metapop/
│       ├── channel_prior_4combos.json # 4 조합 점추정 (탐색)
│       └── spillover_sweep_AB.json    # κ sweep × A·B posterior
│
├── presentations/
│   ├── calibration_identifiability_v3.md/pdf   # v3 (이전)
│   └── calibration_identifiability_v4.md/pdf   # ★ v4 (현재, 28 pages)
│
├── docs/
│   ├── SKELETON_ANALYSIS.md           # kt_epimodel vs kt_epimodel_hira 설계 분석
│   ├── HIRA_VS_ILI_DIFF.md            # ILI 대비 차이점 (단위·연령·γ 의미)
│   ├── GAMMA_STRATEGY.md
│   ├── PRIOR_SPECIFICATION.md
│   ├── identifiability_strategy.md
│   └── TODO_model_improvements.md     # ★ TODO-1~6 + SEM-1~5
│
├── notebooks/                         # EDA, calibration steps
└── tests/                             # pytest
```

---

## 4. 진행 상황 — 현재까지 완료

### 4.1 Calibration 마일스톤

| 단계 | 상태 | 핵심 발견 |
|---|---|---|
| Multi-start 8 시작점 진단 | ✅ | NLL spread 0.57%, φ CV 53%, γ_elder CV 63% → 비식별 확정 |
| R(0) 연령 계단 (0-19: 0.10 / 20-49: 0.30 / 50-64: 0.45 / 65+: 0.65) | ✅ | 65+ peak ratio 1.43 → 0.99 |
| β·φ·γ 분리 (Reparam A: log R0 + simplex) | ✅ | chain swap 해소 |
| Poisson → NB 관측모델 (TODO-3) | ✅ | coverage 12% → 95%, r_hat 2.38 → 1.02, wall 6h25m → 41분 |
| φ 비식별 실증 (r_hat 4.46, ess 5) | ✅ | φ=1.0 고정 정당화 |
| **겨울방학 발견 (NLL +400K)** | ✅ | misspec 가 가짜 비식별 만듦 — school π 0.61 → 0.24, amp 0.3 → 0.9 |
| 채널 식별성 한계 (5 시도) | ✅ | home/work ≤ 0.011 — HIRA 18-44 (27 년 폭) bin 이 home/work peak 흡수 |
| 채널 prior A (NIMS contact) / B (Italy R0) 4 조합 점추정 | ✅ | 정책 효과 robust 판정 (4 조합 averted span) |
| Production NUTS A · B 각 1 (9h 동시) | ✅ | r_hat 1.10, ess_min 31-42, divergence 0 |
| Spillover κ sweep (가구격리 0 ~ 1.0) | ✅ | 병가 부호 전환점 κ ≈ 0.49 |
| 공간 이질성 (주간/야간 ratio) | ✅ | **null** (상업-주거 averted ≤ 0.5 pp), 균일 정책 정당 |

### 4.2 정책 결론 (현재)

| 정책 | 결과 | Robust성 |
|---|---|---|
| **감염 학생 결석** (p_school=0.5) | A +39.7% [11.7, 74.8] / B +55.2% [28.5, 83.6] | ✅ CI 78% overlap, 전 κ 양수 |
| **병가** (p_work=0.4) | A +9.6% / B −10.9% | ⚠️ 채널 prior 가정에 부호 의존 |
| 공간 타겟팅 | 상업/주거 효과 무차이 | ❌ 정책적 의미 없음 (KT mobility backpropagation) |

### 4.3 알려진 한계

- **Peak over-prediction**: 45-64 2.8×, 12-17 2.4×, 18-44 2.2×, 65+ 1.8×, 0-5 0.6×
  - 원인: NB phi_nb=1.3 → 분산 ≫ 평균 → peak 잔차 페널티 약함
  - 영향: averted **%** 비율 상쇄 (영향 작음) / 절대 **카운트** 1.5-3× 과대 → ICER 주의
- **점추정 98% 절벽 artifact**: 4 조합 점추정 (슬라이드 17) 의 학생 결석 98% 는 R_eff 임계 (1.0) 직하 안착 산물. 실제 효과는 production posterior 38-55% (슬라이드 20).

---

## 5. 발표 자료 — v4 (현재)

### 5.1 [presentations/calibration_identifiability_v4.md](presentations/calibration_identifiability_v4.md) (28 pages)

| 섹션 | 슬라이드 | 내용 |
|---|---|---|
| 표지 | 1 | |
| 배경 | 2-3 | 연구 맥락, γ_report 탐지율 |
| 식별성 진단 | 4-10 | 곱셈 함정, multi-start, R(0) 계단, reparam, NB, φ 비식별, Jung 2025 비교 |
| 채널 비식별 + 방학 | 11-14 | 채널 한계 5 시도, ★ 겨울방학 발견, 진단 정직, age-resolution |
| Field knowledge **탐색** | 15-17 | A/B 입력 대칭 표, 4 조합 점추정 채널 mix · 정책 robust |
| **본 추정** Production | 18-20 | 정책 메커니즘, A·B fit 그림 (비식별 시각 callback), 정책 posterior CI |
| 한계 | 21 | ★ Peak over-prediction |
| 심화 | 22-25 | spillover κ, 공간 분류 / null / 결론 |
| 종합 | 26-27 | 1 차 종합, 요약 |

### 5.2 발표 흐름 핵심

**탐색(점추정 4 조합) → 본추정(production posterior)** — 점추정 절벽의 한계를 production 이 희석 검증.

---

## 6. 할 일 (TODO) — 우선순위 순

상세는 [docs/TODO_model_improvements.md](docs/TODO_model_improvements.md).

### 6.1 ★ 세미나 피드백 (2026-06-18, **최우선**)

좌중 요지: home/work=0 결론 수용. **단 대안 원인 배제 검증 요구.**

| ID | 항목 | 우선순위 | 영향 |
|---|---|---|---|
| **SEM-1** | NIMS contact matrix 신뢰도 검증 (vs COMIX/POLYMOD) | 🔴 최우선 | 결론 바꿀 수 있음 |
| **SEM-2** | 4 채널 합산 단일 matrix 피팅 sanity | 🔴 높음 | 결론 강화 (빠름) |
| SEM-3 | 초기조건 — R(0)≈0 (post-pandemic), 초기 감염자 영향 | 🟡 중간 | TODO-1/2 와 통합 |
| SEM-4 | 발표 자료 "98% 절벽" → "posterior 38-87%" 교정 | 🟢 진행중 | v4 에 반영 |
| SEM-5 | 연령별 susceptibility prior 좁게 (φ=1.0 고정) | ✅ 완료 | 기록만 |

**SEM-1 상세**:
- NIMS 전반 접촉 수 낮음 + 20대 work 최저 → 패널 설계 편향 의심
- 손 박사 공유: COMIX 가 POLYMOD (우편 회고) 대비 부각
- 우리 결론에 영향: NIMS 는 (a) unit_R0 계산 (b) 채널 prior A 에 쓰임 → 편향이면 둘 다 흔들림
- work=0 이 "데이터 해상도" 가 아니라 "NIMS 20대 work 과소측정" 일 수도

**SEM-2 상세**:
- 4 채널 합산 → 단일 contact matrix → R0 + 계절성 + 방학만 추정
- 수렴 잘 됨 → "채널 분해가 과파라미터화" 확인 → 결론 강화
- 수렴 안 됨 → 초기조건/데이터 문제

### 6.2 기존 모델 개선

| ID | 항목 | 우선순위 |
|---|---|---|
| **TODO-1** | 백신을 S 뿐 아니라 R(0) 에도 적용 (노인 R(0) 0.65 + 접종률 82%) | 🔴 높음 |
| TODO-2 | post-pandemic 낮은 R(0) 시즌 검증 (2022-23) | 🟡 중간 |
| TODO-3 | NB 관측모델 도입 | ✅ 완료 |
| TODO-4 | (생략) | |
| TODO-5 | spillover κ 외부 추정 (가정격리 시 실제 가구 전파) | 🟡 중간 |
| **TODO-6** | work:other 비율 외부 추정 (KOSIS 직장 패턴 / 한국 surveillance) | 🔴 높음 — 병가 정책 부호 좌우 |

### 6.3 향후 연구 (2 차)

- **ICER + PSA**: 외부 단가 (백신, 입원, 결근, 결석) 확보 후
- HIRA 외래/입원 split
- KDCA serosurvey 연계 (γ level 외부 검증)

---

## 7. 진행 시 주의사항

### 7.1 코드 스타일

- **Polars** 우선 (Pandas 회피) — kt_data 컨벤션과 일관
- **NumPy** 행렬 연산
- Type hint 사용
- 한글 라벨 회피 (그래프·변수명 모두 영어)
- 테스트: pytest

### 7.2 절대 금지

- `model/`, `simulation/`, `calibration/param_vector.py` 본체 수정 — bug fix 시 **kt_epimodel 쪽에 먼저 적용 후 본 프로젝트 동기화**
- ILI 매핑/함수 import — 본 프로젝트는 HIRA 전용
- 14군 컨벤션 (15군 확정)
- per-1000 곱셈/나눗셈 — HIRA 는 절대 카운트 단위

### 7.3 JAX/numpyro 수치 안정성

(메모리 [Numerical safety: prior over NaN](file:///Users/hwcho/.claude/projects/-Users-hwcho-Documents-python-NIMS-kt-epimodel-hira/memory/feedback_numerical_safety.md))
- 극단 영역 진입 방지: prior 로 제어 (NaN 발생 후 보정 X)
- 유한 페널티 사용 (NaN X)
- ODE max_steps tight 유지

### 7.4 자주 헷갈리는 점

- HIRA 6 연령군 (ILI 7 군 아님)
- 시도 18 개 catalog 지만 실 데이터 17 개
- `gamma_report_assumed` 기본값 0.2 (ILI 200.0 과 자릿수 다름)

---

## 8. 즉시 착수 가능한 다음 작업 (추천)

다음 셋 중 하나 선택:

### A. **SEM-2 단일 matrix sanity** (빠름, 결론 강화)
```
scripts/sem2_single_matrix_sanity.py 신규
4 채널 합산 → 단일 contact matrix → R0 + 계절성 + 방학만 추정
수렴 (r_hat, NLL) + fit 품질 비교
→ 4 채널 합산도 잘 수렴하면 "channel 분해 과파라미터" 확인
```

### B. **SEM-1 NIMS vs COMIX/POLYMOD 비교** (결론 좌우)
```
docs/SEM1_NIMS_VALIDATION.md 신규
NIMS 20대 work contact vs COMIX/POLYMOD 패턴
대안 contact 으로 unit_R0 · 채널 prior 재계산
→ work π 살아나면 결론 수정 필요
```

### C. **TODO-1 백신 R(0) 적용** (모델 정직성)
```
src/kt_epimodel_hira/model/parameters.py · simulation 수정
S+R 양쪽에 인구 비례 접종, R 접종은 효과 0
노인 백신 효과 과대평가 교정
```

---

## 9. 최근 commits (참고)

```
4eeaf62  Terminology: 'school closure' -> 'sick student absence'
9017a74  Channel ratio field-knowledge prior: 4-combo robustness analysis
20ae09e  Channel sensitivity analysis: policy robustness differs
4608ab5  Channel normalization: resolve 3 non-identifiabilities
177312b  TODO-3: Negative Binomial obs model — coverage 12%→95%, convergence fix
```

---

## 10. 빠른 참조 — 핵심 숫자

| 지표 | 값 |
|---|---|
| 4 시즌 R0 mean | ~1.90-1.99 (시즌별 [1.74, 1.98]) |
| 채널 π (target_A NIMS) | h 0.29, w 0.29, s 0.06, o 0.36 |
| 채널 π (target_B literature) | h 0.47, w 0.17, s 0.11, o 0.25 |
| unit_R0 (h,w,s,o) | 8.70, 6.21, 25.40, 9.33 |
| γ_report level | 0.6 (CDC 연령비 × 어린이 1.45 / 성인 0.65 / 노인 0.90) |
| φ | 1.0 고정 (비식별 r_hat 4.46) |
| amp 선호 | 0.9 (방학 도입 후) |
| 겨울방학 NLL gain | +400,000 (잡음 80K 의 5 배) |
| NB phi_nb | 1.30-1.59 (식별) |
| 학생 결석 averted (production) | A +39.7% / B +55.2% |
| 병가 averted (production) | A +9.6% / B −10.9% |
| 공간 effect (상업-주거) | < 0.5 pp |
