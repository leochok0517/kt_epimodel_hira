# kt_epimodel_hira — 종합 참조 문서

> 본 프로젝트의 목적·방법론·코드 구성을 한 문서로 정리. **2026-06-19 기준.**
> 새 Claude 세션이 이 파일 1 개로 프로젝트 전체 맥락을 파악하고 코드 검증·판단에
> 활용할 수 있도록 작성. 상태가 빠르게 바뀌는 항목 (실험 결과·TODO 진행 등) 은
> [PROJECT_BRIEF.md](../PROJECT_BRIEF.md), [TODO_model_improvements.md](TODO_model_improvements.md) 에서 별도 추적.

---

## 목차

1. [프로젝트 정체 / 목적](#1-프로젝트-정체--목적)
2. [데이터](#2-데이터)
3. [수학적 모델](#3-수학적-모델)
4. [식별성 전략 — 본 연구의 방법론적 핵심](#4-식별성-전략--본-연구의-방법론적-핵심)
5. [Calibration 파이프라인](#5-calibration-파이프라인)
6. [정책 시뮬레이션](#6-정책-시뮬레이션)
7. [공간 분석 (metapop)](#7-공간-분석-metapop)
8. [코드 구성 (디렉토리·파일별)](#8-코드-구성)
9. [규약 / 컨벤션](#9-규약--컨벤션)
10. [수치 안정성](#10-수치-안정성)
11. [출력물 / 산출물 지도](#11-출력물--산출물-지도)
12. [방법론 교훈 (반복 진단의 자기수정)](#12-방법론-교훈)
13. [세미나 피드백 / 미해결 검증](#13-세미나-피드백--미해결-검증)
14. [용어집 / 핵심 숫자](#14-용어집--핵심-숫자)

---

## 1. 프로젝트 정체 / 목적

### 1.1 정체

- **이름**: `kt_epimodel_hira`
- **위치**: `~/Documents/python/NIMS/kt_epimodel_hira`
- **자매 프로젝트**: [`kt_epimodel`](../kt_epimodel) (ILI 전용) — 모델 코어 공유, calibration target 만 ILI → HIRA 교체
- **외부 의존성**: `kt_data` (editable, `../kt_data`) — 정제된 데이터 + 표준 로더

### 1.2 연구 목적 (4 단계)

1. **Calibration**: HIRA 진료 청구 카운트를 SVEIR + 4 채널 FOI 모델에 fit
2. **식별성 진단·해소**: β · φ · γ 곱셈 ridge 분해 가능성 정직 평가
3. **정책 시뮬레이션**: 감염 학생 결석 / 병가 정책의 averted 감염 수 추정 (PSA)
4. **공간 이질성 (metapop)**: KT mobility 기반 행정동 시뮬, 거주지 기준 정책 효과

### 1.3 향후 (2 차)

- ICER + PSA (외부 단가 확보 후)
- HIRA 외래/입원 split
- KDCA serosurvey 와 γ_report level 외부 검증

### 1.4 kt_epimodel (ILI) 와의 관계

| 영역 | 공유 여부 | 비고 |
|---|---|---|
| `model/` (SEIRV, FOI, dynamics, mobility, parameters) | **그대로 복제** (코드 동일) | ODE 자체는 fit target 무관 |
| `simulation/` (solver, runner) | **그대로 복제** | |
| `calibration/param_vector.py` (33-D 벡터 layout) | **그대로 복제** | 벡터 layout 동일, bounds 만 조정 |
| `calibration/` 의 나머지 (hira_target/loss/optimizer/simple_model) | **새로 작성** (HIRA 어댑터) | ILI → HIRA 변환 |
| `jax_model/` (NUTS 본추정) | **HIRA 단독 개발** | ILI 쪽엔 없음 |
| 노트북 | **본체 복제 + 함수 호출 교체** | 시각화 단위 변경 |

설계 근거: [docs/SKELETON_ANALYSIS.md](SKELETON_ANALYSIS.md) (kt_epimodel 의 동기화 사본).

---

## 2. 데이터

### 2.1 HIRA 진료 청구 (calibration target)

| 항목 | 값 |
|---|---|
| 소스 | 건강보험심사평가원 J09–J11 (인플루엔자 진료에피소드) |
| 기간 | 2018-2024 (4 정상 시즌: 2017-18, 2018-19, 2019-20, 2022-23) |
| 단위 | **절대 카운트** (인구 분모 없음, ILI 의 per-1000 분율과 다름) |
| 연령군 | **6 군**: 0-5 / 6-11 / 12-17 / 18-44 / 45-64 / 65+ |
| 지역 | 수도권: 서울 (11) / 인천 (28) / 경기 (41) |
| 로더 | `kt_data.data.load_hira.load_hira_episodes`, `aggregate_hira_weekly`, `extract_hira_season` |

**ILI 와의 본질 차이**:
- ILI 단위: 외래환자 1,000 명당 분율 → `gamma_report` 가 분모 흡수 + scaling 항이라 해석 모호
- HIRA 단위: 절대 카운트 → `gamma_report` 가 "**감염자 중 진료 청구된 비율**" 로 **직접 해석 가능**
- 그래서 HIRA 에선 `gamma_report` bound `(0.05, 0.5)` 가 의미 있음

상세: [docs/HIRA_VS_ILI_DIFF.md](HIRA_VS_ILI_DIFF.md).

### 2.2 인구 / 접촉 / 모빌리티 (NIMS / KT)

| 데이터 | 출처 | 차원 | 비고 |
|---|---|---|---|
| 인구 | NIMS | 15 군 (0-4, 5-9, …, 70+) × 1154 행정동 | 5 세 단위 |
| 접촉 행렬 | NIMS | 4 채널 (home / work / school / other) × 15 × 15 | row-sum h=23.26, w=17.03, s=14.24, o=31.77 |
| 모빌리티 | KT | M_work, M_school, M_other (시간대별 위치 이동) | M_home = identity |
| ρ (노동자 비율) | 경험 추정 | (15,) | NGM 에서 work mask 가중 |

**시간대 정의** (KT):
- work: 9-17 시 노동자 위치
- other: 17-22 시 (work 외 시간)

### 2.3 캘린더

| 데이터 | 출처 | 비고 |
|---|---|---|
| 한국 공휴일·평일 | `kt_data.data.load_calendar.classify_date` | seasonality / spillover 계산 |
| **겨울방학** | 모델 상수 (start=113d, min_start=127d, min_end=162d, end=183d) | 12월말~2월, smooth ramp |

### 2.4 연령별 회복률 γ_15 (CDC Reed 2015)

γ_15 = **[0.40, 0.18, 0.25]** (child / adult / elder) — CDC Reed 2015 원값.

| 연령군 | NIMS idx | γ (1/일) | 감염기간 1/γ |
|---|---|---|---|
| 어린이 (0-19) | 0-3 | **0.40** | 2.5일 |
| 성인 (20-64) | 4-12 | **0.18** | 5.6일 |
| 노인 (65+) | 13-14 | **0.25** | 4.0일 |

출처: CDC Reed et al., PLOS One 2015 (symptomatic multiplier inversion).
레지스트리: [src/kt_epimodel_hira/calibration/gamma_registry.py](../src/kt_epimodel_hira/calibration/gamma_registry.py) — `ACTIVE_GAMMA = "cdc_reed2015"`.

---

## 3. 수학적 모델

### 3.1 컴파트먼트: SVEIR

$$
S \xrightarrow{v(t)} V \xrightarrow{(1-VE)\lambda} E \xrightarrow{\sigma} I \xrightarrow{\gamma} R
$$

$$
\frac{dS}{dt} = -\lambda S - v(t) S
$$
$$
\frac{dV}{dt} = v(t) S - (1-VE) \lambda V
$$
$$
\frac{dE}{dt} = \lambda S + (1-VE) \lambda V - \sigma E
$$
$$
\frac{dI}{dt} = \sigma E - \gamma I
$$
$$
\frac{dR}{dt} = \gamma I
$$

- 백신 V → S 만 받음 (R 은 안 받음 — **TODO-1 에서 수정 예정**)
- VE = 0.5 (vaccine efficacy)
- 백신 출시: peak ISO week 42, spread 4 주
- 초기조건: R(0) **연령 계단** — 0-19: 0.10 / 20-49: 0.30 / 50-64: 0.45 / 65+: 0.65

구현: [src/kt_epimodel_hira/jax_model/dynamics_jax.py](../src/kt_epimodel_hira/jax_model/dynamics_jax.py).

### 3.2 4 채널 FOI

$$
\lambda(t) = \lambda_h(t) + \lambda_w(t) + \lambda_s(t) + \lambda_o(t)
$$

각 채널은 **구조적으로 다른 mechanics** ([src/kt_epimodel_hira/jax_model/foi_jax.py](../src/kt_epimodel_hira/jax_model/foi_jax.py)):

| 채널 | 적용 연령 | mobility | 부가 메커니즘 | 정책 레버 |
|---|---|---|---|---|
| **home** | 전 연령 (0-70+) | 없음 | spillover `1 + κ·φ_spill(p_school, p_work, ρ)` | — |
| **school** | **STUDENT_SLICE (0-19, idx 0:4)** | 없음 | `I_eff_school = p_school × I_student` | `p_school` |
| **work** | **WORKER_SLICE (20-69, idx 4:14)** | `M_work` einsum 통근 | ρ 가중, `p_work × I_worker`; FOI 가 직장지 → 거주지로 backpropagate | `p_work` |
| **other** | 전 연령 | `M_other` einsum | — | — |

**핵심 주의**: 4 채널 = 4 행렬 + 4 가지 연령 마스크 + 2 가지 mobility 패턴 + 2 가지 정책 레버. **단순 행렬 합으로 1 채널 등가화 불가능** (SEM-2 단일 matrix 검증에서 옵션 B 로 2-channel collapse 채택한 이유).

#### 채널별 수식

**Home** (전 연령, spillover):
$$
\text{spill}_{ji} = 1 + \kappa_j \phi_{\text{spill}}(p_{sch}, p_{wrk}, \rho)
$$
$$
\lambda_h(a, j) = \beta_h \cdot s(t) \cdot \phi(a) \sum_{a'} C^h_{aa'} \frac{I(a',j) \cdot \text{spill}_{ji}}{N(a',j)}
$$

**School** (학생만, 정책 레버):
$$
I_{\text{eff}}^{sch}(a, j) = \mathbb{1}[a \in \text{STUDENT}] \cdot p_{sch} \cdot I(a, j)
$$
$$
\lambda_s(a, j) = \mathbb{1}[a \in \text{STUDENT}] \cdot \beta_s \cdot s(t) \cdot \phi(a) \sum_{a'} C^s_{aa'} \frac{I_{\text{eff}}^{sch}(a', j)}{N(a',j)}
$$

**Work** (근로자만, 통근 backpropagation):
$$
I_{\text{at j}}(a) = \sum_k M^w(a, k, j) \rho(a, k) I(a, k)
$$
$$
\lambda_w(a, i) = \mathbb{1}[a \in \text{WORKER}] \cdot \beta_w \cdot s(t) \cdot \phi(a) \cdot \rho(a, i) \sum_j M^w(a, i, j) \sum_{a'} C^w_{aa'} \frac{p_{wrk} I_{\text{at j}}(a')}{N_{\text{at j}}(a')}
$$

**Other** (전 연령, 통근):
$$
\lambda_o(a, i) = \beta_o \cdot s(t) \cdot \phi(a) \sum_j M^o(a, i, j) \sum_{a'} C^o_{aa'} \frac{I_{\text{at j}}(a')}{N_{\text{at j}}(a')}
$$

### 3.3 계절성

코사인 형태 ([dynamics_jax.py](../src/kt_epimodel_hira/jax_model/dynamics_jax.py) `seasonal_factor_cosine`):
$$
s(t) = 1 + \text{base} + \text{amp} \cdot \cos\!\left(\frac{2\pi (t - \text{peak\_day})}{\text{period}}\right)
$$
- `amp` = **0.9** (방학 도입 후 데이터 선호값. 이전 방학 누락 환경에서는 0.3 nominal)
- `peak_day` ≈ 105 (12월 중순 — 2019-12-16, ISO 36주 원점 +105일)
- `base` = 0

### 3.4 겨울방학 (winter break) ★

방학 누락 → NLL +400K misspec 발견 후 도입.

```python
school_mult(t) = smooth ramp from 1.0 (학기) → school_holiday_amp (방학)
```

기간 (default):
- start 113 일 → min_start 127 → min_end 162 → end 183 (smooth ramp)
- `school_holiday_amp = 0.7` (방학 중 70% 감소? 실은 `1 - 0.7 = 0.3` 수준으로 떨어짐)

방학 감소분을 **β_s 만 스케일** ([dynamics_jax.py:60-94](../src/kt_epimodel_hira/jax_model/dynamics_jax.py#L60-L94)):
```python
beta_s_eff = beta_s * mult_via_beta
p_school_eff = p_school * mult_via_p
# realloc=0: β_s 만 scale (학교 접촉 손실)
# realloc=1: p_school scale (학교 접촉 → home spillover, 기존 κ 재사용)
```
production 은 `realloc = 1.0` (학교 접촉 손실을 가정 spillover 로 재배분).

**가짜 비식별의 원흉**: 방학 누락 시 모델이 amp/π 를 misspec 으로 보정 → γ level=0.6, work:other=0.349 등 일부 "fix" 가 사실 방학 누락 보정 효과였음 ([12. 방법론 교훈](#12-방법론-교훈) 참조).

### 3.5 NGM (Next-Generation Matrix) — R0 ↔ β 변환

[numpyro_model.py:45-104](../src/kt_epimodel_hira/jax_model/numpyro_model.py#L45-L104):

$$
K = \frac{s_f}{\gamma} \cdot \text{diag}(N \cdot \phi \cdot S_{\text{frac}}) \cdot C_{\text{eff}} \cdot \text{diag}(1/N)
$$

$$
R_0 = \rho(K) \quad \text{(spectral radius)}
$$

`C_eff` 는 4 채널 가산:
$$
C_{\text{eff}}(a, a') = \beta_h C_h + \beta_w C_w \cdot (\rho \cdot \text{work\_mask})_a \cdot \rho_{ok, a'} + \beta_s C_s \cdot \text{school\_mask} + \beta_o C_o
$$

**1-homogeneity**: ρ 는 β 에 대해 선형 → `derive_beta_from_R0_simplex` 에서 한 번의 NGM 평가로 β 정확히 역산 가능 (sampler 가 R0 와 π 만 다루도록 reparam).

### 3.6 관측 모델

**Negative Binomial-2** (TODO-3 에서 Poisson 교체):
$$
\text{obs}(a, t) \sim \text{NegBin}\!\left(\mu = \text{pred}(a, t),\; \text{Var} = \mu + \mu^2 / \phi_{nb}\right)
$$

- `phi_nb` ~ HalfNormal(10), 데이터가 결정 (1.30-1.59 식별)
- coverage 12% (Poisson) → **95.2%** (NB)
- r_hat 2.38 → **1.02**, wall 6h25m → **41 min**

구현: [src/kt_epimodel_hira/jax_model/loss_jax.py](../src/kt_epimodel_hira/jax_model/loss_jax.py) `make_multi_season_loss_fn_nb`.

### 3.7 HIRA 6 군 ↔ NIMS 15 군 매핑

[src/kt_epimodel_hira/calibration/hira_target.py](../src/kt_epimodel_hira/calibration/hira_target.py) `HIRA_GROUP_TO_NIMS_WEIGHTED`:

| HIRA bin | NIMS 15 인덱스 | 가중치 |
|---|---|---|
| 0-5 | 0, 1 (부분) | 인구 비례 |
| 6-11 | 1 (부분), 2 (부분) | 인구 비례 |
| 12-17 | 2 (부분), 3 (부분) | 인구 비례 |
| 18-44 | 3 (부분), 4, 5, 6, 7, 8 (부분) | 인구 비례 |
| 45-64 | 8 (부분), 9, 10, 11, 12 (부분) | 인구 비례 |
| 65+ | 12 (부분), 13, 14 | 인구 비례 |

각 NIMS idx 가중치 합 = 1.0 (unit test 검증).

**구조적 한계**: HIRA 18-44 bin = **27 년 폭** → home peak (30-44) + work peak (25-44) **둘 다 흡수** → home/work 채널 분리 불가.

---

## 4. 식별성 전략 — 본 연구의 방법론적 핵심

### 4.1 곱셈의 함정

모델 관측 신호:
$$
\text{청구 카운트} \propto \beta \cdot \phi \cdot \gamma \cdot s(t)
$$

**같은 청구를 다른 (β·φ, γ) 곱이 생성** → 데이터만으로 분해 불가 (**구조적 비식별**).

### 4.2 데이터가 보는 차원 / 못 보는 차원

| 파라미터 | 의미 | 데이터 정보량 | 결정 방법 |
|---|---|---|---|
| **R0** | 시즌 평균 전파력 | **강함** (peak 시점·높이) | **NUTS 추정** |
| **β_channel** (4 채널 비율) | school / other | **강함** | NUTS 추정 |
| **β_channel** (home, work) | 가정 / 직장 | **없음** (HIRA 6 군 한계) | **field knowledge prior** |
| **γ_report** | 탐지율 | **없음** (분모 미관측) | **외부 고정** (CDC + PSA) |
| **φ_age** | 연령별 감수성 | **약함** | **고정** (φ=1.0, 비식별 실증 후) |

### 4.3 Reparam A (현재 사용)

곱셈 ridge 의 primary axis 를 **log R0** 로 회전 — chain swap 방지.

**Sampled** (per season):
- `log_R0` (4): TruncatedNormal(log 1.5, 0.3) [log 0.8, log 2.8]
- `logit_pi` (4 × 4): Normal(0, 0.3) softmax → 4-simplex
- `phi_nb`: HalfNormal(10)

**Derived**:
- β_4 = `derive_beta_from_R0_simplex(ngm_eigval_fn, R0, π, φ)` (NGM 1-homogeneity)
- γ_3 = registry 고정 (CDC × level 0.6)
- φ_14 = 1.0 고정

### 4.4 모델 변형 (NUTS 모델 4 가지)

| 모델 | 정의 위치 | 사용처 |
|---|---|---|
| `hira_model` | numpyro_model.py:107 | Poisson 기본 (deprecated) |
| `hira_model_nb` | numpyro_model.py:345 | NB 기본 (free 4-channel) |
| `hira_model_nb_chprior` | numpyro_model.py:197 | NB + 채널 prior toward target_A/B (production) |
| `hira_model_nb_wo` | numpyro_model.py:272 | NB + work:other = NIMS 0.349 고정 (3-simplex, 폐기) |
| `hira_model_nb_2ch` | numpyro_model_sem2.py | SEM-2: 4→2 channel collapse (home_total + school) |

### 4.5 채널 prior — A vs B (대칭)

[scripts/diagnose_channel_prior.py](../scripts/diagnose_channel_prior.py) `build_target_A/B`:

| 출처 | 입력 quantity | 의미 | 공통 변환 |
|---|---|---|---|
| **A (NIMS)** | contact share `[0.27, 0.20, 0.17, 0.37]` (h, w, s, o) | 접촉빈도 (감염효율 미반영) | **÷ unit_R0** `[8.70, 6.21, 25.40, 9.33]` → normalize → π |
| **B (literature)** | Italy 2009 R0 기여 `[0.40, 0.10, 0.27, 0.23]` | H1N1 fit (감염효율 포함) | **÷ unit_R0** → normalize → π |

→ **target_A π = [0.29, 0.29, 0.06, 0.36]**, **target_B π = [0.47, 0.17, 0.11, 0.25]**

**prior 적용**: centered logit 에 Gaussian penalty:
```python
centered = logit_pi - mean(logit_pi)
dev = centered - logit_target
penalty = -0.5 * Σ dev² / σ²
```
- σ_weak = (0.01, 0.01, 0.30, 0.30) — home/work pin, school/other 추정
- σ_strong = (0.01, 0.01, 0.01, 0.01) — 전 채널 pin

### 4.6 점추정 vs posterior 의 절벽 주의 ⚠️

**4 조합 점추정** (slide 17, [outputs/metapop/channel_prior_4combos.json](../outputs/metapop/channel_prior_4combos.json)):
- 학생 결석 (p_school=0.5): A/B 모두 98.3-98.6% averted
- 원인: 4 조합 모두 R_eff ≈ 1.0 임계 직하에 안착 → 50% 결석으로 epidemic 완전 진압 (절벽)

**Production posterior** (slide 20):
- 학생 결석 (p_school=0.5): A +39.7% [11.7, 74.8] / B +55.2% [28.5, 83.6]
- posterior 가 R_eff 분포 → 절벽 희석

**→ 정책 결론은 posterior (38-87%) 사용. 점추정 98% 는 robust 판정 (span 0.3%) 만 의미 있음.**

---

## 5. Calibration 파이프라인

### 5.1 데이터 파라미터 (33-D 벡터)

[src/kt_epimodel_hira/calibration/param_vector.py](../src/kt_epimodel_hira/calibration/param_vector.py):

```
vec_33 = [phi(14), gamma(3), beta(16)]
       = [phi_0..phi_13,
          γ_child, γ_adult, γ_elder,
          β_2017_h, β_2017_w, β_2017_s, β_2017_o,
          β_2018_h, ..., β_2022_o]
```

- φ_5 ≡ 1.0 (anchor) — Reparam A 에서 trivial
- γ 는 시즌 무관 (3 그룹 단일값)
- β 는 시즌별 4 채널 (16 dim)

### 5.2 build_aggregated_inputs (시즌·지역 집계)

[src/kt_epimodel_hira/calibration/simple_model.py](../src/kt_epimodel_hira/calibration/simple_model.py) `build_aggregated_inputs`:

수도권 1154 행정동 → 1 개 aggregated:
- `pop_15` (15,): NIMS 인구 합산
- `rho` (15,): 노동자 비율 평균
- `matrices`: C_home, C_work, C_school, C_other (15×15)
- `mobility`: M_home, M_work, M_school, M_other (15×15×15)

### 5.3 초기 상태 (시즌별)

`estimate_initial_infected_from_hira(season, ...)`:
- 1 주차 HIRA 청구 / γ × CDC 역산 → 초기 감염 시드 (연령별)

`_build_initial_state_with_age_seed(...)`:
- S, V, E, I, R 모두 (15, n_admdong) shape
- R(0) = `R0_IMMUNITY_PROFILE` 계단 (0-19: 0.10 / 20-49: 0.30 / 50-64: 0.45 / 65+: 0.65)

### 5.4 Loss function

[src/kt_epimodel_hira/jax_model/loss_jax.py](../src/kt_epimodel_hira/jax_model/loss_jax.py) `make_multi_season_loss_fn_nb`:

1. vec_33 → simulate_jax 호출 (시즌별, 4 채널 ODE)
2. 일별 새 감염 → HIRA 6 군 가중 변환 → 주별 집계
3. NB-2 NLL 계산 (weight 적용)

### 5.5 NUTS 본추정

[scripts/m2_production_chprior.py](../scripts/m2_production_chprior.py) — production 표준:
- 4 시즌 합동 fit
- 300 warmup + 300 sample × 4 chain sequential
- `NUTS(target_accept_prob=0.8, max_tree_depth=8, dense_mass=False)`
- A (NIMS) / B (literature) 각 1 회, wall ~9h

### 5.6 진단 / 점추정 스크립트

[scripts/](../scripts/):
- `multistart_*.py` — 8 시작점 multi-start 비식별 진단 (Phase 1)
- `diagnose_*.py` — 비식별·misspec 진단 (channel_decomposition, winter_break, work_channel, channel_prior, 4season_joint, gamma_adult_sweep, …)
- `m2_smoke_*.py` — 짧은 NUTS smoke (50-100 sample, prior/init 검증)
- `metapop_step_*.py` — metapop 단계별 검증 (beta consistency, NBWO, scenarios)
- `sem2_*.py` — 세미나 피드백 검증 (이번 SEM-2 추가)

---

## 6. 정책 시뮬레이션

### 6.1 정책 메커니즘 (대칭)

**규약 (p_school / p_work)**: `p` = **감염자 중 등교/출근 잔존율** (attend-if-sick).
**정책 강도 = 1 − p**. **baseline `p = 1.0`** (감염자 전원 등교/출근 → 정책 없음,
spillover φ_spill = 1 − p = 0 → `spill_factor ≡ 1`). p 는 방학과 무관하며 정책으로만
발동한다 (방학 접촉 감쇠는 C(t) term↔vacation 전환이 전담 — §3.4).

| 정책 | 의미 | FOI 적용 |
|---|---|---|
| **감염 학생 결석** | 감염 학생만 등교 안 함 | `I_eff_school = p_school × I_student` |
| **병가** (감염 근로자 결근) | 감염 근로자만 출근 안 함 | `I_eff_work = p_work × I_worker` |

- p_school = 0.5 → 감염 학생 50% 결석 (50% 는 등교), 정책 강도 1−p = 0.5
- p_work = 0.4 → 감염 근로자 60% 결근 (40% 는 출근), 정책 강도 1−p = 0.6
- **물리적 "학교 폐쇄" 아님**: C_school 그대로, β_s 그대로. θ 가 **감염자 (I) 에만** 적용 → 건강 학생은 정상 등교.
- **Spillover**: 결석/결근 감염자 → 가족 노출 증가 `1 + κ × (1 - θ) × φ_spill`

### 6.2 κ (spillover) sweep

[scripts/diagnose_break_realloc_sweep.py](../scripts/diagnose_break_realloc_sweep.py), [outputs/metapop/spillover_sweep_AB.json](../outputs/metapop/spillover_sweep_AB.json):
- κ_scale = 0 (완전 가구 격리) ~ 1.0 (default)
- A·B 둘 다 학생 결석은 전 구간 양수
- 병가는 B 가 κ ≈ 0.49 에서 부호 반전 (B 는 work π 작음 → home spillover 우세)

### 6.3 채널 가정 robust성 4 조합 (탐색)

L-BFGS 점추정, 분 단위:
- A_strong / A_weak / B_strong / B_weak
- school 결석 98.3-98.6% (span 0.3%) ← ⚠️ 절벽 artifact
- 병가 -2.8% ~ +2.2% (span 5.0%) ← 부호 가정 의존

→ robust 판정만 사용 (실제 효과 크기는 production posterior).

### 6.4 Production posterior 비교 (본추정)

[outputs/calibration/m2_prod_A_samples.npz](../outputs/calibration/m2_prod_A_samples.npz), `m2_prod_B_samples.npz`:

| 정책 | A (NIMS) | B (literature) | CI overlap |
|---|---|---|---|
| 감염 학생 결석 | +39.7% [11.7, 74.8] | +55.2% [28.5, 83.6] | 78.3% (강건) |
| 병가 | +9.6% [+8.0, +10.7] | −10.9% [−12.9, −8.7] | 0% (부호 분리) |

---

## 7. 공간 분석 (metapop)

### 7.1 지역 분류 (주간 / 야간 인구비)

KT mobility 시간대별 노동자 위치:
$$
\text{ratio}_j = \frac{\text{주간 인구}_j (9{-}17\text{시})}{\text{야간 인구}_j (\text{거주})}
$$

- **상업** (ratio > 1.20, 385개): 종로 명동 21.8, 중구 19.5
- **혼합** (0.41 - 1.20, 384개)
- **주거** (ratio < 0.41, 385개): 서대문 0.07, 관악 0.08

range 0.07-21.8 (300× 차이) — 잘 분류됨.

### 7.2 공간 정책 효과 = **null**

[outputs/metapop/spatial_heterogeneity_AB_v2.json](../outputs/metapop/spatial_heterogeneity_AB_v2.json):

| label | region | 병가 | 학생 결석 |
|---|---|---|---|
| A | commercial | +10.07% | +37.19% |
| A | residential | +9.59% | +37.41% |
| B | commercial | −11.02% | +56.12% |
| B | residential | −11.13% | +56.20% |

**commercial / residential ratio**: 0.99-1.05 → **무차이 (<0.5 pp)**

### 7.3 메커니즘 — KT mobility backpropagation

```
직장지(상업) FOI → einsum(M_work, ...) → 거주지(i) S 노출
                                   ↑ 통근으로 거주지에 균질 분배
```

통근 사회에서는 직장 노출 차단의 혜택이 **거주지(전역) 로 분산** → 거주지 기준 효과 균질.

**→ 공간 타겟팅 무가치, 균일 정책 정당.** "공간 모델로 공간 무관성을 입증" 한 의미 있는 null.

---

## 8. 코드 구성

```
kt_epimodel_hira/
├── PROJECT_BRIEF.md                   # 빠른 현황 (이번 회의에 들고 갈)
├── docs/
│   ├── PROJECT_REFERENCE.md           # ★ 본 문서 (전체 참조)
│   ├── SKELETON_ANALYSIS.md           # kt_epimodel 동기화 분석
│   ├── HIRA_VS_ILI_DIFF.md            # ILI 대비 차이점
│   ├── GAMMA_STRATEGY.md              # γ_report 전략
│   ├── PRIOR_SPECIFICATION.md         # prior 사양
│   ├── AGE_DEPENDENT_GAMMA.md         # 연령별 γ 설계
│   ├── identifiability_strategy.md
│   ├── CALIBRATION_COMPARISON.md
│   ├── CONVENTIONS.md
│   └── TODO_model_improvements.md     # ★ TODO + 세미나 피드백
│
├── src/kt_epimodel_hira/
│   ├── model/                         # ODE / FOI 본체 (kt_epimodel 동기화)
│   │   ├── compartments.py            # S/V/E/I/R 인덱스
│   │   ├── dynamics.py                # ODE rhs (numpy 버전 — 통합 안 씀)
│   │   ├── foi.py                     # 4 채널 FOI (numpy)
│   │   ├── mobility_tensor.py         # M_* einsum 헬퍼
│   │   └── parameters.py              # ★ ModelParameters / CalibrationParameters
│   │
│   ├── simulation/                    # solver, runner (numpy — 통합 안 씀)
│   │
│   ├── calibration/                   # 33-D 벡터 fit (numpy/scipy)
│   │   ├── param_vector.py            # ★ 벡터 layout (수정 금지)
│   │   ├── simple_model.py            # build_aggregated_inputs, initial state
│   │   ├── hira_target.py             # ★ HIRA 6 → NIMS 15 매핑
│   │   ├── gamma_registry.py          # γ active / source / CHILD_IDX 등
│   │   ├── loss.py                    # numpy NLL (deprecated, jax_model 으로 이전)
│   │   └── optimizer.py               # L-BFGS-B 점추정
│   │
│   ├── jax_model/                     # ★ 본추정 스택 (JAX / numpyro / diffrax)
│   │   ├── dynamics_jax.py            # ★ ODE rhs (JAX, 방학 ramp 포함)
│   │   ├── foi_jax.py                 # ★ 4 채널 FOI (JAX)
│   │   ├── solver_jax.py              # ★ simulate_jax, daily_new_infection_by_age_jax
│   │   ├── loss_jax.py                # ★ make_multi_season_loss_fn_nb
│   │   ├── numpyro_model.py           # ★ hira_model / hira_model_nb / _chprior / _wo
│   │   └── numpyro_model_sem2.py      # SEM-2: hira_model_nb_2ch (4→2 collapse)
│   │
│   ├── scenarios/                     # 정책 시뮬, metapop (얇은 wrapper)
│   ├── utils/                         # safe_save 등
│   └── viz/                           # plotting
│
├── scripts/                           # 실행 / 진단 스크립트
│   ├── m2_production*.py              # production NUTS (A, B, chprior, wo)
│   ├── m2_smoke_*.py                  # 짧은 smoke (50 sample)
│   ├── m2_fit_vs_data_AB.py           # slide 19 fit 그림
│   ├── m2_posterior_predictive*.py    # coverage 진단
│   ├── m2_policy_compare_AB.py        # 정책 averted 비교
│   ├── diagnose_*.py                  # 비식별·misspec 진단 (다수)
│   ├── multistart_*.py                # 8 시작점 비식별 진단
│   ├── metapop_step_*.py              # metapop 단계별
│   ├── m2_spatial_heterogeneity_AB*.py # 공간 분석
│   ├── sem2_2ch_collapse_sanity.py    # ★ 세미나 SEM-2 (이번 추가)
│   └── build_*.py                     # 노트북 생성 헬퍼
│
├── notebooks/                         # EDA / calibration steps
│
├── outputs/
│   ├── calibration/
│   │   ├── m2_prod_{A,B}_samples.npz      # ★ production NUTS posterior
│   │   ├── m2_prod_{A,B}_posterior.nc     # arviz idata
│   │   ├── m2_prod_{A,B}_result.json      # 요약 (R0, π, rhat, ess)
│   │   ├── sem2_2ch_*_samples.npz         # SEM-2 결과
│   │   └── PROD_{A,B}_DONE.flag           # sentinel
│   ├── metapop/
│   │   ├── channel_prior_4combos.json     # ★ 4 조합 점추정
│   │   ├── spillover_sweep_AB.json        # κ sweep
│   │   ├── spatial_heterogeneity_AB.json  # 공간 v1
│   │   └── spatial_heterogeneity_AB_v2.json # 공간 v2 (주간/야간)
│   ├── eda/
│   └── mlruns/                            # mlflow tracking
│
├── presentations/
│   ├── calibration_identifiability_v3.{md,pdf}  # v3 (이전)
│   └── calibration_identifiability_v4.{md,pdf}  # ★ v4 (현재, 28 pages)
│
├── tests/                             # pytest
└── CLAUDE.md                          # ★ Claude 작업 가이드
```

### 8.1 핵심 모듈 진입점

새 분석 시 먼저 읽을 파일들:

1. **CLAUDE.md** — 작업 규약, 금지사항
2. **이 파일 (PROJECT_REFERENCE.md)** — 전체 맥락
3. **src/kt_epimodel_hira/jax_model/numpyro_model.py** — NUTS 모델 정의
4. **src/kt_epimodel_hira/jax_model/foi_jax.py** — 4 채널 FOI mechanics
5. **src/kt_epimodel_hira/jax_model/dynamics_jax.py** — ODE rhs + 방학 ramp
6. **src/kt_epimodel_hira/calibration/simple_model.py** — 데이터 로드 / 초기 상태
7. **scripts/m2_production_chprior.py** — production 실행 표준 패턴

---

## 9. 규약 / 컨벤션

### 9.1 절대 금지

| 항목 | 이유 |
|---|---|
| `model/`, `simulation/`, `calibration/param_vector.py` 본체 수정 | kt_epimodel 동기화 코어 — bug fix 는 kt_epimodel 에 먼저 |
| `jax_model/` 기존 함수 시그니처 변경 | 모든 production 스크립트 영향 |
| ILI 매핑 / 함수 / 로더 import | 본 프로젝트는 HIRA 전용 |
| 14군 컨벤션 | 15군 확정 |
| per-1000 곱셈/나눗셈 | HIRA 는 **절대 카운트** 단위 |

### 9.2 코드 스타일

- **Polars** 우선 (Pandas 회피) — kt_data 일관
- **NumPy** 행렬 연산
- **Type hint** 사용
- 그래프·변수명 **영어** (한글 라벨 금지)
- 테스트: pytest

### 9.3 신규 모델 변형 추가 방법

- jax_model/ 또는 scripts/ 레벨에서 신규 파일 추가
- 기존 함수 복사-축소 OK, 본체 수정 X
- 예: `jax_model/numpyro_model_sem2.py` 처럼 별도 파일

### 9.4 시도 18 개 / 실 데이터 17 개

- 시도 코드 catalog: 18 개
- 실 HIRA 데이터: 17 개 (어딘가 1 개 결측)
- `SUDOGWON_SIDO_CODES = [11, 28, 41]` (서울 / 인천 / 경기)

---

## 10. 수치 안정성

### 10.1 핵심 규칙 (메모리 `feedback_numerical_safety`)

JAX/numpyro/diffrax 스택에서:

| 규칙 | 이유 |
|---|---|
| **극단 영역 진입 방지: prior 로 제어** | NaN 발생 후 보정보다 사전 차단 |
| **유한 페널티 사용 (NaN 금지)** | sampler 가 NaN 보면 reject 못함 |
| **ODE max_steps tight 유지** | 풀린 모델은 max_steps 폭주 → wall 8h+ |

### 10.2 일반적 trap

- `logit_pi` σ 0.5 → softmax 가 degenerate mix 진입 → β 폭발 → ODE max_steps overrun
- → σ 0.3 으로 좁힘 (numpyro_model.py:156)
- `phi_nb` HalfNormal(10) 적당히 wide — 데이터가 0.1-20 사이 결정

### 10.3 max_steps

- 정상 영역: 100K-200K 충분
- v7 smoke 에서 500K 시도 → warmup leapfrog 느려짐
- production: 적절히 조정 (구체적 값은 [solver_jax.py](../src/kt_epimodel_hira/jax_model/solver_jax.py) 참조)

---

## 11. 출력물 / 산출물 지도

### 11.1 calibration

| 파일 | 출처 스크립트 | 내용 |
|---|---|---|
| `m2_prod_A_samples.npz` | m2_production_chprior.py A | NIMS prior NUTS posterior (~600 draws × 4 chains) |
| `m2_prod_B_samples.npz` | m2_production_chprior.py B | literature prior 동일 |
| `m2_prod_{A,B}_result.json` | 동일 | R0/π/rhat/ess 요약 |
| `m2_prod_{A,B}_posterior.nc` | 동일 | arviz idata netcdf |
| `PROD_{A,B}_DONE.flag` | 동일 | sentinel (완료 표시) |

### 11.2 metapop

| 파일 | 출처 | 내용 |
|---|---|---|
| `channel_prior_4combos.json` | diagnose_channel_prior.py | 4 조합 점추정 채널 mix + 정책 averted |
| `spillover_sweep_AB.json` | spillover sweep | κ 격자 × A·B posterior 정책 효과 |
| `spatial_heterogeneity_AB.json` | spatial v1 | 순유입/인구 분류 (deprecated) |
| `spatial_heterogeneity_AB_v2.json` | spatial v2 | 주간/야간 인구비 분류 |

### 11.3 presentations

| 파일 | 내용 |
|---|---|
| `calibration_identifiability_v4.md` | 마크다운 (Marp) |
| `calibration_identifiability_v4.pdf` | 28 페이지 PDF |

### 11.4 mlflow

- `outputs/mlruns/mlflow.db` — tracking
- experiments: `hira_calibration_m2_prod_chprior_{A,B}` 등

---

## 12. 방법론 교훈

### 12.1 misspecification → 가짜 비식별 (★ 핵심 교훈)

**겨울방학 누락** (NLL +400K 개선) 발견 전:
- school π 0.61 (과대)
- 데이터 선호 amp 0.3 (낮음)
- home π=0.27 을 살리려 γ level=0.6 적용
- work π=0.11 을 살리려 work:other = 0.349 (NIMS row-sum) 고정

방학 도입 후 재평가:
| 이전 결론 | 재평가 |
|---|---|
| γ level=0.6 으로 home 살림 | **방학 누락 보정 효과**였음 — γ CDC 로 충분 |
| work:other = 0.349 고정 | 그 시점 일부 정합화 — 지금은 4 조합 prior 일반화 |
| school π=0.61 → 0.24 | 방학이 "급락" 직접 설명 |

**교훈**: 빠진 메커니즘이 다른 파라미터를 misspec 으로 보정 → "가짜 비식별" 처럼 보이게 함. 진단·해소 반복으로 자기수정.

### 12.2 home/work 구조적 비식별

5 시도 모두 실패:
1. 단일 시즌 4-channel free → home, work ≈ 0
2. Winter break (school_only) → home, work ≈ 0
3. Winter break + 접촉 재배분 → home, work ≤ 0.01
4. amp × realloc sweep (9 조합) → home ≤ 0.004
5. 4 시즌 합동 + 방학 + amp → home, work ≤ 0.011

**원인 (확정)**: HIRA 18-44 bin (27 년 폭) 이 home peak (30-44) + work peak (25-44) 모두 흡수. ILI 동일 해상도. **더 고운 한국 인플루엔자 데이터 없음.**

→ field knowledge prior 4 조합 (A_strong/weak × B_strong/weak) robust 분석으로 정책 결론.

### 12.3 점추정 vs posterior (절벽 주의)

- 점추정 한 점이 R_eff ≈ 1.0 임계 근처 → 정책 averted% 극단 (98%)
- posterior 는 R_eff 분포 → 평균 averted% 희석 (40-55%)
- **→ 정책 결론은 posterior. 점추정은 robust 판정만 (span)**

### 12.4 공간 분류 — denominator 조심

v1: 순유입 / 인구 분류 → dilute 됨 (분모 dominance)
v2: **주간 / 야간 인구비** → 300× dynamic range 회복

→ 분류 지표 선택이 결과를 좌우. 결과가 "noise" 처럼 보이면 분류 지표 의심.

### 12.5 NB 관측모델 — peak over-prediction trade-off

NB phi_nb=1.3 → 분산 ≫ 평균 → 10% 오차의 ΔNLL 가 규모 무관하게 평탄
- coverage 95% 충족 (95% band 안)
- 그러나 posterior mean 이 peak 1.5-3× 과대 (45-64 2.8×, 12-17 2.4×, …)
- **→ 정책 averted % 비율 상쇄로 영향 작음 / 절대 카운트는 1.5-3× 과대 → ICER 주의**

---

## 13. 세미나 피드백 / 미해결 검증

### 13.1 2026-06-18 세미나

좌중 요지: home/work=0 결론 수용. **단 대안 원인 배제 검증 요구.**

상세: [docs/TODO_model_improvements.md](TODO_model_improvements.md) "세미나 피드백" 섹션.

| ID | 항목 | 우선순위 | 영향 |
|---|---|---|---|
| **SEM-1** | NIMS contact matrix 신뢰도 (vs COMIX/POLYMOD) | 🔴 최우선 | **결론 바꿀 수 있음** — NIMS 20대 work contact 과소측정 가능성 |
| **SEM-2** | 4 채널 합산 단일 matrix sanity | 🔴 높음 | **결론 강화** (빠름) — 4→2 collapse 도 fit 동등하면 channel 분해 과파라미터화 확인 |
| SEM-3 | 초기조건 — R(0)≈0 (post-pandemic), 초기 감염자 | 🟡 중간 | TODO-1/2 와 통합 |
| SEM-4 | 발표 자료 "98% 절벽" → "posterior 38-87%" 교정 | 🟢 진행중 | v4 반영 |
| SEM-5 | 연령별 susceptibility prior 좁게 (φ=1.0 고정) | ✅ 완료 | 기록만 |

### 13.2 기존 TODO

| ID | 항목 | 우선순위 |
|---|---|---|
| **TODO-1** | 백신을 S 뿐 아니라 R(0) 에도 적용 (노인 R(0) 0.65 + 접종률 82%) | 🔴 높음 |
| TODO-2 | post-pandemic 낮은 R(0) (2022-23) | 🟡 중간 |
| TODO-3 | NB 관측모델 도입 | ✅ 완료 |
| TODO-5 | spillover κ 외부 추정 | 🟡 중간 |
| **TODO-6** | work:other 비율 외부 추정 (KOSIS) | 🔴 높음 — 병가 정책 부호 좌우 |

### 13.3 진행 중 (현 회의 시점)

**SEM-2 옵션 B 풀런 직전**:
- 모델: [src/kt_epimodel_hira/jax_model/numpyro_model_sem2.py](../src/kt_epimodel_hira/jax_model/numpyro_model_sem2.py) `hira_model_nb_2ch`
- 스크립트: [scripts/sem2_2ch_collapse_sanity.py](../scripts/sem2_2ch_collapse_sanity.py)
- Sanity (50+50×2): wall 745s, r_hat 3.14, div 15, phi_nb 224 — multi-mode 신호
- 사용자 결정 대기 (A 그대로 풀런 / B multi-start init / C 점추정 사전 진단)

---

## 14. 용어집 / 핵심 숫자

### 14.1 약어

| 약어 | 의미 |
|---|---|
| SVEIR | Susceptible → Vaccinated → Exposed → Infectious → Recovered |
| FOI | Force of Infection (감염력) |
| NGM | Next-Generation Matrix |
| NB | Negative Binomial |
| NIMS | 한국 국립수학연구원 (접촉 행렬 출처) |
| HIRA | 건강보험심사평가원 |
| CDC | US Centers for Disease Control (γ 연령 상대비 출처) |
| KT | 한국 통신사 (모빌리티 데이터) |
| ICER | Incremental Cost-Effectiveness Ratio |

### 14.2 핵심 숫자 (production posterior 기준)

| 지표 | 값 |
|---|---|
| 4 시즌 R0 mean | ~1.90-1.99 (시즌별 [1.74, 1.98]) |
| target_A π (NIMS) | h 0.29, w 0.29, s 0.06, o 0.36 |
| target_B π (literature) | h 0.47, w 0.17, s 0.11, o 0.25 |
| unit_R0 (h,w,s,o) | 8.70, 6.21, 25.40, 9.33 (φ=1, sf=1.7) |
| γ_report level | 0.6 (CDC × 어린이 1.45 / 성인 0.65 / 노인 0.90) |
| φ | 1.0 고정 (비식별 r_hat 4.46) |
| amp 선호 | 0.9 (방학 도입 후) |
| 겨울방학 NLL gain | +400,000 (잡음 80K 의 5 배) |
| NB phi_nb | 1.30-1.59 (식별) |
| 학생 결석 averted (production) | A +39.7% / B +55.2% |
| 병가 averted (production) | A +9.6% / B −10.9% |
| 공간 effect (상업-주거) | < 0.5 pp |

### 14.3 시즌 캘린더

- 시즌 시작: 9월 (`day_in_season_offset` 으로 조정)
- 인플루엔자 peak: 1-2월 (day 90-150)
- 겨울방학: day 113-183 (smooth ramp 양끝)
- 백신 출시: ISO week 42 (10월 중순)

### 14.4 연령 인덱스 (15군)

| idx | 연령 | 그룹 |
|---|---|---|
| 0 | 0-4 | 어린이 (CHILD_IDX = [0,1,2,3]) |
| 1 | 5-9 | 어린이 |
| 2 | 10-14 | 어린이 |
| 3 | 15-19 | 어린이 |
| 4 | 20-24 | 성인 (ADULT_IDX = [4..12]) |
| 5 | 25-29 | 성인 |
| 6 | 30-34 | 성인 |
| 7 | 35-39 | 성인 |
| 8 | 40-44 | 성인 |
| 9 | 45-49 | 성인 |
| 10 | 50-54 | 성인 |
| 11 | 55-59 | 성인 |
| 12 | 60-64 | 성인 |
| 13 | 65-69 | 노인 (ELDER_IDX = [13,14]) |
| 14 | 70+ | 노인 |

- STUDENT_SLICE = [0:4] (0-19)
- WORKER_SLICE = [4:14] (20-69)

---

## 부록: Claude 에게 분석·검증 요청 시 권장 흐름

새 Claude 세션이 본 프로젝트를 분석할 때:

1. **먼저 읽기**: `CLAUDE.md`, 본 문서 (`PROJECT_REFERENCE.md`), `PROJECT_BRIEF.md`
2. **현 상태 확인**: `git status`, `git log -5`, [TODO_model_improvements.md](TODO_model_improvements.md)
3. **메모리 활용**: `/Users/hwcho/.claude/projects/...hira/memory/MEMORY.md` — 세미나 피드백, 수치 안정성 가이드
4. **코드 수정 전**: 본체 (model/simulation/calibration/param_vector) 금지 확인, 신규 파일 추가 패턴 사용
5. **풀런 전**: 짧은 sanity (50 sample 정도) → 벽시간 보고 → 사용자 OK 후 백그라운드 실행
6. **결과 보고**: 절벽 / multi-mode / NaN 등 의심 신호 정직 보고, 다음 진단 옵션 제시

**상호작용 원칙** (사용자 선호):
- 사용자 결정 영역에 침범 X — 옵션 제시 + 사용자 선택 대기
- 종료된 결과 (NLL 절벽, 비식별 진단 등) 는 정직 보고 — 미화 X
- 메모리 (특히 [feedback_numerical_safety](../../.claude/projects/-Users-hwcho-Documents-python-NIMS-kt-epimodel-hira/memory/feedback_numerical_safety.md)) 와 모순되는 행동 전 재확인
