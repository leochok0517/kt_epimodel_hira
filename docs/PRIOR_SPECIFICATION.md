# Prior Specification for MAP-Bayesian Calibration

MAP (maximum a posteriori) calibration 진입 전, 각 prior 의 정량적 근거를
확정. 본 문서는 (1) 향후 코드 구현의 사양, (2) 논문 supplement / reviewer
방어의 근거로 사용.

관련 문서:
- [`AGE_DEPENDENT_GAMMA.md`](AGE_DEPENDENT_GAMMA.md) — γ 의 의학·행태 근거
- [`CALIBRATION_COMPARISON.md`](CALIBRATION_COMPARISON.md) — ILI vs HIRA fit
- `outputs/calibration/multistart_final_analysis.json` — 8-start 검증 결과

## 1. 배경 — 왜 prior 가 필요한가

### 1.1 Multimodal 진단 결과

Multi-start 검증 (8 시작점, L-BFGS-B λ=0.1, maxiter=1000):

| 지표 | 값 | 해석 |
|---|---|---|
| NLL spread | 0.57% | 8개 시작점이 거의 같은 NLL 영역 |
| φ 평균 CV | 53.1% | 같은 NLL, 다른 φ — non-identifiable |
| γ_elder CV | 63% | 0.05 ~ 0.24 (4-5배 차이) |
| β 평균 CV | 65.3% | channel swap 빈발 |

8개 결과가 두 corner 로 군집:
- **Corner A** (2/8: warm, bio_prior): γ_elder ≈ 0.22, φ(20+) 평균 0.49
- **Corner B** (6/8: 나머지): γ_elder ≈ 0.05–0.09, φ(20+) 평균 0.89

### 1.2 Multimodal 의 본질

모델 구조: 관측 가능량 = β × φ × γ × seasonal_factor(t)
- **곱셈 결합** → 데이터는 곱만 봄, 개별값 결정 불가
- 두 corner 의 보상:
  - Corner A: "γ_elder 적당 × φ_노인 낮음"
  - Corner B: "γ_elder ≈ 0 × φ_노인 높음"
- 두 corner 의 노인 incidence 예측은 거의 같으므로 NLL 차이 미미

### 1.3 Prior 의 역할

**"외부 지식으로 비현실적 corner 를 배제"** — 데이터로 부족한 식별성을
의학적·생물학적 근거로 보완. 인위적 조작이 아니라 Bayesian 학습의 정공법.

Corner A 가 "정답" 이라고 주장하는 것이 아니라, **Corner B (γ_elder ≈ 0)
가 한국 인구의 의료이용 행태와 양립 불가** 임을 외부 근거로 주장.

---

## 2. 각 파라미터 Prior 의 근거

### 2.1 γ_report — Age-specific informative priors

| Group | Prior | 평균 근거 | σ 근거 |
|---|---|---|---|
| γ_child | Normal(0.40, 0.10) | CDC <18 multiplier 역산, 어린이 진료 행태 | 데이터 학습 여지 |
| γ_adult | Normal(0.18, 0.05) | CDC 18-64 multiplier, 자가관리 경향 | 비교적 좁음 |
| γ_elder | **Normal(0.25, 0.07)** [Phase 3-2 복귀] | CDC 65+ multiplier — age-specific R(0) 도입으로 노인 과대 구조적 해결 | Corner B 배제 + 순수 reporting 해석 |

(평균값의 도출 과정은 [`AGE_DEPENDENT_GAMMA.md` §6](AGE_DEPENDENT_GAMMA.md)
의 Korean medical data + Reed et al. 2015 다배수 역산 참조.)

#### γ_elder = Normal(0.25, 0.07) 의 정당성

**평균 0.25 의 근거**:
- Reed et al. (PLOS One 2015): 65+ 다배수 5.2 → reporting fraction
  ≈ 1/5.2 + 합병증 진료 보정 ≈ 0.25-0.40
- CDC 다배수의 의미: 한 hospitalized case 가 실제로 몇 명의 medically
  attended 환자를 대표하는가
- 노인의 합병증 우려 (폐렴, 심혈관) → 진료 동기 ↑ → 다배수 작음 → reporting
  fraction 큼 (중간값)
- 한국 노인 의료 접근성 (의료보험 보장률, 노인 외래 본인부담 경감) 고려
  시 미국과 유사 또는 약간 높음 → 보수적으로 0.25 채택

**σ = 0.07 의 근거**:
- 0.18 ~ 0.32 의 ±1σ 범위 → 노인 reporting 의 실측 불확실성을 반영
- ±2σ 범위 (0.11 ~ 0.39) 안에서 Corner A 의 0.22 가 정상 영역
- ±2σ 범위 밖에 Corner B 의 0.05 가 위치 → **bayes factor ~0.0006** 으로
  강하게 억제 (계산: exp(-((0.05-0.25)/0.07)²/2) ≈ 0.00018)

**Corner B (γ_elder ≈ 0.05) 가 비현실적인 이유**:

(a) **한국 데이터**: Park et al. JKMS 2023 의 노인 hospitalization rate
35-50% 가 medically-attended 인구 중 비율. γ_elder=0.05 는 "실제 감염
노인의 95% 가 어떤 진료도 받지 않음" 인데, 노인 만성질환 정기 외래에서
인플루엔자 진단이 함께 잡히는 한국 의료 구조와 모순.

(b) **국제 비교**: 미국 (CDC), 일본 (Noda 2022), 유럽 sentinel surveillance
모두 노인의 reporting fraction 이 성인보다 같거나 높음. γ_elder < γ_adult
는 단일 보고 사례 없음.

(c) **모델 구조 정합**: 본 모델은 SVEIR 로 노인 백신접종 (82% coverage)
을 V compartment 로 명시 흡수. 노인 incidence 가 낮은 이유 (백신 보호) 가
이미 모델 안에 있으므로 γ_elder 를 추가로 낮춰 incidence 를 설명하면
**double-counting**. 이는 Corner B 의 구조적 결함.

> **논문 정당성 문장 (제안)**:
>
> "We impose a weakly informative prior γ_elder ~ Normal(0.25, 0.07)
> derived from CDC age-specific symptomatic multipliers (Reed et al. 2015)
> and Korean medical utilization patterns (Park et al. JKMS 2023). This
> prior is necessary to exclude a biologically implausible local mode
> (γ_elder ≈ 0.05) in which 95% of elderly influenza cases would remain
> medically unattended, contradicting the elevated healthcare-seeking
> behavior documented in Korean elderly populations and double-counting
> the vaccination effect already modeled in the SVEIR V compartment."

#### γ_child = Normal(0.40, 0.10), γ_adult = Normal(0.18, 0.05)

이 두 prior 의 표준편차가 elder 보다 차이가 나는 이유:
- γ_child: 어린이는 multistart 8개 중 CV 16% 로 비교적 안정 → 데이터 학습
  여지를 위해 σ=0.10 (느슨)
- γ_adult: CV 20% 로 안정, 데이터 정보량 충분 → σ=0.05 (좁음)
- γ_elder: CV 63% 로 corner 분리 → σ=0.07 (corner B 배제 목적 강함)

---

### 2.2 φ_a — Smoothing prior (Gaussian random walk)

#### 2.2.1 동기

Multi-start φ 평균 CV 53% 의 주된 원인: 인접 연령대 (5세 단위) 의 φ 가
비현실적으로 점프 가능. 8개 fit 중 일부는 φ(20-24)=0.32 vs φ(30-34)=0.60
같은 oscillation. POLYMOD-style contact pattern 의 연속성과 모순.

#### 2.2.2 수학적 형태

```
log p(φ) = -λ_phi · Σ_{i=0}^{12} (φ_{i+1} - φ_i)²  + const
```

이는 **인접 차이 ~ Normal(0, σ_phi)** 의 random walk prior 와 동치:
```
φ_{i+1} - φ_i ~ Normal(0, σ_phi)
σ_phi = 1 / sqrt(2 · λ_phi)
```

| λ_phi | σ_phi | 의미 |
|---|---|---|
| 0.0 | ∞ | smoothing 없음 |
| 0.1 | 2.24 | 현재 default, 약한 smoothing |
| 1.0 | 0.71 | 중간 |
| 10.0 | 0.22 | 강함 (인접 차이 ~0.2 이내) |

#### 2.2.3 λ_phi 선택 근거 (개요)

- 이전 sweep (v2 prototype) 에서 λ=0.1 시도 → 점프 1.11 → 0.79 (28% 감소),
  data NLL 손실 0%
- 충분치 않음 (φ CV 여전히 32%)
- **권장**: MAP 단계에서 λ_phi ∈ {0.1, 1.0, 10.0, 100.0} sweep, optimal 결정
- 근거: 5세 단위 인접 연령 φ 차이가 1.0 을 넘으면 의학적 통념 위반
  (POLYMOD 기반 한국 contact matrix 의 연속성)

#### 2.2.4 φ_reference 정규화 유지

기존: φ_5 (25-29세) = 1.0 reference. 다른 φ_a 는 25-29 대비 상대값.
Smoothing prior 는 이 reference 를 유지하며 인접 연속성만 강제.

---

### 2.3 β — Weak regularization prior

#### 2.3.1 동기

Multi-start β CV 65% (max 122%, 2017-18 work channel). Channel swap 빈발:
"home dominant" 와 "school dominant" 가 같은 NLL 영역에 공존. multimodal
의 주축은 γ_elder 이지만 β corner 도 보조 차단 필요.

#### 2.3.2 권장 형태 (단계적)

**Phase 1 (default)**: 단순 weak Half-Normal regularization
```
β_ch,season ~ HalfNormal(σ = 1.0)
```
- σ=1.0 은 bound [0.001, 0.30] 을 거의 제약 안 함 (genuinely weak)
- penalty: Σ β² / (2·1.0²) per parameter
- 단순히 매우 큰 값 (~5-10) 만 penalty (corner 의 극단값 방지)
- 비고: 이전 권장이었던 σ=0.10 은 strong (β 평균 0.10 으로 강제),
  weak 의미와 모순 → 1.0 으로 정정

**Phase 2 (corner 안 깨질 시)**: channel ratio prior
```
β_h : β_w : β_s : β_o  ~  NIMS contact matrix row sum 비율 (예: 23:17:14:32)
```
- NIMS contact matrix 의 channel별 magnitude 가 측정값 → 한 채널이
  0.001 까지 떨어지는 corner 는 contact 측정값과 모순
- 형태: log(β_ch / β_other) ~ Normal(log(target_ratio), σ_ratio)

**Phase 3 (시즌 일관성)**: 시즌 간 β 변동에 penalty
```
β_h,2018 - β_h,2017  ~  Normal(0, σ_season)
```
- 같은 채널이 시즌마다 크게 다른 것은 strain 차이로만 설명 어려움
- 단, 일부 시즌 효과는 정당 (예: 2017-2018 H3N2 severity) → σ_season 신중

#### 2.3.3 우선순위

Phase 1 부터 시작. NLL impact 작으면 Phase 2 추가. Phase 3 은 시즌 효과
해석을 제한하므로 가장 마지막.

---

## 3. Prior 강도 trade-off + σ sweep 계획

### 3.1 trade-off

| σ 너무 좁음 | σ 너무 넓음 |
|---|---|
| 데이터 무시 (over-regularization) | corner 못 깸 (prior 효과 없음) |
| posterior 가 prior 그대로 | posterior 가 likelihood 그대로 |
| reviewer: "데이터 학습 안 됨" 의심 | reviewer: "왜 prior 썼냐" 의심 |

→ **MAP 단계에서 σ sweep 으로 적정값 탐색**

### 3.2 권장 sweep 범위

| 파라미터 | σ 후보 | 비교 기준 |
|---|---|---|
| γ_elder | {0.03, 0.05, **0.07**, 0.10, ∞} | corner B 차단 여부 + NLL 손실 |
| γ_child | {0.05, 0.07, **0.10**, 0.15, ∞} | 안정성 (이미 stable) |
| γ_adult | {0.03, **0.05**, 0.07, 0.10, ∞} | 안정성 (이미 stable) |
| λ_phi | {0.1, 1.0, **10.0**, 100.0} | φ 점프 max |
| β HalfNormal σ | {0.30, **1.0**, 3.0, ∞} | β CV 감소 (genuinely weak 부터) |

(**Bold** = 본 문서의 default 권장)

### 3.3 검증 지표

각 σ 조합에 대해 multi-start (예: 5-8 시작점) 반복 → 다음 지표 측정:
- **Corner concentration**: 8 starts 가 1개 corner 로 모이는가 (γ_elder CV < 15%)
- **Fit quality**: data NLL 손실 < 1%
- **Bound hits**: bound 도달 파라미터 수 (작을수록 좋음)
- **Posterior plausibility**: corner A 의 값들이 prior ±2σ 안에 있는가

---

## 4. MAP 목적함수 정식화

### 4.1 정식

```
MAP_loss(θ) = NLL_data(θ)
            + Σ_{age ∈ {child, adult, elder}}  (γ_age - μ_age)² / (2 σ_age²)
            + λ_phi · Σ_{i=0}^{12} (φ_{i+1} - φ_i)²
            + Σ_β  β² / (2 σ_β²)
```

벡터 표기 (33-dim multi-season):
```
θ = [φ_0..φ_13, γ_child, γ_adult, γ_elder, β_2017_hwso, β_2018_hwso, β_2019_hwso, β_2022_hwso]
```

### 4.2 ⚠️ Scale 균형 점검 (중요)

**문제**:
- NLL 규모: -28,500,000 (음수, 절댓값 큼)
- Prior penalty 규모 (default): ~10-100 (양수, 작음)
- → MAP 최적화 시 prior 가 numerically negligible

**해결 옵션**:

(a) **Prior weight 증폭**: `prior_weight = N_data` (관측 데이터 점수) 로
penalty 를 곱함. Bayesian 해석은 그대로 (각 데이터 점이 prior 와 동등 가중).

(b) **NLL 정규화**: NLL / N_observations 로 평균 NLL 사용. prior 도 평균 단위.

(c) **Log-likelihood scale 명시 점검**: 현재 NLL = -28.5M 의 양 (4 시즌 ×
26 weighted weeks × 6 ages ≈ 624 점). NLL per point ≈ -45,673 → 매우 큼.
이는 Poisson 의 큰 카운트 효과. prior penalty 가 0.5 (1σ deviation) 라도
weight=N=624 곱하면 312 → 여전히 작음.

→ **권장**: Phase 1 에서 prior penalty 그대로 더하고 (낮은 weight),
σ sweep 동안 weight × {1, 100, 10000} 도 함께 sweep. corner A 로 수렴하는
최소 weight 확정.

**또는 대안**: γ-만 prior 적용 + 강한 weight (γ 3개 × penalty * w),
φ smoothing 은 별도 λ_phi (이미 적용 중), β는 일단 무시.

### 4.3 단계적 도입 권장

| Phase | 적용 prior | 검증 |
|---|---|---|
| **MAP-1** | γ_elder 만 (Normal(0.25, 0.07) × weight) | 8 multistart 가 Corner A 단일 수렴 |
| **MAP-2** | γ 3개 + φ smoothing 강화 | φ 점프 < 0.5, β CV 감소 |
| **MAP-3** | + β weak prior | β CV < 30% |
| **Bayesian** | NUTS/HMC + same priors | 완전 사후 분포 |

---

## 5. Prior 의 한계 + 향후 작업

### 5.1 인식 한계

1. **Informative prior 는 multimodal 의 근본 해결 아님**: corner B 를
   배제할 뿐. 진짜 식별성은 데이터 확장 또는 모델 구조 변경에서 옴.

2. **Point MAP 은 불확실성 정량화 불가**: 다음 단계 MCMC (NUTS) 가 사후 분포
   추정. PSA (Stage 5) 의 입력.

3. **Prior 자체의 불확실성**: μ, σ 값들이 정확하지 않을 수 있음. **Prior
   sensitivity analysis** 로 결론의 robustness 확인 필수.

4. **Circularity 위험 회피**: prior 가 "데이터에 맞추려고" 가 아니라 "외부
   지식" 이어야 함. 본 문서에서 모든 μ 는 CDC/JKMS/POLYMOD 등 외부 근거.

### 5.2 향후 작업

- **MAP σ sweep**: σ_γ_elder ∈ {0.03, 0.05, 0.07, 0.10, ∞} × multistart 검증
- **Bayesian posterior**: NUTS (numpyro) 또는 SMC. Corner A 영역 내부의
  posterior 분산 정량화
- **HIRA 외래환자 데이터 확보 시 γ 직접 추정**: 인플루엔자 진료/전체 외래
  비율 추정 → prior 불필요해짐 (논문 limitation 으로 명시)
- **Prior sensitivity table**: 논문 supplement 의 표준 항목

---

## 6. 참고문헌

1. Reed C, et al. Estimating influenza disease burden from population-based
   surveillance data in the United States. PLOS One. 2015;10(3):e0118369.
2. Park JY, et al. Incidence, Severity, and Mortality of Influenza During
   2010-2020 in Korea. J Korean Med Sci. 2023;38(8):e58.
3. Tokars JI, et al. Seasonal incidence of symptomatic influenza in the
   United States. Clin Infect Dis. 2018;66(10):1511-1518.
4. Mossong J, et al. Social contacts and mixing patterns relevant to the
   spread of infectious diseases. PLOS Med. 2008;5(3):e74. (POLYMOD)
5. Noda T. Incidence Rate of Seasonal Influenza Calculated from Japanese
   Medical Database. MHLW Japan. 2022.

내부 문서:
- [`docs/AGE_DEPENDENT_GAMMA.md`](AGE_DEPENDENT_GAMMA.md) — γ 의학·행태 근거
- [`docs/CALIBRATION_COMPARISON.md`](CALIBRATION_COMPARISON.md) — ILI 비교
- `outputs/calibration/multistart_final_analysis.json` — 8-start 검증 데이터

---

## 부록 A: NUTS 진단 결과 (M1d, 2026-06)

### A.1 발견

짧은 NUTS smoke test (50+50 × 4 chains, target_accept=0.8) 에서 **4 chain
모두 γ_elder ≈ 0.030 으로 수렴** — TruncatedNormal 하한 saturated.

- corner A init chain (warm γ=0.24, bio_prior γ=0.21) → 0.030
- corner B init chain (neutral γ=0.05, distributed γ=0.08) → 0.030
- per-chain std ≈ 0 (mode 정착)

점추정 (corner A 0.22, corner B 0.05) 모두 NUTS 가 발견한 진짜 posterior
mode 와 다름. L-BFGS-B 는 saddle 에 멈췄고, NUTS 가 likelihood 의 진짜
압력을 드러냄.

### A.2 원인 진단 (forward sim 비교, 2019-2020)

| Test | 결과 |
|---|---|
| 65+ pred/obs at γ_elder=0.25 | 누적 2.05×, peak 1.49× — 모델 과대 |
| 65+ raw attack rate (γ 무관) | 7.5% — 합리적 (5-15%) |
| V compartment 효과 | 51.6% 감소 — 백신 정상 작동 |
| 다른 5 연령 fit (γ=0.24) | 0.63 - 1.11 — OK |
| **65+ peak ratio** | **1.43** — outlier |

→ **(2) 모델 noise misspecification 우세**: 모델이 노인 청구를 과대생성하고,
likelihood 가 γ_elder 로 흡수.

### A.3 두 가지 원인 (가능성)

1. **Contact matrix 의 노인 row 과대**: NIMS 측정값이 한국 노인 실제
   mixing 보다 높을 가능성 (요양시설 격리/fragility 미반영). 단 raw AR
   7.5% 정상 → 영향 제한적.
2. **CDC multiplier 의 한국 노인 적용 한계**: 미국 의료시스템 기준 5.2×
   다배수가 한국 노인 reporting fraction 을 과대평가했을 가능성.

Contact matrix 는 외부 측정값이라 본 연구에서 수정 불가 → γ_elder 가
두 효과를 함께 흡수.

### A.4 결정: γ_elder prior 재조정

| 항목 | 이전 | 신규 |
|---|---|---|
| γ_elder | Normal(0.25, 0.07) [CDC 역산] | **Normal(0.13, 0.05)** [데이터 시사값] |
| TruncatedNormal range | [0.03, 0.50] | [0.03, 0.50] (변경 없음) |

**근거**: 65+ cumulative pred/obs ratio = 1.0 영역 γ ≈ 0.10-0.13.
γ=0.13 은 "순수 reporting" 이 아니라 "reporting + 모델 노인 과대 흡수"
혼합값.

### A.5 정직성 + 논문 limitation 명시

- γ_elder=0.13 은 외부 reference (CDC 0.25) 보다 낮음
- 이유: **모델의 노인 incidence 과대생성을 γ 가 흡수**
- 순수 reporting fraction 으로 해석 불가 — "effective reporting parameter"
- PSA 에서 γ_elder 와 contact matrix 노인 row 의 sensitivity 동시 측정

### A.6 Phase 3-2 후속 진단: 진짜 원인은 R(0) age-uniform 단순화

A.1-A.5 의 결론 (γ_elder=0.13 으로 재조정 + "effective reporting" 해석) 은
Phase 2 짧은 NUTS 재실행 (γ_elder prior 0.13) 에서 **여전히 saturate 발견**
(4 chain 모두 0.030 하한, divergence 6→34 증가, r_hat 1.93→4.19) 으로
부분 기각.

**초기조건 코드 점검 결과**: `_build_initial_state_with_age_seed` 의
`initial_immunity = 0.30` 이 전 연령 동일 — **노인 R(0) 과소설정**.

| 연령 | 현재 R(0) | 추정 실제 |
|---|---|---|
| 0-4 | 0.30 | 0.05-0.15 (제한 노출) |
| 70+ | 0.30 | 0.50-0.70 (다년 누적 + 백신) |

→ S_70+(0) 가 약 0.40 만큼 과대 → 모델 노인 감염 과대 → γ_elder 가 흡수
→ 0.03 으로 saturate.

**Forward sim 검증** (γ=CDC 0.25 고정, warm vec):

| R(0) profile | 65+ peak ratio |
|---|---|
| 0.30 uniform (현재) | 1.47 (과대) |
| 4-step .10/.30/.45/.65 | **0.99** (목표 ~1.0) |

### A.7 결정: Step-function R(0) 도입 + γ_elder CDC 복귀

**채택**:

| 구간 | NIMS idx | R(0) | 근거 |
|---|---|---|---|
| Children 0-19 | 0-3 | **0.10** | POLYMOD 학령기 노출 제한 |
| Young/middle adults 20-49 | 4-9 | **0.30** | 성인 baseline |
| Middle-aged 50-64 | 10-12 | **0.45** | 정기 백신 권고 시작 + 누적 |
| Elderly 65+ | 13-14 | **0.65** | 백신 누적 (82%) + 1957/1968 cross-immunity (Vandegrift 2014) |

**γ_elder prior**: Normal(0.13, 0.05) → **Normal(0.25, 0.07)** (CDC 복귀).
R(0) 가 노인 과대를 구조적으로 해결 → γ_elder 가 순수 reporting fraction
해석 회복.

### A.8 논문 정당성 문장 (최종)

> "We introduce a step-function age-specific initial immunity profile
> R(0) = {0.10 [0-19y], 0.30 [20-49y], 0.45 [50-64y], 0.65 [65+y]}
> based on contact-pattern accumulation (Mossong et al. 2008, POLYMOD)
> and elderly cross-reactive immunity from past pandemic strains
> (Vandegrift et al. 2014). This profile resolves the structural
> overprediction of 65+ episodes under the previous uniform R(0) = 0.30
> assumption, restoring the CDC-derived γ_elder ~ Normal(0.25, 0.07)
> prior to its biologically motivated range without absorbing structural
> model misspecification."

### A.9 한계 명시 (논문 limitation)

1. **POLYMOD / Vandegrift 가 한국 직접 측정값 아님** — 미국/유럽 기준 적용
2. **KDCA serosurvey 등 한국 자체 면역 측정 접근 불가** — 향후 정밀화 가능
3. **R(0) 고정 (fit 안 함)**: Stage 5 PSA 에서 sensitivity analysis 로 robustness 확인
4. **65+ 외 연령 ratio 의 refit 후 정착**: 본 R(0) 도입 후 β/φ 재 fit 필요
