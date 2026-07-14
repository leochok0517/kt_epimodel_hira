---
marp: true
theme: default
paginate: true
header: 'HIRA 인플루엔자 모델 — Calibration 식별성 보고'
footer: 'NIMS | 2026-06'
math: katex
style: |
  section { font-size: 22px; padding: 40px 50px; font-family: 'Apple SD Gothic Neo', sans-serif; }
  h1 { color: #1a5490; font-size: 32px; }
  h2 { color: #2c6cb0; font-size: 26px; }
  h3 { color: #4a7ab8; font-size: 22px; }
  table { font-size: 18px; margin: 0 auto; }
  th { background-color: #e8f0fa; }
  .small { font-size: 16px; color: #555; }
  .red { color: #c0392b; font-weight: bold; }
  .blue { color: #1a5490; font-weight: bold; }
  .green { color: #27ae60; font-weight: bold; }
  .center { text-align: center; }
  img { display: block; margin: 0 auto; }
  blockquote { border-left: 4px solid #1a5490; padding-left: 16px; color: #444; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# 수도권 인플루엔자 모델 Calibration
## 모수 식별성 — 진단·해결·결과

조현우
데이터: 국민건강보험공단 J09–J11 진료에피소드 (2018–2024)

**2026년 6월**

---

# 1. 연구 맥락

**정책 목표**: 정상 시기 인플루엔자 sick-leave / 학교 결석 정책의 ICER

**모델**:
- SVEIR (S → V → E → I → R) + 4-channel FOI (가정 / 직장 / 학교 / 기타)
- 연령 NIMS 15군 (5세 단위)
- 수도권 (서울 + 인천 + 경기), 4 정상 시즌

**데이터**: HIRA J09–J11 진료에피소드, 6 연령군

![bg right:42% w:500](figures/data_overview.png)

<div class="small">

HIRA 의 장점:
- **카운트 단위** (인구 분모 깨끗)
- γ_report 해석이 명확 ("감염자 중 청구 비율")

</div>


---

# 2. γ_report 다시 — 탐지율의 의미

**정의**: 실제 감염자 중 HIRA 진료 청구로 잡힌 비율

$$
\gamma_{\text{report}} = \frac{\text{청구 카운트}}{\text{실제 감염자}}
$$

**빙산 비유**:
- 보이는 부분 = HIRA 청구 (분자, 측정 가능)
- 숨은 부분 = 무증상 + 자가관리 + 미진료 (분모, 측정 불가)

**γ = level × 연령 상대비**:
- **level**: Jung et al. (2025) q = 0.67 (감염 → 증상) 에서 무증상·미청구 제외 → **0.6** (비식별 anchor, R0 흡수 차원)
- **연령 상대비**: 데이터가 식별 가능한 차원 (sweep 에서 CDC 비율 NLL 최저 확인)

| 연령군 | CDC 상대비 | γ (level 0.6 적용) |
|---|---|---|
| 어린이 | **1.45** | 0.87 |
| 성인 | **0.65** | 0.39 |
| 노인 | **0.90** | 0.54 |

---

# 3. 식별성 문제 — 곱셈의 함정

**모델 관측 신호**:

$$
\text{청구 카운트} \;\propto\; \beta \times \phi \times \gamma \times s(t)
$$

**두 시나리오, 같은 청구**:

| | 시나리오 A | 시나리오 B |
|---|---|---|
| 실제 감염 (β·φ) | 100 | 250 |
| 탐지율 (γ) | 0.25 | 0.10 |
| **HIRA 청구** | **25** | **25** |

<div class="red center" style="margin-top: 8px">

데이터만 보면 A 와 B 구분 불가 — **구조적 비식별**

</div>

**핵심**: 데이터를 더 모아도 **곱** 만 결정될 뿐 개별 분해 불가.
모델 구조 자체의 성질.

---

# 4. 진단 — Multi-start 가 보여준 다봉성

**검증**: 8개 시작점 multi-start (warm / bio_prior / 극단 / 무작위 등)

| 지표 | 값 | 의미 |
|---|---|---|
| NLL spread | **0.57%** | 8개 fit 거의 동일 적합 |
| φ 평균 CV | **53%** | 같은 NLL 위에 여러 mode |
| γ_elder CV | **63%** | 분해 자유도 큼 |
| β 평균 CV | **65%** | channel split swap |


---
![w:780](figures/multistart_final_8.png)

<div class="blue center" style="margin-top: 4px">

같은 데이터 fit, 다른 (β, φ, γ) — **곱은 식별, 분해는 안 됨**

</div>

---

# 5. R(0) 연령 구조 누락

**2019-2020 시즌 fit**:

| Test | 결과 | 의미 |
|---|---|---|
| 65+ peak ratio (pred/obs) | **1.43** | 노인 구조적 과대 |
| 65+ raw attack rate | 7.5% (정상) | 감염 수준 자체는 적절 |
| 다른 5 연령 fit | 0.63 – 1.11 | 65+ 만 outlier |

**원인 추적**:
1. R(0) (초기 면역) = **균일 30%** (전 연령 동일)
2. 그러나 노인 실제 면역 (이전 시즌 누적) >> 30%
3. → 노인 S 모집단 **과대**, I 과대생성
4. → 데이터 fit 위해 γ_elder 가 흡수 (0.25 → 0.05 corner)

<div class="green center" style="margin-top: 8px">

**모델 내부 단순화 문제** — 외부 데이터가 흡수 → 수정 가능

</div>

---

# 6. 해결 ① — R(0) 계단 형식

**변경**: 균일 30% → 연령 계단

| 연령군 | R(0) 기존 | R(0) **계단** | 근거 |
|---|---|---|---|
| 0–19 | 0.30 | **0.10** | 최근 시즌 노출 적음 + 백신 coverage 낮음 |
| 20–49 | 0.30 | **0.30** | 기준 |
| 50–64 | 0.30 | **0.45** | 누적 노출 + 백신 |
| 65+ | 0.30 | **0.65** | 백신 정책 + 누적 면역 |

**근거**: POLYMOD 이전 시즌 누적 + Vandegrift 2010 미국 추정

**효과**:
- 65+ peak ratio **1.43 → 0.99** 
- γ_elder CDC band 안: 2/8 → 7/8 시작점
- γ_elder CV: 63% → 25%

---

# 7. 해결 ② — 추정 대상 분리

곱 **β · φ · γ** 중 어느 차원을 데이터로 결정할 수 있나?

| 파라미터 | 의미 | 데이터 정보량 | 결정 방법 |
|---|---|---|---|
| **γ_report** | 탐지율 | **없음** (분모 미관측) | **외부 고정** (CDC + PSA) |
| **φ_age** | 연령별 감수성 | **약함** | **prior 좁게** (Cauchemez ±15%) |
| **β_channel** | 채널별 전파율 | **강함** (peak 시점·높이) | **데이터로 추정** (NUTS) |

**Reparam**: β·φ 곱셈 ridge → primary axis 를 **log R0** 로 회전
- log R0 (시즌별 4) + 채널 비율 (시즌별 4) + φ (14)
- β 는 (R0, 채널 비율, φ) 에서 NGM 역산으로 **derive**

---

# 8. <span class="green">결과 ①</span> R0 식별 (production NUTS, 채널 정상화)

**Production NUTS** (NB 관측모델 + 채널 정상화, 300 warmup + 300 sample × 4 chain, wall 64분)

| 시즌 | R0 mean | 95% CI |
|---|---|---|
| 2017–18 | **1.79** | [1.74, 1.83] |
| 2018–19 | **1.95** | [1.91, 1.98] |
| 2019–20 | **1.90** | [1.85, 1.95] |
| 2022–23 | **1.74** | [1.68, 1.84] |

<br>

**전체**: R0 mean **1.84**, range **[1.68, 1.98]** — 전형적 seasonal flu (1–2.5)
**수렴**: divergence **0/2400**, r_hat **1.007**, ess **908**

<div class="blue center" style="margin-top: 8px">

4 chain 일치 + 완벽 수렴 — R0 가 데이터로 식별되는 primary axis

</div>

---

# 8b. <span class="green">결과 ①′</span> Posterior predictive — 연령별 fit (피드백 3)

![w:1000](figures/posterior_predictive_nb.png)

**구성**: 4 시즌 × 6 HIRA 연령군. 빨간 점 = 관측 청구, 파란 선·띠 = posterior 중앙값 + 95% credible (parameter + **NB 관측 노이즈**)

**판독**:
- **95% coverage = 95.2%** (nominal 95% 와 일치) — 모든 시즌·연령
- 연령별 coverage: 0–5 89%, 6–11 99%, 12–17 92%, 18–44 99%, 45–64 99%, 65+ 93%
- 모든 panel에서 peak 시점·형태·진폭 재현

<div class="blue center" style="margin-top: 8px">

**모델 구조 (4채널 contact + 계단 R(0) + γ CDC + φ=1.0 + NB 관측) 가 데이터 재현**

</div>

---

# 8c. <span class="green">결과 ①″</span> 관측모델: Poisson → NB

**문제**: Poisson 가정 (분산=평균) 으로는 청구 데이터 변동 못 잡음
- 주별 변동·보고 지연·요일 효과 → 실측 **분산 > 평균** (과분산)
- 결과: posterior 띠 좁음, coverage 12%, 거기에 수렴까지 악화 (r_hat 2.38)

**해법**: 관측을 **Negative Binomial-2** 로
$$\text{obs} \sim \text{NegBin}\!\left(\mu = \text{pred},\; \text{Var} = \mu + \mu^2/k\right)$$

**결과**:

| 지표 | Poisson | **NB** |
|---|---|---|
| 95% coverage | 12% | **95%** |
| r_hat max | 2.38 | **1.02** |
| ess_min | 5 | **581** |
| R0 mean | 1.96 | 1.90 (거의 동일) |
| wall | 6h 25min | **41분** |

**NB 분산 파라미터** `phi_nb = 1.44 [1.30, 1.59]` — 식별 (r_hat 1.02)

<div class="blue center" style="margin-top: 4px">

NB가 **coverage 와 수렴을 동시에 해결** — 좁은 Poisson likelihood 가 채널 mix ridge 에 chain 들을 가둔 것까지 풀림

</div>

---

# 8d. <span class="green">결과 ①‴</span> 4 채널 전파 분해 — 비식별 진단 후 정상화

**시즌별 채널 mix** (NB + work:other 비율 고정):

| 시즌 | home | **work** | school | other |
|---|---|---|---|---|
| 2017–18 | 0.36 | **0.12** | 0.31 | 0.22 |
| 2018–19 | 0.23 | **0.11** | 0.45 | 0.21 |
| 2019–20 | 0.23 | **0.11** | 0.44 | 0.21 |
| 2022–23 | 0.27 | **0.15** | 0.30 | 0.28 |

**진단·해소된 3 종 비식별**:

| 증상 | 원인 | 외부 근거 해소 |
|---|---|---|
| home ≈ 0 | γ_adult 절대값 너무 높음 → home β 가 성인 보정 | **γ level = 0.6** (Jung q=0.67 − 무증상/미청구) × CDC 연령 상대비 |
| **work ≈ 0** | work/other 둘 다 성인 타겟 + HIRA 6 연령군 → 분해 정보 부족 | **work:other = 0.349 : 0.651** (NIMS 접촉 row-sum) 고정 |
| school 0.62 과대 | work 흡수 (둘 다 R0 ↑ 효율 비슷) | 위 두 해소로 자동 교정 (0.62 → 0.30–0.45) |

<div class="blue center" style="margin-top: 6px">

데이터가 보는 것(home/school/덩어리 크기) 추정 + 못 보는 것(분배·level) 외부 근거 → single-channel dominance 없음, 역학적으로 타당

</div>

---

# 9. <span class="green">결과 ②</span> φ 비식별 실증

**같은 sample 의 φ posterior**:

| chain | φ[0] (0–4세) | 해석 |
|---|---|---|
| 0 (warm) | 4.90 | |
| 1 (bio_prior) | 5.73 | chain 마다 |
| 2 (distributed) | 6.37 | 서로 다른 mode |
| 3 (home_dominant) | 6.62 | (ridge 위 다른 점) |

| 진단 | 값 | 의미 |
|---|---|---|
| φ posterior mean | 3.17 | prior median 1.0 무시 |
| φ near tail (>2.0) | **67%** | prior 와 무관 |
| r_hat (φ) | **4.46** | chain 분산 |
| ess (φ) | **5** | 거의 안 섞임 |

<div class="green center" style="margin-top: 8px">

**슬라이드 8 의 "φ는 데이터 정보 약함" 진단을 데이터로 확정**
→ φ 외부 고정이 정당 (prior 좁힘 or 점추정)

</div>

---

# 10. 비식별 ridge → field knowledge로 결정

(현재) 데이터만으로는 γ·φ 를 정할 수 없다 (ridge)



- γ (탐지율): CDC reporting multiplier
- φ (연령 감수성): Cauchemez 등 문헌 범위 (±15%)
- β (전파력): 데이터로 추정 (R0)



---

# 11. 선행연구 대비 — Jung et al. (2025)

같은 데이터·모델 출발점, **다른 식별성 처리**

| 측면 | Jung et al. 2025 | 본 연구 |
|---|---|---|
| 데이터 | HIRA 인플루엔자 청구 | 동일 |
| 모델 | 연령별 SEIR | 동일 |
| 백신 | 한국 청소년·노인 정책 | 동일 |
| **식별성** | reporting q = 0.67 **단일값 우회** | 진단·분해·고정 + PSA |
| 정책 시뮬 | 백신 coverage 시나리오 | sick-leave 정책 (수도권 metapop) |
| 불확실성 정량 | 부분 | β·R0 posterior + γ·φ PSA |

<div class="blue center" style="margin-top: 12px">

**방법론적 기여**: identifiability 정면 진단·해결 + 출처 객체화 (γ_registry)

</div>

---

# 12a. 채널 식별성 한계 — 4시즌·방학 모두 실패

**데이터로 식별** ✅ : R0, school 채널, other 채널, 계절성, **winter break**

**데이터로 식별 불가** ❌ : **home, work 채널**

| 시도 | 결과 |
|---|---|
| 단일 시즌 4-channel free | home, work ≈ 0 |
| Winter break 모델 (school_only) | home, work ≈ 0 |
| Winter break + 접촉 재배분 (κ 재사용) | home, work ≤ 0.01 |
| amp × realloc sweep (9 조합) | home ≤ 0.004 |
| **4 시즌 합동 + 방학 + amp** | **home ≤ 0.011, work ≤ 0.011** |

**원인** (구조적): HIRA 6 연령군 집계에서 30-44 (home/work) 신호가 18-44 bucket에 묻힘. ILI 동일 — 더 고운 데이터 없음.

→ home, work 는 **field knowledge prior 로 채움** (φ, γ 와 같은 논리)

---

# 12b. 진단 여정의 정직 — 이전 결론 재평가

방학 발견 (NLL +400K) 이 이전 진단들의 전제를 흔듦. 정직히 재평가:

| 이전 결론 | 재평가 (방학 ON 진단 후) |
|---|---|
| **γ level=0.6 으로 home 살림 (0.27)** | **방학 누락 보정 효과**였음 — γ CDC 로 충분 |
| **work : other = 0.349 NIMS 고정** | 그 시점 일부 정합화 — 지금은 **4 조합 prior 일반화** |
| **channel mix 4 채널 정상화 표** | 방학 누락 환경의 채널 분리 산물 |

**진짜 발견** (재평가 후에도 robust):
- ✅ **winter break** (NLL +400K) — 새 메커니즘
- ✅ **amp 식별** (0.9 선호) — 데이터가 강한 계절성 + 방학 동시 원함
- ✅ **R0 식별** (시즌별 ~2.0)
- ✅ **home/work 구조적 한계 확정** — 4-시즌·방학·재배분 다 실패

<div class="blue center" style="margin-top: 4px">

교훈: misspecification (방학 누락) 이 "가짜 비식별" 처럼 보이게 함 — 진단·해소를 반복하며 자기수정

</div>

---

# 12b1. Fit vs 관측 — A · B 거의 겹침 (시각 증거)

![w:1020](figures/fit_vs_data_AB.png)

**구성**: 4 시즌 × 6 HIRA 연령군. 검정 점 = 관측 HIRA 청구, **파랑 = A (NIMS)**, **빨강 = B (literature)** posterior 중앙값 + 95% CI

**판독**:
- **A 와 B 곡선 거의 겹침** — 모든 연령·시즌에서 데이터 적합 동일
- **★ 채널 가정 (NIMS 접촉 vs 문헌) 이 달라도 fit 동일** = 비식별의 시각적 증거
- peak 시점·형태 양 가정 모두 재현

<div class="blue center" style="margin-top: 4px">

데이터가 채널을 구분하지 못 함 — 둘 다 똑같이 잘 맞음

</div>

---

# 12b2. 왜 채널을 가정으로? — 연령 해상도 한계

![w:1100](figures/age_resolution_limit.png)

**원인**: HIRA · ILI 데이터의 연령군 구조 (6 군)

| HIRA bin | 폭 | 묻히는 채널 신호 |
|---|---|---|
| 0–5, 6–11, 12–17 | 6년씩 | school (0–19) 분리 OK |
| **18–44** | **27 년** | **home peak (30–44) + work peak (25–44) 모두 흡수** ★ |
| 45–64 | 20년 | work peak (45–64) 흡수 |
| 65+ | 10년+ | other / 노인 |

→ 18-44 bin 이 home / work peak 둘 다 포함 → 둘이 같은 bin 안에서 trade-off 만 보이고 분해 불가

**검증**: 단일 시즌·4 시즌·방학·재배분 — **모두 home/work 식별 실패** (NLL 무차이)
**한계**: ILI · HIRA 해상도 동일, 더 고운 한국 인플루엔자 데이터 없음

→ **데이터로 분해 불가** (구조적) → **field knowledge 로 채움** (다음 슬라이드)

---

# 12c. 채널 비율 — field knowledge prior (4 조합)

**두 근거** (R0 기여 기준):

| | home | work | school | other | 비고 |
|---|---|---|---|---|---|
| **A** (NIMS contact) | 0.27 | 0.20 | 0.17 | 0.37 | unit R0 보정 |
| **B** (Italy 2009 H1N1 문헌) | 0.40 | 0.10 | 0.27 | 0.23 | work 3-15% 불확실 |

**4 조합 × 2 강도 fit (4 시즌 평균)**:

| combo | home | work | school | other | R0 | total obj |
|---|---|---|---|---|---|---|
| A_strong | 0.11 | 0.13 | 0.27 | 0.49 | 1.98 | −28.31M |
| A_weak | 0.14 | 0.15 | 0.28 | 0.43 | 1.98 | −28.35M |
| B_strong | 0.15 | 0.10 | 0.28 | 0.47 | 1.98 | −28.32M |
| B_weak | 0.13 | 0.06 | 0.29 | 0.53 | 1.99 | −28.36M |

prior 추가 NLL 손실 **~85K** (multi-start 잡음 수준) — 데이터 강하게 거부 안 함.

---

# 12d. 정책 효과 robust성 — 핵심 메시지

![w:1000](figures/channel_prior_comparison.png)

| 정책 | averted 범위 (4 combos) | 판정 |
|---|---|---|
| **감염 학생 결석** (p_school=0.5) | **98.3% ~ 98.6%** (span 0.3%) | ✅ **강건** (어떤 가정이든) |
| **병가** (p_work=0.4) | **−2.8% ~ +2.2%** (span 5.0%) | ⚠️ 효과 작음 (robust) / **부호 가정 의존** |

<div class="blue center" style="margin-top: 8px">

**학교 정책 → 강한 결론**  /  **병가 정책 → 한국 직장전파 외부 데이터 필요**

</div>

A 근거 (NIMS, work π 높음) → 병가 양수 / B 근거 (문헌, work π 낮음) → 병가 음수 (home spillover 우세)

---

# 12e. ★ 정책 메커니즘 명확화 — 감염자 결석·결근

두 정책 모두 **"증상 있는 감염자가 집에 머문다"** 메커니즘 (대칭):

| 정책 | 의미 | FOI 적용 |
|---|---|---|
| **감염 학생 결석** | 감염 학생만 등교 안 함 | `I_eff_school = p_school × I_student` |
| **병가** (감염 근로자 결근) | 감염 근로자만 출근 안 함 | `I_eff_work = p_work × I_worker` |

**물리적 "학교 폐쇄" 아님**:
- C_school (접촉 구조) **그대로** — 건강 학생 정상 등교
- θ 가 **감염자 (I) 에만** 적용 — FOI 에서 감염자 기여만 차단
- spillover: 결석/결근 감염자 → 가족 노출 (κ × (1−θ))

<div class="blue center" style="margin-top: 8px">

**약한 개입 (감염자만 결석) 으로도 큰 효과** (38–87%) — 더 인상적

</div>

---

# 12f. ★ A vs B Production posterior 비교 (NUTS, 9h 동시 실행)

NB + holiday + amp=0.9 + γ CDC + 4 시즌, 채널 prior weak 각각 A·B

![w:830](figures/policy_posterior_AB.png)

| 정책 | A (NIMS) | B (literature) | CI overlap |
|---|---|---|---|
| **감염 학생 결석** | **+39.7%** [11.7, 74.8] | **+55.2%** [28.5, 83.6] | **78.3%** (강건) |
| **병가** (감염 근로자 결근) | **+9.6%** [+8.0, +10.7] | **−10.9%** [−12.9, −8.7] | **0%** (부호 분리) |

| 지표 | A | B |
|---|---|---|
| 수렴 r_hat | 1.108 | 1.098 |
| ess_min | 31 | 42 |
| divergence | 0 | 0 |
| R0 mean | 1.92 | 1.93 (동일 — 채널 무관) |

---

# 12g. ★ Spillover κ sweep — 가구 격리 수준에 따른 효과

![w:1000](figures/spillover_sweep_both_policies.png)

**κ_scale**: 0 = 완전 가구 격리 (가족 노출 0) / 1.0 = 슬라이드 15 default

| κ_scale | 감염 학생 결석 (A / B) | 병가 (A / B) |
|---|---|---|
| 0.0 (완전격리) | +61% / +87% | +35% / +15% |
| **0.4** | +52% / +78% | **+23% / +2%** (둘 다 양수) |
| 0.6 | +47% / +73% | +18% / **−3%** |
| 1.0 (현재) | +38% / +59% | +9% / **−11%** |

**부호 전환점**:
- 감염 학생 결석: **A·B 둘 다 전 구간 양수** ✅ robust
- 병가: A 전 구간 양수 / B 는 **κ ≈ 0.49 에서 부호 반전**

<div class="green center" style="margin-top: 6px">

**가구 노출 60%+ 감소 (κ≤0.4) → A·B 둘 다 양수** — "병가 + 가구 격리 가이드" 결합 robust

</div>

---

# 12h. 공간 이질성 — 지역 분류 (주간 / 야간 인구비)

KT mobility (시간대별) 로 행정동 분류:

$$
\text{ratio}_j = \frac{\text{주간 인구}_j \;(9{-}17\text{시 노동자 위치})}{\text{야간 인구}_j \;(\text{거주})}
$$

| 분류 | 식별된 지역 (range 0.07 – 21.8) |
|---|---|
| **상업** (ratio > 1.20, 385개) | 종로 명동 21.8, 종로1·2·3·4가 19.9, 중구 19.5 |
| **혼합** (0.41 – 1.20, 384개) | — |
| **주거** (ratio < 0.41, 385개) | 서대문 0.07, 관악 0.08 |

![w:760](figures/spatial_heterogeneity_AB_v2.png)

<div class="small center" style="margin-top: 4px">

진단 교훈: 초기 분류 (순유입 / 인구) 는 dilute 됨 → **주간 / 야간 인구비로 100× 폭 회복**

</div>

---

# 12i. ★ 공간 이질성 결과 — **null (의미 있는)** ★

**per-capita averted % of baseline** (행정동별, 거주지 기준):

| label | region | sick worker absence | sick student absence |
|---|---|---|---|
| A | commercial | +10.07% | +37.19% |
| A | residential | +9.59% | +37.41% |
| B | commercial | −11.02% | +56.12% |
| B | residential | −11.13% | +56.20% |

**commercial / residential ratio**: A: 1.05 / 0.99,  B: 0.99 / 1.00
**commercial − residential**: 0.1 – 0.5pp (정책 의미 없음)

**메커니즘** — `compute_foi_work_jax` backpropagation:
```
직장지(상업) FOI → einsum(M_work, ...) → 거주지(i) S 노출
                                   ↑ 통근으로 거주지에 균질 분배
```

→ 통근 사회: 직장 노출 차단 혜택이 **거주지(전역) 로 분산** → 거주지 기준 효과 균질

---

# 12j. 공간 분석 결론 — null 도 정직한 기여

**공간 모델 (metapop + KT mobility) 로 입증한 것**:

- ✅ 상업 / 주거 정확히 식별 (주간 / 야간 21.8× 차이)
- ✅ 그럼에도 **per-resident 정책 효과는 지역 유형에 무관** (<0.5pp)
- ✅ 약한 robust 방향성 (sick worker absence 상업>주거, A·B 일치) — 크기 미미

**정책 함의**:

| | |
|---|---|
| 균일 정책 시행 | ✅ **정당** (지역 차등 실익 없음) |
| 공간 타겟팅 | ❌ 정책적 의미 < 1% (실행 비용 대비 무가치) |

<div class="blue center" style="margin-top: 6px">

**"공간 모델로 공간 무관성을 입증"** — null 도 정직한 metapop 가치

</div>

---

# 13. 1차 연구 종합

**Calibration** (식별성 + misspec 반복 진단):
- ✅ R0·school·other·계절성·**winter break (+400K)** 식별
- ❌ home·work·γ·φ 구조적 비식별 → **field knowledge prior 4 조합 robust**
- 교훈: misspec(방학 누락) → 가짜 비식별처럼 보임 (γ level / work:other 재평가)

**정책 효과 (감염자 결석·결근)**:
- ✅ 감염 학생 결석: **robust 양수** (4 combos 98%, posterior 38-87%)
- ⚠️ 병가: 가정격리(κ≤0.4) 동반 → robust 양수, 단독 → **B 음수 (가정 의존)**
- ✅ 공간: **null** (KT mobility backpropagation, 균일 정책 정당)

**2차 (향후)**:
- TODO: 한국 직장 전파 비중 외부 추정 (병가 부호 확정)
- TODO: HIRA 외래/입원 split, KDCA serosurvey
- ICER + PSA (외부 단가 확보 후)

---

<!-- _class: lead -->
<!-- _paginate: false -->

# 14. 요약 — 1차 연구 완결

- **비식별 · misspec 반복 진단** → 데이터가 보는 것 추정 / 못 보는 것 외부 근거
- **진짜 모델 개선**: R(0) 계단 + **winter break** (NLL +400K)
- **데이터 식별**: R0, school, other, 계절성 (amp ~ 0.9), 방학 효과
- **데이터 한계** (HIRA 구조): home, work — **field knowledge prior 4 조합 robust**
- **정책 결론**:
  · 감염 학생 결석: **38-87% robust** (모든 가정 일관)
  · 병가: **가정격리 (κ≤0.4) 동반 시 robust**, 단독 시 가정 의존
  · 공간: **null** (KT mobility backpropagation, 균일 시행 정당)
- **2차**: ICER, 한국 직장전파 비중, 연령구조 공간 (외부 데이터 후)


