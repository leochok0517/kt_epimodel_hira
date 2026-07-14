# 연령별 감수성 φ(age-specific susceptibility) 문헌 근거 정리

> 목적: kt_epimodel_hira 의 φ(연령별 감수성)를 데이터로 추정하지 않고 문헌 기반
> 물리적 값에 고정할 때, "이 값이 어디서 왔는가"에 대한 방어 근거를 정리한다.
> φ 는 FOI 에서 순수 **감수성**(접촉 1건당 감염될 확률의 연령 상대비)이며,
> 접촉(C_channel)·감염성(infectiousness)·면역(R(0))과는 분리된 개념이다.

## 0. 왜 φ 를 데이터로 추정하지 않고 고정하는가

- 본 모델에서 φ 는 구조적으로 비식별(non-identifiable)임이 진단으로 확인됨:
  free 추정 시 NUTS r_hat 3~4.5, multimodal posterior, 15-19 등 경계 연령이
  0.1 하한으로 붕괴. 채널(π) pin 후에도 φ 비식별은 독립적으로 잔존.
- 데이터가 φ 를 정하지 못하므로(비식별), γ_report·κ 를 외부 고정한 것과 동일한
  논리로 φ 를 문헌 기반 값에 고정하는 것이 정당.
- 문헌 근거: 접촉 패턴과 사전면역만으로는 연령별 감염 위험을 설명하지 못하며,
  별도의 age-specific susceptibility 항이 필요함이 반복 확인됨
  (Merler/Italy H1N1; Cauchemez 2014 — 소아는 사전 HI 항체가 높아도 성인보다
  감수성이 높음, 즉 측정된 면역만으로 설명 안 됨).

## 1. 핵심 정량 근거 — 소아 감수성 배수 (성인 기준)

| 출처 | 지역/균주 | 소아:성인 감수성비 | 비고 |
|---|---|---|---|
| Cauchemez et al. 2009, NEJM (household) | 미국 / 2009 H1N1 | **1.96배** (95% CI 1.05–3.78), ≤18 vs 19–50세 | 표준 참조, 정책 타겟팅에 유용하다 명시 |
| Cauchemez et al. 2023, PNAS (household, 홍콩) | 홍콩 / 2009 H1N1 | **3.2배** (95% CrI 2.8–3.7) | 가장 정밀. 지역사회 감염확률 소아33% vs 성인12% |
| Jayasundara et al. (메타분석, via IRV 2018) | 다국 | ~**4.3배** (attack rate 15.2% vs 3.5%) | attack rate 기반(감수성 상한 추정) |
| Vietnam household cohort (PLoS Pathog 2014) | 베트남 | **2배+** (≤14 vs ≥15세) | HI 항체 보정 후에도 유지 |
| 리뷰(HH transmission, PMC4733423) | 종합 | "younger age = higher susceptibility" | 정성적 합의 |

→ **합의: 소아 φ ≈ 성인의 2–3배.** 보수적 하한 2.0(Cauchemez US),
   정밀 추정 3.2(Cauchemez HK). 본 모델은 계절 인플루엔자이므로 중간값(≈2.0–2.5) 권장.

## 2. 감수성 vs 감염성 — φ 에 넣지 말아야 할 것

- 소아는 감수성은 높지만 **감염성(infectiousness)은 성인과 비슷**함
  (Cauchemez 2004/2009/2014; PLoS Pathog 2014 Vietnam: "children are more
  susceptible but as infectious as adults").
- Cauchemez 2009: 감염 소아의 감염성이 성인보다 높다는 별도 추정(0.48 vs 0.26/일)도
  있으나, 이는 **infectivity**이며 본 모델 φ(susceptibility)와 다른 항.
- **결론: φ 에는 "감수성" 배수만 반영. 감염성 차이는 φ 로 넣지 않는다.**

## 3. 노인 — 균주 의존적, 주의 요함

| 맥락 | 노인 감수성 방향 | 근거 |
|---|---|---|
| 계절 인플루엔자 | **높음** (면역노화 immunosenescence) | 노인 접촉당 전파확률 3배+ (PMC2876165) |
| 2009 H1N1 pandemic | **낮음** (과거 유사주 교차면역) | Italy H1N1 (PMC3792117); 낮은 노인 attack rate |

- 본 모델 데이터: **계절 인플루엔자 2017–2020 (+2022-23)** → 노인 감수성 **높은** 방향.
- ⚠️ 역할 분담 주의: 본 모델은 이미 R0_IMMUNITY(65+ = 0.65)로 노인 사전면역을
  반영. 노인의 "면역으로 인한 방어"는 R(0)이 담당하고, φ 는 **순수 감수성**만
  담당해야 이중 계산을 피함. → 노인 φ 는 과도하게 높이지 말 것(≈1.3–1.5 수준).

## 4. 한국 지역 정합성 근거

| 출처 | 내용 | 본 모델과의 관계 |
|---|---|---|
| 한국 Omicron susceptibility (PMC9684890) | 한국 접촉행렬+역학조사+Bayesian 으로 age-specific FOI·susceptibility 직접 추정 | **방법론 동일**(연령구조+접촉행렬+Bayesian). 한국에서 이 접근의 선례 |
| 한국 인플루엔자 R0 (BMC 2021, PMC8265026) | 한국 H1N1 3연령군 전파율, R0 1.4–1.6, 감염기간 3.8일 | γ=0.25(4일), R0≈2.0 과 정합 |
| 한국 2009 대응평가 (PMC4064639) | 한국 인플루엔자 3연령군(0-19/20-64/65+) 모델 | 연령 그룹핑 선례 |
| 한국 접촉조사 2023-24 (NIMS, Son/Lee et al.) | 한국 접촉행렬 실측, 노인 접촉률 유독 높음 | C_channel 근거. φ 아님(접촉≠감수성) |

## 5. φ 15군 구성 원칙 (권장)

- **형태**: U자 (소아 높음 → 청·장년 최저 → 노인 다시 높음).
- **기준(anchor)**: 25–29세 = 1.0 (현 코드 anchor idx 5, 청년 성인).
- **소아(0–14)**: 성인의 2.0–2.5배. 가장 어린 0–4 를 정점으로 완만 감소.
- **청소년(15–19)**: 소아와 성인 사이 전이 (경계, 비식별 심한 구간 → 이웃 보간).
- **성인(20–64)**: 최저 수준(0.7–1.0), 완만.
- **노인(65+)**: 1.3–1.5 (계절 인플루엔자, 단 R(0) 면역과 중복 피해 과도 금지).
- **매끄러움**: 5세 단위 15군을 급변 없이 연속적으로(단일 점 튐 방지).

## 6. 핵심 인용 (방어용 primary refs)

1. Cauchemez S, et al. Household transmission of 2009 pandemic influenza A (H1N1)
   virus in the United States. N Engl J Med. 2009;361(27):2619–27.
   → 소아 상대감수성 1.96; 연령별 감수성의 정책 활용 명시.
2. Cauchemez S, et al. Reconstructing household transmission dynamics ... PNAS.
   2023;120(33):e2304750120. → 소아 3.2배; 지역사회 감염확률 소아33%/성인12%.
3. (한국) Age-specific susceptibility, Korea, Bayesian + contact matrix
   (PMC9684890). → 한국에서 동일 방법론 선례.
4. (노인/계절) 노인 감수성·전파 근거 (PMC2876165); Italy H1N1 대비 (PMC3792117).
5. (감수성≠감염성) Vietnam HH cohort, PLoS Pathog 2014.
