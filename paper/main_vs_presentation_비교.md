# main_v3 (논문) vs presentation (발표) — 맥락 비교

> 목적: 두 문서의 전체 논조·강조점을 대조하고, 특히 **Limitations**에서
> 발표본에만 있는(= 저자가 말하고 싶은) 내용을 논문에 반영할지 판단할 근거를
> 정리한다. **어느 파일도 수정하지 않음.**

---

## 1. 한눈에

| | **main_v3.tex** (논문, 25쪽) | **presentation.tex** (발표, 32 본문 슬라이드) |
|---|---|---|
| 성격 | 완성된 학술 원고 | 구두 발표용 요약 |
| 구조 | Abstract → Intro → Model Description → Parameters → Methods → Results(7절) → Discussion(7절) → Conclusion | Background → Model assumptions → Model & methods → Results → Discussion |
| 논조 | 학술적·유보적(hedged), 문단 서술 | 직접적·단정적 bullet, 슬라이드 |
| 최신성 | **재생산수 분해 + 치명률 overlay 포함** | 이 둘은 **없음**(발표본이 더 이전 맥락) |

핵심 메시지는 두 문서가 동일하다: **"학교결석은 효과적·확실, 병가는 작고 불확실"**,
그리고 **"효과크기 하나가 아니라 누구에게·얼마나 확실히·언제로 평가하라"**.
차이는 강조점과 세부 caveat에 있다.

---

## 2. 내용 커버리지 차이

### 논문(main_v3)에만 있고 발표에 없는 것
- **재생산수 분해**(채널 R_c / 연령 R_b / 시점별 R_e(t)) — 발표본 작성 이후 추가됨.
- **치명률 overlay**(연령별 IFR 곱, "참고용" caveat) — Discussion 정책함의 뒤.
- **NUTS posterior·π_work 식별** 상세, **방법론적 기여** 절(presymptomatic 표현,
  C(t) 분리, 식별성).
- 여러 방법론적 한계: 보고분율 US 기반, presenteeism 일반질환 데이터, 인구
  robustness, 유행폭 과대예측·0–5 under-fit.

### 발표(presentation)에만 있거나 더 강조된 것
- **첫 슬라이드 제목 "The evidence for sick leave is limited"** — 증거 부족을
  전면에 세움(논문은 Intro에서 서술로 완만하게).
- **κ vs Ferguson** 대비가 sensitivity 슬라이드에서 더 전면적:
  "우리 추정(0.34–0.40)은 threshold(0.51) 아래라 net-beneficial, **그러나
  Ferguson류 +50–100%면 net-harmful을 배제할 수 없다**". (논문에도 있으나
  Results/Discussion에 흩어져 있고 발표만큼 또렷하지 않음.)
- **Limitations 2가지가 논문에 없음** → 아래 3절.

---

## 3. Limitations 상세 비교 (핵심)

### 발표 Limitations (5개, 슬라이드)
1. presymptomatic weight는 인플루엔자 A 기반(B는 없음 → 보수적).
2. 밀집 κ는 time-use 자료에서 도출(아픈 사람 접촉일기 아님); **κ와 baseline
   retention p 둘 다 sweep**.
3. **★ 접촉조사가 설 연휴 근처(평일)에 수행 → 접촉행렬에 연휴 편향 가능성.**
4. **★ 결석을 전체 infectious 기간으로 모델링 → 실제 결석 기간과 다를 수 있음.**
5. 병가 효과가 0 근처라 정확한 크기·카운트 비율이 부정확 → 방향·확실성을 강조.

### 논문(main_v3) Limitations (문단, 8개 요소)
- **병가 순효과의 통계적 불확실성**(2/3 시즌 0 포함)을 "가장 중요"로 먼저.
- 방학중 심화는 방향이 아니라 **크기 변화**(부호 안 뒤집힘).
- **school-vs-sick 비율의 가정 민감성**(baseline symmetry, immunity, κ,
  presymptomatic weight).
- presymptomatic A 기반(보수적), w sweep.
- κ 정의적 선택 → μ sweep.
- workplace 전파는 실용적 식별만.
- **보고분율 US 기반·presenteeism 일반질환 데이터·인구 robustness**.
- **유행폭 과대예측·0–5 under-fit**.

### 겹침 / 차이 정리

| 항목 | 발표 | 논문 |
|---|---|---|
| presymptomatic A 기반(보수적) | ✔ | ✔ |
| κ 도출 방식 + sweep | ✔ (p도 sweep 명시) | ✔ (μ sweep; p sweep은 Methods) |
| 병가 0 근처 → 방향·확실성 강조 | ✔ | ✔ (첫 문장) |
| **설 연휴 접촉조사 편향** | **✔ (발표만)** | ✘ 없음 |
| **결석기간 = 전체 infectious 가정** | **✔ (발표만)** | ✘ 없음 |
| 비율의 가정 민감성 | (sensitivity 슬라이드로) | ✔ |
| 보고분율/presenteeism/인구 robustness | ✘ | ✔ |
| 유행폭 과대예측·0–5 under-fit | ✘ | ✔ |

> **결론적으로 발표본에만 있는 실질적 한계는 두 가지다:**
> **(1) 설 연휴 근처 접촉조사로 인한 접촉행렬 편향 가능성,**
> **(2) 결석을 전체 감염기간으로 잡은 모델링 가정.**
> 저자가 "발표 쪽이 내가 말하고 싶은 맥락"이라 느끼는 지점이 이 둘일 가능성이 큼.
> 둘 다 데이터·모델 구성의 정직한 한계이고 논문에 넣어도 자연스럽다.

---

## 4. 논조/프레이밍 차이

- **병가에 대한 태도**: 두 문서 모두 "해롭다고 단정하지 않음 + 작고 불확실".
  발표는 첫 슬라이드에서 **"증거 부족"**을 전면화해 조금 더 회의적 톤,
  논문은 "certainty의 대비"로 프레이밍.
- **presymptomatic의 역할**: 두 문서 모두 최종적으로 병가가 작은 **주원인은
  household spillover**로 귀속(이전 수정 반영). presymptomatic은 "일반적
  ceiling"으로만 유지 — 발표의 model-assumptions 슬라이드와 논문 methodological
  contribution이 서로 대응.
- **κ 위험(net-harmful 가능성)**: 발표가 더 또렷하게 "배제할 수 없다"고 말함.
  논문은 같은 내용을 더 완만하게 서술.

---

## 5. 제안 (반영 판단용)

논문(main_v3)에 발표의 맥락을 더 담고 싶다면, 우선순위:

1. **설 연휴 접촉조사 편향** → 논문 Limitations에 한 문장 추가(접촉행렬 대표성).
2. **결석기간 = 전체 감염기간 가정** → 논문 Limitations에 한 문장 추가
   (정책 강도 해석과 직결되는 정직한 한계).
3. (선택) **κ vs Ferguson "net-harmful 배제 불가"**를 Discussion에서 한 단계
   더 전면화 — 발표 수준의 회의적 톤을 원하면.

반대로, 발표에는 없지만 논문에 있는 재생산수 분해·치명률 overlay·유행폭/보고분율
한계 등은 발표를 업데이트할 때 반영 후보다.

*(본 문서는 비교용이며 어떤 원본도 수정하지 않았습니다.)*
