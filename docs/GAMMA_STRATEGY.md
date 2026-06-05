# γ (Reporting / Detection Rate) Strategy

확정 결정: **γ 는 추정하지 않고 외부 출처에서 고정**. PSA 로 불확실성 처리.

## §1. 결론 (확정)

- **γ 는 β·φ·γ 곱셈 비식별** → MCMC posterior 가 γ→0 로 끌림 (M1d / Phase 2
  smoke 진단 확인)
- L-BFGS 점추정은 init valley 에 머무를 뿐, posterior 의 진짜 mode 와 다름
- 비식별은 prior 만으로 부분 차단 (강한 σ 필요) — 정공법은 γ 자체를 fit
  대상에서 빼는 것
- **default**: CDC Reed et al. 2015 (`gamma_child=0.40, gamma_adult=0.18,
  gamma_elder=0.25`)
- 불확실성: Stage 5 PSA 에서 γ 흔들기 (`gamma_psa_samples()`)

연결 문서:
- [`docs/PRIOR_SPECIFICATION.md`](PRIOR_SPECIFICATION.md) §A.5+ — NUTS 진단 기록
- [`docs/AGE_DEPENDENT_GAMMA.md`](AGE_DEPENDENT_GAMMA.md) §4 — CDC multiplier 역산
- [`src/kt_epimodel_hira/calibration/gamma_registry.py`](../src/kt_epimodel_hira/calibration/gamma_registry.py) — 구현

## §2. 선행연구 대비 (Jung et al. 2025)

| 항목 | Jung 2025 | 본 연구 |
|---|---|---|
| γ 처리 | 단일 상수 q=0.67 (증상비율) 로 고정. 연령별 미도입 | 연령별 (child/adult/elder), 외부 출처 명시 |
| 식별성 진단 | 명시 없음 (사실상 γ 추정 회피로 우회) | NUTS 로 비식별 명시 측정 + 외부 고정 결정 |
| 미보고 처리 | 초기 면역 (L2/L3) 에선 미보고 인정하나 fit 단계에선 무시 | 동일 비대칭 회피 (γ 고정으로 일관) |
| 불확실성 | 단일점 (PSA 명시 없음) | PSA sweeping (Stage 5) |

> 시사: "reporting 은 추정 말고 고정" 은 선행연구도 (암묵적) 실천 →
> 본 연구의 γ 고정 결정을 뒷받침. 본 연구는 이를 **연령별로 정교화** + **비식별
> 진단을 투명하게 기록** + **교체 가능한 구조**로 향후 한국 데이터 흡수 가능.

## §3. γ 교체 경로 (★ 데이터 생길 때)

### 우선순위
1. **한국 자체 측정 데이터** (선배 연구원 데이터 등 — 성격 미정)
2. **한국 문헌** (HIMM 2013-14 성인 등 부분 보정)
3. **CDC default** (현 상태)

### 절차 (3 단계)

```text
[데이터 입수]
   ↓ (성격에 맞는 adapter 호출)
[gamma_adapters.gamma_from_serosurvey(...)
 또는 gamma_from_cohort_attack_rate(...)
 또는 gamma_from_test_positivity(...)
 또는 gamma_direct(child, adult, elder)]
   ↓ (returns {"child": ..., "adult": ..., "elder": ...})
[GAMMA_REGISTRY 에 GammaSource 등록]
   ↓
[ACTIVE_GAMMA = "new_key" 한 줄 변경]
   ↓
[모델/fit 재실행 — 구조 무수정]
```

### 어댑터 인터페이스 (현재 stub)

- `gamma_from_serosurvey(hira_rate, sero_infection_rate)` —
  paired serosurvey. γ = HIRA 청구율 / 항체 양성률.
- `gamma_from_cohort_attack_rate(hira_rate, cohort_attack_rate)` —
  종단 코호트 추적. serosurvey 와 동일 원리.
- `gamma_from_test_positivity(claims, confirmed, total)` —
  검사 양성률 보정 (부분적).
- `gamma_direct(child, adult, elder)` —
  이미 γ 값 형태로 주어진 경우.

각 함수는 데이터 확보 전까지 `NotImplementedError`. 데이터 성격 확정 시
해당 함수만 채우면 됨.

## §4. 한계 + 향후 작업

1. **현재 CDC (US) default**: 한국 직접 측정값 부재 — limitation 명시
2. **한국 시즌별·전연령·수도권 serosurvey 공개 데이터 없음**
   (KDCA 비공개 또는 부분 공개)
3. **모델 노인 incidence 과대 흔적 가능성**: PRIOR_SPEC §A.6 참조.
   γ_elder=0.25 가 "순수 reporting + 모델 잔존 misspecification" 의
   효과적 흡수일 수 있음
4. **선배 데이터 확보 시 정밀화**: 구조는 준비됨 (registry + adapter)

## §5. PSA (Stage 5)

```python
from kt_epimodel_hira.calibration.gamma_registry import gamma_psa_samples

# n_samples × 15 array of γ vectors
gammas = gamma_psa_samples(n_samples=1000, rng=np.random.default_rng(0))
# 각 sample 로 posterior predictive forward sim → ICER credible interval
```

- 활성 source 의 `psa_sd` (default child=0.07, adult=0.05, elder=0.07) 적용
- TruncatedNormal in [0.01, 0.99] per group
- ICER 신뢰구간 + 정책 권고의 robustness 측정

## §6. 본 연구 fit 워크플로우 (γ 고정 후)

```
[ACTIVE_GAMMA = "cdc_reed2015"]
   ↓
[numpyro NUTS — phi(14) + beta(16) 만 sample]
   → posterior of φ, β (γ 는 deterministic constant)
   ↓
[Stage 4 정책 시나리오 forward sim with γ_active]
   ↓
[Stage 5 PSA — γ_psa_samples + φ/β posterior 곱셈]
   → ICER credible interval
```
