# 09 — θ (other-channel differential) 재구현 + θ·κ 스윕

`prompts/99_conventions.md` 를 함께 적용할 것.

## 상황

main_v5 에는 θ 가 **현역**이다 — Methods `sec:theta_methods`, Results
`sec:theta`, Table `tab:theta`, Limitations 의 "40% 차등" 주장, 415 행 연령
재분배 강건성 논증이 전부 θ 에 의존한다. **그런데 이 저장소에 구현이 없다.**
`_foi_other_presymp` 는 `β·C·(w·I₁ + I₂₃)/N` 뿐이고 `(1−θΔ_b)` 항이 없다.
`git log -S` 로도 안 잡히고 삭제된 파일도 없다.

### 이름이 겹치는 셋을 먼저 구분할 것

| 기호 | 정체 | 상태 |
|---|---|---|
| **η (겹침효율)** | κ 에 곱하던 승수 (κ_체류 × η, η∈[0.75,0.86]) | **이미 제거됨.** κ 를 직접 스윕하면서 불필요해졌다. `scripts/kappa_no_eta*.py` 가 그 흔적이고 논문 245 행에 반영돼 있다. **되살리지 말 것, 잔재 찾지 말 것 — 이미 깨끗하다.** |
| **η_b (고용률)** | Eq `eq:increment` 의 η_b. 코드의 `rho` | 현역, 그대로 사용 |
| **θ** | other 채널 differential. κ 와 **무관** | 현역이나 구현 없음 → 이번 작업 대상 |

## 사양 (main_v5 153 행 · Eq `eq:increment`)

other 채널 FOI 의 유증상 블록만 `(1 − θ·Δ_b)` 로 스케일한다. presymptomatic
블록 `w·I₁` 은 영향 없다:

```
λ_other,a = φ_a · s_f(t) · β_other · Σ_b C_other,ab(t) · [ w·I₁,b + (1 − θ·Δ_b)·I₂₃,b ] / N_b
```

Δ_b (policy-attributable increment):

```
Δ_b = p_base_school − p_school            (학생)
    = η_b · (p_base_work − p_work)        (근로자, η_b = 고용률 = rho)
    = 0                                    (70+)
```

**새 파라미터가 필요 없다.** 기존 `compute_phi_spillover(p_school, p_work, rho)`
가 `(1−p)_b` 를 주므로

```python
delta_b = compute_phi_spillover(p_school, p_work, rho) \
        - compute_phi_spillover(p_base_school, p_base_work, rho)
```

가 정확히 Δ_b 다 ( `(1−p) − (1−p_base) = p_base − p` ). 이미 `p_school`,
`p_work`, `rho`, `policy_school_baseline`, `policy_work_baseline` 이 전부
`compute_derivatives_erlang_presymp` 의 kwargs 이므로 배선만 하면 된다.

## 구현

**`erlang_presymp.py` · `foi_jax.py` 수정 금지.** 신규
`src/kt_epimodel_hira/jax_model/erlang_presymp_theta.py` 를 만들되:

- 바뀌는 함수만 재정의한다 — `_foi_other_presymp` 와 그것을 호출하는
  FOI 조립부. **나머지는 전부 원본에서 import** 할 것. 통째 복사는 drift 를
  부르므로 금지.
- `theta` 는 기본값 0.0 의 kwarg 로 받는다.

## 검수 — 과학 하기 전에 반드시 통과시킬 것

### T1. θ=0 동등성

같은 입력에서 `erlang_presymp_theta` (θ=0) 와 원본 `erlang_presymp` 의 궤적이
수치적으로 동일해야 한다 (`np.allclose`, rtol 1e-10). 어긋나면 배선 오류다.

### T2. baseline 불변성

`p = p_base` (정책 없음) 이면 Δ_b ≡ 0 이므로 other 항이 1 이 되어 **baseline
유행이 θ 전 구간에서 불변**이어야 한다. 논문 346 행이 명시적으로 주장하는
성질이다. θ ∈ {−0.25, 0, 0.75} 에서 확인.

### T3. ★ v5 표 재현 (합격 판정선)

**v5 조건 그대로** 재현할 것 — A+B 타깃, v5 3 시즌 (2016-17, 2017-18, 2019-20),
v5 사후 `outputs/eda/nuts_seasonpop_merged.npz`, 경험적 κ, p=0.4.
main_v5 `sec:theta` 의 값과 대조:

| θ | 2016-17 | 2017-18 | 2019-20 |
|---|---|---|---|
| 0 | +0.46% | +0.23% | +0.40% |
| 0.75 | +1.07% | +1.04% | +1.07% |
| CI 가 0 제외하는 θ* | −0.04 | **+0.39** | +0.09 |

**재현되면** 구현이 옳다 → 다음 단계로.
**재현 안 되면 거기서 멈추고 보고할 것.** 논문 수치의 출처가 불명이라는 뜻이고,
그건 재구현 실패보다 큰 문제다. 임의로 맞추려 들지 말 것.

## 본 실행 — fluA4 사후로

### Stage A — θ 스윕

`outputs/eda/nuts_fluA4_raw.npz`, 4 시즌, θ ∈ [−0.25, 0.75] (0.05 간격 권장,
최소 −0.25/0/0.25/0.5/0.75), N=500 draw, p=0.4, 경험적 κ.

기록: 시즌별 총 averted (병가·학생결석 × 학기·방학), **연령별 Δattack
(학기·방학)**, CI 가 0 을 제외하는 θ*.

### Stage B — κ 스윕

κ_성인 ∈ {0.20, 0.34, 0.40, 0.50, 0.70, 1.00} (논문과 동일), θ=0.
main_v5 415 행의 **"방학 중 아동 쪽 이동이 750 draw 중 749 에서 양수"** 를
fluA4 에서 다시 세어 보고할 것. 지금 살아남은 유일한 재분배 결론이다.

**재피팅 금지** — 두 스윕 모두 사후 draw 재사용 forward 계산이다.
`jax.jit` 필수 (`prompts/99_conventions.md` 의 54GB 사고 참조).

## 이번에 답해야 하는 것

fluA4 에서 **학기 중 연령 재분배가 사라졌다** (아동 값이 0 이거나 음수, 유의한
것 없음). 방학은 유지된다. v5 는 학기에도 아동 양수·유의였다.

1. **음의 θ 에서 학기 성분이 되살아나는가?** 되살아나면 "학기 효과 없음" 은
   θ=0 가정의 산물이고, 안 되살아나면 θ 와 무관한 robust 한 결과다.
2. 방학 성분은 θ 전 구간에서 유지되는가? (v5 의 핵심 주장)
3. 새 θ* 는 얼마인가? v5 의 2017-18 θ*=+0.39 가 Limitations "40% 차등" 문장의
   근거인데, fluA4 는 θ=0 에서 이미 3/4 시즌이 0 을 제외한다 → 그 문장이
   유지되는지 폐기되는지 판정할 것.

## 산출물

- `src/kt_epimodel_hira/jax_model/erlang_presymp_theta.py`
- `tests/test_theta_equivalence.py` — T1·T2 를 pytest 로
- `outputs/fluA/theta_v5_reproduction.json` — T3 결과 (v5 값과 나란히)
- `outputs/fluA/sens_theta_fluA4.json` / `.png`
- `outputs/fluA/sens_kappa_fluA4.json` / `.png`

## 보고

1. T1·T2·T3 통과 여부 — **T3 실패면 여기까지만 보고**
2. 위 "답해야 하는 것" 3 개
3. main_v5 에서 수정이 필요한 문장 목록 (해당 절·행 번호와 함께)
