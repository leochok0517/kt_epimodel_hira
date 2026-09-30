# 15. v8 — 공저자 피드백 중 계산이 필요한 것

작성 2026-09-29. 회사 Mac 전용 (JAX + `outputs/eda/nuts_fluA4_raw.npz`). **재적합은 없다.**
전부 기존 4시즌 사후분포 위에서 전방 시뮬레이션만 돌린다. 합쳐서 1시간 안쪽으로 본다.

---

## 0. 맥락

공저자가 `main_v6` 주석본을 보냈다 (`sickleave_InfluenzaA/manuscript/COAUTHOR_FEEDBACK_v6.md`,
14건). 표현·출처 수정 9건은 이미 `main_v8.tex` 에 반영했다. 남은 것 중 계산이 필요한 것이
아래 둘이고, 그 과정에서 **원고가 존재하지 않는 결과를 인용하고 있는 것**이 하나 드러났다.

### 0.1 Figure 3 (공저자 #14)

> credible interval 이 있어야 하는데 바뀌지 않았다. axis tick label 이 바뀌지 않았다.
> 0-50 의 Epidemic week 라 36주 시작이라는 점이 드러나지 않는다.

`outputs/fluA/fit_total_curves_fluA4.json` 에는 시즌별 **사후 평균 곡선과 관측점만** 있다
(`fig_fit_total_fluA4.py` 가 N=60 subsample 의 평균만 저장). 분위수가 없으니 띠를 못 그린다.

### 0.2 존재하지 않는 민감도 — 이게 더 급하다

`main_v8.tex` §4.5 (Methods, Sensitivity analysis) 는 이렇게 적는다.

> we test the robustness ... while varying the presymptomatic weight $w$, the crowding
> coefficient $\kappa$, the workplace share $\pi_{\mathrm{work}}$, adult initial immunity,
> the symmetric baseline retention $p$, and the number of Erlang sub-stages.
> The analyses of $w$, $\kappa$, and the Erlang sub-stage count reuse the posterior draws
> and re-derive $\beta$ without re-fitting.

그리고 §5.6 (Results, Robustness) 는

> These conclusions held when we varied the presymptomatic weight $w$, adult initial
> immunity (Supplement, Section~S9), the symmetric baseline retention $p$, and the
> Erlang shape, with no sign reversal in any age group.

**그런데 `w`, `p`, Erlang 단계 수의 fluA4 산출물이 없다.** 있는 것은

```
outputs/eda/sens_w_sweep_seasonpop.json        ← v5, 전 인플루엔자, 3시즌
outputs/eda/sens_erlang_n_seasonpop.json       ← v5
outputs/eda/sens_baseline_p_seasonpop.json     ← v5
```

뿐이다. fluA4 로 포팅된 민감도는 `outputs/fluA/sens_{kappa,kappa_upper,piwork_kappa,r0_immunity,theta*}_fluA4.json`
다섯 종류이고, `w`·`p`·Erlang 은 빠졌다. v6 개정 때 §4.5·§5.6 문장이 v5 에서 그대로 넘어온 것이다.

이건 공저자 #7·#11 (Erlang(3) 의 근거가 없다)에 대한 답과도 직결된다. "n 을 바꿔도
결론이 유지된다"고 답하려면 **fluA4 사후분포에서 돌린 결과**가 있어야 한다.

---

## 1. 먼저 읽을 것

- `scripts/sens4_w_sweep.py`, `scripts/sens6_erlang_n.py`, `scripts/sens5_baseline_p.py` —
  v5 원본. **수정 금지.** 설계만 가져온다.
- `scripts/sens_kappa_fluA4.py` — v5 민감도를 fluA4 로 포팅한 **표준 패턴**. 새 스크립트는
  이것과 같은 구조로 쓴다 (`fluA_setup`, `nuts_fluA4_raw.npz`, `jax.vmap` over draws,
  `jax.jit`, 결과는 `outputs/fluA/*_fluA4.json`).
- `scripts/fig_fit_total_fluA4.py` — Figure 3 의 현재 생성 경로.
- `prompts/99_conventions.md`.

v5 산출 JSON 의 `meta` 를 보면 설계가 나온다.

| v5 파일 | 격자 | draw | 비고 |
|---|---|---|---|
| `sens_w_sweep_seasonpop` | w ∈ {0.05, 0.08, …, 0.45} (15점) | 50 | |
| `sens_erlang_n_seasonpop` | n ∈ {1, 2, 3, 5} | 50 | **w 를 n 에 따라 바꿈** — `presymp_fraction=0.1` 을 고정하기 위해 `w_by_n = {1: 1.0, 2: 0.111, 3: 0.222, 5: 0.444}` |
| `sens_baseline_p_seasonpop` | (p_work, p_school) 6조합 | — | 점적합. `pi_work_pin=0.29` |

---

## 2. 할 일 A — 민감도 세 개를 fluA4 로 포팅

신규 파일 셋. 4시즌 전부, 결과는 `outputs/fluA/` 에.

```
scripts/sens4_w_sweep_fluA4.py      → outputs/fluA/sens_w_sweep_fluA4.json
scripts/sens6_erlang_n_fluA4.py     → outputs/fluA/sens_erlang_n_fluA4.json
scripts/sens5_baseline_p_fluA4.py   → outputs/fluA/sens_baseline_p_fluA4.json
```

**격자는 v5 와 같게.** 비교 가능해야 한다. 단 Erlang 의 `w_by_n` 규칙은 그대로 따르되,
**n=3 행이 주 분석(w=0.22)과 정확히 같은 설정인지 확인**하고 JSON `meta` 에 적을 것.
v5 의 `w_by_n[3] = 0.222` 와 주 분석의 `W` 상수가 같은 값인지 `nuts_seasonpop.W` 로 대조.

**보고할 지표** — §5.6 의 문장이 요구하는 것에 맞춘다.

1. 시즌·연령대·기간(school-term / winter-break)별 sick-leave Δattack 의 **부호**와, 격자
   전 구간에서 부호 반전이 있는지. §5.6 의 `no sign reversal in any age group` 이 fluA4 에서
   성립하는지가 핵심 질문이다.
2. 시즌별 총 averted % (sick, school) 의 격자별 값.
3. Erlang 에 대해서는 추가로 **n 별 함의된 평균 generation interval** 을 `meta` 에 넣을 것.
   지수 잠복기 2.0 일 + Erlang(n) 감염기 4.0 일에서 단계 k 의 전파는 평균 k·(4/n) 일에
   일어나고, 단계별 가중치 (w, 1, …, 1) 로 평균낸다. n=3, w=0.22 에서 5.14 일이 나와야 한다.
   (공저자 #7 의 "generation interval 과 어떤 관계인지"에 대한 답이다.)

**p 민감도는 점적합**이라 시간이 좀 든다. v5 와 같이 `pi_work` 를 고정하고 한다.
고정값은 v5 의 0.29 가 아니라 **fluA4 사후 median 0.314** 를 쓸 것 (`outputs/fluA/per_season_pi_fluA4.json` 의 `shared.pi.work.median`). `meta` 에 명시.

**결과가 v5 와 달라도 놀라지 말 것.** 2018-19 가 추가됐고 타깃이 influenza A 다.
부호 반전이 나오면 그건 발견이지 오류가 아니다 — 멈추지 말고 그대로 보고.

---

## 3. 할 일 B — Figure 3 에 credible interval 과 ISO 주 축

`fig_fit_total_fluA4.py` 를 **수정하지 말고** 새 파일로.

```
scripts/fig_fit_total_ci_fluA4.py
  → outputs/fluA/fit_total_curves_fluA4.json   (기존 키 유지 + q05/q95/q25/q75 추가)
  → figures/paper_fluA4/{pdf,png}/fit_total.{pdf,png}   (덮어씀)
```

1. 시즌별로 `nuts_fluA4_raw.npz` 에서 **N=500 draw** (60 이 아니라) 를 뽑아 전방 시뮬레이션,
   주간 총 claim 곡선의 q05/q95 (90% CI) 와 q25/q75 를 계산. JSON 의 기존 `mean`·`obs` 키는
   그대로 두고 분위수 키를 **추가**한다 (`_fig_Re_readable_fluA4.py` 가 `mean` 을 읽는다).
2. 그림: 관측점, 사후 평균 선(검정 — 이전에 시즌별 색을 검정으로 통일했다), **90% CI 띠**
   (`fill_between`, 회색 alpha 0.25). 범례에 `90% credible interval` 명시.
3. **x 축.** 지금은 `Epidemic week` 0~50. 주 0 이 ISO 36주라는 것이 안 보인다. 눈금을
   ISO 주 번호로 찍을 것 — 36, 40, 44, 48, 52, 4, 8, 12, 16, 20 처럼 4주 간격, 연도가 바뀌는
   52→1 을 넘어간다. 축 라벨은 `ISO week` 또는 `Week (ISO, from week 36)`. 라벨 영어.
4. 4시즌 2×2 배치와 크기는 기존과 같게. 파일명 `fit_total` **그대로** (원고가 그 이름을 집는다).

그린 뒤 `~/Documents/cowork/sickleave_InfluenzaA/manuscript/figures_fluA4/{pdf,png}/` 로
복사. 안 하면 원고가 못 집는다.

---

## 4. 할 일 C (선택, 여력 있으면)

- `prompts/14` §2 — 공통 φ 에서의 NLL 정리. 원고가 인용하지 않으니 후순위.
- π_work 사전분포 민감도 (σ 0.10→0.20 재적합). 이건 3.6h 짜리 **재적합**이라 이 프롬프트
  범위 밖. 하게 되면 따로.

---

## 5. 보고

본문으로. 파일 경로만 남기지 말 것.

1. **§5.6 의 문장이 fluA4 에서 성립하는가.** w·p·Erlang 각각에 대해 "부호 반전 없음" 이
   맞는지, 아니면 어느 시즌·연령대·격자점에서 뒤집히는지.
2. Erlang: n=1,2,3,5 각각의 함의된 평균 GI 와, n 에 따라 결론이 어떻게 움직이는지.
   n=3 행이 주 분석을 정확히 재현하는지 (재현 안 되면 멈추고 보고).
3. Figure 3 의 CI 폭이 어느 정도인지 — peak 에서 관측점이 띠 안에 드는 시즌과 밖인 시즌.
   (φ_NB=1.38 이면 띠가 꽤 넓을 것이다. 그게 정상이다.)
4. 이 문서와 실제 코드가 어긋난 지점.

---

## 6. 하지 말 것

- **재적합 금지.** 사후분포는 `nuts_fluA4_raw.npz` 하나다. (p 점적합은 예외)
- v5 스크립트 (`sens4_w_sweep.py`, `sens5_baseline_p.py`, `sens6_erlang_n.py`) 수정 금지.
  `nuts_seasonpop*.py`, `season_pop_setup.py`, `sens_common.py`, `policy_seasonpop.py`,
  `model/`, `simulation/`, `calibration/hira_target.py`, `foi_jax.py`, `erlang_presymp.py`,
  `nuts_fluA.py` 수정 금지.
- `outputs/eda/` 의 v5 산출물 덮어쓰기 금지.
- **원고 수정 금지** (`main_v8.tex`, `supplement_v8.tex`). 반영은 Cowork 에서.
- 관측값 정수 반올림 금지. 그림 라벨 영어. `R_0` 라고 적지 말 것 (추정량은 `R_peak`).
- joint loss 를 `jax.jit` 밖에서 호출하지 말 것.
