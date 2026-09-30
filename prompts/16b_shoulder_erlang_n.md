# 16b. 진단 추가 — 감염기 Erlang 단계 수를 늘리면 하강이 따라오는가

작성 2026-09-29. `prompts/16` 의 추가 항목 하나. 점적합만, 5~10분.

## 0. 왜

`prompts/16` 은 **잠복기** Erlang(2) 를 시험했고 효과가 없었다 (V2 ≈ V0). 감염기의
단계 수 n 은 `prompts/15` 에서 n ∈ {2,3,5} 를 돌렸지만 그건 **기존 사후분포의 R_peak 를
그대로 쓴 정책 forward sim** 이라 적합도·하강 모양은 재지 않았다. 감염기 n 을 늘려
점적합했을 때 2016-17 의 하강이 따라오는지가 아직 답이 없다.

기제상 답은 "아니오"일 가능성이 크다 — 단계 수는 대기시간 분산을 줄여 상승·peak 을
뾰족하게 하지, peak 이후의 감수성자 소진 속도를 바꾸지 않는다. 그래도 한 줄이면 되니
확인한다.

## 1. 할 일

`scripts/diag_shoulder_fluA4.py` 를 **수정하지 말고** `scripts/diag_shoulder_erlang_n_fluA4.py`
를 새로 만든다. 같은 점적합 설정(π 는 공유 median 고정, log R_peak 와 log φ_NB 둘만
최적화, 4시즌), 변형만 다르다.

| | 잠복기 | 1/σ | 1/γ | 감염기 n | w | 평균 GI |
|---|---|---|---|---|---|---|
| V0 | Erlang(1) | 2.0 | 4.0 | 3 | 0.22 | 5.14 |
| **V4** | Erlang(1) | 2.0 | 4.0 | **5** | 0.44 | 4.60 |
| **V5** | Erlang(1) | 2.0 | 4.0 | **8** | 0.78 | 4.28 |

w 는 발증전 비율 0.10 을 고정하는 `sens6_erlang_n_fluA4.py` 의 `w_by_n` 규칙을 따른다
(n=5 → 0.444). n=8 은 그 규칙을 연장해 계산할 것 (w = presymp_fraction·n / (1 − presymp_fraction) 형태인지
`sens6` 의 정의를 읽고 그대로 적용). GI 는 같은 공식으로 `meta` 에.

**모형 코드는 이미 있다.** `kt_epimodel_hira/jax_model/erlang_n.py` 의
`simulate_jax_erlang_n`, `daily_new_onset_by_age_erlang_n`, `ngm_factor` — `sens6_erlang_n_fluA4.py`
의 `make_fn` 이 쓰는 그대로. n 은 static 이라 n 마다 컴파일된다. **원본 수정 금지.**

V0 는 `diag_shoulder_fluA4.json` 의 값을 다시 계산하지 말고 그 파일에서 읽어 같은
표에 넣는다 (같은 설정이므로 같은 값이 나와야 하고, 다르면 멈추고 보고).

## 2. 산출

```
outputs/fluA/diag_shoulder_erlang_n_fluA4.json
figures/paper_fluA4/{pdf,png}/diag_shoulder_erlang_n.{pdf,png}   # 원고 폴더로 복사하지 말 것
```

지표는 `prompts/16` §3 과 동일: R_peak, φ_NB, NLL, week_obs/model, peak_ratio, total_ratio,
decline_2wk·4wk (own peak 과 observed peak 두 정의 모두).

## 3. 보고 (본문으로, 짧게)

1. 2016-17 2주 하강비: 관측 0.217, V0 0.625, **V4 = ?, V5 = ?** — 판정 기준은 `prompts/16` §5 와 같다.
2. R_peak 와 φ_NL 가 n 에 따라 어떻게 움직이는가. 4시즌 NLL 합.
3. 한 줄 판정 — 감염기 단계 수가 하강의 지렛대인가 아닌가.

## 4. 하지 말 것

`prompts/16` §6 과 동일. NUTS 없음, 원본 모형 파일 수정 금지, 원고 수정 금지, 그림을
원고 폴더로 복사하지 말 것.
