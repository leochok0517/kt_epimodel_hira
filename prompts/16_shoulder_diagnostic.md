# 16. 진단 — 하강(shoulder) 미스가 구조 변경으로 고쳐지는가

작성 2026-09-29. 회사 Mac 전용 (JAX). **NUTS 없음. 점적합만.** 30분 안쪽.

**이건 원고에 들어가는 계산이 아니다.** v9 로 갈지 말지를 정하는 진단이다. 원고
(`main_v8.tex`, `supplement_v8.tex`)는 건드리지 않는다.

---

## 0. 맥락

`prompts/15` 로 Figure 3 에 사후예측 90% 띠를 넣었더니 커버리지가 38 / 54 / 69 / 88%
(2016-17 / 17-18 / 18-19 / 19-20)로 나왔다. 미스는 전부 **peak 이후 하강 구간**이다.
2016-17: peak 2주 뒤 관측 205k → 45k (4.5배 감소), 모형 255k → 130k (2배).

**하강이 γ 에 묶인 게 아니다.** 1/γ = 4 일이면 2주에 30배까지 떨어질 수 있는데 그보다
훨씬 느리므로, 모형 안에서 peak 이후에도 R_eff 가 1 근처에 오래 머문다는 뜻이다.
감수성자가 충분히 소진되지 않는다.

**가설.** 잠복기 2.0 일 + 감염기 4.0 일이 함의하는 평균 generation interval 이 5.1 일로
문헌(≈3 일)보다 길다(`outputs/fluA/sens_erlang_n_fluA4.json` meta). GI 가 길면 같은
상승 속도에 대해 R_peak 가 높게 맞춰지고, R 이 높으면 final size 가 커서 peak 뒤에도
감염될 사람이 많이 남아 shoulder 가 두꺼워진다. 즉 **GI 문제와 width 문제가 같은
문제**일 수 있다. 여기에 잠복기가 단일 지수 구획(Erlang(1))이라는 것도 폭을 넓힌다
(Wearing 2005 의 논지이고, 우리는 I 에만 Erlang 을 적용했다).

**질문.** 이 둘을 고치면 하강이 관측을 따라오는가. 따라오면 v9 재적합이 정당화된다.
안 따라오면 v8 대로 가고 "시험해 봤으나 구조 변경으로는 안 됐다"를 적는다.

---

## 1. 먼저 읽을 것

- `kt_epimodel_hira/jax_model/erlang_presymp.py` — 현재 ODE. **수정 금지.** E 는 단일 구획.
- `scripts/sens5_baseline_p_fluA4.py` — `fit_pi_pin` 점적합 패턴. 이걸 따른다.
- `scripts/fluA_setup.py` — `C` 의 구조. `shared_by_s` 에 σ, γ 가 어떻게 들어가는지.
- `scripts/nuts_seasonpop.py` — `PHI`, `GAMMA`, `KAP`, `BASE`, `W`, `NGM_F`. **재유도 금지, import.**
- `kt_epimodel_hira/jax_model/numpyro_model.py::derive_beta_from_R0_simplex` — γ 가 바뀌면
  NGM 정규화(`s_f/γ`)가 같이 바뀌므로, 변형에서도 R_peak 가 여전히 dominant eigenvalue 인지 확인.
- `prompts/99_conventions.md`.

---

## 2. 변형 네 가지

| | 잠복기 구조 | 1/σ | 1/γ | n | w | 평균 GI |
|---|---|---|---|---|---|---|
| **V0** 현재 | Erlang(1) | 2.0 | 4.0 | 3 | 0.22 | 5.14 |
| **V1** 짧게 | Erlang(1) | 1.5 | 3.0 | 3 | 0.22 | 3.85 |
| **V2** 잠복 Erlang | Erlang(2) | 2.0 | 4.0 | 3 | 0.22 | 5.14 |
| **V3** 둘 다 | Erlang(2) | 1.5 | 3.0 | 3 | 0.22 | 3.85 |

GI 공식: 잠복기 평균 + Σ_k w_k·k·(1/γ)/n / Σ_k w_k, 가중치 (w, 1, 1). V1/V3 에서 3.85 가
나와야 한다. 각 변형의 GI 를 결과 JSON `meta` 에 넣을 것.

**V1 은 코드 변경 없이 파라미터만 바꾸면 된다.** `shared_by_s` 의 σ, γ 를 덮어쓴다.
발증전 창이 1/(3γ) = 1.0 일이 된다는 점을 `meta` 에 적을 것.

**V2/V3 은 새 ODE 파일이 필요하다.** `kt_epimodel_hira/jax_model/erlang_presymp_e2_fluA4.py`
를 신규로 만든다 — `erlang_presymp.py` 를 복사해 E 를 `E1 → E2` 두 구획(각각 rate 2σ)으로
나눈다. 상태 배열 레이아웃, `split_seed_to_erlang` (E(0) 를 두 구획에 균등 배분),
`daily_new_onset_by_age_*` (I1→I2 전이는 그대로), `simulate_*` 를 전부 새 파일 안에서
맞춘다. **원본은 한 줄도 건드리지 않는다.**

V2 가 V0 를 정확히 재현하는 검증은 불가능하다(구조가 다르니까). 대신 **V2 코드에서
σ→∞ 극한이 아니라, E1·E2 를 합친 잠복기 평균이 2.0 일인지** 시뮬레이션으로 확인할 것
(초기 E 만 넣고 감염 없이 돌려 E 가 반으로 줄어드는 시간 = 2.0·ln2 ≈ 1.39 일이 아니라
Erlang(2) 의 중앙값 ≈ 1.68 일이어야 한다).

---

## 3. 점적합

**π 는 고정한다.** 4시즌 공유 사후 median `[0.422, 0.314, 0.073, 0.189]`
(`outputs/fluA/per_season_pi_fluA4.json` → `shared.pi.*.median`). 구조 변경의 효과를
배분 변경과 섞지 않기 위해서다. 자유 파라미터는 **log R_peak 와 log φ_NB 둘**뿐.
시즌마다 2차원 최적화 — 초 단위다.

시즌: **네 시즌 전부** (싸니까). 보고는 2016-17 을 중심으로.

변형 × 시즌 = 16 점적합. 각각에서 저장할 것:

- 적합된 R_peak, φ_NB, NLL
- 주간 총 곡선 (모형) 과 관측
- 지표: `week_obs`, `week_model`, `peak_ratio` (obs/model), `total_ratio` (Σobs/Σmodel),
  그리고 **하강 지표** — `decline_2wk = obs(peak+2)/obs(peak)` 대 같은 것의 모형값,
  `decline_4wk` 도. 2016-17 관측은 2주에 약 0.22 (205k→45k) 다. 이 숫자가 모형에서
  몇이 되는지가 이 진단의 핵심 산출이다.

**φ_NB 를 같이 맞추는 이유** — 하강이 좋아지면 φ 가 올라가야 한다(잔차가 줄어드니).
V0 의 점적합 φ 와 V3 의 φ 를 비교하면 misfit 이 얼마나 줄었는지 한 숫자로 나온다.

---

## 4. 산출

```
outputs/fluA/diag_shoulder_fluA4.json
figures/paper_fluA4/{pdf,png}/diag_shoulder.{pdf,png}
scripts/diag_shoulder_fluA4.py
kt_epimodel_hira/jax_model/erlang_presymp_e2_fluA4.py   (V2/V3 용, 신규)
```

그림: 2×2 시즌, 각 패널에 관측점 + V0~V3 네 곡선(V0 검정, 나머지 색). x 축은 ISO 주
(`fig_fit_total_ci_fluA4.py` 의 눈금 코드를 가져올 것). **원고 폴더로 복사하지 말 것** —
원고에 안 들어간다.

---

## 5. 보고 (본문으로)

1. **V3 에서 2016-17 의 2주 하강비가 관측(≈0.22)에 얼마나 가까워지는가.** V0 는 ≈0.5 다.
   0.3 이하면 "따라온다", 0.45 위면 "안 따라온다", 사이면 "부분적".
2. V1 과 V2 를 따로 — 둘 중 어느 쪽이 하강을 더 움직이는가 (GI 길이 vs 잠복기 분포).
3. R_peak 가 변형별로 어떻게 움직이는가. V1/V3 에서 내려가야 정상이다(GI 가 짧아졌으니).
   내려간 R_peak 가 문헌의 계절 인플루엔자 R (1.2~1.6) 근처인지.
4. φ_NB 가 변형별로 어떻게 움직이는가. 올라가면 misfit 이 준 것.
5. `total_ratio` — 시즌 총량 과대예측(현재 1.5~2.1배)이 얼마나 줄었는가.
6. **판정 한 줄**: v9 재적합을 권하는가, 아닌가. 권한다면 어느 변형으로, 그리고 재적합
   후 바뀔 것(R_peak 값, §S11 의 R_0 유도값, 정책 %, 민감도 전부)을 나열.
7. 이 문서와 코드가 어긋난 지점.

---

## 6. 하지 말 것

- **NUTS 돌리지 말 것.** 점적합만.
- `erlang_presymp.py`, `model/`, `simulation/`, `nuts_seasonpop*.py`, `nuts_fluA.py`,
  `foi_jax.py`, `calibration/`, `season_pop_setup.py`, `sens_common.py`,
  `policy_seasonpop.py` 수정 금지. V2 의 ODE 는 **새 파일**.
- `outputs/eda/`, `outputs/fluA/` 의 기존 파일 덮어쓰기 금지.
- 원고 수정 금지. 그림을 원고 폴더로 복사하지 말 것.
- joint loss 를 `jax.jit` 밖에서 호출하지 말 것. 관측값 정수 반올림 금지. 라벨 영어.
- `R_0` 라고 적지 말 것 — 추정량은 `R_peak`.
