# 05 — 정책 시뮬 · 민감도 · 논문 그림 재생성

`prompts/99_conventions.md` 를 함께 적용할 것. 선행: `04_compare_v5.md`.

## 배경

논문의 결론은 전부 사후 draw 를 쓰는 forward simulation 에서 나온다. 사후가
바뀌었으므로 아래를 전부 다시 돌려야 한다. v5 경로는 건드리지 말고 fluA 사후를
읽는 병렬 경로를 만든다.

## 재생성 대상 (main_v5.tex 기준)

| 논문 요소 | 원 스크립트 | 비고 |
|---|---|---|
| fit_total | `paper_figures.py` | 04 에서 이미 생성 |
| 정책별 averted (총량 · CI) | `policy_seasonpop.py` | π = 결합 배분, N=500 draw/시즌 |
| 연령별 averted, rate vs number | `fig_byage_*.py` | |
| 학기 / 겨울방학 층화 | `holiday_reversal_multiseason.py` | |
| 연령 재분배 | `paper_figures_extra.py` | |
| R_e 채널·연령 분해 | `reproduction_analysis.py` | |
| 민감도 (w, κ, π_work, 초기면역, p, Erlang n) | `sens_*` | w·κ·Erlang 은 재피팅 없이 draw 재사용 |
| θ (other-channel 차등) 스윕 | `kappa_no_eta_presymp.py` 계열 | −0.25 ~ 0.75 |

## 할 일

1. fluA 사후를 읽는 얇은 어댑터 `scripts/fluA_posterior.py` 작성:
   `load_fluA_posterior(n_draw=500)` → v5 하위 스크립트가 기대하는 것과
   같은 형태의 draw dict 반환. `policy_seasonpop.py` 등은 `MERGED_NPZ =
   ED / "nuts_seasonpop_merged.npz"` 를 하드코딩하고 있으므로, 경로를 인자로
   받는 fluA 사본을 만드는 편이 안전하다. 시즌 수가 3 → 5(또는 7) 로 늘어난 것을
   하위 스크립트가 견디는지 먼저 확인하고, 못 견디면 그 스크립트의
   **fluA 사본**을 만들 것 (원본 수정 금지).

2. 위 표를 순서대로 재생성. 출력은 전부 `outputs/fluA/` 또는
   `manuscript/figures_fluA/`.

3. 결론이 바뀌었는지 항목별로 판정해서 표로 보고:
   - 학생 결석 > 병가 (효과 크기) — 유지되는가
   - 병가의 population benefit CI 가 0 을 포함하는가 (v5: 3 시즌 중 2 개에서 포함)
   - 병가가 per-capita 위험을 아동 쪽으로 옮기는가, 겨울방학에 강화되는가
   - 공간 타겟팅 null 유지
   - **부호가 바뀐 항목이 있으면 최우선 보고.**

4. 시즌이 늘어난 만큼 새로 할 수 있는 분석을 제안:
   - 시즌별 R0 분산이 커졌으므로 "π 가 다년 상수인가" 검정이 강해진다
   - 시즌 간 이질성을 랜덤효과로 볼 여지

## 완료 기준

- 재생성된 그림·JSON 일습
- v5 결론 대비 변경/유지 판정표
- 부호가 바뀐 항목의 원인 분석
