# 04 — v5 사후분포와 대조

`prompts/99_conventions.md` 를 함께 적용할 것. 선행: `03_production_fit.md`.

## 배경

v5 논문 수치 (main_v5.tex / PROJECT_BRIEF.md):
- 시즌: 2016-17, 2017-18, 2019-20 (3 개)
- 공유 채널 배분 π = [home 0.442, work 0.298, school 0.065, other 0.195]
- 관측모델 NB, `phi_nb` 시즌 공유
- v5 논문 사후 (본문 "4000 post-warm-up draws"): `outputs/eda/nuts_seasonpop_merged.npz`
  — shape (4000, 3). 이것이 **논문 기준선**이다.
- 참고: `outputs/eda/nuts_seasonpop_v7reparam_raw.npz` (8000, 3) 은 이후의
  재모수화 (W-2) 실행. 좌표계가 fluA 쪽과 같으므로 sampler 거동 비교에는 이쪽이
  적절하지만, **논문 수치와의 대조는 merged 로** 할 것.

## 할 일

1. `scripts/compare_v5_fluA.py` 신규 작성. v5 npz 와 fluA npz 를 읽어:
   - π 4 채널의 사후 평균 · 95% CI 를 나란히
   - 공통 3 시즌의 R0 사후를 나란히
   - `phi_nb` 사후를 나란히
   - 겹치는 시즌에 대해 CI overlap 비율

2. 그림 `outputs/fluA/compare_v5_fluA.png`:
   - 상단: π 4 채널 forest plot (v5 vs fluA)
   - 하단: 시즌별 R0 forest plot. fluA 에만 있는 시즌은 다른 색으로 구분

3. fit 그림 `outputs/fluA/fit_total_fluA.png`:
   - 시즌별 관측 A 카운트 (점) vs 사후 평균 모델 (선)
   - 참고용으로 보정 전 A+B 카운트를 옅은 회색으로 함께
   - `scripts/paper_figures.py` 의 `fit_total` 스타일을 따르되 **원본은 수정 금지**

4. 해석해서 보고:
   - π 가 v5 와 통계적으로 구분되는가? 특히 `work` — 병가 정책 부호를 좌우한다.
   - **2017-18 R0 가 얼마나 이동했는가** (A 점유 0.50 인 시즌). 사전 기대는
     "떨어진다" 인데, 근거는 `gamma_report` 가 **고정**이라는 점이다 — 관측
     카운트가 절반이 되면 모델이 그 수준 변화를 탐지율로 흡수할 수 없고,
     감염 총량 자체를 줄여야 하며, 최종 크기는 R0 가 결정한다. 초기 시드도
     같이 절반이 되지만 (`estimate_initial_infected_from_fluA`) 최종 크기는
     시드가 아니라 R0 의 함수이므로 이 논리는 유지된다.
     **그런데 이건 가설이지 사실이 아니다.** NB 관측모델의 과산포가 크면
     수준 불일치를 상당히 용인하고 (v5 의 "peak over-prediction" 한계가 바로
     그 증상이다), 그 경우 fit 이 성장률 위주로 결정돼 R0 가 거의 안 움직일 수
     있다. 어느 쪽이든 **결과를 보고 판정하고, 안 움직였으면 fit 그림에서
     2017-18 의 peak 잔차를 확인**할 것 — 모델이 관측보다 위에 있으면 후자다.
   - 2016-17 / 2019-20 R0 는 거의 안 움직여야 정상 (A 점유 ~1.0).
     크게 움직였다면 **시즌 확장이나 인구 쪽에 버그**가 있다는 신호 — 조사할 것.
   - `phi_nb` 변화 — A 보정으로 B 유행이 만들던 잔차가 사라졌으면 과산포가
     줄어(= phi_nb 증가) 있어야 한다. 반대면 설명 필요.

## 완료 기준

- 비교 표 + 두 그림
- 위 4 의 네 항목에 대한 판단, 특히 "안 움직여야 할 것이 움직였는지" 검증
