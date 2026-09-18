# 07 — 월요일 결과 정리 보고

`prompts/99_conventions.md` 를 함께 적용할 것.

## 상황

주말 동안 `scripts/weekend_run.sh` 가 무인으로 돌았다. 체인은
population → production NUTS → v5 비교 → 정책 forward → 리포트 순이고,
**단계별 게이트**가 있어 중간에 죽어도 거기까지의 산출물과
`outputs/fluA/REPORT.md` 는 남는다.

## 먼저 읽을 것

```
outputs/fluA/run_status.json      단계별 성공/실패 + wall time + 로그 경로
outputs/fluA/REPORT.md            스크립트가 자동 생성한 보고서 (수치·표)
logs/weekend.log                  오케스트레이터 로그
logs/weekend_*_<stamp>.log        단계별 상세 로그
```

## 할 일

1. **체인이 어디까지 갔는지 먼저 확정.** `run_status.json` 의 `aborted_at` 이
   null 이 아니면 그 단계의 로그를 읽고 원인과 재개 명령을 제시. 추측 금지 —
   로그의 실제 traceback 을 인용할 것.

2. **`REPORT.md` 를 검증.** 자동 생성물이라 수치는 맞지만 해석이 없다. 다음을
   직접 판단해서 보고에 추가:
   - `compare_v5_fluA.json` 의 `flags` 가 비어 있는가. **비어 있지 않으면
     그것이 이 보고의 헤드라인이다** — 다른 결론을 먼저 말하지 말 것.
   - A 점유율이 1.0 에 가까운 시즌 (2016-17, 2019-20) 의 R0 가 v5 와
     사실상 같은가. 다르면 A 보정 외의 무언가가 바뀐 것이다.
   - `phi_nb` 가 v5 대비 올라갔는가 (B 잔차 제거 → 과산포 감소가 기대 방향).
     내려갔다면 설명 필요.
   - π_work 의 95% CI 가 v5 와 겹치는가. 병가 정책 부호를 좌우한다.

3. **v5 결론 대비 변경/유지 판정표**를 만들 것:
   | 결론 | v5 | 이번 | 판정 |
   - 학생 결석 > 병가 (효과 크기)
   - 병가의 population benefit CI 가 0 을 포함
   - 병가가 per-capita 위험을 아동 쪽으로 이동
   - 겨울방학에 그 이동이 강화
   **부호가 바뀐 항목은 최우선으로, 원인 가설과 함께 보고.**

4. 새로 늘어난 시즌이 만든 것을 짚을 것:
   - 시즌별 R0 분산이 커졌으므로 "π 가 다년 상수인가" 검정이 강해졌다.
     시즌 간 π 이질성을 볼 여지가 생겼는지 판단.
   - 새 시즌 중 fit 품질이 나쁜 것이 있는지 (`fit_total_fluA.png`).

5. `outputs/fluA/REPORT.md` 를 위 해석을 반영해 **다시 쓸 것**. 자동 생성본은
   `REPORT_auto.md` 로 보존하고, 해석이 들어간 최종본을 `REPORT.md` 로.

6. 다음 단계 제안:
   - 남은 downstream (연령별 그림, holiday reversal, R_e 분해, 민감도 스윕,
     θ 스윕) — `prompts/05_downstream.md`
   - 원고 갱신 — `prompts/06_manuscript.md`
   - 7 시즌이 아니었다면 `prompts/02_population_2013_2014.md`

## 하지 말 것

- 실행되지 않은 단계의 수치를 추정해서 채우지 말 것. "미실행" 이라고 쓸 것.
- flags 가 떠 있는데 결론부터 말하지 말 것.
- v5 산출물 (`outputs/eda/nuts_seasonpop_*`, `policy_posterior_seasonpop*`)
  덮어쓰지 말 것.
