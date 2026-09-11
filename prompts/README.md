# prompts/ — Claude Code 작업 프롬프트

인플루엔자 A 전용 재피팅 작업을 단계별로 쪼갠 프롬프트 모음. 각 파일은
**단독으로 붙여넣어도 동작하도록** 배경·경로·완료 기준을 자체 포함한다.

## 사용법

```
cd ~/Documents/python/NIMS/kt_epimodel_hira
claude
```
프롬프트 파일 내용을 그대로 붙여넣거나 `@prompts/01_smoke.md` 로 참조.

프로젝트 전반 규칙은 `../CLAUDE.md` 의 "Influenza A 전용 재피팅" 절에 있다.
프롬프트는 그 규칙을 **대체하지 않고 전제**한다.

## 순서

| # | 파일 | 무엇 | 선행 |
|---|---|---|---|
| 01 | `01_smoke.md` | 스모크 실행 + 타깃 정합성 확인 | — |
| 02 | `02_population_2013_2014.md` | 2013-15 인구 확보 → 7 시즌 (보류) | 대안 KOSIS 표 |
| 03 | `03_production_fit.md` | 본 NUTS 실행 + 수렴 진단 | 01 (02 는 선택) |
| 04 | `04_compare_v5.md` | v5 사후분포와 대조 | 03 |
| 05 | `05_downstream.md` | 정책 시뮬 · 논문 그림 재생성 | 04 |
| 06 | `06_manuscript.md` | main_v5 → v6 원고 갱신 | 05 |
| 07 | `07_monday_report.md` | 주말 무인 실행 결과 정리 | 주말 체인 |
| 08 | `08_seasonality_offset.md` | 2015-16 misfit 이 고정 캘린더 때문인지 | 07 |
| 09 | `09_theta_reimplementation.md` | θ 재구현 + θ·κ 스윕 | 4시즌 피팅 |
| 99 | `99_conventions.md` | 매 세션 공통 주의사항 | — |

01 → 03 → 04 → 05 → 06 이 주 경로. 02 는 병렬로 언제든.

## 무인 실행 (주말·야간)

03 → 04 → 05 를 사람 없이 이어 돌리려면 프롬프트 대신 오케스트레이터를 쓴다:

```
nohup caffeinate -ims bash scripts/weekend_run.sh > logs/weekend.log 2>&1 &
disown
```

시즌 집합은 실행 시점에 `pop_all_seasons_fluA.csv` 를 보고 자동 결정된다
(현재는 5 시즌: 2015-16 ~ 2019-20). 첫 게이트가 접촉행렬 상호성 보정을
확인하므로, 보정이 풀린 상태로 몇 시간짜리 피팅이 돌아가는 일은 없다. 단계별 게이트가 있어
중간에 죽어도 `outputs/fluA/REPORT.md` 는 항상 생성된다.
끝나고 나서는 `07_monday_report.md` 로 해석을 붙인다.
