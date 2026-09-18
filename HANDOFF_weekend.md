# 주말 무인 실행 — 퇴근 전 체크리스트

작성 2026-08-28. 이 파일 하나만 따라 하면 된다.

## 1. 스모크 — 종료 완료, 원인 수정 완료

스모크가 182 분 멈추고 메모리가 54GB 까지 갔다. 원인은 스모크의 **점추정**이
`make_joint_loss_fluA` 를 **jit 없이** Nelder-Mead 로 수백 번 호출한 것 —
매 호출마다 5 시즌 diffrax ODE 가 eager 로 다시 돌았다.

`setup` 까지는 정상이었고 시즌별 통계도 기대값과 일치했다
(2017-18 obs_A/obs_all = 0.502, 2016-17 = 0.998).

**주말 체인에는 영향이 없다.** `--smoke` 없이 실행하면 점추정을 건너뛰고
NUTS 로 직행하며, NUTS 는 numpyro 가 모델을 한 번 트레이스해 jit 한다 —
v5 가 성공적으로 돌린 것과 같은 경로다.

수정: `--smoke` 에서 최적화를 걷어내고 **jit 된 NLL 을 R0 세 점에서 평가**하는
probe 로 교체했다. 컴파일 성공 · NLL 유한성 · R0 에 대한 응답성을 확인한다.

### 스모크 결과 (2026-08-28 실행 완료)

probe 는 통과했다:

```
compile + first evaluation: 5.3s
NLL at R0=1.6 = 1,482,515.2
NLL at R0=2.0 =    20,014.8      <- 최소가 2.0 근처
NLL at R0=2.6 =    46,851.6
finite: True   per-evaluation: 26 ms
```

그 뒤 NUTS warmup 이 65-91 s/it 로 느리게 돌았는데 **버그가 아니다.** NUTS 는
warmup 초반에 step size 를 찾기 전까지 매 iteration 트리를 최대 깊이
(2^8 - 1 = 255 leapfrog) 까지 펼친다. v7 진단값과 대조:

| | v7 (3 시즌, warmup 1000) | smoke (5 시즌, warmup 20) |
|---|---|---|
| mean_num_steps | 29.1 | 255 (포화) |
| saturation | 0.125% | 사실상 100% |
| 속도 | 2.74 s/it 평균 | 65 → 91 s/it |

v7 도 이 구간을 지났지만 1000 회 평균에 묻혀 0.125% 로 보인다.
**warmup 20 회짜리 스모크는 이 구간을 벗어날 수 없다** — 스모크가 짧아서 느린
것이지, 느려서 짧은 것이 아니다.

→ **스모크의 NUTS 부분은 여기서 끊어도 된다.** 필요한 검증은 probe 가 이미
끝냈고, warmup iteration 이 실제로 진행되는 것까지 확인됐다.

본 실행 추정은 그대로 (v7 실측을 시즌 수로 스케일):
warmup 2743s x 5/3 = 1.3h + sampling 5703s x 5/3 = 2.6h = **약 3.9 시간**.
초기 포화 구간이 포함된 수치다.

## 2. 시즌은 5 개로 확정 — 할 일 없음

KOSIS `행정구역(읍면동)별 5세별 주민등록인구` 표가 **2015 년부터 시작**한다
(2026-08-28 확인). 그 시점에 집계 방식이 바뀐 것으로 보이며 재-export 로는 더
내려가지 않는다. 따라서 **2015-16 ~ 2019-20 의 5 시즌**으로 진행한다.

2013-14 · 2014-15 를 살리려면 시도/시군구 단위 연령별 표에서 인구를 가져와야
한다 (읍면동 해상도는 쓰지 않으므로 충분하다). 월요일 이후 별도 작업 —
`prompts/02_population_2013_2014.md`.

체인은 `pop_all_seasons_fluA.csv` 를 보고 시즌 집합을 자동 결정하므로
손댈 것은 없다.

## 2b. 접촉행렬 상호성 — 확인 완료

`deploy_reciprocity_matrices.py` 가 2026-08-24 에 방식 A 보정본을 kt_data 표준
경로에 제자리 교체해 뒀다. 보정이 **호출부 플래그가 아니라 데이터 파일에**
있으므로 `load_contact_matrices()` 를 쓰는 모든 경로가 자동으로 받는다 —
influenza A 경로 포함, 겨울방학 행렬 포함.

측정값: 상호성 `G_i C_ij = G_j C_ji` 가 ~1e-10 (T 스케일 1e7) 수준으로 성립,
method A 재적용은 ~1e-17 로 no-op (즉 v7 의 `--corrected 1` 도 이중 보정이
아니었다).

체인의 **첫 게이트**가 `scripts/check_reciprocity_fluA.py` 다. 백업본이
되돌려졌거나 행렬이 재-export 돼 보정이 풀렸으면 거기서 멈춘다 — 몇 시간짜리
피팅을 돌린 뒤에 알게 되는 일은 없다.

## 3. 체인을 띄운다

```bash
cd ~/Documents/python/NIMS/kt_epimodel_hira
mkdir -p logs
nohup caffeinate -ims bash scripts/weekend_run.sh > logs/weekend.log 2>&1 &
disown
```

띄운 직후 5 분쯤 확인:

```bash
tail -f logs/weekend.log
```

`season set: ...` 줄에 기대한 시즌이 찍히고 `STAGE production fit` 이 시작되면
끝난 것이다. 터미널을 닫아도 된다 (`disown`).

## 3b. 맥 프로가 잠들지 않게

`caffeinate -ims` 가 프로세스가 도는 동안 idle sleep · 디스플레이 sleep ·
디스크 sleep 을 막아준다. 다만 **시스템 설정 쪽이 이를 덮어쓰는 경우**가 있어
아래를 함께 걸어두는 편이 확실하다.

GUI: 시스템 설정 → 에너지 절약(또는 배터리 → 옵션) 에서
"디스플레이가 꺼져 있을 때 자동으로 잠자기 방지" 켜기.

터미널 (관리자 암호를 묻는다):

```bash
sudo pmset -a sleep 0           # 시스템 잠자기 끔
sudo pmset -a disksleep 0       # 디스크 잠자기 끔
sudo pmset -a womp 1            # (선택) 네트워크로 깨우기
sudo pmset -a autorestart 1     # (선택) 정전 복구 후 자동 재시작
```

확인:

```bash
pmset -g custom | egrep " sleep| disksleep| displaysleep"
pmset -g assertions | head -20   # caffeinate 가 잡고 있는지
```

디스플레이는 꺼져도 무방하다 — `displaysleep` 은 건드릴 필요 없다.

주말이 끝나면 되돌리는 것을 권한다:

```bash
sudo pmset -a sleep 1
sudo pmset -a disksleep 10
```

**주의**: `pmset` 은 잠자기만 막는다. 소프트웨어 업데이트 자동 재시작이 켜져
있으면 그것이 세션을 죽일 수 있다. 시스템 설정 → 일반 → 소프트웨어 업데이트 →
자동 업데이트에서 "macOS 업데이트 설치" 를 주말 동안 꺼두면 확실하다.

## 4. 예상 시간

v7 실행 기준 (3 시즌, warmup 1000 + sample 2000 × 4 chain) warmup 46 분 +
sampling 95 분 = 2.4 시간. 시즌 수에 거의 비례한다.

| 단계 | 5 시즌 |
|---|---|
| 상호성 체크 + 인구 | 1 분 미만 |
| production NUTS | 약 4 시간 |
| v5 비교 + 그림 | 약 5 분 |
| 정책 forward (N=500) | 약 4 시간 |
| **합계** | **약 8 시간** |

금요일 저녁에 띄우면 토요일 새벽에 끝난다. 주말 내내 돌 필요는 없다.

## 5. 월요일에 받는 것

```
outputs/fluA/REPORT.md              자동 생성 보고서 (수치·표·그림 링크)
outputs/fluA/run_status.json        단계별 성공/실패 + wall time
outputs/fluA/compare_v5_fluA.png    v5 대비 π·R0 forest plot
outputs/fluA/fit_total_fluA.png     시즌별 fit
outputs/fluA/policy_posterior_fluA.json
outputs/eda/nuts_fluA_raw.npz       사후 draw
```

월요일 아침 9시 (KST) 예약 세션이 이 파일들을 읽고 해석을 붙인 보고서를 올린다.

**한 가지 확인이 필요하다.** 예약 작업을 만들 때 device 바인딩이 아직
승인되지 않아 지금은 클라우드 전용으로 잡혀 있다 — 그 상태로 실행되면 맥의
결과 파일을 읽지 못한다. Claude 데스크톱 앱의 예약 작업 목록에서
"월요일 아침 — influenza A 재피팅 결과 정리" 를 열어 이 컴퓨터에 연결하도록
승인해 둘 것. 승인이 안 되어 있으면 월요일에 세션이 "맥에 접근할 수 없다" 고
알려주며, 그때 아래 수동 경로로 돌리면 된다.

수동 경로: `prompts/07_monday_report.md` 를 Claude Code 에 붙여넣는다.

## 6. 중간에 죽으면

단계별 게이트가 있어 **어디서 죽든 `REPORT.md` 는 생성된다.** 재개는:

- NUTS: warm-up state 가 `outputs/eda/nuts_fluA_warmup_state.pkl` 에 남는다.
  같은 시즌 집합이면 체인을 그냥 다시 띄우면 warmup 을 건너뛴다.
  (시즌 집합이 다르면 자동으로 stale 처리하고 새로 warmup 한다.)
- 정책 forward: `outputs/fluA/policy_posterior_fluA_partial.jsonl` 에
  draw 단위로 쌓인다. 다시 띄우면 이어서 간다.

## 7. 안 건드리는 것

`nuts_seasonpop*.py`, `season_pop_setup.py`, `sens_common.py`,
`policy_seasonpop.py`, `outputs/eda/nuts_seasonpop_*`,
`outputs/eda/policy_posterior_seasonpop*` — 전부 v5 재현 경로다. 체인은
읽기만 하고 쓰지 않는다.

---

## 부록 — 다른 시즌 집합으로 다시 돌리기 (2026-08-31 추가)

체인이 `TAG` 와 `SEASONS_OVERRIDE` 를 받는다. 파일을 하나도 고치지 않고
시즌 집합을 바꿔 돌릴 수 있고, **출력이 `TAG` 로 분리돼 이전 실행을 덮어쓰지
않는다.**

```bash
cd ~/Documents/python/NIMS/kt_epimodel_hira
TAG=fluA4 SEASONS_OVERRIDE="2016-2017 2017-2018 2018-2019 2019-2020" \
nohup caffeinate -ims bash scripts/weekend_run.sh > logs/weekend_fluA4.log 2>&1 &
disown
```

- `TAG=fluA` (기본) 는 기존 파일명을 그대로 쓴다. 그 외 태그는 전부 접미사가 붙는다:
  `nuts_<TAG>_raw.npz`, `nuts_<TAG>_diagnostics.json`, `compare_v5_<TAG>.json/.png`,
  `fit_total_<TAG>.png`, `policy_posterior_<TAG>.json`, `run_status_<TAG>.json`,
  `REPORT_<TAG>.md`.
- warm-up state 도 태그별이고, 시즌 수가 다르면 자동으로 stale 처리된다.
- `SEASONS_OVERRIDE` 의 시즌은 `pop_all_seasons_fluA.csv` 에 인구가 있어야 한다.
  없으면 stage 0 에서 중단한다.

5 시즌 실행 산출물은 `outputs/fluA/run_5season/` 에 보존해 뒀다.
