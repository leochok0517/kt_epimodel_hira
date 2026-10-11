# 13. 시즌별 독립 π 적합 (per-season channel allocation)

작성 2026-09-21. 회사 Mac 전용 (HIRA 원자료와 JAX 환경이 여기에만 있다).

---

## 0. 왜 하는가

공저자 질문 #18 이다.

> 개별 연도에서 channel share 를 추정했을 때에도 각 channel 의 비중이 많이 달라지지
> 않는가?

원고 §5.4 는 다음과 같이 주장한다.

> A single shared channel allocation reproduced all four seasons at negligible
> cost in fit.

**이 주장을 뒷받침하는 수치가 원고에도 `outputs/fluA/` 에도 없다.** §4.3 이 말하는 것
(단일 시즌에서 π_work 가 실질적으로 비식별이라는 것)은 다른 이야기이므로 답이 되지
않는다.

계산 하나로 두 가지가 풀린다.

1. 공저자 질문에 정량적으로 답한다. 시즌별 π 가 얼마나 흔들리는지 수치가 나온다.
2. **2015-16 이 안 맞는 이유가 갈린다.** 공유 배분 탓인지, 그 시즌 자체가 다른지.
   지금 Discussion 은 "no adjustment of the shared parameters accommodates it" 이라고만
   적고 있어 리뷰어가 모형 실패로 읽을 여지가 있다.

---

## 1. 먼저 읽을 것

코드를 쓰기 전에 아래를 읽고, 이 문서가 적은 내용과 실제가 맞는지 확인하라.
다르면 **실제 코드를 따르고**, 어긋난 지점을 보고에 적어라.

- `scripts/nuts_fluA.py` — `make_joint_loss_fluA(C)`, `build_model_fluA(C, ...)`
- `scripts/nuts_fluA_run.py` — 4시즌 합동 적합의 실행부. 체인 수, 워밍업, 샘플 수,
  타깃 수용률, 시드, 저장 형식이 여기 있다. **이번 적합도 같은 설정을 쓴다.**
- `scripts/fluA_setup.py` (또는 setup 빌더가 있는 파일) — `C` 의 구조.
  특히 `C['seasons']`, `C['obs']`, `C['w']`, `C['st_by_s']`, `C['ngm3_by_s']`,
  `C['shared_by_s']`, `C['nw']`
- `prompts/99_conventions.md`

**핵심 관찰**: `nuts_fluA.py` 는 시즌 목록을 모듈 상수가 아니라 `C['seasons']` 에서
가져온다. 따라서 **시즌 하나만 담은 `C` 를 만들면 그대로 단일 시즌 적합이 된다.**
`loss_joint` 도 `build_model_fluA` 도 고칠 필요가 없다.

---

## 2. 만들 것

### 2.1 신규 파일

```
scripts/nuts_per_season_pi_fluA4.py
```

**기존 파일은 하나도 수정하지 않는다.** v5 재현 경로 수정 금지 규칙이 그대로 적용된다
(`nuts_seasonpop*.py`, `season_pop_setup.py`, `sens_common.py`, `policy_seasonpop.py`,
`model/`, `simulation/`, `calibration/hira_target.py`, `foi_jax.py`,
`erlang_presymp.py`). `nuts_fluA.py` 도 건드리지 말고 import 만 한다.

### 2.2 하는 일

1. 전체 setup `C` 를 한 번 만든다.
2. 시즌 하나씩, `C` 에서 그 시즌만 남긴 얕은 복사본 `C_s` 를 만든다.
   `seasons` 는 길이 1 리스트, `obs`/`w` 는 해당 인덱스 하나만, dict 들은 그 키만.
3. `build_model_fluA(C_s)` 로 NUTS 를 돌린다. 설정은 `nuts_fluA_run.py` 와 동일.
4. 사후분포를 저장한다.

**대상 시즌은 5개다**: 2015-16, 2016-17, 2017-18, 2018-19, 2019-20.
2015-16 은 본 적합에서 제외된 시즌이지만, **이번에는 반드시 포함한다.** 그 시즌 전용
π 로는 정점이 맞는지가 §0 의 두 번째 질문이다.

### 2.3 지켜야 할 것

- **`jax.jit` 밖에서 joint loss 를 호출하지 말 것.** 2026-08-28 에 54GB 를 먹고 멈춘
  적이 있다. `nuts_fluA_run.py` 가 쓰는 패턴을 그대로 따른다.
- 고정 벡터(`PHI`, `GAMMA`, `KAP`, `BASE`, `W`, `NGM_F`, `LOGIT_REF`, `SIGMA_PIN`)를
  다시 유도하지 말 것. `nuts_seasonpop` 에서 import 한다.
- `outputs/eda/nuts_seasonpop_*`, `outputs/eda/policy_posterior_seasonpop*` 를
  덮어쓰지 말 것.
- 관측값을 정수로 반올림하지 말 것.

---

## 3. 출력

### 3.1 `outputs/fluA/per_season_pi_fluA4.json`

시즌마다 아래를 담는다. 요약값은 median 과 q05/q95 (원고가 쓰는 90% CI 규약).

- `pi_home`, `pi_work`, `pi_school`, `pi_other` — 4채널 사후 요약
- `R_peak` — 해당 시즌의 추정 재생산수 (`exp(log_R0)`). **이름을 `R_0` 로 적지 말 것.**
  추정량은 `R_peak` 이고 `R_0` 는 유도값이다
- `phi_nb`
- 진단: `r_hat` 최대값, 최소 ESS(bulk/tail), divergence 수, 샘플·워밍업·체인 수
- 적합: 사후 median 에서의 NLL, 시즌 총 관측 대비 모형 총합의 비(peak ratio 와
  season-total ratio). 4시즌 합동 적합과 같은 방식으로 계산할 것

여기에 더해 최상위에 대조 블록을 넣는다.

- `shared` — 4시즌 합동 적합의 π 와 `phi_nb` (기존 산출물에서 읽어온다. 새로 돌리지 말 것)
- `spread` — 시즌별 π 의 채널별 범위(min~max)와, 각 시즌 π 가 공유 π 의 90% CI 안에
  들어오는지 여부
- `delta_fit` — 공유 π 대비 시즌별 π 의 적합 개선분. 시즌별 NLL 합 vs 합동 적합의 NLL.
  "negligible cost in fit" 주장이 성립하는지가 여기서 갈린다

원고에 들어갈 수치는 전부 이 JSON 에서 인용한다. 추정값·임의값을 쓰지 말 것.

### 3.2 그림

```
figures/paper_fluA4/{pdf,png}/per_season_pi.{pdf,png}
```

채널 4개를 x축(또는 4패널)으로, 시즌 5개를 색·마커로. 각 점에 90% CI 막대.
합동 적합의 공유 π 를 수평선(또는 회색 띠로 CI)으로 깔아서 시즌별 값이 그 안에
들어오는지 한눈에 보이게 한다. 2015-16 은 나머지와 구분되게(빈 마커 등) 표시하고
범례에 "excluded from the main fit" 을 적는다.

라벨은 영어. 한글 금지.

### 3.3 보고

작업이 끝나면 아래를 본문으로 보고하라. 파일 경로만 남기지 말 것.

1. 시즌별 π 4채널 median (90% CI) 표
2. 공유 π 와의 차이가 CI 안인지 밖인지, 채널별로
3. `delta_fit` — 시즌별 π 를 허용했을 때 적합이 얼마나 좋아지는가
4. **2015-16 전용 π 로 정점이 맞는가.** 맞으면 "공유 배분 때문"이고, 여전히 안 맞으면
   "그 시즌 자체가 다르다"이다. 이 한 줄이 Discussion 의 2015-16 문단을 바꾼다
5. 수렴 진단에 문제가 있는 시즌
6. §1 에서 이 문서와 실제 코드가 어긋났던 지점

---

## 4. 순서

1. §1 의 파일들을 읽고 `C` 의 구조와 실행 설정을 확인한다.
2. **스모크 런 먼저.** 한 시즌(2017-18), 워밍업·샘플을 아주 작게(예: 50/50, 체인 1개)
   잡아 끝까지 돌아가는지 본다. 여기서 메모리나 shape 문제를 잡는다.
3. 스모크가 통과하면 5시즌 본 적합. 시즌마다 독립이므로 순차로 돌리고, 시즌 하나가
   끝날 때마다 중간 결과를 저장한다. 중간에 끊겨도 처음부터 다시 돌리지 않게 할 것.
4. JSON 을 쓰고 그림을 그린다.
5. 그림을 `manuscript/figures_fluA4/{pdf,png}/` 로 복사한다 —
   `~/Documents/cowork/sickleave_InfluenzaA/manuscript/figures_fluA4/`.
   복사하지 않으면 원고가 집지 못한다.
6. 보고.

**원고(`main_v6.tex`, `supplement_v6.tex`)는 이번에 수정하지 않는다.** 수치가 나오면
원고 반영은 Cowork 쪽에서 한다.

---

## 5. 막히면

- 시즌 하나만 담은 `C` 를 만드는 과정에서 dict 키가 시즌 문자열이 아니라 인덱스로
  들어가 있을 수 있다. `C['seasons']` 의 원소를 그대로 키로 쓰는지 먼저 확인할 것.
- 2015-16 의 인플루엔자 A 비율이 다른 시즌과 크게 다르므로(그 시즌 제외 사유다)
  관측 스케일이 달라 초기값이 안 잡힐 수 있다. `log_R0` 사전분포는 그대로 두고,
  초기값만 조정해 볼 것. 사전분포를 바꿨다면 보고에 반드시 적을 것.
- 메모리가 터지면 체인을 병렬(`chain_method="parallel"`)이 아니라 순차로 돌린다.
- 2시간을 넘기면 멈추고 어디까지 됐는지 보고하라.
