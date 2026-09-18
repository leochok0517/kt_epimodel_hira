# 01 — 스모크 실행 + A 타깃 정합성 확인

`prompts/99_conventions.md` 를 함께 적용할 것.

## 배경

인플루엔자 A 전용 재피팅 파이프라인이 새로 들어왔다 (`CLAUDE.md` 의
"Influenza A 전용 재피팅" 절 참조). 타깃 생성 경로는 컨테이너에서 7 시즌 모두
검증했지만, **JAX 스택 (`fluA_setup` → `nuts_fluA` → NUTS) 은 아직 한 번도
실행된 적이 없다.** 이 세션의 목적은 그 경로를 처음 돌려보는 것이다.

## 할 일

1. 시즌 인구 파일 생성 (없으면):
   ```
   .venv/bin/python scripts/build_pop_seasons_fluA.py
   ```
   기존 3 시즌 (2016_17 / 2017_18 / 2019_20) 값이 `pop_all_seasons.csv` 와
   정확히 일치한다는 assert 가 통과해야 한다. 실패하면 **거기서 멈추고 보고**.

2. 스모크:
   ```
   .venv/bin/python scripts/nuts_fluA_run.py --smoke 2>&1 | tee logs/fluA_smoke.log
   ```
   점추정 + 20 warmup / 20 sample × 2 chain. 실패하면 traceback 을 그대로 보고.

3. 타깃 정합성 확인 — 다음 값을 표로:
   - 시즌별 A-share coverage, `obs_A / obs_all`, 피크 주, 인구
   - v5 의 3 시즌 (2016-17, 2017-18, 2019-20) 에 대해 `hira_counts_all` 이
     `load_hira_target_by_age` 의 원래 카운트와 **정확히 일치**하는지
     (A 보정 외에 아무것도 안 바뀌었음을 증명)
   - 초기 시드 `estimate_initial_infected_from_fluA` 총합이 시즌별로 1e3 자릿수인지

4. **R0 · π 는 이 단계에서 판단하지 말 것.** 스모크는 최적화를 하지 않는다 —
   jit 된 NLL 을 R0 세 점에서 평가하는 probe 만 있다 (이유는 아래 "흔한 함정").
   20 draw NUTS 의 사후평균은 r_hat 이 1.5 를 훌쩍 넘으므로 근거로 쓸 수 없다.
   probe 에서 볼 것은 세 가지뿐:
   - `finite: True`
   - R0 에 대한 응답 (`spread` 가 크고, R0=2.0 이 양옆보다 낮은가)
   - `per-evaluation` 이 수십 ms 수준인가 (수 초면 본 실행 추정이 어긋난다)

   실제 점추정이 필요하면 `final_pipeline_confirmed.fit_season_pi` 를 쓸 것
   (jit + value_and_grad + L-BFGS). R0 · π 의 v5 대조는 `prompts/04_compare_v5.md`
   에서 수렴한 사후분포로 한다.

## 완료 기준

- `outputs/fluA/nuts_fluA_smoke_diagnostics.json` 생성
- 위 3·4 의 표와 해석이 보고에 포함
- 실패 시: 어디서 왜 깨졌는지와 최소 수정안 (기존 파일 수정 없이)

## 흔한 함정

- `obs` 는 정수가 아니다. 정수 캐스팅·반올림하지 말 것.
- `n_weeks` 는 52 고정. 시즌 주 수가 52 가 아니면 `fluA_setup` 이 명시적으로
  에러를 낸다 — 조용히 자르지 말 것.
- 스모크의 r_hat / ESS 는 20 draw 라 의미 없다. 판단 근거로 쓰지 말 것.
- **loss 를 jit 밖에서 반복 호출하지 말 것.** 2026-08-28 에 스모크의 점추정이
  `make_joint_loss_fluA` 를 jit 없이 Nelder-Mead 로 수백 번 호출해 182 분 멈추고
  메모리가 54GB 까지 갔다. 그래서 점추정을 제거했다.
- NUTS warmup 초반이 60-90 s/it 로 느린 것은 **정상**이다 — step size 가 잡히기
  전까지 트리가 최대 깊이 (255 leapfrog) 로 포화한다. 적응 후 4-6 s/it 로 떨어진다.
  속도로 건강성을 판단하지 말고 진단 json 의 `mean_num_steps` (~29 기대) 와
  `saturation_pct` (sampling 0% 기대) 로 볼 것.
