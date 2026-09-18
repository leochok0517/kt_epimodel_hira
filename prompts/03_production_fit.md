# 03 — 본 NUTS 실행 + 수렴 진단

`prompts/99_conventions.md` 를 함께 적용할 것. 선행: `01_smoke.md` 통과.

## 할 일

1. detach 실행:
   ```
   mkdir -p logs
   nohup caffeinate -i .venv/bin/python scripts/nuts_fluA_run.py \
       --warmup 1000 --samples 2000 --chains 4 --target-accept 0.95 \
       > logs/fluA_prod.log 2>&1 & disown
   ```
   v5 기준 wall time 은 수 시간. warmup state 는
   `outputs/eda/nuts_fluA_warmup_state.pkl` 에 저장되므로 sampling 만 늘리려면
   `--resume-from` 으로 재개.

2. 진행 중 주기적으로 로그 확인. 다음이 보이면 **중단하고 보고**:
   - warmup 중 divergence 가 지속 발생
   - `saturation` (max tree depth 도달 비율) 이 30% 초과
   - NaN / inf

3. 완료 후 `outputs/fluA/nuts_fluA_diagnostics.json` 기준으로 판정:
   | 기준 | 목표 |
   |---|---|
   | r_hat | ≤ 1.01 |
   | ess_bulk / ess_tail | ≥ 400 |
   | sampling divergence | 0 |
   | saturation | 낮을수록 좋음 |

4. 목표 미달이면 원인 가설을 세워 보고. 자주 나오는 순서:
   - 시즌 수가 늘어 log_R0 차원이 커진 만큼 warmup 부족 → warmup 증량
   - A 보정으로 obs 스케일이 줄어 `phi_nb` 사후가 이동 → prior 재검토
   - 특정 시즌만 r_hat 나쁨 → 그 시즌 fit 그림부터 확인

## 완료 기준

- `outputs/fluA/nuts_fluA_diagnostics.json` + `outputs/eda/nuts_fluA_raw.npz`
- 진단 표와 판정, 미달 시 다음 수순 제안
