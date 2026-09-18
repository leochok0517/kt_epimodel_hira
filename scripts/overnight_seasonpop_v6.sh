#!/usr/bin/env bash
# Overnight orchestrator v6 — 상호성 보정(방식 A) 접촉행렬 + fresh NUTS warmup.
# scripts/overnight_seasonpop.sh 의 사본. v5 원본은 건드리지 않음(재현용 보존).
# 바뀐 것: (1) S1 warmup 재사용 제거(fresh warmup, --resume-from 없음,
#   timeout 39600s) — 단일 4ch×500 실행 후 사전 등록 기준으로 2번째 실행
#   필요 여부를 판정 (2) preflight 에 전국 G 체크섬 + 접촉행렬 상호성 확인 추가.
# 첫째 밤: DO_S3_ONWARD=0 으로 S1+S2까지만.
# Usage:
#   caffeinate -i -s nohup bash scripts/overnight_seasonpop_v6.sh > /dev/null 2>&1 &
#   DO_S3_ONWARD=0 caffeinate -i -s nohup bash scripts/overnight_seasonpop_v6.sh > /dev/null 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/logs/overnight_seasonpop_v6.log"
mkdir -p "$REPO/logs"

ED="$REPO/outputs/eda"
V4_WARMUP="$ED/nuts_v4_warmup_state.pkl"

# Config
DO_S5="${DO_S5:-1}"    # sens1+sens2 seasonpop
DO_S6="${DO_S6:-1}"    # sens3+sens3ext+sens4+sens5+sens6 seasonpop
DO_S3_ONWARD="${DO_S3_ONWARD:-1}"   # 0 = 첫째 밤(S1+S2까지만)
GLOBAL_TIMEOUT="${GLOBAL_TIMEOUT:-50400}"   # 14h (첫째 밤은 이 값 그대로도 충분; 둘째 밤 전체는 86400 권장)

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
ntfy() { curl -s -d "$1" ntfy.sh/hwcho-nuts >/dev/null 2>&1 || true; }

export PYTHONPATH="scripts:${PYTHONPATH:-}"

# Env for NUTS steps
export XLA_FLAGS='--xla_force_host_platform_device_count=4'
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2

# Run a step with timeout + resume check + ntfy.
# args: STEP_NAME  SENTINEL_FILE  TIMEOUT_SEC  CMD...
run_step() {
    local name="$1"; shift
    local sentinel="$1"; shift
    local timeout_s="$1"; shift
    if [ -f "$sentinel" ]; then
        log "[$name] SKIP (sentinel exists: $sentinel)"
        return 0
    fi
    log "[$name] START  (timeout=${timeout_s}s)  cmd: $*"
    ntfy "overnight $name START"
    local start_ts=$(date +%s)
    ( "$@" ) >> "$LOG" 2>&1 &
    local pid=$!
    ( sleep "$timeout_s"; if kill -0 $pid 2>/dev/null; then
        kill $pid; log "[$name] TIMEOUT (${timeout_s}s) — killed pid $pid"; fi ) &
    local wd=$!
    wait $pid
    local es=$?
    kill $wd 2>/dev/null || true
    local dur=$(( $(date +%s) - start_ts ))
    if [ $es -eq 0 ] && [ -f "$sentinel" ]; then
        log "[$name] DONE  wall=${dur}s  exit=$es"
        ntfy "overnight $name DONE (${dur}s)"
    else
        log "[$name] FAIL  wall=${dur}s  exit=$es  sentinel_exists=$([ -f "$sentinel" ] && echo y || echo n)"
        ntfy "overnight $name FAIL exit=$es"
        return 1
    fi
    return 0
}

# ────────────── PREFLIGHT ──────────────
log "============================================================"
log "OVERNIGHT SEASONPOP ORCHESTRATOR START"
log "  DO_S5=$DO_S5 DO_S6=$DO_S6 GLOBAL_TIMEOUT=$GLOBAL_TIMEOUT"
log "============================================================"

check_pop() {
    local season=$1; local expected=$2
    local f="$REPO/data/population/pop_${season}.csv"
    if [ ! -f "$f" ]; then log "[preflight] MISSING $f"; return 1; fi
    local sum=$(python -c "
import csv
tot=0
with open('$f') as fp:
    for r in csv.DictReader(fp):
        try: tot += int(round(float(r['population'])))
        except: pass
print(tot)")
    if [ "$sum" != "$expected" ]; then
        log "[preflight] $f checksum $sum != $expected"; return 1
    fi
    log "[preflight] $f sum=$sum ok"
    return 0
}
check_pop 2016_17 25590465 || { ntfy "overnight preflight FAIL"; exit 1; }
check_pop 2017_18 25679863 || { ntfy "overnight preflight FAIL"; exit 1; }
check_pop 2019_20 25925799 || { ntfy "overnight preflight FAIL"; exit 1; }

check_pop_national() {
    local f="$REPO/data/population/national/pop_national_2024_01.csv"
    if [ ! -f "$f" ]; then log "[preflight] MISSING $f"; return 1; fi
    local sum=$(python -c "
import csv
tot=0
with open('$f') as fp:
    for r in csv.DictReader(fp):
        tot += int(r['population'])
print(tot)")
    if [ "$sum" != "51313912" ]; then
        log "[preflight] $f checksum $sum != 51313912 (전국 G 아님)"; return 1
    fi
    log "[preflight] $f sum=$sum ok (전국 G)"
    return 0
}
check_pop_national || { ntfy "overnight v6 preflight FAIL national G"; exit 1; }

check_contact_reciprocity() {
    python -c "
import sys, numpy as np
sys.path.insert(0, 'scripts')
from kt_data.data.load_contact import load_contact_matrices
import phase1_reciprocity_v2 as P1
G, _ = P1.load_G('pop_national_2024_01.csv', 51_313_912)
ok_all = True
for path_label, path in [('term', None), ('vacation', P1.VAC_NPZ)]:
    m = load_contact_matrices(path=path)
    for k in ('C_home','C_work','C_school','C_other'):
        M = m[k].T
        T = G[:, None] * M
        ok = np.allclose(T, T.T, atol=1e-8, rtol=1e-6)
        ok_all &= ok
        if not ok:
            print(f'RECIPROCITY FAIL {path_label}/{k}')
sys.exit(0 if ok_all else 1)
"
}
if check_contact_reciprocity; then
    log "[preflight] 접촉행렬 상호성(방식 A) 확인 — OK (학기+방학)"
else
    log "[preflight] 접촉행렬 상호성 확인 FAIL — 보정 행렬이 배치돼 있지 않거나 손상됨"
    ntfy "overnight v6 preflight FAIL reciprocity"; exit 1
fi

if [ ! -f "$V4_WARMUP" ]; then
    log "[preflight] v4 warmup pkl missing: $V4_WARMUP (참고용 존재 확인 — v6은 이걸 재사용하지 않음)"
fi

log "[preflight] OK"
ntfy "overnight v6 preflight OK — starting chain"

# ────────────── S1+S2 (v6): fresh warmup, 조건부 2번째 실행 ──────────────
V6_WARMUP="$ED/nuts_seasonpop_v6_warmup_state.pkl"
RAW_A="$ED/nuts_seasonpop_raw_v6a.npz"
RAW_B="$ED/nuts_seasonpop_raw_v6b.npz"
MERGED_SENTINEL="$ED/nuts_seasonpop_merged.npz"

if [ -f "$MERGED_SENTINEL" ]; then
    log "[S1_S2_v6] SKIP (merged sentinel exists: $MERGED_SENTINEL)"
else
    # S1a: fresh warmup + sampling, 4ch x 500 (--resume-from 없음)
    run_step "S1a_nuts_v6_fresh" "$RAW_A" 39600 \
        uv run python -u scripts/nuts_seasonpop.py \
            --warmup 500 --samples 500 --chains 4 \
            --seed 601 --tag v6a

    # nuts_seasonpop.py 는 warmup pkl 을 고정 이름으로 저장함(태그 무관).
    # v4 pkl 을 덮어쓰지 않도록 별도 이름으로 복사(데이터 파일 조작, 코드 미수정).
    if [ -f "$ED/nuts_seasonpop_warmup_state.pkl" ] && [ ! -f "$V6_WARMUP" ]; then
        cp "$ED/nuts_seasonpop_warmup_state.pkl" "$V6_WARMUP"
        log "[S1a_nuts_v6_fresh] warmup state copied -> $V6_WARMUP (nuts_v4_warmup_state.pkl 은 건드리지 않음)"
    fi

    # S1b: 단일 실행(2000 draws) 사전 등록 기준 판정
    log "[S1b_check_single] START"
    uv run python -u scripts/nuts_seasonpop_v6_check_single.py "$RAW_A" >> "$LOG" 2>&1
    CHECK_ES=$?
    if [ $CHECK_ES -eq 0 ]; then
        log "[S1b_check_single] PASS — 단일 4ch x 500 실행으로 충분 (2000 draws), 두 번째 실행 생략"
        uv run python -u scripts/nuts_seasonpop_v6_finalize_single.py "$RAW_A" >> "$LOG" 2>&1
        log "[S1c_finalize_single] DONE -> $MERGED_SENTINEL"
    else
        log "[S1b_check_single] FAIL — 사전 등록 기준 미달, 두 번째 4ch x 500 실행 필요(v6 warmup 재사용)"
        run_step "S1d_nuts_v6_second" "$RAW_B" 18000 \
            uv run python -u scripts/nuts_seasonpop.py \
                --warmup 500 --samples 500 --chains 4 \
                --resume-from "$V6_WARMUP" --seed 602 --tag v6b
        # nuts_seasonpop_merge.py 는 reuse/extended 파일명을 하드코딩하고 있어
        # (핵심 코드 미수정 원칙상 고치지 않음) 데이터 파일 복사로 자리를 맞춘다.
        # v5의 reuse/extended.npz는 이미 outputs/eda_v5_uncorrected/ 에 백업돼 있음.
        cp "$RAW_A" "$ED/nuts_seasonpop_raw_reuse.npz"
        cp "$RAW_B" "$ED/nuts_seasonpop_raw_extended.npz"
        log "[S1e_prep_merge] v6a/v6b -> reuse/extended.npz 복사 (merge.py 원본 재사용 위함)"
        XLA_FLAGS='--xla_force_host_platform_device_count=1' \
        OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
        run_step "S2_merge" "$MERGED_SENTINEL" 600 \
            uv run python -u scripts/nuts_seasonpop_merge.py
    fi
fi

if [ "$DO_S3_ONWARD" != "1" ]; then
    log "============================================================"
    log "DO_S3_ONWARD=0 — 첫째 밤 종료 (S1+S2까지). S3 이후는 여기서 멈춤."
    log "============================================================"
    ntfy "overnight v6 첫째 밤 완료 (S1+S2) — logs/overnight_seasonpop_v6.log 확인"
    exit 0
fi

# ────────────── S3: policy_seasonpop forward ──────────────
XLA_FLAGS='--xla_force_host_platform_device_count=1' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
run_step "S3_policy" "$ED/policy_posterior_seasonpop.json" 10800 \
    uv run python -u scripts/policy_seasonpop.py

# ────────────── S4: refit_compare (old vs new) ──────────────
run_step "S4_compare" "$ED/refit_compare_old_vs_new.json" 300 \
    uv run python -u scripts/refit_compare.py

# ────────────── S5: sens1 + sens2 seasonpop ──────────────
if [ "$DO_S5" = "1" ]; then
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S5_sens1" "$ED/sens_piwork_kappa_seasonpop.json" 7200 \
        uv run python -u scripts/sens_seasonpop_runner.py sens1
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S5_sens2" "$ED/sens_kappa_upper_seasonpop.json" 10800 \
        uv run python -u scripts/sens_seasonpop_runner.py sens2
fi

# ────────────── S6: sens3..sens6 seasonpop ──────────────
if [ "$DO_S6" = "1" ]; then
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S6_sens3" "$ED/sens_R0_transition_seasonpop.json" 7200 \
        uv run python -u scripts/sens_seasonpop_runner.py sens3
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S6_sens3ext" "$ED/sens_R0_transition_ext_seasonpop.json" 7200 \
        uv run python -u scripts/sens_seasonpop_runner.py sens3ext
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S6_sens4" "$ED/sens_w_sweep_seasonpop.json" 21600 \
        uv run python -u scripts/sens_seasonpop_runner.py sens4
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S6_sens5" "$ED/sens_baseline_p_seasonpop.json" 3600 \
        uv run python -u scripts/sens_seasonpop_runner.py sens5
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S6_sens6" "$ED/sens_erlang_n_seasonpop.json" 7200 \
        uv run python -u scripts/sens_seasonpop_runner.py sens6
fi

# ────────────── S7: final summary ──────────────
log "============================================================"
log "OVERNIGHT SEASONPOP FINISHED"
log "============================================================"
summary=$(python -c "
import json, os
from pathlib import Path
ED = Path('$ED')
lines = []
for name, f in [
    ('S2 diag', 'nuts_seasonpop_merged_diagnostics.json'),
    ('S3 pol',  'policy_posterior_seasonpop.json'),
    ('S4 cmp',  'refit_compare_old_vs_new.json'),
    ('S5 sens1', 'sens_piwork_kappa_seasonpop.json'),
    ('S5 sens2', 'sens_kappa_upper_seasonpop.json'),
    ('S6 sens3', 'sens_R0_transition_seasonpop.json'),
    ('S6 sens3ext', 'sens_R0_transition_ext_seasonpop.json'),
    ('S6 sens4', 'sens_w_sweep_seasonpop.json'),
    ('S6 sens5', 'sens_baseline_p_seasonpop.json'),
    ('S6 sens6', 'sens_erlang_n_seasonpop.json'),
]:
    p = ED / f
    lines.append(f'  {name}: ' + ('OK ' + str(p.stat().st_size) if p.exists() else 'MISS'))
print('\n'.join(lines))
")
log "SUMMARY:"
echo "$summary" >> "$LOG"

# refit_compare 핵심 라인
if [ -f "$REPO/logs/refit_compare.md" ]; then
    log "--- refit_compare.md (첫 40줄) ---"
    head -40 "$REPO/logs/refit_compare.md" >> "$LOG"
fi

ntfy "overnight_seasonpop 최종 완료 — logs/overnight_seasonpop.log 확인"
log "END"
