#!/usr/bin/env bash
# Overnight orchestrator: S1..S7 for season-pop refit.
# resume-aware: each step skipped if durable output already exists.
# Single log: logs/overnight_seasonpop.log
# S5=1 S6=1 (per user answers)
# Usage:
#   caffeinate -i -s nohup bash scripts/overnight_seasonpop.sh > /dev/null 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/logs/overnight_seasonpop.log"
mkdir -p "$REPO/logs"

ED="$REPO/outputs/eda"
V4_WARMUP="$ED/nuts_v4_warmup_state.pkl"

# Config
DO_S5="${DO_S5:-1}"    # sens1+sens2 seasonpop
DO_S6="${DO_S6:-1}"    # sens3+sens3ext+sens4+sens5+sens6 seasonpop
GLOBAL_TIMEOUT="${GLOBAL_TIMEOUT:-50400}"   # 14h

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

if [ ! -f "$V4_WARMUP" ]; then
    log "[preflight] v4 warmup pkl missing: $V4_WARMUP"; ntfy "overnight preflight FAIL warmup"; exit 1
fi

log "[preflight] OK"
ntfy "overnight preflight OK — starting chain"

# ────────────── S1: NUTS-ext (seed 137) ──────────────
# If a nuts_seasonpop.py extended process is already running, wait for its output.
S1_SENTINEL="$ED/nuts_seasonpop_raw_extended.npz"
if [ -f "$S1_SENTINEL" ]; then
    log "[S1_nuts_ext] SKIP (sentinel exists)"
elif pgrep -f "nuts_seasonpop.py.*--tag extended" >/dev/null 2>&1; then
    RUNPID=$(pgrep -f "nuts_seasonpop.py.*--tag extended" | head -1)
    log "[S1_nuts_ext] existing pid=$RUNPID — waiting for sentinel (max 18000s)"
    ntfy "overnight S1 detected running pid=$RUNPID — waiting"
    S1_WAIT_START=$(date +%s)
    while [ ! -f "$S1_SENTINEL" ]; do
        if ! kill -0 "$RUNPID" 2>/dev/null; then
            log "[S1_nuts_ext] pid $RUNPID gone but sentinel missing"
            if [ ! -f "$S1_SENTINEL" ]; then
                log "[S1_nuts_ext] FAIL — pid gone, no output"
                ntfy "overnight S1 FAIL — pid died before output"
                exit 1
            fi
            break
        fi
        NOW=$(date +%s)
        ELAPSED=$(( NOW - S1_WAIT_START ))
        if [ $ELAPSED -gt 18000 ]; then
            log "[S1_nuts_ext] wait timeout 18000s — killing pid"
            kill "$RUNPID" 2>/dev/null || true
            ntfy "overnight S1 timeout wait"
            exit 1
        fi
        sleep 60
    done
    log "[S1_nuts_ext] existing run produced sentinel"
    ntfy "overnight S1 DONE (via existing run)"
else
    run_step "S1_nuts_ext" "$ED/nuts_seasonpop_raw_extended.npz" 18000 \
        uv run python -u scripts/nuts_seasonpop.py \
            --warmup 500 --samples 500 --chains 4 \
            --resume-from "$V4_WARMUP" --seed 137 --tag extended
fi

# ────────────── S2: merge + diagnostics ──────────────
if [ -f "$ED/nuts_seasonpop_raw_reuse.npz" ] && [ -f "$ED/nuts_seasonpop_raw_extended.npz" ]; then
    XLA_FLAGS='--xla_force_host_platform_device_count=1' \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    run_step "S2_merge" "$ED/nuts_seasonpop_merged_diagnostics.json" 600 \
        uv run python -u scripts/nuts_seasonpop_merge.py
else
    log "[S2_merge] SKIP — raw files missing"
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
