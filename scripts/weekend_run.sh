#!/usr/bin/env bash
# Influenza-A refit — unattended weekend chain.
#
#   population -> production NUTS -> v5 comparison -> policy forward -> report
#
# Design notes:
#   * Every stage is gated. A stage that fails stops the chain but the report
#     is still written, so Monday always has something to read.
#   * The season set is decided at run time: if the KOSIS export carries the
#     2013 and 2014 June columns the chain fits seven seasons, otherwise five.
#     No edit needed either way.
#   * The NUTS warm-up state and the policy partial file both persist, so a
#     stage that dies can be resumed rather than restarted.
#   * caffeinate keeps the machine awake for the whole chain.
#
# Usage (from the repo root):
#   nohup caffeinate -ims bash scripts/weekend_run.sh > logs/weekend.log 2>&1 &
#   disown
#
# Watch:  tail -f logs/weekend.log
# Stop:   pkill -f weekend_run.sh   (then also pkill -f nuts_fluA_run.py)

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-$REPO/.venv/bin/python}"
LOGS="$REPO/logs"
OUTD="$REPO/outputs/fluA"
mkdir -p "$LOGS" "$OUTD"

WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
CHAINS="${CHAINS:-4}"
NPOST="${NPOST:-500}"
# TAG keeps runs with different season sets from overwriting each other.
# SEASONS_OVERRIDE pins the season set instead of taking every season that has
# a population — use it to drop a season without editing any file.
TAG="${TAG:-fluA}"
SEASONS_OVERRIDE="${SEASONS_OVERRIDE:-}"
# default tag keeps the original filenames; any other tag gets a suffix
if [ "$TAG" = "fluA" ]; then SUF=""; else SUF="_$TAG"; fi
STATUS="$OUTD/run_status${SUF}.json"

STAMP="$(date +%Y%m%d_%H%M%S)"
STAGE_JSON=""
ABORT_AT=""
ABORT_WHY=""

say() { echo "[$(date '+%F %T')] $*"; }

record() {   # name result wall log
  local entry
  entry="{\"name\":\"$1\",\"result\":\"$2\",\"wall_sec\":$3,\"log\":\"$4\"}"
  if [ -z "$STAGE_JSON" ]; then STAGE_JSON="$entry"; else STAGE_JSON="$STAGE_JSON,$entry"; fi
  write_status
}

write_status() {
  {
    printf '{\n'
    printf '  "started": "%s",\n' "$STAMP"
    printf '  "updated": "%s",\n' "$(date '+%F %T')"
    printf '  "seasons": %s,\n' "${SEASONS_JSON:-[]}"
    printf '  "tag": "%s",\n' "$TAG"
    printf '  "config": {"warmup": %s, "samples": %s, "chains": %s, "n_post": %s},\n' \
           "$WARMUP" "$SAMPLES" "$CHAINS" "$NPOST"
    printf '  "aborted_at": %s,\n' "$( [ -n "$ABORT_AT" ] && echo "\"$ABORT_AT\"" || echo null )"
    printf '  "abort_reason": %s,\n' "$( [ -n "$ABORT_WHY" ] && echo "\"$ABORT_WHY\"" || echo null )"
    printf '  "stages": [%s]\n' "$STAGE_JSON"
    printf '}\n'
  } > "$STATUS"
}

finish() {
  say "writing report"
  "$PY" scripts/weekend_report.py "$TAG" 2>&1 | tee -a "$LOGS/weekend_report_$STAMP.log"
  say "REPORT: $OUTD/REPORT${SUF}.md"
  if [ -n "$ABORT_AT" ]; then
    say "CHAIN INCOMPLETE — stopped at $ABORT_AT: $ABORT_WHY"
    exit 1
  fi
  say "CHAIN COMPLETE"
  exit 0
}

abort() { ABORT_AT="$1"; ABORT_WHY="$2"; write_status; finish; }

run_stage() {   # name log_basename command...
  local name="$1"; shift
  local logf="$LOGS/$1_$STAMP.log"; shift
  say "STAGE $name -> $logf"
  local t0 t1
  t0=$(date +%s)
  "$@" > "$logf" 2>&1
  local rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    say "STAGE $name FAILED (exit $rc). Last 25 lines:"
    tail -25 "$logf"
    record "$name" "failed (exit $rc)" $((t1 - t0)) "$logf"
    return 1
  fi
  say "STAGE $name ok ($(( (t1 - t0) / 60 )) min)"
  record "$name" "ok" $((t1 - t0)) "$logf"
  return 0
}

say "=============================================================="
say "influenza-A weekend chain   repo=$REPO"
say "python=$PY  warmup=$WARMUP samples=$SAMPLES chains=$CHAINS n_post=$NPOST"
say "=============================================================="

[ -x "$PY" ] || { echo "python not found at $PY"; exit 2; }

# ---- stage 0a: contact matrices carry the reciprocity correction -----------
# The correction lives in the kt_data npz files, not in a call-site flag, so a
# restored backup or a re-exported matrix would silently un-correct the fit.
# Check it before spending hours on a fit that would have to be thrown away.
run_stage "reciprocity check" "weekend_0a_reciprocity" \
  "$PY" scripts/check_reciprocity_fluA.py || abort "reciprocity check" \
  "contact matrices are not the reciprocity-corrected ones — see the log; \
re-run scripts/deploy_reciprocity_matrices.py before fitting"

# ---- stage 0: populations, and decide the season set -----------------------
run_stage "population" "weekend_0_pop" \
  "$PY" scripts/build_pop_seasons_fluA.py || abort "population" \
  "build_pop_seasons_fluA.py failed — the KOSIS export is probably malformed"

SEASONS=$("$PY" - <<'EOF'
import csv, sys
from pathlib import Path
p = Path("data/population/pop_all_seasons_fluA.csv")
have = set()
for row in csv.DictReader(open(p)):
    y0 = row["season"].split("_")[0]
    have.add(f"{y0}-{int(y0)+1}")
# every season that passes the influenza-A single-wave screen, in order
screen = [f"{y}-{y+1}" for y in range(2013, 2020)]
sel = [s for s in screen if s in have]
print(" ".join(sel))
EOF
)
SEASONS_JSON=$(printf '%s' "$SEASONS" | awk '{for(i=1;i<=NF;i++) printf "%s\"%s\"", (i>1?",":""), $i}')
SEASONS_JSON="[$SEASONS_JSON]"
if [ -n "$SEASONS_OVERRIDE" ]; then
  # every requested season must actually have a population
  for s in $SEASONS_OVERRIDE; do
    echo "$SEASONS" | tr ' ' '\n' | grep -qx "$s" || abort "population" \
      "season $s was requested but has no population in pop_all_seasons_fluA.csv"
  done
  say "season set OVERRIDDEN: $SEASONS_OVERRIDE  (available: $SEASONS)"
  SEASONS="$SEASONS_OVERRIDE"
fi
say "season set: $SEASONS  ($(echo $SEASONS | wc -w | tr -d ' ') seasons)   tag=$TAG"
write_status

if [ -z "$SEASONS" ]; then
  abort "population" "no season has both a population and a passing single-wave screen"
fi

# ---- stage 1: production NUTS ---------------------------------------------
RESUME=""
WPKL="$REPO/outputs/eda/nuts_${TAG}_warmup_state.pkl"
WSEA="$REPO/outputs/eda/nuts_${TAG}_warmup_state.seasons.json"
if [ -f "$WPKL" ]; then
  # A warm-up state is only valid for the season set it was tuned on.
  if [ -f "$WSEA" ] && "$PY" -c "
import json,sys
prev=json.load(open('$WSEA')); want='''$SEASONS'''.split()
sys.exit(0 if list(prev)==want else 1)"; then
    say "reusing the existing warm-up state (season set matches)"
    RESUME="--resume-from $WPKL"
  else
    say "existing warm-up state is for a different season set — warming up fresh"
    mv "$WPKL" "$WPKL.stale_$STAMP"
    [ -f "$WSEA" ] && mv "$WSEA" "$WSEA.stale_$STAMP"
  fi
fi

NPZ="$REPO/outputs/eda/nuts_${TAG}_raw.npz"
rm -f "$NPZ"          # so a stale npz cannot be mistaken for this run's output

if ! run_stage "production fit" "weekend_1_fit" \
     "$PY" scripts/nuts_fluA_run.py --seasons $SEASONS --tag "$TAG" \
     --warmup "$WARMUP" --samples "$SAMPLES" --chains "$CHAINS" $RESUME; then
  # The posterior is written to npz immediately after sampling, before the
  # arviz summary and the diagnostics json. So a non-zero exit with the npz
  # present means sampling succeeded and something cosmetic downstream of it
  # failed — worth a warning, not worth discarding hours of sampling.
  if [ -f "$NPZ" ]; then
    say "WARNING: the fit exited non-zero but the posterior npz exists."
    say "         Sampling finished; the failure is after it (summary or"
    say "         diagnostics json). Continuing — check weekend_1_fit log."
    record "production fit (partial)" "npz written, post-processing failed" 0 \
           "$LOGS/weekend_1_fit_$STAMP.log"
  else
    abort "production fit" \
      "NUTS did not finish and wrote no posterior — the warm-up state is saved, so resume rather than restart"
  fi
fi

[ -f "$NPZ" ] || abort "production fit" \
  "NUTS exited cleanly but wrote no posterior npz"

# ---- stage 2: compare against v5 ------------------------------------------
run_stage "v5 comparison" "weekend_2_compare" \
  "$PY" scripts/compare_v5_fluA.py --seasons $SEASONS --tag "$TAG" \
  --fluA-npz "$NPZ" \
  || say "comparison failed — continuing to the policy stage anyway"

# A flag here does not stop the chain: the policy numbers are still worth
# having, and the report puts the flag above them.
if [ -f "$OUTD/compare_v5_${TAG}.json" ]; then
  NFLAG=$("$PY" -c "import json;print(len(json.load(open('$OUTD/compare_v5_${TAG}.json'))['flags']))" 2>/dev/null || echo 0)
  [ "$NFLAG" != "0" ] && say "WARNING: $NFLAG flag(s) raised in the v5 comparison"
fi

# ---- stage 3: policy forward ----------------------------------------------
run_stage "policy forward" "weekend_3_policy" \
  "$PY" scripts/policy_fluA.py --seasons $SEASONS --n-post "$NPOST" \
  --tag "$TAG" --npz "$NPZ" \
  || say "policy forward failed — the partial file lets it resume on Monday"

# ---- stage 4: report -------------------------------------------------------
finish
