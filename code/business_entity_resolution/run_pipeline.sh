#!/usr/bin/env bash
# End-to-end: data -> normalisation -> synonym learning -> blocking -> matcher -> decoded submission.
# Usage: bash run_pipeline.sh <dataset_dir> <work_dir> <output_dir>
#
# The defaults reproduce the submitted output/ files:
#   MAX_ROWS=0        training S1 cap (0 = full train set; e.g. 200000 for a quick run, S2/S3 sampled
#                     to keep the distractor ratio; use TRAIN_PCT=70 DEV_PCT=30 with it)
#   TRAIN_PCT=8       % of train S1 entities used to fit (~177k entities, ~7.9M candidate pairs)
#   DEV_PCT=4         % held out for the dev report
# The test split is never sampled: every test S1 entity is scored.
# On a 15 GB machine, wrap the call in `systemd-run --user --scope -p MemoryMax=10G ...` so a memory spike
# can only kill this job. Peak RAM is about 9 GB (training stage B, and the validator with --check-ids).
set -euo pipefail
DATA=${1:-dataset}; WORK=${2:-work}; OUT=${3:-output}
MAX_ROWS=${MAX_ROWS:-0}; TRAIN_PCT=${TRAIN_PCT:-8}; DEV_PCT=${DEV_PCT:-4}
SRC="$(cd "$(dirname "$0")" && pwd)/src"
PY=${PYTHON:-python}
$PY "$SRC/prep.py"    --data "$DATA" --work "$WORK" --max-rows "$MAX_ROWS"                 # parquet + pass-1 normalisation
$PY "$SRC/canon.py"   --data "$DATA" --work "$WORK"                                       # learn synonym tables, canonicalise
$PY "$SRC/block.py"   --work "$WORK" --split train                                        # train candidates + competition features
$PY "$SRC/train.py"   --data "$DATA" --work "$WORK" --train-pct "$TRAIN_PCT" --dev-pct "$DEV_PCT"   # two-stage XGBoost + dev report
$PY "$SRC/block.py"   --work "$WORK" --split test                                         # test candidates
$PY "$SRC/predict.py" --data "$DATA" --work "$WORK" --out "$OUT"                          # score, decode, write both TSVs
$PY "$SRC/validate_submission.py" --matching "$OUT/matching_results.tsv" \
    --candidate "$OUT/candidate_pairs.tsv" --test-dir "$DATA/test" --check-ids            # official checks (separate process)
