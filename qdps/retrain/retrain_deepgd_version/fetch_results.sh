#!/bin/bash
# Copy the DeepGD-protocol retraining results from a cluster into the local results tree,
# then rebuild TABLES.md locally.
#
#   bash qdps/retrain/retrain_deepgd_version/fetch_results.sh <ssh-host>     # e.g. fir, narval
#
# Local layout (qdps/results/retrain_deepgd/):
#   subjects/<subject>.json     raw metrics (original accuracy, every run per condition, selections)
#   logs/<job>-<id>.out         SLURM logs of the retraining and preparation jobs
#   pilot/                      pilot_settings.py runs (per-epoch curves)
#   generated_check/            validity check of the generated test sets (grids, REPORT.md)
#   TABLES.md                   rebuilt from subjects/ by report.py
set -euo pipefail
HOST="${1:?usage: fetch_results.sh <ssh-host>}"
cd "$(git rev-parse --show-toplevel)"
LOCAL=qdps/results/retrain_deepgd
mkdir -p $LOCAL/subjects $LOCAL/logs $LOCAL/generated_check $LOCAL/pilot

rsync -a "$HOST:QDPS/$LOCAL/subjects/" $LOCAL/subjects/ --include='*.json' --exclude='*'
rsync -a "$HOST:QDPS/$LOCAL/generated_check/" $LOCAL/generated_check/
rsync -a "$HOST:QDPS/$LOCAL/pilot/" $LOCAL/pilot/ 2>/dev/null || true
# remote $SCRATCH is /scratch/<user> on the Alliance clusters
rsync -a "$HOST:/scratch/*/QDPS/logs/" $LOCAL/logs/ --include='rt-*.out' --include='prep-*.out' --include='pilot-*.out' --exclude='*'

.venv/bin/python -m qdps.retrain.retrain_deepgd_version.report
echo "Fetched from $HOST into $LOCAL:"
ls $LOCAL/subjects $LOCAL/logs
