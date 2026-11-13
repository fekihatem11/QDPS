#!/bin/bash
# Load the cluster settings from <repo>/.env (see .env.example at the repository root).
# Sourced by the SLURM job scripts and by the submit scripts. Exports:
#   SBATCH_ACCOUNT    allocation account, read natively by sbatch
#   QDPS_PROJECT_DIR  project directory holding .venv/ and retrain/
QDPS_REPO_DIR="${QDPS_REPO_DIR:-$HOME/QDPS}"
if [ -f "$QDPS_REPO_DIR/.env" ]; then
    set -a
    source "$QDPS_REPO_DIR/.env"
    set +a
fi
: "${SBATCH_ACCOUNT:?SBATCH_ACCOUNT is not set: copy .env.example to .env and fill it in}"
: "${QDPS_PROJECT_DIR:?QDPS_PROJECT_DIR is not set: copy .env.example to .env and fill it in}"
