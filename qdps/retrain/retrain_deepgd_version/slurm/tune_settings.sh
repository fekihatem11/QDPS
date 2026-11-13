#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
# One tune_settings.py run (see that file). GPU type, memory and time come from
# submit_tuning.sh.
set -euo pipefail
source "${QDPS_REPO_DIR:-$HOME/QDPS}/qdps/retrain/slurm/env.sh"  # cluster settings from .env
PROJECT=$QDPS_PROJECT_DIR
export QDPS_RETRAIN_DIR=$PROJECT/retrain
module load StdEnv/2023 python/3.10 gcc/12.3 opencv/4.10.0 cuda cudnn
source $PROJECT/.venv/bin/activate
cd $HOME/QDPS
echo "args=$* node=$(hostname) commit=$(git rev-parse --short HEAD)"
python -u -m qdps.retrain.retrain_deepgd_version.tune_settings "$@"
