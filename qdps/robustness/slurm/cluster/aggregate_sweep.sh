#!/bin/bash
# Aggregate per_config_<NN>.json files produced by cluster_sweep_array.sh
# into one sweep_results.json (same format as the sequential --sweep mode).
# Tiny job: pure JSON I/O + ranking, no UMAP/HDBSCAN.
#
# Submit as a dependent job after the sweep array:
#     ARRAY_JOBID=$(sbatch --parsable --export=ALL,SUBJECT=mnist_LeNet1 \
#                          qdps/robustness/slurm/cluster/cluster_sweep_array.sh)
#     sbatch --dependency=afterok:$ARRAY_JOBID \
#             --export=ALL,SUBJECT=mnist_LeNet1 \
#             qdps/robustness/slurm/cluster/aggregate_sweep.sh
#
# Logs: $SCRATCH/QDPS/slurm_logs/sweep-agg-<jobid>.out

#SBATCH --account=def-manel131
#SBATCH --job-name=sweep-agg
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/QDPS/slurm_logs/sweep-agg-%j.out
#SBATCH --error=/scratch/%u/QDPS/slurm_logs/sweep-agg-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fekihatem72@gmail.com

set -e

module purge
module load python/3.10

source "$SCRATCH/QDPS/.venv/bin/activate"

cd "$HOME/QDPS"

SUBJECT=${SUBJECT:-mnist_LeNet1}
SEED=${SEED:-0}

echo "----- SLURM job info -----"
echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Host:     $(hostname)"
echo "Subject:  $SUBJECT"
echo "Seed:     $SEED"
echo "Started:  $(date -Iseconds)"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "--------------------------"

python -u qdps/robustness/cluster.py \
        --subject "$SUBJECT" --seed "$SEED" --aggregate-sweep

echo "Finished: $(date -Iseconds)"
