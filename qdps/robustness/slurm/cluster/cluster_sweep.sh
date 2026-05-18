#!/bin/bash
# One-time UMAP/HDBSCAN hyperparameter sweep on (subject, seed=0).
# Tries SETS's 80-config grid, ranks by closeness to target cluster count.
# After this finishes, pick the winning config and update LOCKED_CONFIG in
# qdps/robustness/cluster.py, then submit slurm/cluster/cluster.sh for all seeds.
#
# This is the sequential single-job sweep -- much slower than the parallel
# job-array version (slurm/cluster/cluster_sweep_array.sh). Kept as fallback.
#
# Submit:   sbatch --export=ALL,SUBJECT=mnist_LeNet1 qdps/robustness/slurm/cluster/cluster_sweep.sh
# Logs:     $SCRATCH/QDPS/slurm_logs/cluster-sweep-<jobid>.out

#SBATCH --account=def-manel131
#SBATCH --job-name=cluster-sweep
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/QDPS/slurm_logs/cluster-sweep-%j.out
#SBATCH --error=/scratch/%u/QDPS/slurm_logs/cluster-sweep-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fekihatem72@gmail.com

set -e

module purge
module load python/3.10

source "$SCRATCH/QDPS/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_CPP_MIN_LOG_LEVEL=2

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

python qdps/robustness/cluster.py --subject "$SUBJECT" --seed "$SEED" --sweep

echo "Finished: $(date -Iseconds)"
