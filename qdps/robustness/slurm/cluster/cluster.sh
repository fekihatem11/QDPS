#!/bin/bash
# Per-instance fault clustering. Uses LOCKED_CONFIG from cluster.py
# (set after the one-time --sweep on seed 0). Runs as a 5-task job array.
#
# Submit:   sbatch --export=ALL,SUBJECT=mnist_LeNet1 qdps/robustness/slurm/cluster/cluster.sh
# Logs:     $SCRATCH/QDPS/slurm_logs/cluster-<jobid>_<task>.out

#SBATCH --account=def-manel131
#SBATCH --job-name=cluster
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/QDPS/slurm_logs/cluster-%A_%a.out
#SBATCH --error=/scratch/%u/QDPS/slurm_logs/cluster-%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=fekihatem72@gmail.com

set -e

module purge
module load python/3.10

source "$SCRATCH/QDPS/.venv/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TF_CPP_MIN_LOG_LEVEL=2

cd "$HOME/QDPS"

SUBJECT=${SUBJECT:-mnist_LeNet1}

echo "----- SLURM job info -----"
echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "Host:     $(hostname)"
echo "Subject:  $SUBJECT"
echo "Seed:     $SLURM_ARRAY_TASK_ID"
echo "Started:  $(date -Iseconds)"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "--------------------------"

python qdps/robustness/cluster.py --subject "$SUBJECT" --seed "$SLURM_ARRAY_TASK_ID"

echo "Finished: $(date -Iseconds)"
