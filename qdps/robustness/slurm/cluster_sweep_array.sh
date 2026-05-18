#!/bin/bash
# Parallelized version of the SETS 80-config sweep: each array task runs
# ONE config of cluster.py and writes per_config_<NN>.json. After this
# array finishes, submit aggregate_sweep.sh to combine into one
# sweep_results.json.
#
# Submit:
#     sbatch --export=ALL,SUBJECT=mnist_LeNet1 \
#             qdps/robustness/slurm/cluster_sweep_array.sh
#     sbatch --dependency=afterok:<jobid_above> \
#             --export=ALL,SUBJECT=mnist_LeNet1 \
#             qdps/robustness/slurm/aggregate_sweep.sh
#
# Logs: $SCRATCH/QDPS/slurm_logs/sweep-arr-<jobid>_<task>.out

#SBATCH --account=def-manel131
#SBATCH --job-name=sweep-arr
#SBATCH --array=0-79
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/QDPS/slurm_logs/sweep-arr-%A_%a.out
#SBATCH --error=/scratch/%u/QDPS/slurm_logs/sweep-arr-%A_%a.err
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

# Stagger task starts modulo 8 (within each cohort of 8 tasks, start 5s
# apart) to spread the Lustre I/O burst when tasks happen to land on the
# same node. The first task in each cohort starts immediately; the worst
# case wait is 7*5 = 35s.
STAGGER=$(( (SLURM_ARRAY_TASK_ID % 8) * 5 ))
echo "Staggering start by ${STAGGER}s..."
sleep "$STAGGER"

echo "----- SLURM job info -----"
echo "Job:      $SLURM_JOB_NAME ($SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "Host:     $(hostname)"
echo "Subject:  $SUBJECT"
echo "Seed:     $SEED"
echo "Config:   $SLURM_ARRAY_TASK_ID (of 80)"
echo "Started:  $(date -Iseconds)"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "--------------------------"

python -u qdps/robustness/cluster.py \
        --subject "$SUBJECT" --seed "$SEED" \
        --sweep --config-index "$SLURM_ARRAY_TASK_ID"

echo "Finished: $(date -Iseconds)"
