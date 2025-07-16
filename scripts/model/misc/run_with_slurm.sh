# run_with_slurm.sh
#!/bin/bash

# Usage: ./run_with_slurm.sh <partition> <gpus> <mem> <cpus> <job_name>
PARTITION=${1:-gpuA100}
GPUS=${2:-1}
MEM=${3:-32G}
CPUS=${4:-4}
JOB_NAME=${5:-train_job}

cat <<EOF | sbatch
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mail-user=your.email@example.com
#SBATCH --mail-type=END,FAIL

bash train_model.sh
EOF
