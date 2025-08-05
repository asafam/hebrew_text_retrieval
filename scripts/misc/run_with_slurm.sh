#!/bin/bash

# Load .env file from current working directory
if [ -f ".env" ]; then
    export $(grep -E '^EMAIL=' ".env" | xargs)
else
    echo "❌ Error: .env file not found in current directory"
    exit 1
fi

if [[ -z "$EMAIL" ]]; then
    echo "❌ Error: EMAIL variable not set in .env"
    exit 1
fi

# Default values
PARTITION="A100-4h"
GPUS=1
MEM="32G"
CPUS=4

# Required values (initially empty)
JOB_NAME=""
SCRIPT=""
SCRIPT_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --script) SCRIPT="$2"; shift 2 ;;
    --) shift; SCRIPT_ARGS=("$@"); break ;;   # Everything after -- is for the script
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Validate required arguments
if [[ -z "$JOB_NAME" ]]; then
  echo "❌ Error: --job-name is required"
  exit 1
fi

if [[ -z "$SCRIPT" ]]; then
  echo "❌ Error: --script is required"
  exit 1
fi

# Auto-create logs directory
mkdir -p logs

# Submit SLURM job
cat <<EOF | sbatch
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=END,FAIL

bash ${SCRIPT} ${SCRIPT_ARGS[@]}
EOF
