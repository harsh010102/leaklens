#!/bin/bash
#SBATCH --job-name=leaklens-lenstable
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL
set -uo pipefail
CONFIG="${1:-configs/lenstable_quant_8b.yaml}"
PROJ="$HOME/leaklens"; CONDA_SH="${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
cd "$PROJ"; mkdir -p logs; source "$CONDA_SH"; conda activate myenv
export HF_HOME=/mnt/ssd2/harshparikh/huggingface
[ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ] && export HF_TOKEN="$(cat "$HF_HOME/token")"
echo "===== $(date '+%F %T') lenstable-quant  config=$CONFIG ====="
python experiments/lenstable_quant.py --config "$CONFIG"
echo "===== $(date '+%F %T') done ====="
