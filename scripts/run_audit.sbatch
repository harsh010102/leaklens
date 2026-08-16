#!/bin/bash
#SBATCH --job-name=leaklens-audit
#SBATCH --partition=a100_only
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

# Run a LeakLens audit on one A100 (model loading needs a GPU; the login node cannot run torch).
#   sbatch scripts/run_audit.sbatch configs/tier0_smoke.yaml
set -uo pipefail
CONFIG="${1:-configs/tier0_smoke.yaml}"

PROJ="$HOME/leaklens"
CONDA_SH="${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
cd "$PROJ"; mkdir -p logs
source "$CONDA_SH"; conda activate myenv

export HF_HOME=/mnt/ssd2/harshparikh/huggingface
[ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ] && export HF_TOKEN="$(cat "$HF_HOME/token")"

python -m pip install -q -e . 2>/dev/null || true
echo "===== $(date '+%F %T') leaklens audit  config=$CONFIG ====="
python -m leaklens.cli audit --config "$CONFIG"
echo "===== $(date '+%F %T') done ====="
