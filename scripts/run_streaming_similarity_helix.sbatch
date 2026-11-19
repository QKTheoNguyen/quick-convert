#!/usr/bin/env bash
#SBATCH --partition=h200
#SBATCH --nodelist=helix03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200_nvl_1g.35gb:1
#SBATCH --job-name=stream-spk-sim

set -euo pipefail
if (( $# < 3 || $# > 5 )); then
    echo "Usage: sbatch $0 ORIGINAL.wav ANONYMIZED.wav OUTPUT_DIR [WINDOW_SECONDS [HOP_SECONDS]]" >&2
    exit 2
fi
original=$(realpath -- "$1")
anonymized=$(realpath -- "$2")
output_dir=$(realpath -m -- "$3")
window=${4:-4.0}
hop=${5:-2.0}
project=/medias/speech/projects/chandra/quick-convert-hackathon
cache_root=/medias/speech/projects/chandra/.cache
export http_proxy=http://proxy.eurecom.fr:8080
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export HF_HOME="$cache_root/huggingface"
export TORCH_HOME="$cache_root/torch"
export XDG_CACHE_HOME="$cache_root"
export UV_CACHE_DIR="$cache_root/uv"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR"
cd "$project"
# Provision this environment separately with uv; jobs never sync/install packages.
: "${ECAPA_CHECKPOINT:?Set ECAPA_CHECKPOINT to an existing SpeechBrain embedding_model.ckpt for CUDA inference}"
: "${ALIGNMENT_MODE:?Set ALIGNMENT_MODE=aligned or unavailable after checking audio alignment}"
case "$ALIGNMENT_MODE" in
    aligned) alignment=(--aligned) ;;
    unavailable) alignment=(--alignment-unavailable) ;;
    *) echo "ALIGNMENT_MODE must be aligned or unavailable" >&2; exit 2 ;;
esac
if [[ ! -f "$original" || ! -f "$anonymized" || ! -f "$ECAPA_CHECKPOINT" ]]; then
    echo "Both WAV files and ECAPA_CHECKPOINT must exist" >&2
    exit 2
fi
echo "Job=${SLURM_JOB_ID:-manual} host=$(hostname)"
echo "original=$original anonymized=$anonymized output=$output_dir window=$window hop=$hop"
echo "encoder=ecapa device=cuda checkpoint=$ECAPA_CHECKPOINT alignment=$ALIGNMENT_MODE output_start=${OUTPUT_START_SECONDS:-0}"
nvidia-smi
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: refusing CPU fallback"; print("CUDA:", torch.cuda.get_device_name(0)); torch.empty(1, device="cuda")'
uv run --no-sync python scripts/evaluate_streaming_speaker_similarity.py \
    "$original" "$anonymized" "$output_dir" \
    --encoder ecapa --checkpoint "$ECAPA_CHECKPOINT" --device cuda \
    "${alignment[@]}" --output-start-seconds "${OUTPUT_START_SECONDS:-0}" \
    --window-seconds "$window" --hop-seconds "$hop"
