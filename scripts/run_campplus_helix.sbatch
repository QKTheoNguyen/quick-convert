#!/bin/bash
#SBATCH --partition=h200
#SBATCH --nodelist=helix03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=dev
#SBATCH --gres=gpu:h200_nvl_1g.18gb:1
#SBATCH --job-name=campplus-sim
#SBATCH --output=/medias/speech/projects/chandra/quick-convert-hackathon/logs/%j.out
#SBATCH --error=/medias/speech/projects/chandra/quick-convert-hackathon/logs/%j.err
set -euo pipefail

PROJECT=/medias/speech/projects/chandra/quick-convert-hackathon
cd "$PROJECT"

export UV_CACHE_DIR=/medias/speech/projects/chandra/uv_cache
export HF_HOME="$PROJECT/.cache/huggingface"
export TORCH_HOME="$PROJECT/.cache/torch"

export http_proxy=http://proxy.eurecom.fr:8080
export https_proxy=http://proxy.eurecom.fr:8080

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

source .venv/bin/activate

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Encoder: CAMPPlus"
echo "Device: CPU"

srun python -u scripts/evaluate_streaming_speaker_similarity.py \
    /medias/speech/projects/chandra/vpc2024_vctk/data/VCTK/wav/p225_001.wav \
    /medias/speech/projects/chandra/vpc2024_vctk/data/VCTK_nac_vctk_global/wav/p225_001.wav \
    "$PROJECT/results/streaming_similarity/p225_001" \
    --aligned \
    --encoder campplus \
    --device cpu \
    --window-seconds 1.5 \
    --hop-seconds 0.5 \
    --chunk-seconds 0.5

echo "Completed: $(date)"