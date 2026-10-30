#!/bin/bash -l
#SBATCH --job-name=synth-landmark-dataset
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

# Builds the synthetic nnU-Net landmark dataset; all arguments are forwarded to
# JunctionDetection/PreProcessing/create_synthetic_nnunet_landmark_dataset.py.
#
# Usage (from repo root):
#   sbatch JunctionDetection/PreProcessing/synthetic_nnunet_landmark_dataset_job.sh \
#       --model-dir <model dir> --output-dir <dataset dir> [--exclude-case-mapping <json>] [--hf-repo <repo>]

set -euo pipefail

export nnUNet_raw="/home/jhehli/data/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/jhehli/data/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/jhehli/data/datasets/nnUNet/nnUNet_results"
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/jhehli/data/ForkSight"
cd "$REPO_ROOT"

echo "CWD: $(pwd)"
echo "Args: $*"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

mkdir -p "/scratch/jhehli/logs"

source ~/.nnUNet_env/bin/activate
srun python -u -m JunctionDetection.PreProcessing.create_synthetic_nnunet_landmark_dataset "$@"
