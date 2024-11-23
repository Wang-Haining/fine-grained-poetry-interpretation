#!/bin/sh
#SBATCH --job-name=eval_gemma2_2b_it
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
python -m eval_it --model google/gemma-2-2b-it --corpus poem_interpretation --batch_size 20 --output_file results/eval_gemma2_2b_it.parquet