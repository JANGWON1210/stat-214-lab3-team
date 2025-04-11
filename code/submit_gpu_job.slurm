#!/bin/bash
#SBATCH --job-name=ridge_gpu
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1        # 또는 v100-16도 가능
#SBATCH --cpus-per-task=4
#SBATCH --mem=60000                 # 60000M = 60GB 이하
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 로그 폴더 없으면 생성
mkdir -p logs

# 환경 설정
module load anaconda3
source activate env214

# 실행
python part_2.py