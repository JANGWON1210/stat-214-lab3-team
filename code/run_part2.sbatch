#!/bin/bash
#SBATCH --job-name=test-run
#SBATCH --partition=RM-shared
#SBATCH --account=mth240012p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.log

module load anaconda3
source activate env_214

python part_2.py