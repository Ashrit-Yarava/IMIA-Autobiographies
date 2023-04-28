#!/bin/bash



#SBATCH --partition=main
#SBATCH --job-name=question
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ary24/Autobiographies/logs/output.%N.%j.out

cd /scratch/ary24/Autobiographies/
source ~/.bashrc
conda activate ai
python3 question_answer.py
