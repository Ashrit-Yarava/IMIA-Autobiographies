#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=question
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ary24/Autobiographies/logs/output.%N.%j.out

cd /scratch/ary24/Autobiographies/
source ~/.bashrc
conda activate ai
module load cuda cudnn
python3 question_answer.py
