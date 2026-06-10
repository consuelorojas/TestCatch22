#!/bin/bash
#SBATCH -p Total
#SBATCH -N 1
#SBATCH -c 39
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH -J sine22

# Initialize conda (IMPORTANT)
source ~/.bashrc

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate testcatch22

# Debug (keep for now)
 echo "Running on $(hostname)"
 which python
 python --version
 pwd
 ls

# Run your script
srun ~/miniconda3/envs/testcatch22/bin/python runner.py
