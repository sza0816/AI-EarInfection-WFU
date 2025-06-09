#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:45:00               # specify job time
#SBATCH --mem=32G         # specify memory
#SBATCH --output=job_resNet34_output.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python main_resNet34.py  # run your job