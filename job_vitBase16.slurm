#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:45:00       # specify job time
#SBATCH --mem=16G                 # specify memory
#SBATCH --output=output_vitbase16/job_vitBase16_output.log           # vitbase16 = model name

module load cuda-toolkit/12.2.2  # load necessary modules

srun python main_vitBase16.py  # run your job