#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:45:00       # specify job time
#SBATCH --mem=16G                 # specify memory
#SBATCH --output=output_efficientnetb0/job_efficientNet_output.log           # efficientnetb0 = model name

module load cuda-toolkit/12.2.2  # load necessary modules

srun python main_efficientNet.py  # run your job