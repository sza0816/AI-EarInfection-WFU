#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=ciaq                      # gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=3-03:00:00       # specify job time
#SBATCH --mem=16G                 # specify memory
#SBATCH --output=output_param_search/job_tune_parameter_output.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python parameter_search_only_best_left.py  # run your job
