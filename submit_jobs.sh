#!/bin/bash 

declare -a MODELS=("resnet34" "convnext" "efficientnetb0" "swint" "vitbase16" "efficientvitb0") 
declare -a KEYFRAMES=("auto" "human") 

DELAY=1200   # 20 min delay 

for model in "${MODELS[@]}"; do 
  for keyframe in "${KEYFRAMES[@]}"; do 
    echo "Submitting: $model $keyframe" 
    sbatch job.slurm "$model" "$keyframe" 
    echo "Waiting ${DELAY}s before next job..." 
    sleep $DELAY 
  done 
done 


echo "✅ All jobs submitted with delays." 

# bash submit_jobs.sh

# beware there is still a chance of crashing if the delay period is not long enough 
# to run single model, use: sbatch job.slurm <model_name> <data_frame>