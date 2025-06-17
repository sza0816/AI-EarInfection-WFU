#!/bin/bash 

declare -a MODELS=("resnet34" "convnext" "efficientnetb0" "swint" "vitbase16" "efficientvitb0") 
declare -a KEYFRAMES=("auto" "human") 

# 2 min is not enough

DELAY=240   # test 4 min delay 

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