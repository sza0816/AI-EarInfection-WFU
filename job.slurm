#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:45:00       # specify job time
#SBATCH --mem=16G                 # specify memory
#SBATCH --output=/dev/null

module load cuda-toolkit/12.2.2  # load necessary modules

MODEL_NAME=$1
KEYFRAME_MODE=$2

OUTPUT_DIR="output_${MODEL_NAME}/${KEYFRAME_MODE}"
mkdir -p "$OUTPUT_DIR"

echo "Training model: $MODEL_NAME"
echo "Keyframe mode: $KEYFRAME_MODE"
echo "Output path: $OUTPUT_DIR/output.log"

srun python main.py --model "$MODEL_NAME" --keyframe "$KEYFRAME_MODE" > "$OUTPUT_DIR/output.log" 2>&1 

# ex: sbatch job.slurm resnet34 auto