#!/bin/bash
#SBATCH --job-name=gpu-job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=gyang16.log
#SBATCH --partition=gpu-l40s # Use the appropriate GPU partition
#SBATCH --nodelist=g101          # Request nodes with L40S GPUs
#SBATCH --gres=gpu:1          # Request 1 GPU

conda activate research

# Run the script
python mwe.py &

# get the PID of the training script
TRAINING_PID=$!

# only monitor the GPU usage while the training script is running
while kill -0 $TRAINING_PID 2> /dev/null; do
    nvidia-smi >> gpu_usage.log
done