#!/bin/bash
#SBATCH --job-name=rcnn_training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpg-turin
#SBATCH --gpus=2
#SBATCH --mem=32gb
#SBATCH --time=72:00:00
#SBATCH --output=logs/rcnn_training_newmodel7_%j.log

#Checking GPUS
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
pwd; hostname; date
echo "Starting RCNN training job..."

# Activate the pre-existing python environment
source .venv/bin/activate


# Run the training script with regularization parameters
echo "Running train.py..."
python train.py \
    --device cuda \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.00001 \
    --weight-decay 1e-4 \
    --early-stop-patience 20 \
    --scheduler-patience 10 \
    --output-folder "output/Newmodel7_Run1_200Epochs" \
    --model-name "rcnn_svhn.pth"

# Print end time
date
echo "Training job finished."
