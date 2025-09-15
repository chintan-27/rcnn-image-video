#!/bin/bash
#SBATCH --job-name=rcnn_training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/rcnn_training_%j.log

# Print job information
pwd; hostname; date
echo "Starting RCNN training job..."

# Activate the pre-existing python environment
source .venv/bin/activate

# Run the training script
echo "Running train.py on a single GPU..."
python train.py --device cuda --epochs 200 --batch-size 128 --output-folder "output/Run3_200Epochs" --model-name "rcnn_svhn.pth"

# Print end time
date
echo "Training job finished."
