#!/bin/bash
#SBATCH --job-name=rcnn_testing
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=8gb
#SBATCH --time=00:15:00
#SBATCH --output=logs/rcnn_testing_%j.log

# Print job information
pwd; hostname; date
echo "Starting RCNN testing job..."

# Activate the pre-existing python environment
source .venv/bin/activate

# Run the testing script
echo "Running test.py on the trained model..."
python test.py --model-path rcnn_svhn_best_loss.pth --device cuda --data-folder "output/Newmodel9_Run1_200Epochs"

# Print end time
date
echo "Testing job finished."
