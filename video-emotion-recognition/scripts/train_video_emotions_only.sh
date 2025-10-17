#!/bin/bash
#SBATCH --job-name=video_rcnn_emotions
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gpus=2
#SBATCH --mem=64gb
#SBATCH --time=96:00:00
#SBATCH --output=logs/video_rcnn_emotions_only_%j.log

# Checking GPUS
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p data

# Print job information
pwd; hostname; date
echo "Starting Video RCNN Emotions-Only training job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Activate the pre-existing python environment
source .venv/bin/activate

# Check if required files exist
if [ ! -f "data/train_emotions.csv" ]; then
    echo "ERROR: Training CSV file not found at data/train_emotions.csv"
    exit 1
fi

if [ ! -f "data/val_emotions.csv" ]; then
    echo "ERROR: Validation CSV file not found at data/val_emotions.csv"
    exit 1
fi

if [ ! -d "data/videos" ]; then
    echo "ERROR: Video directory not found at data/videos/"
    exit 1
fi

# Run the training script - Option B: Discrete Emotions Only
echo "Running Video RCNN training for Discrete Emotion Classification..."
python main.py \
    --model_type emotions_only \
    --train_csv data/ckvideo_out/splits/train.csv \
    --val_csv data/ckvideo_out/splits/val.csv \
    --video_root data/ckvideo_out/ \
    --batch_size 12 \
    --num_epochs 250 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --num_frames 10 \
    --input_size 112 112 \
    --num_workers 8 \
    --top_k_emotions 50 \
    --early_stopping_patience 35 \
    --save_every 5 \
    --scheduler_type plateau \
    --device cuda \
    --save_dir checkpoints \
    --log_dir logs \
    --experiment_name "emotions_only_50classes_10frames_250epochs" \
    --label_smoothing 0.1

# Print end time and job statistics
date
echo "Emotions-Only training job finished."
echo "Job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
