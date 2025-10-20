#!/bin/bash
#SBATCH --job-name=video_rcnn_va_only
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gpus=2
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --output=logs/video_rcnn_va_only_%j.log

# Checking GPUS
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p data

# Print job information
pwd; hostname; date
echo "Starting Video RCNN VA-Only training job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Activate the pre-existing python environment
source .venv/bin/activate

# Check if required files exist
# If [ ! -f "data/train_emotions.csv" ]; then
#     echo "ERROR: Training CSV file not found at data/train_emotions.csv"
#     exit 1
# Fi
# 
# If [ ! -f "data/val_emotions.csv" ]; then
#     echo "ERROR: Validation CSV file not found at data/val_emotions.csv"
#     exit 1
# Fi
# 
# If [ ! -d "data/videos" ]; then
#     echo "ERROR: Video directory not found at data/videos/"
#     exit 1
# Fi
DST="/blue/ruogu.fang/chintan.acharya/RCNN/video-emotion-recognition/data/ckvideo_out"

[ -f "$DST/splits/train.csv" ] || { echo "Missing $DST/splits/train.csv"; exit 1; }
[ -f "$DST/splits/val.csv" ]   || { echo "Missing $DST/splits/val.csv"; exit 1; }
[ -d "$DST/frames/train" ]     || { echo "Missing $DST/frames/train"; exit 1; }
[ -d "$DST/frames/val" ]       || { echo "Missing $DST/frames/val"; exit 1; }



# Run the training script - Option A: Valence/Arousal Only
echo "Running Video RCNN training for Valence/Arousal prediction..."
python main.py \
    --model_type va_only \
    --train_csv $DST/splits/train.csv \
    --val_csv $DST/splits/val.csv \
    --video_root $DST \ 
    --batch_size 16 \
    --num_epochs 200 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --num_frames 10 \
    --input_size 112 112 \
    --num_workers 8 \
    --early_stopping_patience 30 \
    --save_every 5 \
    --scheduler_type plateau \
    --device cuda \
    --save_dir checkpoints \
    --log_dir logs \
    --experiment_name "va_only_10frames_112px_200epochs" \
    --valence_weight 1.0 \
    --arousal_weight 1.0

# Print end time and job statistics
date
echo "VA-Only training job finished."
echo "Job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
