#!/bin/bash
#SBATCH --job-name=video_rcnn_efficient
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/video_rcnn_efficient_%j.log

# Checking GPUS
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p data

# Print job information
pwd; hostname; date
echo "Starting Video RCNN Efficient Model training job..."
echo "Job ID: $SLURM_JOB_ID"

# Activate the pre-existing python environment
source .venv/bin/activate

DST="/blue/ruogu.fang/chintan.acharya/RCNN/video-emotion-recognition/data/ckvideo_out"

[ -f "$DST/splits/train.csv" ] || { echo "Missing $DST/splits/train.csv"; exit 1; }
[ -f "$DST/splits/val.csv" ]   || { echo "Missing $DST/splits/val.csv"; exit 1; }
[ -d "$DST/frames/train" ]     || { echo "Missing $DST/frames/train"; exit 1; }
[ -d "$DST/frames/val" ]       || { echo "Missing $DST/frames/val"; exit 1; }


# Run the training script - Efficient Model for Real-Time
echo "Running Video RCNN Efficient training for Real-Time deployment..."
python main.py \
    --model_type multitask \
    --efficient \
    --train_csv $DST/splits/train.csv \
    --val_csv $DST/splits/val.csv \
    --video_root $DST \
    --batch_size 16 \
    --num_epochs 150 \
    --learning_rate 2e-4 \
    --weight_decay 5e-5 \
    --num_frames 8 \
    --input_size 96 96 \
    --num_workers 6 \
    --top_k_emotions 30 \
    --early_stopping_patience 25 \
    --save_every 5 \
    --scheduler_type plateau \
    --device cuda \
    --save_dir checkpoints \
    --log_dir logs \
    --experiment_name "efficient_realtime_30classes_8frames_96px" \
    --valence_weight 1.0 \
    --arousal_weight 1.0 \
    --emotion_weight 1.5 \
    --adaptive_weights

# Print end time and job statistics
date
echo "Efficient training job finished."
echo "Job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
