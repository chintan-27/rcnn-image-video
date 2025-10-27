#!/bin/bash
#SBATCH --job-name=video_rcnn_multitask
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gpus=2
#SBATCH --mem=64gb
#SBATCH --time=120:00:00
#SBATCH --output=logs/video_rcnn_multitask_%j.log

# Checking GPUS
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p data

# Print job information
pwd; hostname; date
echo "Starting Video RCNN Multi-Task training job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Available GPU memory:"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Activate the pre-existing python environment
source .venv/bin/activate

# Print Python and PyTorch versions
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check if required files exist
# if [ ! -f "data/train_emotions.csv" ]; then
#     echo "ERROR: Training CSV file not found at data/train_emotions.csv"
#     exit 1
# fi
# 
# if [ ! -f "data/val_emotions.csv" ]; then
#     echo "ERROR: Validation CSV file not found at data/val_emotions.csv"
#     exit 1
# fi
# 
# if [ ! -d "data/videos" ]; then
#     echo "ERROR: Video directory not found at data/videos/"
#     exit 1
# fi
DST="/blue/ruogu.fang/chintan.acharya/RCNN/video-emotion-recognition/data/ckvideo_out"

[ -f "$DST/splits/train.csv" ] || { echo "Missing $DST/splits/train.csv"; exit 1; }
[ -f "$DST/splits/val.csv" ]   || { echo "Missing $DST/splits/val.csv"; exit 1; }
[ -d "$DST/frames/train" ]     || { echo "Missing $DST/frames/train"; exit 1; }
[ -d "$DST/frames/val" ]       || { echo "Missing $DST/frames/val"; exit 1; }


# Run the training script - Option C: Multi-Task Learning
echo "Running Video RCNN Multi-Task training (Valence + Arousal + Emotions)..."
python main.py \
    --model_type multitask \
    --train_csv $DST/splits/train.csv \
    --val_csv $DST/splits/val.csv \
    --video_root $DST \
    --batch_size 8 \
    --num_epochs 300 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --num_frames 10 \
    --input_size 112 112 \
    --num_workers 8 \
    --top_k_emotions 50 \
    --early_stopping_patience 40 \
    --save_every 5 \
    --scheduler_type plateau \
    --device cuda \
    --save_dir checkpoints \
    --log_dir logs \
    --experiment_name "multitask_complete_50classes_10frames_300epochs" \
    --valence_weight 1.0 \
    --arousal_weight 1.0 \
    --emotion_weight 2.0 \
    --adaptive_weights \
    --label_smoothing 0.1

# Print end time and job statistics
date
echo "Multi-Task training job finished."
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
echo "Job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
