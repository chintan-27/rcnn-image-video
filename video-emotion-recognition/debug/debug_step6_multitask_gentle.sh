#!/bin/bash
#SBATCH --job-name=video_rcnn_multitask_gentle
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1                   # use 1 GPU unless DDP is implemented
#SBATCH --mem=48gb
#SBATCH --time=72:00:00
#SBATCH --output=logs/video_rcnn_multitask_gentle_%j.log

# -------- Memory allocator: reduce fragmentation on long runs --------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Check GPUs
nvidia-smi

# Dirs
mkdir -p logs checkpoints data

# Print job info
pwd; hostname; date
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Env
source .venv/bin/activate
python --version
python - <<'PY'
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
PY

# Data roots
DST="/blue/ruogu.fang/chintan.acharya/RCNN/video-emotion-recognition/data/ckvideo_out"
[ -f "$DST/splits/train.csv" ] || { echo "Missing $DST/splits/train.csv"; exit 1; }
[ -f "$DST/splits/val.csv" ]   || { echo "Missing $DST/splits/val.csv"; exit 1; }
[ -d "$DST/frames/train" ]     || { echo "Missing $DST/frames/train"; exit 1; }
[ -d "$DST/frames/val" ]       || { echo "Missing $DST/frames/val"; exit 1; }

echo "Running Video RCNN Multitask (gentle balancing)..."
python main.py \
    --model_type multitask \
    --train_csv "$DST/splits/train.csv" \
    --val_csv   "$DST/splits/val.csv" \
    --video_root "$DST" \
    --batch_size 8 \
    --num_epochs 300 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --num_frames 8 \
    --input_size 96 96 \
    --num_workers 4 \
    --top_k_emotions 50 \
    --early_stopping_patience 40 \
    --save_every 5 \
    --scheduler_type plateau \
    --device cuda \
    --mixed_precision \
    --save_dir checkpoints \
    --log_dir logs \
    --experiment_name "multitask_gentle_top50_8f_96x96" \
    --valence_weight 1.0 \
    --arousal_weight 1.0 \
    --emotion_weight 0.3 \
    --label_smoothing 0.0

date
echo "Training finished."
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State

