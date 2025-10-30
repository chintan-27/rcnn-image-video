#!/bin/bash
#SBATCH --job-name=step4_5_micro_overfit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=chintan.acharya@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mem=24gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/step4_5_micro_overfit_%j.log

set -euo pipefail

echo "==============================="
echo " SLURM: Step 4 & 5 Micro-Overfit"
echo "==============================="
date
pwd
hostname
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# -------- Config --------
# Dataset root. You can override at submit time:
#   sbatch --export=ALL,DST=/abs/path/to/ckvideo_out debug/debug_step4_and_step5_micro.sh
DST="${DST:-/blue/ruogu.fang/chintan.acharya/RCNN/video-emotion-recognition/data/ckvideo_out}"
export DST

# We'll create ONE tiny CSV using train rows; use it for both train and val.
# This ensures the dataset reads from frames/train/... for both loaders.
TINY_TRAIN_CSV="${DST}/splits/_tiny_train.csv"

# Training knobs (safe defaults for tiny runs)
VA_EPOCHS=${VA_EPOCHS:-200}
EMO_EPOCHS=${EMO_EPOCHS:-150}
LR_VA=${LR_VA:-3e-4}
LR_EMO=${LR_EMO:-3e-4}
TOPK_EMO=${TOPK_EMO:-10}
NUM_FRAMES=${NUM_FRAMES:-10}
INPUT_H=${INPUT_H:-112}
INPUT_W=${INPUT_W:-112}
# Important for tiny set: keep batch size <= #rows (10) and avoid workers
BS_VA=${BS_VA:-8}
BS_EMO=${BS_EMO:-8}
NUM_WORKERS=${NUM_WORKERS:-0}
USE_EFFICIENT=${USE_EFFICIENT:-0}   # set to 1 to add --efficient

echo ""
echo "Config:"
echo "  DST          = ${DST}"
echo "  TINY_TRAIN   = ${TINY_TRAIN_CSV}"
echo "  VA_EPOCHS    = ${VA_EPOCHS}"
echo "  EMO_EPOCHS   = ${EMO_EPOCHS}"
echo "  LR_VA        = ${LR_VA}"
echo "  LR_EMO       = ${LR_EMO}"
echo "  TOPK_EMO     = ${TOPK_EMO}"
echo "  NUM_FRAMES   = ${NUM_FRAMES}"
echo "  INPUT_SIZE   = ${INPUT_H}x${INPUT_W}"
echo "  BS_VA        = ${BS_VA}"
echo "  BS_EMO       = ${BS_EMO}"
echo "  NUM_WORKERS  = ${NUM_WORKERS}"
echo "  USE_EFFICIENT= ${USE_EFFICIENT}"

# -------- Env / dirs --------
mkdir -p logs checkpoints
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "WARNING: .venv not found; using system Python"
fi

python --version
python - <<'PY' || true
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

# -------- Sanity checks --------
[ -f "${DST}/splits/train.csv" ] || { echo "ERROR: Missing ${DST}/splits/train.csv"; exit 1; }
[ -f "${DST}/splits/val.csv" ]   || { echo "ERROR: Missing ${DST}/splits/val.csv"; exit 1; }
[ -d "${DST}/frames/train" ]     || { echo "ERROR: Missing ${DST}/frames/train"; exit 1; }

# -------- Create tiny CSV (TRAIN-NAMED) --------
# Use a single base_id from train split (so frames exist under frames/train/<base_id>).
# We then pass this same CSV to both train_csv and val_csv to avoid split-dir mismatches.
echo ""
echo "Creating tiny CSV at ${TINY_TRAIN_CSV} ..."
python - <<PY
import os, pandas as pd
dst=os.environ["DST"]
train_src=f"{dst}/splits/train.csv"
tiny_train=f"{dst}/splits/_tiny_train.csv"

tdf=pd.read_csv(train_src)
# Normalize column name casing and find the ID column
idcol=[c for c in tdf.columns if c.lower()=="id"][0]
base=tdf[idcol].astype(str).str.split("_").str[0]
# Pick the first base_id; you can change this to a specific id if you like
b0=base.iloc[0]
tiny_tr=tdf[base==b0].copy()
if tiny_tr.empty:
    raise SystemExit("No rows found for selected base_id in train.csv")
tiny_tr.to_csv(tiny_train, index=False)
print(f"Wrote {tiny_train} rows={len(tiny_tr)} base_id={b0}")
PY

# -------- Common flags --------
EFFICIENT_FLAG=""
if [ "${USE_EFFICIENT}" = "1" ]; then
  EFFICIENT_FLAG="--efficient"
fi

# ==========================================
# Step 4 — VA-only micro-overfit
# ==========================================
echo ""
echo ">>> Step 4: VA-only micro-overfit starting..."
python main.py \
  --model_type va_only \
  --train_csv "${TINY_TRAIN_CSV}" \
  --val_csv   "${TINY_TRAIN_CSV}" \
  --video_root "${DST}" \
  --batch_size ${BS_VA} \
  --num_epochs ${VA_EPOCHS} \
  --learning_rate ${LR_VA} \
  --weight_decay 0 \
  --num_frames ${NUM_FRAMES} \
  --input_size ${INPUT_H} ${INPUT_W} \
  --num_workers ${NUM_WORKERS} \
  --scheduler_type none \
  --device cuda \
  --save_dir checkpoints \
  --log_dir logs \
  --experiment_name "debug_va_micro_${NUM_FRAMES}f_${INPUT_H}x${INPUT_W}" \
  ${EFFICIENT_FLAG}

echo ">>> Step 4: Completed."
date

# ==========================================
# Step 5 — Emotions-only micro-overfit
# ==========================================
echo ""
echo ">>> Step 5: Emotions-only micro-overfit starting..."
python main.py \
  --model_type emotions_only \
  --train_csv "${TINY_TRAIN_CSV}" \
  --val_csv   "${TINY_TRAIN_CSV}" \
  --video_root "${DST}" \
  --batch_size ${BS_EMO} \
  --num_epochs ${EMO_EPOCHS} \
  --learning_rate ${LR_EMO} \
  --weight_decay 0 \
  --label_smoothing 0.0 \
  --top_k_emotions ${TOPK_EMO} \
  --num_frames ${NUM_FRAMES} \
  --input_size ${INPUT_H} ${INPUT_W} \
  --num_workers ${NUM_WORKERS} \
  --scheduler_type none \
  --device cuda \
  --save_dir checkpoints \
  --log_dir logs \
  --experiment_name "debug_emotions_micro_top${TOPK_EMO}_${NUM_FRAMES}f_${INPUT_H}x${INPUT_W}" \
  ${EFFICIENT_FLAG}

echo ">>> Step 5: Completed."
date

echo ""
echo "All done. Results:"
echo "  - Checkpoints: checkpoints/debug_va_micro_* and checkpoints/debug_emotions_micro_*"
echo "  - Logs:        logs/debug_va_micro_* and logs/debug_emotions_micro_*"
echo ""
echo "Job statistics (may be partial until completion):"
sacct -j ${SLURM_JOB_ID} --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES | sed -n '1,200p'

