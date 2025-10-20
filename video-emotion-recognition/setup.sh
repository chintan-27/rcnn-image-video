#!/usr/bin/env bash
set -euo pipefail

echo "Setting up Video RCNN training environment..."
echo "============================================="

# ---- Config: set your restructured dataset root here (ABSOLUTE PATH RECOMMENDED) ----
DST="${DST:-/blue/ruogu.fang/chintan.acharya/RCNN/video-emotion-recognition/data/ckvideo_out}"   # or export DST before running:  export DST=/path/to/ckvideo_out
echo "Dataset root (DST) = $DST"

# ---- Make training scripts executable (they live at repo root in your codebase) ----
echo "Making scripts executable..."
chmod +x submit_all_video.sh || true
chmod +x scripts/train_video_va_only.sh || true
chmod +x scripts/train_video_emotions_only.sh || true
chmod +x scripts/train_video_multitask.sh || true
chmod +x scripts/train_video_efficient.sh || true

# (If you actually keep them under scripts/, keep these as well)
[ -f scripts/train_video_va_only.sh ] && chmod +x scripts/train_video_va_only.sh || true
[ -f scripts/train_video_emotions_only.sh ] && chmod +x scripts/train_video_emotions_only.sh || true
[ -f scripts/train_video_multitask.sh ] && chmod +x scripts/train_video_multitask.sh || true
[ -f scripts/train_video_efficient.sh ] && chmod +x scripts/train_video_efficient.sh || true

# ---- Create run dirs ----
echo "Creating directories..."
mkdir -p logs checkpoints

# ---- Optional helpers ----
[ -f "inspect_ck_data.py" ] && chmod +x inspect_ck_data.py
[ -f "data_inspect.py" ] && chmod +x data_inspect.py

echo ""
echo "Checking environment..."
echo "----------------------"

# Python
if command -v python &>/dev/null; then
  echo "FOUND: $(python --version 2>&1)"
else
  echo "ERROR: Python not found"
fi

# VENV (optional)
if [ -f ".venv/bin/activate" ]; then
  echo "FOUND: .venv/ — activating"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "WARNING: No .venv/ found"
fi

echo "Inspecting data"
python data_inspect.py

# Key packages
echo "Checking packages..."
python - <<'PY' 2>/dev/null || echo "ERROR: package check failed"
import importlib, sys
for m in ("torch","cv2","pandas"):
    try:
        v = importlib.import_module(m)
        print(f"FOUND: {m} {getattr(v,'__version__','')}".strip())
    except Exception as e:
        print(f"ERROR: {m} not available: {e}", file=sys.stderr)
import torch
print("CUDA available:", torch.cuda.is_available())
PY

echo ""
echo "Checking project files..."
echo "------------------------"

# Your repo uses these file names/locations:
[ -f "main.py" ]            && echo "FOUND: main.py"            || echo "ERROR: main.py missing"
[ -f "data_utils/dataset.py" ]         && echo "FOUND: dataset.py"         || echo "ERROR: dataset.py missing"
[ -f "models/video_rcnn.py" ]      && echo "FOUND: video_rcnn.py"      || echo "ERROR: video_rcnn.py missing"
[ -f "training/trainer.py" ]         && echo "FOUND: trainer.py"         || echo "ERROR: trainer.py missing"
[ -f "training/metrics.py" ]         && echo "FOUND: metrics.py"         || echo "WARN: metrics.py missing (optional)"
[ -f "utils/helpers.py" ]         && echo "FOUND: helpers.py"         || echo "WARN: helpers.py missing (optional)"

# If you instead moved them under subfolders, also check those (non-fatal):
[ -f "data_utils/dataset.py" ]            && echo "ALT FOUND: data/dataset.py"            || true
[ -f "models/video_rcnn.py" ]       && echo "ALT FOUND: models/video_rcnn.py"       || true
[ -f "training/trainer.py" ]        && echo "ALT FOUND: training/trainer.py"        || true

echo ""
echo "Checking restructured dataset (ckvideo_out)..."
echo "----------------------------------------------"
# New layout checks
[ -f "$DST/splits/train.csv" ] || { echo "ERROR: Missing $DST/splits/train.csv"; exit 1; }
[ -f "$DST/splits/val.csv" ]   || { echo "ERROR: Missing $DST/splits/val.csv";   exit 1; }
[ -f "$DST/splits/test.csv" ]  && echo "FOUND: $DST/splits/test.csv" || echo "WARN: test.csv not present (ok if not needed)"

[ -d "$DST/frames/train" ] || { echo "ERROR: Missing $DST/frames/train"; exit 1; }
[ -d "$DST/frames/val" ]   || { echo "ERROR: Missing $DST/frames/val";   exit 1; }
[ -d "$DST/frames/test" ]  && echo "FOUND: $DST/frames/test" || echo "WARN: frames/test not present (ok if not needed)"

[ -f "$DST/metadata/manifest.json" ] && echo "FOUND: manifest.json" || echo "WARN: manifest.json missing"
[ -f "$DST/metadata/CowenKeltnerEmotionalVideos.csv" ] && echo "FOUND: CowenKeltnerEmotionalVideos.csv" || echo "WARN: emotions CSV missing (required for emotions_only/multitask)"
[ -f "$DST/metadata/Header.txt" ] && echo "FOUND: Header.txt" || echo "INFO: Header.txt not required post-restructure"

# Quick sanity counts (non-fatal)
echo "Counting sample items..."
for s in train val; do
  vids=$(find "$DST/frames/$s" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
  echo "  $s videos: $vids"
done

echo ""
echo "Setup complete!"
echo "==============="
echo ""
echo "Next steps:"
echo "1) Export your DST if not set yet:  export DST=$DST"
echo "2) Submit training jobs, e.g.:"
echo "   sbatch train_video_va_only.sh"
echo "   sbatch train_video_multitask.sh"
echo ""
echo "If your job scripts still reference old paths, update:"
echo "  --train_csv \"$DST/splits/train.csv\""
echo "  --val_csv   \"$DST/splits/val.csv\""
echo "  --video_root \"$DST\""
