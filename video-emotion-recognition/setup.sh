#!/bin/bash

echo "Setting up Video RCNN training environment..."
echo "============================================="

# Make all scripts executable
echo "Making scripts executable..."
chmod +x submit_all_video.sh
chmod +x scripts/train_video_va_only.sh
chmod +x scripts/train_video_emotions_only.sh  
chmod +x scripts/train_video_multitask.sh
chmod +x scripts/train_video_efficient.sh

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p checkpoints

# Make existing scripts executable if they exist
[ -f "inspect_ck_data.py" ] && chmod +x inspect_ck_data.py
[ -f "data_inspect.py" ] && chmod +x data_inspect.py

# Basic environment check
echo "Checking environment..."
echo "----------------------"

# Check Python environment
if command -v python &> /dev/null; then
    echo "FOUND: Python $(python --version 2>&1)"
else
    echo "ERROR: Python not found"
fi

# Check virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "FOUND: Virtual environment at .venv/"
    source .venv/bin/activate
    echo "ACTIVATED: Virtual environment"
else
    echo "WARNING: No virtual environment found at .venv/"
fi

# Check key Python packages
echo "Checking packages..."
python -c "import torch; print('FOUND: PyTorch', torch.__version__)" 2>/dev/null || echo "ERROR: PyTorch not available"
python -c "import cv2; print('FOUND: OpenCV', cv2.__version__)" 2>/dev/null || echo "ERROR: OpenCV not available"  
python -c "import pandas; print('FOUND: Pandas', pandas.__version__)" 2>/dev/null || echo "ERROR: Pandas not available"

# Check CUDA if available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "CUDA check failed"

echo ""
echo "Checking project files..."
echo "------------------------"

# Check critical files
[ -f "main.py" ] && echo "FOUND: main.py" || echo "ERROR: main.py missing"
[ -f "data_utils/dataset.py" ] && echo "FOUND: data_utils/dataset.py" || echo "ERROR: data_utils/dataset.py missing"
[ -f "models/video_rcnn.py" ] && echo "FOUND: models/video_rcnn.py" || echo "ERROR: models/video_rcnn.py missing"
[ -f "training/trainer.py" ] && echo "FOUND: training/trainer.py" || echo "ERROR: training/trainer.py missing"

# Check data files
[ -f "data/ckvideo/ckvideo_middleframe_train.csv" ] && echo "FOUND: Training CSV" || echo "ERROR: Training CSV missing"
[ -f "data/ckvideo/ckvideo_middleframe_val.csv" ]   && echo "FOUND: Validation CSV" || echo "ERROR: Validation CSV missing"
[ -d "data/ckvideo/ckvideo_middleframe_train" ]     && echo "FOUND: Training videos" || echo "ERROR: Training videos missing"
[ -d "data/ckvideo/ckvideo_middleframe_val" ]       && echo "FOUND: Validation videos" || echo "ERROR: Validation videos missing"

echo ""
echo "Setup complete!"
echo "==============="
echo ""
echo "Next steps:"
echo "1. Check your data: python inspect_ck_data.py"
echo "2. Submit training jobs: ./submit_all_video.sh"
echo ""
echo "Individual training scripts:"
echo "- sbatch scripts/train_video_va_only.sh"
echo "- sbatch scripts/train_video_emotions_only.sh" 
echo "- sbatch scripts/train_video_multitask.sh"
echo "- sbatch scripts/train_video_efficient.sh"
