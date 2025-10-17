#!/bin/bash

echo "Submitting all Video RCNN training jobs..."
echo "================================================"

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

echo "Checking CK Video dataset..."
python data_inspect.py

echo ""
read -p "Does the data look correct? Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then  # FIXED: Added $ before REPLY
    echo "Aborted by user."
    exit 1
fi

# Submit jobs and capture job IDs
echo "Submitting Option A: Valence/Arousal Only..."
JOB_VA=$(sbatch --parsable scripts/train_video_va_only.sh)  # FIXED: Added scripts/ path
echo "  Job ID: $JOB_VA"

echo "Submitting Option B: Discrete Emotions Only..."
JOB_EMOTIONS=$(sbatch --parsable scripts/train_video_emotions_only.sh)  # FIXED: Added scripts/ path
echo "  Job ID: $JOB_EMOTIONS"

echo "Submitting Option C: Multi-Task Learning..."
JOB_MULTITASK=$(sbatch --parsable scripts/train_video_multitask.sh)  # FIXED: Added scripts/ path
echo "  Job ID: $JOB_MULTITASK"

echo "Submitting Efficient Model..."
JOB_EFFICIENT=$(sbatch --parsable scripts/train_video_efficient.sh)  # FIXED: Added scripts/ path
echo "  Job ID: $JOB_EFFICIENT"

echo ""
echo "All jobs submitted successfully!"
echo "================================================"
echo "Job Summary:"
echo "  VA Only:     $JOB_VA"
echo "  Emotions:    $JOB_EMOTIONS" 
echo "  Multi-Task:  $JOB_MULTITASK"
echo "  Efficient:   $JOB_EFFICIENT"
echo ""

# Create monitoring script
cat > monitor_jobs.sh << EOF
#!/bin/bash
echo "Video RCNN Training Job Status:"
echo "=================================="
squeue -u \$USER --format="%.10i %.15j %.8t %.10M %.6D %R" | grep "video_rcnn"
echo ""
echo "Resource Usage:"
echo "=================="
sacct -j $JOB_VA,$JOB_EMOTIONS,$JOB_MULTITASK,$JOB_EFFICIENT --format=JobID,JobName,State,MaxRSS,Elapsed,CPUTime | head -20
EOF

chmod +x monitor_jobs.sh

echo "Monitor jobs with: ./monitor_jobs.sh"
echo "View logs in: logs/ directory"
echo "Checkpoints will be saved in: checkpoints/ directory"
echo ""
echo "Useful commands:"
echo "  squeue -u \$USER                    # Check job queue"
echo "  scancel <job_id>                   # Cancel a job"  
echo "  tail -f logs/video_rcnn_*_*.log    # Monitor training progress"
echo ""
echo "Expected training times:"
echo "  VA Only:     ~24-48 hours"
echo "  Emotions:    ~36-60 hours" 
echo "  Multi-Task:  ~48-96 hours"
echo "  Efficient:   ~12-24 hours"
