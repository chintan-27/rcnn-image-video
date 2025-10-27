#!/usr/bin/env bash
set -euo pipefail

echo "Submitting all Video RCNN training jobs..."
echo "================================================"

# Dataset root (absolute path recommended). Override by: export DST=/abs/path/to/data/ckvideo_out
DST="${DST:-$(readlink -f data/ckvideo_out)}"
echo "Using dataset root: $DST"

# Create necessary directories
mkdir -p logs checkpoints

echo "Checking CK Video dataset..."
# If your inspector accepts --root, pass DST; otherwise it will use its default.
if python -c 'import sys; import data_inspect as m; sys.exit(0)'; then
  python data_inspect.py
else
  python data_inspect.py  # keep as-is if your script has no CLI
fi

echo ""
# Confirm only if running interactively
if [ -t 0 ]; then
  read -p "Does the data look correct? Continue with training? (y/N): " -n 1 -r
  echo
  if [[ ! ${REPLY:-N} =~ ^[Yy]$ ]]; then
      echo "Aborted by user."
      exit 1
  fi
else
  echo "Non-interactive shell detected; proceeding without prompt."
fi

# Submit jobs and capture job IDs (export DST to jobs)
echo "Submitting Option A: Valence/Arousal Only..."
JOB_VA=$(sbatch --export=ALL,DST="$DST" --parsable scripts/train_video_va_only.sh)
echo "  Job ID: $JOB_VA"

echo "Submitting Option B: Discrete Emotions Only..."
JOB_EMOTIONS=$(sbatch --export=ALL,DST="$DST" --parsable scripts/train_video_emotions_only.sh)
echo "  Job ID: $JOB_EMOTIONS"

echo "Submitting Option C: Multi-Task Learning..."
JOB_MULTITASK=$(sbatch --export=ALL,DST="$DST" --parsable scripts/train_video_multitask.sh)
echo "  Job ID: $JOB_MULTITASK"

echo "Submitting Efficient Model..."
JOB_EFFICIENT=$(sbatch --export=ALL,DST="$DST" --parsable scripts/train_video_efficient.sh)
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
#!/usr/bin/env bash
set -euo pipefail
echo "Video RCNN Training Job Status"
echo "=================================="
# Show all your jobs; adjust the pattern if your #SBATCH -J names differ
squeue -u "\$USER" --format="%.10i %.20j %.8t %.10M %.6D %R" | sed -n '1,200p'

echo ""
echo "Resource Usage (sacct):"
echo "========================"
# sacct may not have data until jobs start/finish
sacct -j ${JOB_VA},${JOB_EMOTIONS},${JOB_MULTITASK},${JOB_EFFICIENT} --format=JobID,JobName%24,State,Elapsed,MaxRSS,MaxVMSize,AllocTRES | sed -n '1,200p'
EOF
chmod +x monitor_jobs.sh

echo "Monitor jobs with: ./monitor_jobs.sh"
echo "View logs in: logs/ (SLURM outputs are named via %x-%j.out by your scripts)"
echo "Checkpoints will be saved in: checkpoints/"
echo ""
echo "Useful commands:"
echo "  squeue -u \$USER"
echo "  scancel <job_id>"
