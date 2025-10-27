#!/usr/bin/env bash
set -euo pipefail
echo "Video RCNN Training Job Status"
echo "=================================="
# Show all your jobs; adjust the pattern if your #SBATCH -J names differ
squeue -u "$USER" --format="%.10i %.20j %.8t %.10M %.6D %R" | sed -n '1,200p'

echo ""
echo "Resource Usage (sacct):"
echo "========================"
# sacct may not have data until jobs start/finish
sacct -j 16290552,16290553,16290554,16290555 --format=JobID,JobName%24,State,Elapsed,MaxRSS,MaxVMSize,AllocTRES | sed -n '1,200p'
