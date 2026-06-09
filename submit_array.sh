#!/bin/bash
#SBATCH --account=surrgrad                    # Currently Zen Lab is running everything under this account
#SBATCH --partition=several                      # for CPU only change this, e.g. to cpu_short, see also sinfo command output
#SBATCH --job-name=traversal
#SBATCH --array=0-4            # set to n_jobs - 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --signal=USR1@60        # send SIGUSR1 60s before timeout for clean requeue
#SBATCH --output=logs/job-%j.out
#   //// SBATCH --exclude=pcl1004
#   //// SBATCH --nodelist=pcl1003

# ---- configure these paths before submitting ----
SHAPENET_PATH=/tachyon/groups/gzenke/datasets/ShapeNetCoreV2 
OUTPUT_DIR=/tachyon/groups/gzenke/datasets/3DLT/dev_v6
SPLITS_DIR=splits_debug            # directory produced by split_objects.py
# -------------------------------------------------

mkdir -p logs
mkdir -p $OUTPUT_DIR

source $HOME/data/synseqsizer/venv/bin/activate

TASK_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

echo "Processing ${SPLITS_DIR}/objects_${TASK_ID}.npy ..."

blenderproc run generate_traversals.py \
  --models-path   "$SHAPENET_PATH" \
  --output-dir    "$OUTPUT_DIR" \
  --objects       "${SPLITS_DIR}/objects_${TASK_ID}.npy" \
  --metadata-name "metadata_${TASK_ID}" \
  --n-frames      32 \
  --render-samples 50 \
  --seqs-per-object 150 \
  --image-size    128 \
  --velocity-stdev 0.5 \
  --velocity-dist uniform \
  --random-offset \
  --freeze-prob 0.5 --multi-factor \
  --skip-list skip-list.txt \
  --seed    $SLURM_ARRAY_TASK_ID \
  --factors 2 3 4 5 6 7 8 9 



## Misc info 
# LATENT_NAMES = ['rot_x', 'rot_y', 'rot_z', 'floor_hue',
#                 'spot_theta', 'spot_phi', 'spot_hue',
#                 'trans_x', 'trans_y', 'trans_z']
