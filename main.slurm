#!/bin/bash

#SBATCH --job-name=MichaelGPT_Train          # Job name
#SBATCH --qos=coc-ice                        # Quality of Service
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --gres=gpu:h200:2                    # Request 2 H200 GPUs
#SBATCH --mem=200GB                          # Total memory for the job
#SBATCH --time=7:59:00                       # Time limit (7 hours)
#SBATCH --cpus-per-task=8                    # CPUs per task (adjust based on needs)
#SBATCH --output=training_%j.out             # Output file
#SBATCH --error=training_%j.err              # Error file
#SBATCH --mail-type=END,FAIL                 # Email notifications
#SBATCH --mail-user=mwang764@gatech.edu    # Replace with your email

# Change to your working directory (replace with your path)
cd /home/hice1/mwang764/scratch/MichaelGPT

# Load necessary modules (adjust based on your environment)
module load cuda/12.6.1  # Example CUDA version; check available with 'module avail'
module load anaconda3  # Or your Python module

# Run the training script
# Adjust arguments as needed; num_workers set to match cpus-per-task for efficiency
# (num_workers in DataLoader controls parallel data loading; set <= cpus-per-task to avoid issues)
python train.py \
  --batch_size 32 \
  --batch_size 32 \
  --gradient_accum_steps 4 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --epochs 5 \
  --num_workers 8 \
  --log_interval 100 \
  --checkpoint_interval 10000 \
  --use_cosine_annealing \
  --eval_interval 1000 \
  --patience 5 \
#  --resume_from_checkpoint None  # Set to checkpoint path if resuming

# If resuming, replace --resume_from_checkpoint with the path, e.g., checkpoint_epoch_0