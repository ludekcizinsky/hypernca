#!/bin/bash
#SBATCH --job-name=hypernca-train
#SBATCH --output=outputs/slurm/train_%j.out       # Save output to logs with job ID
#SBATCH --error=outputs/slurm/train_%j.err        # Save errors to separate file
#SBATCH --time=20:00:00                  # Max runtime (hh:mm:ss)
#SBATCH --partition=gpu                  # Partition name (adjust if needed)
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=8                # Adjust CPU count if needed
#SBATCH --mem=16G                        # RAM requested
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --qos=cs-503
#SBATCH --account=cs-503


USER=$1

cd /home/$USER/x-to-nif/
source /home/$USER/venvs/hypernca/bin/activate

nvidia-smi

# Run your training script
python train_nca.py
