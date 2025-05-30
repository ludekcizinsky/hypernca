#!/bin/bash
#SBATCH --job-name=eval_post_train
#SBATCH --output=outputs/slurm/%x.%j.out       # Save output to logs with job ID
#SBATCH --error=outputs/slurm/%x.%j.err       # Save errors to separate file
#SBATCH --time=02:00:00                  # Max runtime (hh:mm:ss)
#SBATCH --partition=gpu                  # Partition name (adjust if needed)
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --cpus-per-task=10                # Adjust CPU count if needed
#SBATCH --mem=16G                        # RAM requested
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --account=master

cd /home/cizinsky/x-to-nif/
source /home/cizinsky/venvs/hypernca/bin/activate

nvidia-smi

# Baseline
python eval_post_train.py --run_id="8l52fbpf"

# Ablations
# - clip
python eval_post_train.py --run_id="nacm7ml4"

# - clip-128
python eval_post_train.py --run_id="u75r16fh"

# - temb-32
python eval_post_train.py --run_id="lkvwsmle"

# - cond on nca
python eval_post_train.py --run_id="468nzy6k"
