#!/bin/bash
#SBATCH --account=def-dprecup
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=10000M               # memory (per node)
#SBATCH --time=09:00:00            # time (DD-HH:MM)
module load python/3.7.0
module load scipy-stack
source mujoco_env/bin/activate
cd /home/samin/projects/def-dprecup/samin/git_files/SparceReward/engine
export PYTHONPATH=~/projects/def-dprecup/samin/git_files/SparceReward/
python3 main.py --num_steps 3000000 --sparse_reward --threshold_sparcity 15 --algo SAC --lambda 0.35 --sigma_squared 0.017 --betta 0.002 --seed 100 --output_path /home/samin/hoss_results
