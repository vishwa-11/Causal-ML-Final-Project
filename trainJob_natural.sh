#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=train_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --account=eecs598s009f23_class

# load environment and modules
module purge
module load tensorflow
source envs/tf_207_bert_309/bin/activate

# train script
python train.py --train-natural -f BERT_natural_noMMD --train-ds-filename nat_train_small.npz --val-ds-filename nat_val_small.npz

# deactivate after done
deactivate
module purge