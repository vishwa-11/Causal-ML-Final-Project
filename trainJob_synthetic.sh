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
python train.py -m 1.000 -f "BERT_synthetic_mMMD_1_000" -r

# deactivate after done
deactivate
module purge