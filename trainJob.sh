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
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --account=eecs598s009f23_class

module purge
module load tensorflow

# for now test pythong run file
python test_tensorflow.py > test_output.txt

module purge