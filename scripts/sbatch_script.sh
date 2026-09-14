#!/bin/bash

# Slurm job script for Galvani 

#SBATCH -J bagnet_imagenet              # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=32              # Number of CPU cores per task
#SBATCH --nodes=1                       # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=<PARTITION_NAME>    # Which partition will run your job
#SBATCH --time=0-16:00                  # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:8                    # (optional) Requesting type and number of GPUs
#SBATCH --output=<RESULTS_PATH>.out     # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=<RESULTS_PATH>.err      # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<YOUR_EMAIL>        # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

# Setup Phase
source .venv/bin/activate

echo Training bagnet33 on ImageNet...
torchrun --standalone --nproc_per_node=8 -m bagnetsv2.train_multigpu model.variant=bagnet33 dataset.name=imagenet dataset.dir=<IMAGENET_DIR> train.batch_size=1024 train.epochs=90 train.num_workers=4

deactivate
