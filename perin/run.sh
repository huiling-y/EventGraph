#!/bin/bash

#SBATCH --job-name=ACE
#SBATCH --account=ec30
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gpus=1


# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones

module purge

module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/1.7.1-foss-2019b-cuda-11.1.1-Python-3.7.4
module load nlpl-transformers/4.14.1-foss-2019b-Python-3.7.4
module load nlpl-nlptools/2021.01-foss-2019b-Python-3.7.4
module load nlpl-scipy-ecosystem/2021.01-foss-2019b-Python-3.7.4
module load sentencepiece/0.1.96-foss-2019b-Python-3.7.4
module load nlpl-nltk/3.5-foss-2019b-Python-3.7.4
module load nlpl-wandb/0.12.6-foss-2019b-Python-3.7.4


#nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
#	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
#NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process

TRANSFORMERS_OFFLINE=1 python3 train.py --log_wandb --config "$1" --name "$2"
