#!/bin/bash

# Runs cycle GAN

#SBATCH --job-name=cycle_GAN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=nickmatt@live.unc.edu

unset OMP_NUM_THREADS

# Set SIMG name
SIMG_NAME=/proj/dschridelab/SparseNets/pytorch1.4.0-py3-cuda10.1-ubuntu16.04_production.simg

echo singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 train.py --use_cuda
singularity exec --nv -B /pine -B /proj $SIMG_NAME python3 train.py --use_cuda
