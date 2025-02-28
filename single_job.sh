#!/bin/bash
#SBATCH --job-name=vanillabo         # Job name
#SBATCH --output=logs/%j.log          # Output file (with job id)
#SBATCH --error=logs/%j.log            # Error file (with job id)
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=T4:2
#SBATCH -A NAISS2024-22-1613

# Path to your Singularity container image (replace with the actual path)
SINGULARITY_IMAGE="vanillabo.sif"

# Command to run inside the Singularity container
# Replace 'your_command' with the command you want to execute inside the container
singularity run $SINGULARITY_IMAGE $*
