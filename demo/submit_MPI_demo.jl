#!/bin/bash -l
#SBATCH --job-name=MPI
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:10:00
#SBATCH --partition=high
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail

module load julia

srun julia IIBLMM_broadcast.jl
