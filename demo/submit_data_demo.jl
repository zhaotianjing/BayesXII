#!/bin/bash -l
#SBATCH --job-name=data
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --partition=high
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your email

module load julia

# srun julia create_data_demo.jl
# srun julia split_column.jl
srun julia pa_calc_demo.jl