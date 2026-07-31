#!/bin/bash
#------------------------------------------------------------
# Frontera (TACC) job script for Nek5000
# Partition: normal — 56 cores/node
# Adjust -N, -n, -t, and -A before submitting
#------------------------------------------------------------
#SBATCH -J annulus
#SBATCH -o logfile.out
#SBATCH -e logfile.err
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 112
#SBATCH -t 04:00:00
#SBATCH -A YOUR_ALLOCATION

module load cmake gcc/11.2.0 openmpi

export NEK_SOURCE_ROOT=/path/to/Nek5000
export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR
ibrun ./nek5000
