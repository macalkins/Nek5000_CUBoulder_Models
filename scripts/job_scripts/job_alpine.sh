#!/bin/bash
#------------------------------------------------------------
# Alpine (CU Boulder) job script for Nek5000
# Partition: amilan — 64 cores/node
# Adjust --nodes, --ntasks, --time, and --account before submitting
#------------------------------------------------------------
#SBATCH --nodes=4
#SBATCH --ntasks=256
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --job-name=annulus
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ALLOCATION
#SBATCH -o logfile.out
#SBATCH -e logfile.err

module purge
ml cmake
ml gcc/11.2.0
ml openmpi/4.1.1

export NEK_SOURCE_ROOT=/path/to/Nek5000
export OMP_NUM_THREADS=1

cd $SLURM_SUBMIT_DIR
$NEK_SOURCE_ROOT/bin/nekmpi annulus $SLURM_NTASKS
