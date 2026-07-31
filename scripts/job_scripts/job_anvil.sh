#!/bin/bash -l
#------------------------------------------------------------
# Anvil (Purdue/ACCESS) job script for Nek5000
# Partition: wholenode — 128 cores/node
# Adjust --nodes, --ntasks, --time, and --account before submitting
#------------------------------------------------------------
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --partition=wholenode
#SBATCH --job-name=annulus
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ALLOCATION
#SBATCH -o logfile.out
#SBATCH -e logfile.err

ml cmake
ml gcc/11.2.0
ml openmpi

export NEK_SOURCE_ROOT=/path/to/Nek5000
export OMP_NUM_THREADS=1

# Prevent conda from polluting the linker path
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v conda | tr '\n' ':')

cd $SLURM_SUBMIT_DIR
$NEK_SOURCE_ROOT/bin/nekmpi annulus $SLURM_NTASKS
