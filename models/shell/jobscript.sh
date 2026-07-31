#!/bin/bash
#SBATCH --job-name=hydro_par
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --account=phy180013

module purge
module load gcc/11.2.0
module load openmpi/4.0.6

cd $SLURM_SUBMIT_DIR

echo shell > SESSION.NAME
echo $SLURM_SUBMIT_DIR/ >> SESSION.NAME

mpirun -np 16 ./nek5000 > logfile 2>&1
