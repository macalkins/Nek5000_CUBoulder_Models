#!/bin/bash
# Vista (TACC) production batch script for the cylinder case. See VISTA.md.
#SBATCH -J cyl_prod
#SBATCH -p gg              # Grace-Grace CPU partition (144 cores/node)
#SBATCH -N 4               # 4 nodes -> 576 ranks, ~50 elements/rank (good efficiency)
#SBATCH -n 576
#SBATCH -t 24:00:00        # qgg max is 48h; run is walltime-limited
#SBATCH -A YOUR_ALLOCATION        # TACC project/allocation (UPPERCASE)
#SBATCH -o cyl_prod.%j.out

# Load the SAME MPI stack that nek5000 was compiled with (see makenek / VISTA.md).
module load nvidia/24.7 openmpi/5.0.5

cd $SLURM_SUBMIT_DIR

# SESSION.NAME: line 1 = case name, line 2 = absolute run dir with trailing slash
echo rrbc_cylinder          >  SESSION.NAME
echo "$SLURM_SUBMIT_DIR/"   >> SESSION.NAME

ibrun ./nek5000 > logfile 2>&1

# To continue past 24h: uncomment `startFrom = <last rrbc_cylinder0.f#####>` in
# rrbc_cylinder.par and resubmit.
