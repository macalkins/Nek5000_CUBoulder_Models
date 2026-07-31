# Building and running on Vista (TACC)

Step-by-step for the `cylinder` model on Vista. Most of the gotchas below are
general to this Nek5000 build on Vista, not specific to the cylinder case.

Assumptions:
- Nek5000 source at `$WORK/Nek5000` (set `NEK_SOURCE_ROOT` to it).
- This model directory copied into a run directory on `$SCRATCH`.

## 0. One-time: build the mesh tools

`gmsh2nek` and `genmap` are usually not pre-built. Build them once:

```bash
cd $WORK/Nek5000/tools
./maketools genmap
./maketools gmsh2nek
# binaries land in $WORK/Nek5000/bin
```

## 1. Generate the mesh

`make_cylinder_geo.py` needs only numpy; `gmsh` produces the `.msh`.

```bash
python scripts/make_cylinder_geo.py            # writes cylinder.geo (+ a BL report)
gmsh -3 cylinder.geo -o cylinder.msh -format msh2 -bin 0
```

On Vista, `gmsh` is a module that requires a `gcc` module first:

```bash
module load gcc/14.2.0 gmsh
```

Note: a `miniconda` entry on `PATH` can shadow the module's `gmsh`. If
`which gmsh` does not point into the module, generate the `.msh` on a machine
where gmsh works and copy `cylinder.msh` over — it is portable ASCII.

## 2. Convert to Nek format

`gmsh2nek` is interactive. Its prompts, in order, are:

```
Enter mesh dimension:                            -> 3
Input fluid .msh file name:                      -> cylinder      (no .msh)
Do you have solid mesh ? (0 no, 1 yes)           -> 0
Enter number of periodic boundary surface pairs: -> 0             (closed cylinder)
please give re2 file name:                       -> rrbc_cylinder
```

Scripted:

```bash
export PATH=$WORK/Nek5000/bin:$PATH
printf '3\ncylinder\n0\n0\nrrbc_cylinder\n' | gmsh2nek     # -> rrbc_cylinder.re2
printf 'rrbc_cylinder\n0.2\n'               | genmap       # -> rrbc_cylinder.ma2
```

`gmsh2nek` should report boundary IDs `WALL(1) WALL_BOTTOM(2) WALL_TOP(3)`.

## 3. Compile

```bash
export NEK_SOURCE_ROOT=$WORK/Nek5000
./makenek                                # builds ./nek5000
```

Two things this build depends on (both handled for you):

1. **Force the MPI wrappers.** Vista's `nvidia` module pre-exports
   `FC=nvfortran` / `CC=nvc`. Nek expects `FC`/`CC` to be the MPI wrappers and
   adds no MPI include paths itself, so `makenek` sets `FC=mpif90 CC=mpicc`.
   Without this the build fails with `cannot open source file mpi.h` /
   `Unable to open include file mpif.h`.

2. **`bcrobin_mhd` stub.** This Nek build has a patched `induct.f` that always
   calls `bcrobin_mhd`, so every case (even hydro-only) must define it or the
   link fails with `undefined reference to bcrobin_mhd_`. `rrbc_cylinder.usr`
   includes a no-op stub at the end. (The annulus/channel/shell models do the
   same.)

## 4. Run

The mesh has 28,672 elements and `SIZE` uses `lelg=30000`, `lpmin=64`
(`lelt = lelg/lpmin + 3 = 471`), so it needs at least
`ceil(28672/471) = 61` MPI ranks. One `gg` (Grace CPU) node is plenty.

For a first (cold) start, comment out `startFrom` in the `.par` — there is no
restart file shipped:

```bash
sed -i 's/^startFrom/#startFrom/' rrbc_cylinder.par
```

Then submit `job.sh` (partition `gg`, account `YOUR_ALLOCATION` — **TACC project names
are UPPERCASE**, MPI stack matching the build, launched with `ibrun`):

```bash
sbatch job.sh
squeue --me
tail -f logfile
```

A successful run ends with `run successful: dying` and
`TACC:  Shutdown complete`.

### Smoke test

Set `numSteps = 20` in the `.par` for a quick end-to-end check before a
production run. The 20-step cold-start smoke test above ran on 64 ranks of one
`gg` node in ~70 s.

### Production

The shipped `job.sh` uses **4 `gg` nodes / 576 ranks** (~50 elements/rank), a
good efficiency/throughput balance for this 28,672-element mesh (~0.45 s/step,
extrapolated from the 64-rank smoke test at 3.46 s/step). Up to 8 nodes / 1152
ranks is still reasonable if faster turnaround is worth some efficiency.

`rrbc_cylinder.par` is set for a walltime-limited run: `numSteps` is large and
`writeInterval = 8000` writes a field checkpoint roughly once per wall-hour.
The `.usr` writes `nusselt.csv` (time, bottom Nu, top Nu) and `reynolds.csv`
every timestep. To continue past a 24 h job, set
`startFrom = <last rrbc_cylinder0.f#####>` in the `.par` and resubmit.
