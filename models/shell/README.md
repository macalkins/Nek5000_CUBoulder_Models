# Rotating Convection in a Spherical Shell (Hydro)

Hydrodynamic rotating convection in a spherical shell using Nek5000.
Reference: Yadav et al. (PNAS, 2016) hydro case -- Re ~ 75.7, Nu ~ 2.19.

## Parameters

| Parameter | Value |
|-----------|-------|
| Ekman number (Ek) | 1e-4 |
| Rayleigh number (Ra) | 5e6 |
| Prandtl number (Pr) | 1 |
| Radius ratio (eta) | 0.35 |
| Shell thickness (d) | 1 (nondimensional) |
| Inner radius (ri) | eta/(1-eta) = 0.5385 |
| Outer radius (ro) | 1/(1-eta) = 1.5385 |

No-slip velocity and fixed temperature (T=1 inner, T=0 outer) on both walls.
Temperature BCs set in `userbc` (not in the mesh).
Buoyancy with gravity proportional to radius, Coriolis along z-axis.

## Mesh

Cubed-sphere: 6 cube faces projected onto concentric spherical shells.
Default: 6x6 angular elements per face, 8 radial shells = 1728 elements.
Radial distribution uses Gauss-Lobatto-Legendre (GLL) node spacing
(denser near inner and outer boundaries).
Polynomial order N=7 (lx1=8), dealiasing lxd=12, ~885k DOFs.

## Quick Start

```bash
# 1. Generate mesh (writes .re2 binary directly, no .rea needed)
python3 gen_cubed_sphere.py --nphi 6 --nr 8 --eta 0.35 -o shell.re2

# 2. Generate partition map
printf 'shell\n0.05\n' | $NEK_SOURCE_ROOT/bin/genmap

# 3. Build solver
./makenek

# 4. Run
sbatch jobscript.sh
```

## Mesh Generator (`gen_cubed_sphere.py`)

Python script that generates a cubed-sphere spherical shell mesh and writes
it directly as a Nek5000 `.re2` binary file. No intermediate `.rea` file
or `reatore2` conversion is needed.

### Algorithm

1. Create a uniform Cartesian grid on each of the 6 faces of a unit cube.
2. Normalize each grid point to obtain unit direction vectors on the sphere.
3. Build a radial grid using GLL node spacing between ri and ro.
4. Scale direction vectors by radial values to get 3-D element corners.
5. Mark inner and outer element faces as spherically curved ('s').
6. Assign velocity and temperature BCs on inner/outer boundaries.
7. Write the .re2 binary file.

### Options

```
-o FILE              Output .re2 filename (default: sphere.re2)
--nphi N             Angular elements per cube face side (default: 6)
--nr N               Number of radial shells (default: 8)
--eta FLOAT          Radius ratio ri/ro (default: 0.35)
--thickness FLOAT    Shell thickness d (default: 1.0)
--bc-v-inner TYPE    Velocity BC on inner wall: no-slip or stress-free (default: no-slip)
--bc-v-outer TYPE    Velocity BC on outer wall: no-slip or stress-free (default: no-slip)
--bc-t-inner TYPE    Temperature BC on inner wall: userbc or fixed (default: userbc)
--bc-t-outer TYPE    Temperature BC on outer wall: userbc or fixed (default: userbc)
--T-inner FLOAT      Fixed temperature at inner wall (default: 0.0)
--T-outer FLOAT      Fixed temperature at outer wall (default: 0.0)
```

Total elements = 6 * nphi^2 * nr. For the default (nphi=6, nr=8): 1728 elements.

### Examples

```bash
# Default Jackson benchmark mesh
python3 gen_cubed_sphere.py -o shell.re2

# Higher resolution (6144 elements)
python3 gen_cubed_sphere.py --nphi 8 --nr 16 -o shell.re2

# Stress-free outer boundary
python3 gen_cubed_sphere.py --bc-v-outer stress-free -o shell.re2

# Fixed temperature BCs in the mesh (instead of userbc)
python3 gen_cubed_sphere.py --bc-t-inner fixed --T-inner 1.0 \
                            --bc-t-outer fixed --T-outer 0.0 -o shell.re2
```

### Note on genmap tolerance

Use a tolerance of 0.05 (tighter than the default 0.2) when running genmap
on the cubed-sphere mesh. This prevents false connectivity matches at
cube-face edges where element faces from different cube faces come close
together but are not actually connected.

## Parameter File (`shell.par`)

The `.par` file controls the simulation at runtime.

| Setting | Value | Description |
|---------|-------|-------------|
| `numSteps` | 5000 | Total timesteps |
| `dt` | 1e-4 | Initial timestep |
| `variableDt` / `targetCFL` | yes / 0.5 | Adaptive timestepping |
| `timeStepper` | BDF3 | 3rd-order backward differentiation |
| `writeInterval` | 100 | Checkpoint every 100 steps |
| `userParam01` | 1e-4 | Ekman number |
| `userParam02` | 5e6 | Rayleigh number |
| `userParam03` | 0.35 | Radius ratio eta |
| `userParam04` | 5 | Diagnostic output frequency (steps) |

Filtering is disabled by default (commented out). Uncomment the
`filtering`, `filterWeight`, and `filterModes` lines if needed for
stability at higher resolution.

Viscosity and conductivity are both 1.0 (nondimensional, Pr = 1).

To restart from a checkpoint, add `startFrom = shell0.f00XXX` under `[GENERAL]`.

## Building the Solver

```bash
module purge
module load gcc/11.2.0
module load openmpi/4.0.6
./makenek
```

## Running

Generate the partition map before the first run:
```bash
printf 'shell\n0.05\n' | $NEK_SOURCE_ROOT/bin/genmap
```

Submit the job:
```bash
sbatch jobscript.sh
```

The job runs on 1 node with 16 MPI ranks (`shared` partition, account
`phy180013`). To change core count, edit `--ntasks` in the job script
and adjust `lpmin` in `SIZE`.

## Diagnostics

The `userchk` routine writes two CSV files every `userParam04` timesteps:

- `reynolds.csv`: time, Re, Re_x, Re_y, Re_z (volume-averaged RMS Reynolds number)
- `nusselt.csv`: time, Nu_in, Nu_out (Nusselt number on inner/outer walls)

## Files

| File | Description |
|------|-------------|
| `gen_cubed_sphere.py` | Mesh generator (Python, writes .re2 directly) |
| `shell.usr` | User routines (forcing, BCs, ICs, diagnostics) |
| `shell.par` | Runtime parameters |
| `SIZE` | Compile-time dimensions |
| `makenek` | Build script |
| `jobscript.sh` | SLURM submission script |
