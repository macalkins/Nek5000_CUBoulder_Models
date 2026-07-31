# Getting Started

This guide walks you through setting up Nek5000 on an HPC cluster and
running your first simulation (the spherical shell model). It assumes you
are comfortable with the Linux command line and batch job submission, but
have not used Nek5000 before.

---

## 1. What is Nek5000?

Nek5000 is a spectral element code for incompressible fluid dynamics. Unlike
finite-difference or finite-volume codes, it uses high-order polynomial
(Gauss-Legendre-Lobatto) basis functions within each element. The key
consequence for you: the mesh is relatively coarse (fewer elements than an
equivalent FD grid), but each element contains many GLL quadrature points
set by the polynomial order `lx1`. Resolution is controlled by both the
number of elements *and* the polynomial order.

Nek5000 is written in Fortran 77 and C, parallelises with MPI, and requires
a small amount of recompilation each time you change the mesh or polynomial
order (because array sizes are set at compile time in a file called `SIZE`).

---

## 2. Cluster setup

### Modules

You will need CMake, a C/Fortran compiler, and MPI. The exact module names
vary by cluster — check the cluster documentation. Typical examples:

```bash
# Frontera (TACC)
module load cmake gcc/11.2.0 openmpi

# Alpine (CU Boulder)
module load gcc openmpi cmake

# Anvil (Purdue)
module load gcc openmpi cmake
```

Add the appropriate lines to your `~/.bashrc` so they load automatically.

### Nek5000

Clone and build the Nek5000 tools, then add the `bin/` directory to your
PATH so you can call `makenek` and `genmap` from anywhere:

```bash
git clone https://github.com/Nek5000/Nek5000.git
cd Nek5000/tools && ./maketools genmap gmsh2nek
export PATH=/path/to/Nek5000/bin:$PATH   # put this in ~/.bashrc
```

For the hemisphere model, first apply the anti-periodic patches to the
Nek5000 source — see `patches/README.md`.

---

## 3. Clone this repository

Clone to a persistent directory (not scratch — scratch is for run output
and may be purged):

```bash
git clone https://github.com/macalkins/Nek5000_CUBoulder_Models.git
```

Each model directory contains a `.usr` file, a `.par` file, `SIZE`, and a
`README.md` with model-specific detail. The `README.md` files are
comprehensive — refer to them once you are comfortable with the basics.

---

## 4. The Nek5000 workflow

Every Nek5000 simulation follows the same four steps:

```
1. Generate mesh  →  2. Compile  →  3. Run  →  4. Post-process
```

### Step 1 — Generate the mesh

Each model provides a Python mesh generator; see the model README for
details. For the spherical shell the generator writes the `.re2` binary
mesh directly:

```bash
cd models/shell
python3 gen_cubed_sphere.py --nphi 6 --nr 8 --eta 0.35 -o shell.re2
```

Next, generate the connectivity map with `genmap`:
```bash
genmap
# when prompted: enter "shell" (without extension)
# when prompted for tolerance: enter 0.05
```
This produces `shell.ma2`.

(The cylinder model instead uses gmsh: its generator writes a `.geo` file,
gmsh produces a `.msh`, and `gmsh2nek` converts it to `.re2` — see
`models/cylinder/README.md`.)

### Step 2 — Compile

```bash
makenek shell
```

This compiles `shell.usr` against the Nek5000 source and produces a
`nek5000` executable. You need to recompile whenever you change `SIZE` or
`shell.usr`. You do *not* need to recompile to change runtime parameters
in `shell.par`.

### Step 3 — Run

For a quick test on a login node (small problems only, a few minutes max):
```bash
nekmpi shell 4    # run with 4 MPI ranks
```

For production runs, submit a batch job. Template job scripts for
Frontera, Alpine, and Anvil are provided in `scripts/job_scripts/` —
copy the appropriate one into your run directory, fill in your
allocation code and paths, and submit:

```bash
cp /path/to/repo/scripts/job_scripts/job_alpine.sh run00/job.sh
# edit job.sh to set -A, paths, and node/core counts
sbatch job.sh
```

The key differences between clusters are the partition name, cores per
node, and the MPI launcher (`ibrun` on Frontera/TACC, `mpirun` on
Alpine and Anvil). The job scripts include comments explaining each.

Submit with `sbatch job.sh`. Monitor with `squeue -u yourusername`.

### Step 4 — Post-process

Nek5000 writes binary field files named `shell0.f00001`, `shell0.f00002`,
etc., each containing a snapshot of velocity and temperature. The
`scripts/` directory contains Python utilities for reading and plotting
these. See Section 6 below.

---

## 5. Setting up a run directory

Rather than running directly in the model source directory, keep each run in
its own directory on scratch. The shared script `create_run_directory.sh`
automates this:

```bash
# First, copy the compiled binary and model files to your scratch case directory
cp /path/to/repo/models/shell/nek5000 \
   /path/to/repo/models/shell/shell.par \
   /path/to/repo/models/shell/SIZE \
   /scratch/MyCase/

cd /scratch/MyCase/
bash /path/to/repo/scripts/create_run_directory.sh shell
```

This creates `run00/` (or the next available `runNN/`) containing the
binary, restart file, and a `SESSION.NAME` pointing to the correct path.

---

## 6. Post-processing scripts

All scripts live in `scripts/` and use standard Python (numpy, matplotlib;
pymech for reading field files):

| Script | What it does |
|--------|-------------|
| `plot_nusselt.py` | Time series of Nusselt number |
| `radial_profiles.py` | Shell-averaged mean/RMS radial profiles from field files |
| `equatorial_spectra.py` | Azimuthal energy spectra in the equatorial plane |
| `sphere_resolution.py` | Boundary-layer resolution check for shell meshes |
| `hemisphere_resolution.py` | Same, for the hemisphere wedge |
| `paraview_files.sh` | Generate the metadata file ParaView needs to open field files |

---

## 7. Workflow summary

```bash
# One-time setup (add to ~/.bashrc)
module load cmake gcc openmpi
export PATH=/path/to/Nek5000/bin:$PATH

# Generate mesh and compile (in the model source directory)
cd /path/to/repo/models/shell
python3 gen_cubed_sphere.py --nphi 6 --nr 8 --eta 0.35 -o shell.re2
genmap          # input: shell, tolerance 0.05  →  shell.ma2
makenek shell   # compiles  →  nek5000

# Set up run directory on scratch
cp nek5000 shell.par SIZE /scratch/MyCase/
cd /scratch/MyCase/
bash /path/to/repo/scripts/create_run_directory.sh shell
cd run00/
sbatch job.sh
```

---

## 8. Further reading

- Nek5000 documentation: https://nek5000.github.io/NekDoc/
- Each model's `README.md` has physics details, parameter tables, and
  full mesh/build/run instructions.
