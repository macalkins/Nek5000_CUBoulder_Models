# Nek5000 Rotating Convection Models

Working [Nek5000](https://nek5000.mcs.anl.gov/) model configurations for
rotating convection in cylindrical and spherical geometries, from the
Calkins group (University of Colorado Boulder).

## Models

| Model | Directory | Description |
|-------|-----------|-------------|
| Cylinder | `models/cylinder/` | Rotating Rayleigh–Bénard convection in an upright circular cylinder; gmsh O-grid mesh with Tilgner boundary-layer stretching |
| Spherical shell | `models/shell/` | Rotating convection in a spherical shell; cubed-sphere mesh generated directly in Python (no external mesher) |
| Hemisphere | `models/hemisphere/` | Rotating convection in a 180° spherical wedge with anti-periodic (rotational-symmetry) boundary conditions |

## Repository layout

```
models/
  cylinder/           # .usr, .par, SIZE, mesh generator, README
  shell/              # .usr, .par, SIZE, mesh generator, README
  hemisphere/         # .usr, .par, SIZE, mesh generators, README
scripts/
  job_scripts/        # SLURM job script templates (Frontera, Alpine, Anvil)
  ...                 # shared mesh-resolution and post-processing utilities
patches/              # Nek5000 source patches (required for the hemisphere model)
```

Each model directory contains a `.usr` file, a `.par` file, a `SIZE` file,
and a `README.md` with the physics, parameter tables, and step-by-step
mesh/build/run instructions.

New to Nek5000? Start with [GETTING_STARTED.md](GETTING_STARTED.md).

## Requirements

- [Nek5000](https://github.com/Nek5000/Nek5000) (v19 or later)
- Fortran/C compilers + MPI
- Python 3 with numpy (mesh generation); matplotlib, pymech (post-processing)
- [gmsh](https://gmsh.info/) (cylinder model only)

The hemisphere model additionally requires the anti-periodic patches in
`patches/` to be applied to the Nek5000 source — see `patches/README.md`.

## License

GPL-3.0 — see [LICENSE](LICENSE).
