# Cylinder — rotating convection in a circular cylinder

Rotating Rayleigh–Bénard convection in an upright circular cylinder, meshed with
a gmsh O-grid (butterfly) cross-section extruded in the axial direction.

- **Mechanical BCs:** no-slip, impenetrable on all walls.
- **Thermal BCs:** fixed temperature on the top and bottom plates, insulating on
  the curved sidewall.
- **Non-dimensionalization:** length by height `H=1`; the Ekman number carries a
  factor of 2, `E = nu / (2 Omega H^2)`, matching the Coriolis term `uy/Ek` in
  `rrbc_cylinder.usr`. The buoyancy uses the reduced Rayleigh number
  `R_tilde = Ra * E^{4/3}`.

## Parameters (`rrbc_cylinder.par`)

| userParam | meaning |
|-----------|---------|
| `userParam01` | reduced Rayleigh number, `R_tilde = Ra * E^{4/3}` |
| `userParam02` | Ekman number, `E = nu / (2 Omega H^2)` |
| `userParam03` | aspect ratio `D/H` (used by the mesh generator) |
| `userParam04` | initial-condition seed: `0` = random noise, `1` = Cartesian hex tessellation at the critical wavenumber `a_c = 1.3048 Ek^{-1/3}` |

`conductivity` in `[TEMPERATURE]` is `1/Pr`.

Both ICs start from rest with the conductive temperature profile `T = 1 - z`
plus a small seed. The random seed lets the flow select its own planform; the
hex seed (amplitude `1e-2`, `sin(pi*z)` vertical structure) jump-starts
convection at the critical wavelength for a faster, cleaner near-onset transient.

## Mesh generation

`scripts/make_cylinder_geo.py` is the **canonical** mesh generator. It reads
`Ek` and the aspect ratio from `rrbc_cylinder.par` and writes `cylinder.geo`
using **Tilgner (1999) sine stretching** (the same clustering used by the
`annulus` and `channel` box generators):

- **axial (z):** two-sided Tilgner (`beta_z`) clusters elements at both the top
  and bottom plates to resolve the **Ekman layers**;
- **radial:** one-sided Tilgner (`beta_r`) clusters elements at `r=R` to resolve
  the **Stewartson layer**.

```
python scripts/make_cylinder_geo.py [Nc Nr Nz beta_z beta_r]   # defaults: 16 10 32 0.9 0.9
gmsh -3 cylinder.geo -o cylinder.msh -format msh2 -bin 0
gmsh2nek        # cylinder.msh -> rrbc_cylinder.re2   (enter base name: rrbc_cylinder)
genmap          # rrbc_cylinder.re2 -> rrbc_cylinder.ma2
```

The generator prints a boundary-layer resolution report (GLL points within the
Ekman and Stewartson layers, assuming `lx1=8`) and the total element count.
Update `lelg` in `SIZE` if the element count changes.

## Build & run

```
makenek rrbc_cylinder            # compile
ibrun ./nek5000                  # or a batch job (>= lpmin MPI ranks)
```

## Vista (TACC)

See [VISTA.md](VISTA.md) for the full build/mesh/run workflow on Vista,
including the required MPI-wrapper and bcrobin_mhd build fixes.
