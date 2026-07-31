# Hemisphere (Cubed-Sphere Wedge)

180-degree spherical shell wedge with anti-periodic boundary conditions
for rotating convection and dynamo simulations.

## Geometry

- Cubed-sphere mesh: 3 cube faces for the hemisphere (y >= 0)
- Cut plane at y = 0 with 180-degree anti-periodic BCs
- Spherical shell: inner radius ri, outer radius ro (default eta = ri/ro = 0.35)

## Key Requirements

- **PnPn formulation** (lx2 = lx1): PnPn-2 pressure solver cannot converge
  with periodic BCs
- **Patched Nek5000**: requires anti-periodic flip patches in
  `core/hmholtz.f` and `core/navier1.f` (see `patches/` directory)
- **ifcyclic = .false.**: the built-in rotate_cyc fails at 180 degrees

## Mesh Generation

```bash
python gen_hemisphere.py --na 20 --nr 12 --radial_dist chebyshev --no-mhd --output shell
```

Parameters:
- `--na`: angular elements per cube face edge (must be even)
- `--nr`: radial element layers
- `--ri/--ro`: inner/outer radius
- `--radial_dist`: uniform, chebyshev, or geometric
- `--no-mhd`: omit magnetic field BCs (hydro only)

Total elements: `nelg = 3 * na^2 * nr`

The script generates both `.re2` and `.ma2` files directly, bypassing genmap.
A `_mesh_info.txt` file is also written with all generation parameters.

## Setup New Case

1. Generate mesh with `gen_hemisphere.py`
2. Copy `shell.usr` and `shell.par` (adjust parameters)
3. Update SIZE: set `lelg` to match mesh, adjust `lpmin` for target core count
4. Compile with patched Nek5000: `makenek shell`

## Validated Cases

| Ek | Ra | na | nr | nelg |
|----|------|----|----|------|
| 1e-4 | 4e6 | 16 | 10 | 7,680 |
| 1e-5 | 1e8 | 20 | 12 | 14,400 |
| 1e-6 | 8e8 | 28 | 18 | 42,336 |
