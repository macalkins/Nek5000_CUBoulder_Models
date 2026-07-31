# Nek5000 Core Patches

Patches against upstream Nek5000 for anti-periodic boundary conditions
in the 180-degree hemisphere wedge geometry.

## Anti-Periodic Flip Patches

Required for the hemisphere model. These patches add a `dssum_ap_flip`
subroutine that applies sign flips to velocity components (ux, uy) at
periodic boundaries before/after direct stiffness summation, implementing
true anti-periodic (rotational) symmetry.

### Files Modified

- `core/hmholtz.f`: adds `dssum_ap_flip` and calls it in the Helmholtz
  solver and its preconditioner
- `core/navier1.f`: adds pflip-aware `opdssum` and `opdsop` variants

### Applying the Patches

```bash
cd /path/to/Nek5000
patch -p0 < patches/hmholtz_anti_periodic.patch
patch -p0 < patches/navier1_anti_periodic.patch
```

### Notes

- Only needed for hemisphere (anti-periodic) cases
- Standard periodic cases (annulus, box) do not need these patches
- The patches assume the periodic vertex pairs are set up by `gen_hemisphere.py`
