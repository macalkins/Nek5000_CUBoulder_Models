#!/usr/bin/env python3
"""
Generate a cubed-sphere spherical shell mesh for Nek5000.

Outputs a .re2 binary mesh file directly (no intermediate .rea needed).

The cubed-sphere mesh is constructed by projecting the faces of a cube
onto concentric spherical shells. This gives a quasi-uniform angular
resolution over the full sphere without polar singularities.

Algorithm:
  1. Create a uniform grid on each of the 6 faces of a unit cube
     (equidistant in Cartesian coordinates on the cube face).
  2. Normalize each grid point to obtain unit direction vectors.
  3. Build a radial grid using Gauss-Lobatto-Legendre (GLL) node
     spacing (denser near the inner and outer boundaries).
  4. Scale direction vectors by the radial grid values to get the
     3-D element corner coordinates.
  5. Assign spherical curvature ('s') to the inner and outer faces
     of each element.
  6. Assign boundary conditions on the inner and outer surfaces:
     - Velocity: no-slip ('v  ') on both walls (default), or
       stress-free ('SYM') if requested.
     - Temperature: set in userbc ('t  ') by default, or
       fixed Dirichlet ('T  ') with specified values.
  7. Write the .re2 binary file.

Usage:
    python gen_cubed_sphere.py [options]

Examples:
    # Default Jackson benchmark mesh (eta=0.35, 6 angular, 8 radial)
    python gen_cubed_sphere.py -o jackson.re2

    # Higher resolution
    python gen_cubed_sphere.py --nphi 12 --nr 16 -o jackson.re2

    # Stress-free outer, no-slip inner
    python gen_cubed_sphere.py --bc-v-outer stress-free -o jackson.re2

Author: Auto-generated for Nek5000 cubed-sphere spherical shell cases.
Based on the Fortran cubed_sphere.f90 by Jan Vormann and the Python
belt.py mesh generator by Tobias.
"""

import argparse
import struct
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Gauss-Lobatto-Legendre (GLL) nodes
# ---------------------------------------------------------------------------

def gll_nodes(n):
    """
    Compute the n Gauss-Lobatto-Legendre nodes on [-1, 1].

    Uses the eigenvalue method: the interior GLL nodes are eigenvalues
    of a symmetric tridiagonal matrix (the Jacobi matrix for the
    weight function associated with P'_N).

    Parameters
    ----------
    n : int
        Number of nodes (polynomial order + 1).

    Returns
    -------
    z : ndarray, shape (n,)
        GLL nodes in ascending order, z[0] = -1, z[-1] = 1.
    """
    if n < 2:
        raise ValueError("Need at least 2 GLL nodes")
    z = np.zeros(n)
    z[0] = -1.0
    z[-1] = 1.0
    if n == 2:
        return z
    if n == 3:
        z[1] = 0.0
        return z
    # Interior nodes via eigenvalue of symmetric tridiagonal matrix
    p = n - 1  # polynomial degree
    M = np.zeros((p - 1, p - 1))
    for i in range(p - 2):
        j = i + 1
        val = 0.5 * np.sqrt(j * (j + 2) / ((j + 0.5) * (j + 1.5)))
        M[i, i + 1] = val
        M[i + 1, i] = val
    interior = np.sort(np.linalg.eigvalsh(M))
    z[1:-1] = interior
    return z


def map_gll_to_interval(n, a, b):
    """
    Map n GLL nodes from [-1, 1] to the interval [a, b].

    Parameters
    ----------
    n : int
        Number of nodes.
    a, b : float
        Interval endpoints.

    Returns
    -------
    ndarray, shape (n,)
    """
    z = gll_nodes(n)
    return 0.5 * (1.0 - z) * a + 0.5 * (1.0 + z) * b


# ---------------------------------------------------------------------------
# Cube-face grid
# ---------------------------------------------------------------------------

def cube_face_grid(nphi):
    """
    Generate the grid points on the 6 faces of a unit cube centered at the
    origin (side length = 1, faces at +/-0.5).

    Uses equidistant spacing on the cube face (phi_method=2 in the
    Fortran code).

    Parameters
    ----------
    nphi : int
        Number of elements per cube face side direction.

    Returns
    -------
    cube_grid : ndarray, shape (6, nphi+1, nphi+1, 3)
        (face, i, j, xyz) coordinates on the cube faces.
    """
    n1 = nphi + 1
    grid_1d = np.linspace(-0.5, 0.5, n1)

    cube = np.zeros((6, n1, n1, 3))

    # Face 1 (front):  y = -0.5, vary x and z
    cube[0, :, :, 1] = -0.5
    for i in range(n1):
        for j in range(n1):
            cube[0, i, j, 0] = grid_1d[i]
            cube[0, i, j, 2] = grid_1d[j]

    # Face 2 (right):  x = +0.5, vary y and z
    cube[1, :, :, 0] = 0.5
    for i in range(n1):
        for j in range(n1):
            cube[1, i, j, 1] = grid_1d[i]
            cube[1, i, j, 2] = grid_1d[j]

    # Face 3 (back):   y = +0.5, vary x and z
    cube[2, :, :, 1] = 0.5
    for i in range(n1):
        for j in range(n1):
            cube[2, i, j, 0] = grid_1d[i]
            cube[2, i, j, 2] = grid_1d[j]

    # Face 4 (left):   x = -0.5, vary y and z
    cube[3, :, :, 0] = -0.5
    for i in range(n1):
        for j in range(n1):
            cube[3, i, j, 1] = grid_1d[i]
            cube[3, i, j, 2] = grid_1d[j]

    # Face 5 (bottom): z = -0.5, vary x and y
    cube[4, :, :, 2] = -0.5
    for i in range(n1):
        for j in range(n1):
            cube[4, i, j, 0] = grid_1d[i]
            cube[4, i, j, 1] = grid_1d[j]

    # Face 6 (top):    z = +0.5, vary x and y
    cube[5, :, :, 2] = 0.5
    for i in range(n1):
        for j in range(n1):
            cube[5, i, j, 0] = grid_1d[i]
            cube[5, i, j, 1] = grid_1d[j]

    # Project to unit sphere (normalize to unit vectors)
    norms = np.sqrt(np.sum(cube ** 2, axis=3, keepdims=True))
    cube /= norms

    return cube


# ---------------------------------------------------------------------------
# Build the mesh
# ---------------------------------------------------------------------------

def build_cubed_sphere(nphi, nr, eta, d=1.0,
                       bc_v_inner='v  ', bc_v_outer='v  ',
                       bc_t_inner='t  ', bc_t_outer='t  ',
                       T_inner=0.0, T_outer=0.0):
    """
    Build a cubed-sphere spherical shell mesh.

    Parameters
    ----------
    nphi : int
        Number of angular elements per cube face side direction.
    nr : int
        Number of radial shells (elements in the radial direction).
    eta : float
        Radius ratio r_i / r_o.
    d : float
        Shell thickness (nondimensional, default 1.0).
    bc_v_inner, bc_v_outer : str
        Velocity BC type on inner/outer walls ('v  ' for no-slip,
        'SYM' for stress-free).
    bc_t_inner, bc_t_outer : str
        Temperature BC type ('t  ' for userbc, 'T  ' for fixed).
    T_inner, T_outer : float
        Fixed temperature values if bc_t is 'T  '.

    Returns
    -------
    coords : ndarray, shape (nel, 8, 3)
        Element corner coordinates (x, y, z) for all elements.
        Corner ordering follows Nek5000 convention.
    curves : list of tuples
        Each entry: (elem_1based, face_1based, radius)
        Curved side data for spherical faces.
    vel_bcs : list of tuples
        Each entry: (elem_1based, face_1based, bc_type, bc_vals)
        Non-internal velocity boundary conditions.
    temp_bcs : list of tuples
        Same format for temperature boundary conditions.
    nel : int
        Total number of elements.
    nelv : int
        Number of velocity (fluid) elements (= nel for no mantle).
    """
    r_o = d / (1.0 - eta)
    r_i = r_o - d
    nel = 6 * nphi * nphi * nr
    nelv = nel

    print(f"Inner radius r_i = {r_i:.6f}")
    print(f"Outer radius r_o = {r_o:.6f}")
    print(f"Total elements: {nel}")

    # Direction vectors on cube faces
    cube_grid = cube_face_grid(nphi)

    # Radial grid with GLL spacing
    rad_grid = map_gll_to_interval(nr + 1, r_i, r_o)
    print(f"Radial grid: {rad_grid}")

    # Allocate arrays
    coords = np.zeros((nel, 8, 3))
    curves_list = []
    vel_bcs = []
    temp_bcs = []

    elcount = 0

    for ie in range(6):       # cube face (0-based: 0..5 -> Fortran 1..6)
        for in_ in range(nphi):
            for im in range(nphi):
                for ir in range(nr):
                    # Get the 4 angular direction vectors for this element
                    # (corners on the cube face grid)
                    d00 = cube_grid[ie, in_,     im,     :]
                    d10 = cube_grid[ie, in_ + 1, im,     :]
                    d01 = cube_grid[ie, in_,     im + 1, :]
                    d11 = cube_grid[ie, in_ + 1, im + 1, :]

                    r_inner = rad_grid[ir]
                    r_outer = rad_grid[ir + 1]

                    # The corner ordering depends on which cube face we are
                    # on, matching the Fortran code exactly.
                    # Fortran corners are 1-8, Python corners are 0-7.
                    #
                    # Nek5000 hex convention:
                    #   corners 1-4: one face of the hex
                    #   corners 5-8: opposite face
                    #
                    # Each cube face (ie) has a specific mapping of
                    # (angular position, radius) to corner indices that
                    # ensures correct face orientation.

                    if ie == 0:  # Face 1 (front, y=-0.5)
                        coords[elcount, 0, :] = d00 * r_outer
                        coords[elcount, 1, :] = d10 * r_outer
                        coords[elcount, 2, :] = d10 * r_inner
                        coords[elcount, 3, :] = d00 * r_inner
                        coords[elcount, 4, :] = d01 * r_outer
                        coords[elcount, 5, :] = d11 * r_outer
                        coords[elcount, 6, :] = d11 * r_inner
                        coords[elcount, 7, :] = d01 * r_inner
                        # Curved faces: face 3 = inner, face 1 = outer
                        inner_face = 3
                        outer_face = 1

                    elif ie == 1:  # Face 2 (right, x=+0.5)
                        coords[elcount, 0, :] = d00 * r_inner
                        coords[elcount, 1, :] = d00 * r_outer
                        coords[elcount, 2, :] = d10 * r_outer
                        coords[elcount, 3, :] = d10 * r_inner
                        coords[elcount, 4, :] = d01 * r_inner
                        coords[elcount, 5, :] = d01 * r_outer
                        coords[elcount, 6, :] = d11 * r_outer
                        coords[elcount, 7, :] = d11 * r_inner
                        # face 4 = inner, face 2 = outer
                        inner_face = 4
                        outer_face = 2

                    elif ie == 2:  # Face 3 (back, y=+0.5)
                        coords[elcount, 0, :] = d00 * r_inner
                        coords[elcount, 1, :] = d10 * r_inner
                        coords[elcount, 2, :] = d10 * r_outer
                        coords[elcount, 3, :] = d00 * r_outer
                        coords[elcount, 4, :] = d01 * r_inner
                        coords[elcount, 5, :] = d11 * r_inner
                        coords[elcount, 6, :] = d11 * r_outer
                        coords[elcount, 7, :] = d01 * r_outer
                        # face 1 = inner, face 3 = outer
                        inner_face = 1
                        outer_face = 3

                    elif ie == 3:  # Face 4 (left, x=-0.5)
                        coords[elcount, 0, :] = d00 * r_outer
                        coords[elcount, 1, :] = d00 * r_inner
                        coords[elcount, 2, :] = d10 * r_inner
                        coords[elcount, 3, :] = d10 * r_outer
                        coords[elcount, 4, :] = d01 * r_outer
                        coords[elcount, 5, :] = d01 * r_inner
                        coords[elcount, 6, :] = d11 * r_inner
                        coords[elcount, 7, :] = d11 * r_outer
                        # face 2 = inner, face 4 = outer
                        inner_face = 2
                        outer_face = 4

                    elif ie == 4:  # Face 5 (bottom, z=-0.5)
                        coords[elcount, 0, :] = d00 * r_outer
                        coords[elcount, 1, :] = d10 * r_outer
                        coords[elcount, 2, :] = d11 * r_outer
                        coords[elcount, 3, :] = d01 * r_outer
                        coords[elcount, 4, :] = d00 * r_inner
                        coords[elcount, 5, :] = d10 * r_inner
                        coords[elcount, 6, :] = d11 * r_inner
                        coords[elcount, 7, :] = d01 * r_inner
                        # face 6 = inner, face 5 = outer
                        inner_face = 6
                        outer_face = 5

                    elif ie == 5:  # Face 6 (top, z=+0.5)
                        coords[elcount, 0, :] = d00 * r_inner
                        coords[elcount, 1, :] = d10 * r_inner
                        coords[elcount, 2, :] = d11 * r_inner
                        coords[elcount, 3, :] = d01 * r_inner
                        coords[elcount, 4, :] = d00 * r_outer
                        coords[elcount, 5, :] = d10 * r_outer
                        coords[elcount, 6, :] = d11 * r_outer
                        coords[elcount, 7, :] = d01 * r_outer
                        # face 5 = inner, face 6 = outer
                        inner_face = 5
                        outer_face = 6

                    # Curved sides: both inner and outer faces are spherical
                    # Store positive radius (ABS convention from Fortran).
                    curves_list.append((elcount + 1, inner_face,
                                        r_inner))
                    curves_list.append((elcount + 1, outer_face,
                                        r_outer))

                    # Boundary conditions on innermost/outermost shells
                    if ir == 0:
                        # Inner boundary
                        vel_bcs.append((elcount + 1, inner_face,
                                        bc_v_inner,
                                        [0.0, 0.0, 0.0, 0.0, 0.0]))
                        temp_bcs.append((elcount + 1, inner_face,
                                         bc_t_inner,
                                         [T_inner, 0.0, 0.0, 0.0, 0.0]))
                    if ir == nr - 1:
                        # Outer boundary
                        vel_bcs.append((elcount + 1, outer_face,
                                        bc_v_outer,
                                        [0.0, 0.0, 0.0, 0.0, 0.0]))
                        temp_bcs.append((elcount + 1, outer_face,
                                         bc_t_outer,
                                         [T_outer, 0.0, 0.0, 0.0, 0.0]))

                    elcount += 1

    assert elcount == nel, f"Element count mismatch: {elcount} != {nel}"
    print(f"Velocity BCs: {len(vel_bcs)}")
    print(f"Temperature BCs: {len(temp_bcs)}")
    print(f"Curved sides: {len(curves_list)}")

    return coords, curves_list, vel_bcs, temp_bcs, nel, nelv


# ---------------------------------------------------------------------------
# Write .re2 binary file
# ---------------------------------------------------------------------------

def write_re2(filename, coords, curves, vel_bcs, temp_bcs, nel, nelv, ndim=3):
    """
    Write a Nek5000 .re2 binary mesh file.

    The .re2 format (version #v002) layout:
      1. 80-byte ASCII header: '#v002 nel ndim nelv this is the hdr'
      2. 4-byte float32 endianness test value: 6.54321
      3. For each element (nel total):
         - 8 bytes: group index (float64)
         - ndim * 8 * 8 bytes: corner coordinates (float64),
           written as x(1:8), y(1:8), z(1:8) for corners 1-4 then 5-8
      4. 8 bytes: number of curved sides (float64)
      5. For each curved side:
         - 64 bytes (8 float64): elem, face, v1, v2, v3, v4, v5, ccurve
           where ccurve is an 8-byte field with the character 's' padded
      6. For each BC field (velocity, then temperature):
         - 8 bytes: number of non-internal BCs (float64)
         - For each non-internal BC:
           64 bytes (8 float64): elem, face, bc_val(1:5), bc_type_str

    Parameters
    ----------
    filename : str
        Output .re2 file path.
    coords : ndarray, shape (nel, 8, 3)
        Element corner coordinates.
    curves : list of (elem, face, radius)
        Curved side data.
    vel_bcs : list of (elem, face, bc_type, bc_vals)
        Velocity boundary conditions (non-internal only).
    temp_bcs : list of (elem, face, bc_type, bc_vals)
        Temperature boundary conditions.
    nel : int
        Total number of elements.
    nelv : int
        Number of velocity elements.
    ndim : int
        Number of spatial dimensions (always 3 for this generator).
    """
    with open(filename, 'wb') as f:
        # --- Header (80 bytes) ---
        hdr = f"#v002{nel:9d}{ndim:3d}{nelv:9d} this is the hdr"
        hdr = hdr.ljust(80)
        f.write(hdr.encode('ascii'))

        # --- Endianness test float (4 bytes, float32) ---
        f.write(struct.pack('<f', 6.54321))

        # --- Element data ---
        for ie in range(nel):
            # Group index (float64)
            f.write(struct.pack('<d', 0.0))

            # Corner coordinates: x(1:8), y(1:8), z(1:8)
            # Each coordinate is written as a contiguous block of 8 doubles.
            # This matches how Nek5000's reader unpacks the data:
            #   xc(1:8) from 8 consecutive doubles
            #   yc(1:8) from next 8 consecutive doubles
            #   zc(1:8) from next 8 consecutive doubles
            for dim in range(ndim):
                for k in range(8):
                    f.write(struct.pack('<d', coords[ie, k, dim]))

        # --- Curved sides ---
        ncurve = len(curves)
        f.write(struct.pack('<d', float(ncurve)))

        for elem, face, radius in curves:
            # buf2: elem, face, 0, 0, 0, radius, 1.0, ccurve
            # ccurve is stored as 8 bytes with 's' in the first byte
            buf = struct.pack('<7d', float(elem), float(face),
                              0.0, 0.0, 0.0, radius, 1.0)
            # ccurve: 's' followed by spaces to fill 8 bytes
            ccurve = b's   ' + b'    '
            f.write(buf + ccurve)

        # --- Velocity boundary conditions ---
        nbc_v = len(vel_bcs)
        f.write(struct.pack('<d', float(nbc_v)))

        for elem, face, bc_type, bc_vals in vel_bcs:
            buf = struct.pack('<7d', float(elem), float(face),
                              bc_vals[0], bc_vals[1], bc_vals[2],
                              bc_vals[3], bc_vals[4])
            # BC type string: 3 chars padded with spaces to fill 8 bytes
            bc_bytes = bc_type.encode('ascii').ljust(8, b' ')[:8]
            f.write(buf + bc_bytes)

        # --- Temperature boundary conditions ---
        nbc_t = len(temp_bcs)
        f.write(struct.pack('<d', float(nbc_t)))

        for elem, face, bc_type, bc_vals in temp_bcs:
            buf = struct.pack('<7d', float(elem), float(face),
                              bc_vals[0], bc_vals[1], bc_vals[2],
                              bc_vals[3], bc_vals[4])
            # BC type string: 3 chars padded with spaces to fill 8 bytes
            bc_bytes = bc_type.encode('ascii').ljust(8, b' ')[:8]
            f.write(buf + bc_bytes)

    import os
    fsize = os.path.getsize(filename)
    print(f"Wrote {filename} ({fsize} bytes, {nel} elements, {ncurve} curved sides, "
          f"{nbc_v} vel BCs, {nbc_t} temp BCs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a cubed-sphere spherical shell mesh for Nek5000 "
                    "and write it as a .re2 binary file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default Jackson benchmark mesh (1728 elements)
  python gen_cubed_sphere.py -o jackson.re2

  # Higher resolution (6144 elements)
  python gen_cubed_sphere.py --nphi 8 --nr 16 -o jackson.re2

  # Stress-free outer wall
  python gen_cubed_sphere.py --bc-v-outer stress-free -o jackson.re2
""")

    parser.add_argument('-o', '--output', default='sphere.re2',
                        help='Output .re2 filename (default: sphere.re2)')
    parser.add_argument('--nphi', type=int, default=6,
                        help='Angular elements per cube face side (default: 6)')
    parser.add_argument('--nr', type=int, default=8,
                        help='Number of radial shells (default: 8)')
    parser.add_argument('--eta', type=float, default=0.35,
                        help='Radius ratio ri/ro (default: 0.35)')
    parser.add_argument('--thickness', type=float, default=1.0,
                        help='Shell thickness d (default: 1.0)')
    parser.add_argument('--bc-v-inner', choices=['no-slip', 'stress-free'],
                        default='no-slip',
                        help='Velocity BC on inner wall (default: no-slip)')
    parser.add_argument('--bc-v-outer', choices=['no-slip', 'stress-free'],
                        default='no-slip',
                        help='Velocity BC on outer wall (default: no-slip)')
    parser.add_argument('--bc-t-inner', choices=['userbc', 'fixed'],
                        default='userbc',
                        help='Temperature BC on inner wall (default: userbc)')
    parser.add_argument('--bc-t-outer', choices=['userbc', 'fixed'],
                        default='userbc',
                        help='Temperature BC on outer wall (default: userbc)')
    parser.add_argument('--T-inner', type=float, default=0.0,
                        help='Fixed temperature at inner wall (if --bc-t-inner=fixed)')
    parser.add_argument('--T-outer', type=float, default=0.0,
                        help='Fixed temperature at outer wall (if --bc-t-outer=fixed)')

    args = parser.parse_args()

    # Map user-friendly BC names to Nek5000 .rea strings
    bc_v_map = {'no-slip': 'v  ', 'stress-free': 'SYM'}
    bc_t_map = {'userbc': 't  ', 'fixed': 'T  '}

    bc_v_inner = bc_v_map[args.bc_v_inner]
    bc_v_outer = bc_v_map[args.bc_v_outer]
    bc_t_inner = bc_t_map[args.bc_t_inner]
    bc_t_outer = bc_t_map[args.bc_t_outer]

    print(f"Cubed-sphere mesh generator for Nek5000")
    print(f"  nphi = {args.nphi} (angular elements per cube face side)")
    print(f"  nr   = {args.nr} (radial shells)")
    print(f"  eta  = {args.eta} (radius ratio)")
    print(f"  d    = {args.thickness} (shell thickness)")
    print(f"  Velocity BCs: inner={args.bc_v_inner}, outer={args.bc_v_outer}")
    print(f"  Temperature BCs: inner={args.bc_t_inner}, outer={args.bc_t_outer}")
    print()

    coords, curves, vel_bcs, temp_bcs, nel, nelv = build_cubed_sphere(
        nphi=args.nphi,
        nr=args.nr,
        eta=args.eta,
        d=args.thickness,
        bc_v_inner=bc_v_inner,
        bc_v_outer=bc_v_outer,
        bc_t_inner=bc_t_inner,
        bc_t_outer=bc_t_outer,
        T_inner=args.T_inner,
        T_outer=args.T_outer,
    )

    write_re2(args.output, coords, curves, vel_bcs, temp_bcs, nel, nelv)

    print()
    print(f"Next steps:")
    print(f"  1. Generate partition map:  genmap  (enter '{args.output.replace('.re2','')}' and tolerance 0.2)")
    print(f"  2. Compile solver:          ./makenek")
    print(f"  3. Run:                     mpirun -np N ./nek5000")


if __name__ == '__main__':
    main()
