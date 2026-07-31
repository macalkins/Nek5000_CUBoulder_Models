#!/usr/bin/env python3
"""
Generate a cubed-sphere spherical shell mesh in Nek5000 .re2 format.

Produces a full 3D spherical shell mesh using the cubed-sphere approach:
- 6 cube faces, each subdivided into N_angular x N_angular elements
- N_radial layers between r_inner and r_outer
- Curved face data ('s' type) for inner and outer spherical surfaces

Vertex ordering follows Nek5000 convention:
    4---3       8---7
    |   |       |   |     bottom (1-4), top (5-8)
    1---2       5---6

Face numbering (preprocessor notation):
    1: y=min, 2: x=max, 3: y=max, 4: x=min, 5: z=min (inner), 6: z=max (outer)

For a cubed-sphere shell, radial direction maps to the element's local
"z" direction, so face 5 = inner sphere, face 6 = outer sphere.

Usage:
    python generate_cubed_sphere_re2.py [--na 8] [--nr 8] [--ri 0.53846154] [--ro 1.53846154] [--output jackson.re2]
"""

import numpy as np
import struct
import argparse


def normalize(v):
    """Normalize a vector or array of vectors."""
    if v.ndim == 1:
        return v / np.linalg.norm(v)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cube_face_vertices():
    """
    Return the 6 cube faces as arrays of 4 corner unit vectors.

    Each face is defined by 4 corners on the unit cube, ordered so that
    the outward normal points away from the origin. The corners follow
    the right-hand rule (counterclockwise when viewed from outside).

    Returns list of 6 arrays, each shape (4, 3).
    """
    # Cube corners at distance 1/sqrt(3) from origin (unit sphere inscribed)
    # but we'll normalize later, so use unit cube corners.
    faces = []

    # +X face: corners at x=1, (y,z) in {-1,1}
    faces.append(np.array([
        [ 1, -1, -1],
        [ 1,  1, -1],
        [ 1,  1,  1],
        [ 1, -1,  1],
    ], dtype=float))

    # -X face
    faces.append(np.array([
        [-1,  1, -1],
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1,  1],
    ], dtype=float))

    # +Y face
    faces.append(np.array([
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1,  1,  1],
    ], dtype=float))

    # -Y face
    faces.append(np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [-1, -1,  1],
    ], dtype=float))

    # +Z face
    faces.append(np.array([
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=float))

    # -Z face
    faces.append(np.array([
        [-1,  1, -1],
        [ 1,  1, -1],
        [ 1, -1, -1],
        [-1, -1, -1],
    ], dtype=float))

    return faces


def equiangular_grid(face_corners, na):
    """
    Generate an equiangular grid of points on one cube face.

    Uses gnomonic (central) projection: points are uniformly spaced in
    angle on the unit sphere, giving better element uniformity than
    equidistant spacing on the cube face.

    Parameters:
        face_corners: (4,3) array of cube face corner positions
        na: number of elements along each edge

    Returns:
        grid: (na+1, na+1, 3) array of unit vectors on the sphere
    """
    # Bilinear interpolation in parameter space, then project to sphere
    grid = np.zeros((na + 1, na + 1, 3))

    for j in range(na + 1):
        for i in range(na + 1):
            # Use equiangular parameterization
            # Map [0, na] to [-pi/4, pi/4] for gnomonic projection
            alpha = -np.pi/4 + (np.pi/2) * i / na
            beta  = -np.pi/4 + (np.pi/2) * j / na

            # Gnomonic projection on the tangent plane
            xi  = np.tan(alpha)
            eta = np.tan(beta)

            # Bilinear interpolation on the cube face
            s = (xi + 1) / 2   # map [-1,1] to [0,1]
            t = (eta + 1) / 2

            p = ((1 - s) * (1 - t) * face_corners[0]
                 + s * (1 - t) * face_corners[1]
                 + s * t * face_corners[2]
                 + (1 - s) * t * face_corners[3])

            grid[j, i] = normalize(p)

    return grid


def solve_expansion_ratio(delta_bl, n_bl, dr_int):
    """Solve for geometric ratio g such that the last BL cell matches dr_int.

    For a zone with N cells, width W, geometric ratio g:
      h1 = W * (g - 1) / (g^N - 1)    (first/wall cell)
      hN = h1 * g^(N-1)               (last cell, must equal dr_int)

    Solve: f(g) = dr_int * (g^N - 1) / (g^(N-1) * (g - 1)) - delta_bl = 0
    """
    N = n_bl

    def f(g):
        return dr_int * (g**N - 1) / (g**(N-1) * (g - 1)) - delta_bl

    g_lo, g_hi = 1.001, 5.0
    f_lo = f(g_lo)
    f_hi = f(g_hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"Bisection failed: f({g_lo})={f_lo:.4e}, f({g_hi})={f_hi:.4e}. "
            f"Cannot match BL width={delta_bl:.4f} with {N} cells and dr_int={dr_int:.6f}"
        )

    for _ in range(100):
        g_mid = (g_lo + g_hi) / 2.0
        f_mid = f(g_mid)
        if abs(f_mid) < 1e-12:
            break
        if f_lo * f_mid < 0:
            g_hi = g_mid
        else:
            g_lo = g_mid
            f_lo = f_mid

    return g_mid


def generate_radial_layers(ri, ro, nr, Ek=1e-3, n_bl_cells=0, delta_factor=5.0,
                           radial_dist='uniform'):
    """
    Generate radial layer boundaries.

    radial_dist options:
      'uniform'   - equal spacing (default)
      'chebyshev' - Chebyshev distribution (clusters near both walls)
      'geometric' - 3-zone geometric grading (requires n_bl_cells > 0)

    Returns array of nr+1 radii from ri to ro.
    """
    if radial_dist == 'chebyshev':
        # Chebyshev nodes on [-1, 1], mapped to [ri, ro]
        # Clusters points near both boundaries — ideal for Ekman layers
        j = np.arange(nr + 1)
        nodes = -np.cos(np.pi * j / nr)  # Chebyshev nodes [-1, 1]
        radii = 0.5 * (ro + ri) + 0.5 * (ro - ri) * nodes
        dr_min = np.min(np.diff(radii))
        dr_max = np.max(np.diff(radii))
        print(f'  Chebyshev radial distribution:')
        print(f'    dr_min (wall) = {dr_min:.6f}, dr_max (center) = {dr_max:.6f}')
        print(f'    Ratio dr_max/dr_min = {dr_max/dr_min:.1f}')
        return radii

    if n_bl_cells == 0 or radial_dist == 'uniform':
        return np.linspace(ri, ro, nr + 1)

    L = ro - ri
    delta_bl = delta_factor * Ek**(1.0/3.0)

    if 2 * delta_bl >= L:
        print(f"WARNING: 2*delta_bl ({2*delta_bl:.4f}) >= gap ({L:.4f}). Using uniform.")
        return np.linspace(ri, ro, nr + 1)

    n_interior = nr - 2 * n_bl_cells
    if n_interior <= 0:
        raise ValueError(
            f"nr ({nr}) too small for BL grading: need > {2*n_bl_cells}"
        )

    interior_width = L - 2 * delta_bl
    dr_int = interior_width / n_interior

    # Check if grading is feasible (BL zone must be wider than n_bl uniform cells)
    if delta_bl <= n_bl_cells * dr_int:
        print(f'  BL zone ({delta_bl:.4f}) <= {n_bl_cells} uniform cells ({n_bl_cells*dr_int:.4f})')
        print(f'  Grading not needed at this Ek — using uniform spacing.')
        return np.linspace(ri, ro, nr + 1)

    g = solve_expansion_ratio(delta_bl, n_bl_cells, dr_int)
    R = g**(n_bl_cells - 1)
    dr_wall = dr_int / R

    print(f'  Radial grading:')
    print(f'    BL thickness: {delta_bl:.4f} ({n_bl_cells} cells/wall)')
    print(f'    Interior: {n_interior} cells, dr={dr_int:.6f}')
    print(f'    Wall cell: dr={dr_wall:.6f}, expansion ratio={R:.2f}, growth={g:.4f}')

    # Build radial node positions
    radii = [ri]

    # Inner BL: geometric expansion (smallest cell at wall)
    h = dr_wall
    for i in range(n_bl_cells):
        radii.append(radii[-1] + h)
        h *= g

    # Interior: uniform
    for i in range(n_interior):
        radii.append(radii[-1] + dr_int)

    # Outer BL: geometric compression (smallest cell at wall)
    h = dr_int
    for i in range(n_bl_cells):
        h /= g
        radii.append(radii[-1] + h)

    # Fix any floating-point drift
    radii[-1] = ro

    return np.array(radii)


def write_re2(filename, elements, curved_sides, bc_data, ndim=3, nelv=None):
    """
    Write a Nek5000 .re2 binary mesh file (version #v002, 64-bit).

    Binary layout (64-bit / #v002):
      Header:  80 bytes (ASCII) + 4 bytes (endian test float32) = 84 bytes
      Mesh:    per element: 25 float64 = 200 bytes
               (1 igroup_as_f64 + 8 xc + 8 yc + 8 zc)
      Curves:  1 float64 count, then per record: 8 float64 = 64 bytes
               (eg, face, curve(5), ccurve_as_padded_f64)
      BCs:     1 float64 count per field, then per record: 8 float64 = 64 bytes
               (eg, face, bc(5), cbc_as_padded_f64)

    Parameters:
        filename: output file path
        elements: list of dicts with keys 'x', 'y', 'z' (each length 8)
        curved_sides: list of (elem_1based, face_1based, curve_data[5], ccurve_char)
        bc_data: dict mapping field_name to list of (elem_1based, face_1based, bc_params[5], cbc_str)
        ndim: spatial dimension (3)
        nelv: number of fluid (velocity) elements. If None, defaults to nelt.
    """
    nelt = len(elements)
    if nelv is None:
        nelv = nelt

    with open(filename, 'wb') as f:
        # --- Header (84 bytes) ---
        # Format: '#v002 nelt ndim nelv'
        hdr = f'#v002{nelt:9d}{ndim:3d}{nelv:9d} cubed-sphere'
        hdr = hdr.ljust(80)[:80]
        f.write(hdr.encode('ascii'))
        f.write(struct.pack('<f', 6.54321))  # 4-byte endian test

        # --- Mesh data (25 float64 = 200 bytes per element) ---
        for elem in elements:
            # igroup as 1 float64 (reader does copyi4 from 2 int4 words)
            f.write(struct.pack('<d', 0.0))
            for val in elem['x']:
                f.write(struct.pack('<d', val))
            for val in elem['y']:
                f.write(struct.pack('<d', val))
            for val in elem['z']:
                f.write(struct.pack('<d', val))

        # --- Curved sides (8 float64 = 64 bytes per record) ---
        ncurve = len(curved_sides)
        # Count stored as 1 float64
        f.write(struct.pack('<d', float(ncurve)))

        for (eg, iface, curve_vals, ccurve) in curved_sides:
            # 8 float64 words: eg, face, curve(1:5), ccurve_char_in_f64
            f.write(struct.pack('<d', float(eg)))
            f.write(struct.pack('<d', float(iface)))
            for k in range(5):
                f.write(struct.pack('<d', curve_vals[k]))
            # Last word: ccurve character stored in first byte of a float64
            char_word = bytearray(8)
            char_word[0:1] = ccurve.encode('ascii')
            f.write(char_word)

        # --- Boundary conditions (8 float64 = 64 bytes per record) ---
        for field_name, bcs in bc_data.items():
            nbc = len(bcs)
            f.write(struct.pack('<d', float(nbc)))

            for (eg, iface, bc_params, cbc) in bcs:
                f.write(struct.pack('<d', float(eg)))
                f.write(struct.pack('<d', float(iface)))
                for k in range(5):
                    f.write(struct.pack('<d', float(bc_params[k])))
                # Last word: cbc string (3 chars) stored in first bytes of a float64
                char_word = bytearray(8)
                cbc_padded = cbc.ljust(3)[:3]
                char_word[0:3] = cbc_padded.encode('ascii')
                f.write(char_word)

    print(f'Wrote {filename}: nelt={nelt}, nelv={nelv}, {ncurve} curved sides')


def build_shell_elements(face_grids, na, radii, nr, elem_id_start=0,
                         curved=True, keep_set=None):
    """
    Build elements and curved-side data for a spherical shell.

    Parameters:
        face_grids: list of 6 grids from equiangular_grid()
        na: angular resolution per cube-face edge
        radii: array of nr+1 radial node positions
        nr: number of radial layers
        elem_id_start: starting element ID (0-based, will be incremented to 1-based)
        curved: if True, add spherical curvature data. Set False for mantle
                elements to avoid gen_fast degenerate geometry issues.
        keep_set: if not None, a set of (iface_cube, j, i) tuples specifying
                  which angular elements to include. If None, include all.

    Returns:
        elements: list of element dicts
        curved_sides: list of curved-side tuples (1-based elem IDs)
        elem_id: final element ID (1-based)
        elem_map: dict mapping (iface_cube, j, i, k) -> 1-based element ID
    """
    elements = []
    curved_sides = []
    elem_id = elem_id_start
    elem_map = {}

    for iface_cube, grid in enumerate(face_grids):
        for j in range(na):
            for i in range(na):
                if keep_set is not None and (iface_cube, j, i) not in keep_set:
                    continue

                corners = [
                    grid[j,   i],
                    grid[j,   i+1],
                    grid[j+1, i+1],
                    grid[j+1, i],
                ]

                for k in range(nr):
                    r_inner = radii[k]
                    r_outer = radii[k + 1]
                    elem_id += 1

                    vx = np.zeros(8)
                    vy = np.zeros(8)
                    vz = np.zeros(8)

                    for iv, corner in enumerate(corners):
                        vx[iv] = corner[0] * r_inner
                        vy[iv] = corner[1] * r_inner
                        vz[iv] = corner[2] * r_inner
                        vx[iv + 4] = corner[0] * r_outer
                        vy[iv + 4] = corner[1] * r_outer
                        vz[iv + 4] = corner[2] * r_outer

                    elements.append({
                        'x': vx.tolist(),
                        'y': vy.tolist(),
                        'z': vz.tolist(),
                    })

                    elem_map[(iface_cube, j, i, k)] = elem_id

                    # Curved sides for spherical faces
                    if curved:
                        if k == 0:
                            curved_sides.append((
                                elem_id, 5,
                                [0.0, 0.0, 0.0, r_inner, 0.0],
                                's'
                            ))
                        if k == nr - 1:
                            curved_sides.append((
                                elem_id, 6,
                                [0.0, 0.0, 0.0, r_outer, 0.0],
                                's'
                            ))

    return elements, curved_sides, elem_id, elem_map


def compute_wedge_180(face_grids, na):
    """
    Compute the keep set and periodic BC info for a 180-degree wedge (y >= 0).

    For each cube face and each (j, i) element, compute the centroid on the
    unit sphere. Keep elements where centroid y >= 0.

    For elements on the cut plane (y=0 boundary), identify which element face
    is on the cut and find the periodic partner under 180-degree rotation
    around z: (x,y,z) -> (-x,-y,z).

    Parameters:
        face_grids: list of 6 grids from equiangular_grid()
        na: number of angular elements per cube-face edge (must be even)

    Returns:
        keep_set: set of (iface_cube, j, i) tuples to keep
        cut_faces: list of (iface_cube, j, i, local_face, partner_face_idx, pj, pi, partner_local_face)
                   where local_face is the Nek5000 face number (1-4) on the cut plane,
                   and (partner_face_idx, pj, pi) is the matching element under rotation,
                   partner_local_face is the face number on the partner element.
    """
    if na % 2 != 0:
        raise ValueError(f'--na must be even for --wedge 180 (got {na})')

    # Compute centroids for all (face, j, i) elements
    centroids = {}  # (iface, j, i) -> centroid unit vector
    for iface, grid in enumerate(face_grids):
        for j in range(na):
            for i in range(na):
                # Centroid is average of 4 corner unit vectors, re-normalized
                c = (grid[j, i] + grid[j, i+1] + grid[j+1, i+1] + grid[j+1, i]) / 4.0
                c = normalize(c)
                centroids[(iface, j, i)] = c

    # Keep elements with centroid y >= 0
    # Use a small negative tolerance to include elements exactly on the plane
    tol = -1e-12
    keep_set = set()
    for key, c in centroids.items():
        if c[1] >= tol:
            keep_set.add(key)

    # Build full neighbor map across cube faces
    neighbor_map = build_angular_neighbors(face_grids, na)

    # Find cut-plane faces: for each kept element, if a neighbor is not in keep_set,
    # that face is on the y=0 cut plane.
    # Collect cut-plane face info with their midpoint on the unit sphere.
    #
    # Face vertex indices (corners of the angular element):
    #   face 1 (j- side): corners[0], corners[1]  -> grid[j, i], grid[j, i+1]
    #   face 2 (i+ side): corners[1], corners[2]  -> grid[j, i+1], grid[j+1, i+1]
    #   face 3 (j+ side): corners[2], corners[3]  -> grid[j+1, i+1], grid[j+1, i]
    #   face 4 (i- side): corners[3], corners[0]  -> grid[j+1, i], grid[j, i]

    face_vertex_offsets = {
        1: [(0, 0), (0, 1)],    # grid[j, i], grid[j, i+1]
        2: [(0, 1), (1, 1)],    # grid[j, i+1], grid[j+1, i+1]
        3: [(1, 1), (1, 0)],    # grid[j+1, i+1], grid[j+1, i]
        4: [(1, 0), (0, 0)],    # grid[j+1, i], grid[j, i]
    }

    # Collect all cut faces with their midpoints
    cut_face_list = []  # (iface, j, i, local_face, face_midpoint)
    for (iface, j, i) in keep_set:
        grid = face_grids[iface]
        for local_face, (di, dj) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)], start=1):
            ni, nj = i + di, j + dj
            if 0 <= ni < na and 0 <= nj < na:
                neighbor_key = (iface, nj, ni)
            else:
                neighbor_key = neighbor_map.get((iface, j, i, local_face))

            if neighbor_key is None:
                continue

            if neighbor_key not in keep_set:
                # Compute midpoint of this face on the unit sphere
                offsets = face_vertex_offsets[local_face]
                v0 = grid[j + offsets[0][0], i + offsets[0][1]]
                v1 = grid[j + offsets[1][0], i + offsets[1][1]]
                mid = normalize((v0 + v1) / 2.0)
                cut_face_list.append((iface, j, i, local_face, mid))

    # Match periodic partners: for each cut face at midpoint (x, ~0, z),
    # the partner is the face at midpoint (-x, ~0, z) under 180° rotation.
    # Build spatial index of face midpoints for matching.
    if len(cut_face_list) == 0:
        return keep_set, []

    mids = np.array([cf[4] for cf in cut_face_list])
    n_cut = len(cut_face_list)

    # For each face, find its partner by rotating midpoint 180° around z
    matched = [False] * n_cut
    cut_faces = []

    for idx in range(n_cut):
        if matched[idx]:
            continue
        iface, j, i, lf, mid = cut_face_list[idx]
        mid_rot = np.array([-mid[0], -mid[1], mid[2]])

        # Find closest match
        dists = np.linalg.norm(mids - mid_rot, axis=1)
        dists[idx] = 999.0  # exclude self
        for jj in range(n_cut):
            if matched[jj]:
                dists[jj] = 999.0
        best = np.argmin(dists)

        if dists[best] > 0.05:
            raise RuntimeError(
                f'No periodic partner for face ({iface},{j},{i}) lf={lf}, '
                f'mid={mid}, best dist={dists[best]:.6f}')

        pface, pj, pi, plf, _ = cut_face_list[best]

        cut_faces.append((iface, j, i, lf, pface, pj, pi, plf))
        cut_faces.append((pface, pj, pi, plf, iface, j, i, lf))
        matched[idx] = True
        matched[best] = True

    if not all(matched):
        n_unmatched = sum(1 for m in matched if not m)
        raise RuntimeError(f'{n_unmatched} cut-plane faces unmatched')

    return keep_set, cut_faces


def build_angular_neighbors(face_grids, na):
    """
    Build cross-face neighbor connectivity for cube-face edges.

    For elements on the edge of a cube face, find the matching element on the
    adjacent cube face by matching vertex positions.

    Returns:
        neighbor_map: dict mapping (iface, j, i, local_face) -> (neighbor_face, nj, ni)
                      Only populated for cross-face edges.
    """
    # Build lookup from vertex position (rounded) to (face, j, i) list
    # Each element's 4 corners correspond to grid vertices.
    # Two elements share an edge if they share 2 vertices.

    # For each cube face edge element and each of its 4 faces, if the face
    # is on the cube-face boundary, find the matching element on another cube face.

    # Strategy: for each face's boundary edges, store the edge midpoints and match.
    # Simpler: store all grid edge midpoints and match across faces.

    # Actually, simplest approach: for each element on a cube-face edge,
    # compute the midpoint of the boundary face (2 corners) and find the
    # matching element on another cube face whose corresponding face midpoint matches.

    neighbor_map = {}

    # For each cube face boundary, collect the edge info
    # face 1 (local_face=1): j=0 side, edge vertices grid[0, i] and grid[0, i+1]
    # face 3 (local_face=3): j=na side, edge vertices grid[na, i+1] and grid[na, i]
    # face 4 (local_face=4): i=0 side, edge vertices grid[j+1, 0] and grid[j, 0]
    # face 2 (local_face=2): i=na side, edge vertices grid[j, na] and grid[j+1, na]

    # Collect all boundary face midpoints
    boundary_faces = {}  # midpoint_key -> (iface, j, i, local_face)

    def midpoint_key(p1, p2):
        mid = (p1 + p2) / 2.0
        return tuple(np.round(mid, decimals=10))

    for iface, grid in enumerate(face_grids):
        for idx in range(na):
            # j=0 boundary (local_face=1): element (j=0, i=idx)
            mid = midpoint_key(grid[0, idx], grid[0, idx+1])
            boundary_faces.setdefault(mid, []).append((iface, 0, idx, 1))

            # j=na-1 boundary (local_face=3): element (j=na-1, i=idx)
            mid = midpoint_key(grid[na, idx+1], grid[na, idx])
            boundary_faces.setdefault(mid, []).append((iface, na-1, idx, 3))

            # i=0 boundary (local_face=4): element (j=idx, i=0)
            mid = midpoint_key(grid[idx+1, 0], grid[idx, 0])
            boundary_faces.setdefault(mid, []).append((iface, idx, 0, 4))

            # i=na-1 boundary (local_face=2): element (j=idx, i=na-1)
            mid = midpoint_key(grid[idx, na], grid[idx+1, na])
            boundary_faces.setdefault(mid, []).append((iface, idx, na-1, 2))

    # Match pairs
    for mid, entries in boundary_faces.items():
        if len(entries) == 2:
            a, b = entries
            iface_a, ja, ia, lf_a = a
            iface_b, jb, ib, lf_b = b
            neighbor_map[(iface_a, ja, ia, lf_a)] = (iface_b, jb, ib)
            neighbor_map[(iface_b, jb, ib, lf_b)] = (iface_a, ja, ia)
        elif len(entries) > 2:
            # Corner point — skip (not a unique edge match)
            pass

    return neighbor_map


def main():
    parser = argparse.ArgumentParser(description='Generate cubed-sphere .re2 mesh')
    parser.add_argument('--na', type=int, default=8,
                        help='Number of elements per cube-face edge (default: 8)')
    parser.add_argument('--nr', type=int, default=8,
                        help='Number of radial layers (default: 8)')
    parser.add_argument('--ri', type=float, default=7.0/13.0,
                        help='Inner radius (default: 7/13)')
    parser.add_argument('--ro', type=float, default=20.0/13.0,
                        help='Outer radius (default: 20/13)')
    parser.add_argument('--Ek', type=float, default=1e-3,
                        help='Ekman number for BL grading (default: 1e-3)')
    parser.add_argument('--n_bl', type=int, default=0,
                        help='Number of BL cells per wall (0=uniform, default: 0)')
    parser.add_argument('--delta_factor', type=float, default=5.0,
                        help='BL zone extends delta_factor * Ek^(1/3) from wall (default: 5)')
    parser.add_argument('--output', type=str, default='jackson.re2',
                        help='Output filename (default: jackson.re2)')
    parser.add_argument('--radial_dist', type=str, default='uniform',
                        choices=['uniform', 'chebyshev', 'geometric'],
                        help='Radial distribution (default: uniform)')
    parser.add_argument('--no-mhd', action='store_true',
                        help='Omit magnetic field BCs (hydro only, 2 BC fields)')
    parser.add_argument('--nr_mantle', type=int, default=0,
                        help='Number of radial layers in insulating mantle (default: 0, no mantle)')
    parser.add_argument('--r_outer', type=float, default=None,
                        help='Outer radius of mantle (default: same as ro)')
    parser.add_argument('--wedge', type=int, default=0, choices=[0, 180],
                        help='Wedge angle in degrees (0=full sphere, 180=half sphere y>=0)')
    args = parser.parse_args()

    na = args.na
    nr = args.nr
    ri = args.ri
    ro = args.ro
    nr_mantle = args.nr_mantle
    r_outer_mantle = args.r_outer if args.r_outer is not None else ro

    # Validate mantle arguments
    if nr_mantle > 0:
        if r_outer_mantle <= ro:
            raise ValueError(f'--r_outer ({r_outer_mantle}) must be > --ro ({ro}) when using mantle')
        if args.no_mhd:
            raise ValueError('Mantle region requires MHD (incompatible with --no-mhd)')

    # Validate wedge arguments
    wedge = args.wedge
    if wedge == 180 and nr_mantle > 0:
        raise ValueError('--wedge 180 with mantle is not yet supported')

    # Generate the 6 cube face grids
    faces = cube_face_vertices()
    face_grids = []
    for fc in faces:
        grid = equiangular_grid(fc, na)
        face_grids.append(grid)

    # Compute keep set for wedge mode
    keep_set = None
    cut_faces = []
    if wedge == 180:
        keep_set, cut_faces = compute_wedge_180(face_grids, na)
        nel_per_layer = len(keep_set)
        print(f'Generating cubed-sphere mesh (180-degree wedge, y >= 0):')
        print(f'  Angular elements per layer: {nel_per_layer} (of {6*na*na} full sphere)')
    else:
        nel_per_layer = 6 * na * na
        print(f'Generating cubed-sphere mesh:')

    nelv = nel_per_layer * nr  # fluid (core) elements
    nelt = nel_per_layer * (nr + nr_mantle)  # total elements

    print(f'  Angular resolution: {na} x {na} per face')
    print(f'  Radial layers (core): {nr}')
    print(f'  ri = {ri:.8f}, ro = {ro:.8f}')
    if nr_mantle > 0:
        print(f'  Radial layers (mantle): {nr_mantle}')
        print(f'  r_outer (mantle) = {r_outer_mantle:.8f}')
        print(f'  Core elements (nelv): {nelv}')
        print(f'  Mantle elements: {nelt - nelv}')
        print(f'  Total elements (nelt): {nelt}')
    else:
        print(f'  Total elements: {nelt}')

    # Radial layers for core (with optional BL grading)
    radii_core = generate_radial_layers(ri, ro, nr, Ek=args.Ek,
                                        n_bl_cells=args.n_bl,
                                        delta_factor=args.delta_factor,
                                        radial_dist=args.radial_dist)

    # Build core elements
    core_elements, core_curved, elem_id, core_map = build_shell_elements(
        face_grids, na, radii_core, nr, elem_id_start=0, keep_set=keep_set)

    assert elem_id == nelv, f'Core element count mismatch: {elem_id} != {nelv}'

    # Build mantle elements if requested
    mantle_elements = []
    mantle_curved = []
    mantle_map = {}
    if nr_mantle > 0:
        radii_mantle = np.linspace(ro, r_outer_mantle, nr_mantle + 1)
        mantle_elements, mantle_curved, elem_id, mantle_map = build_shell_elements(
            face_grids, na, radii_mantle, nr_mantle, elem_id_start=nelv,
            curved=False, keep_set=keep_set)
        assert elem_id == nelt, f'Total element count mismatch: {elem_id} != {nelt}'

    # Combine elements and curved sides
    elements = core_elements + mantle_elements
    curved_sides = core_curved + mantle_curved

    # --- Boundary conditions ---
    bc_vel = []   # velocity BCs (field 1, only on fluid elements 1..nelv)
    bc_temp = []  # temperature BCs (field 2)
    bc_mag = []   # magnetic field BCs (field 3)

    if nr_mantle == 0:
        # No mantle
        # Velocity: W at ri (face 5 of k=0) and W at ro (face 6 of k=nr-1)
        # Temperature: t at ri and t at ro
        # Magnetic: ON at ri and ON at ro (pseudo-vacuum)
        iter_set = keep_set if keep_set is not None else (
            (f, j, i) for f in range(6) for j in range(na) for i in range(na))
        for (iface_cube, j, i) in iter_set:
            eid_inner = core_map[(iface_cube, j, i, 0)]
            eid_outer = core_map[(iface_cube, j, i, nr - 1)]
            bc_vel.append((eid_inner, 5, [0,0,0,0,0], 'W  '))
            bc_temp.append((eid_inner, 5, [0,0,0,0,0], 't  '))
            if not args.no_mhd:
                bc_mag.append((eid_inner, 5, [0,0,0,0,0], 'ON '))
            bc_vel.append((eid_outer, 6, [0,0,0,0,0], 'W  '))
            bc_temp.append((eid_outer, 6, [0,0,0,0,0], 't  '))
            if not args.no_mhd:
                bc_mag.append((eid_outer, 6, [0,0,0,0,0], 'ON '))
    else:
        # With mantle
        iter_set = keep_set if keep_set is not None else (
            (f, j, i) for f in range(6) for j in range(na) for i in range(na))
        for (iface_cube, j, i) in iter_set:
            eid_inner_core = core_map[(iface_cube, j, i, 0)]
            eid_outer_core = core_map[(iface_cube, j, i, nr - 1)]
            eid_outer_mantle = mantle_map[(iface_cube, j, i, nr_mantle - 1)]

            bc_vel.append((eid_inner_core, 5, [0,0,0,0,0], 'W  '))
            bc_vel.append((eid_outer_core, 6, [0,0,0,0,0], 'W  '))

            bc_temp.append((eid_inner_core, 5, [0,0,0,0,0], 't  '))
            bc_temp.append((eid_outer_core, 6, [0,0,0,0,0], 't  '))
            bc_temp.append((eid_outer_mantle, 6, [0,0,0,0,0], 'I  '))

            bc_mag.append((eid_inner_core, 5, [0,0,0,0,0], 'W  '))
            bc_mag.append((eid_outer_mantle, 6, [0,0,0,0,0], 'W  '))

    # --- Periodic BCs for wedge mode ---
    n_periodic_faces = 0
    if wedge == 180 and len(cut_faces) > 0:
        print(f'  Cut-plane element-faces: {len(cut_faces)}')
        for (iface, j, i, lf, pface, pj, pi, plf) in cut_faces:
            for k in range(nr):
                eid = core_map[(iface, j, i, k)]
                partner_eid = core_map[(pface, pj, pi, k)]
                bc_vel.append((eid, lf, [partner_eid, plf, 0, 0, 0], 'P  '))
                bc_temp.append((eid, lf, [partner_eid, plf, 0, 0, 0], 'P  '))
                if not args.no_mhd:
                    bc_mag.append((eid, lf, [partner_eid, plf, 0, 0, 0], 'P  '))
                n_periodic_faces += 1

    print(f'  Generated {len(elements)} elements')
    print(f'  Curved sides: {len(curved_sides)}')
    if args.no_mhd:
        print(f'  BCs: {len(bc_vel)} (vel), {len(bc_temp)} (temp)')
    else:
        print(f'  BCs: {len(bc_vel)} (vel), {len(bc_temp)} (temp), {len(bc_mag)} (mag)')
    if n_periodic_faces > 0:
        print(f'  Periodic face pairs: {n_periodic_faces}')

    # Write .re2 file
    bc_data = {'velocity': bc_vel, 'temperature': bc_temp}
    if not args.no_mhd:
        bc_data['magnetic'] = bc_mag

    # CHT-style: nelv = core only, nelt = core + mantle
    write_re2(args.output, elements, curved_sides, bc_data,
              nelv=nelv if nr_mantle > 0 else None)

    # Print SIZE file recommendations
    lx1 = 8  # polynomial order + 1 (typical)
    print(f'\nRecommended SIZE parameters:')
    print(f'  parameter (ldim={3})')
    print(f'  parameter (lx1={lx1})')
    print(f'  parameter (lxd={int(3*lx1/2)})')
    print(f'  parameter (lelg={nelt})')
    print(f'  parameter (lelt={max(1, nelt // 4 + 50)})')  # rough estimate
    if nr_mantle > 0:
        print(f'  ! nelv={nelv} (fluid), nelt={nelt} (total)')
        print(f'  ! Mantle elements are solid (no velocity solve)')


if __name__ == '__main__':
    main()
