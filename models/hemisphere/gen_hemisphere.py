#!/usr/bin/env python3
"""
Generate a cubed-sphere HEMISPHERE mesh with periodic BCs in Nek5000 format.

Produces both .re2 (mesh) and .ma2 (connectivity) files, bypassing genmap
which cannot handle the 180-degree rotational periodicity.

The hemisphere covers y >= 0 (180 degrees in longitude). The cut plane at
y = 0 has periodic BCs under 180-degree rotation: (x,y,z) -> (-x,-y,z).

The .ma2 file is written directly with correct periodic vertex identification,
so genmap is NOT needed.

Usage:
    python gen_hemisphere.py [--na 8] [--nr 8] [--ri 0.5385] [--ro 1.5385] [--output hemisphere]
"""

import numpy as np
import struct
import argparse
import math


# ============================================================================
# Cubed-sphere mesh generation (reused from gen_cubed_sphere.py)
# ============================================================================

def normalize(v):
    if v.ndim == 1:
        return v / np.linalg.norm(v)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cube_face_vertices():
    faces = []
    faces.append(np.array([[ 1,-1,-1],[ 1, 1,-1],[ 1, 1, 1],[ 1,-1, 1]], dtype=float))
    faces.append(np.array([[-1, 1,-1],[-1,-1,-1],[-1,-1, 1],[-1, 1, 1]], dtype=float))
    faces.append(np.array([[ 1, 1,-1],[-1, 1,-1],[-1, 1, 1],[ 1, 1, 1]], dtype=float))
    faces.append(np.array([[-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1],[-1,-1, 1]], dtype=float))
    faces.append(np.array([[-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]], dtype=float))
    faces.append(np.array([[-1, 1,-1],[ 1, 1,-1],[ 1,-1,-1],[-1,-1,-1]], dtype=float))
    return faces


def equiangular_grid(face_corners, na):
    grid = np.zeros((na + 1, na + 1, 3))
    for j in range(na + 1):
        for i in range(na + 1):
            alpha = -np.pi/4 + (np.pi/2) * i / na
            beta  = -np.pi/4 + (np.pi/2) * j / na
            xi  = np.tan(alpha)
            eta = np.tan(beta)
            s = (xi + 1) / 2
            t = (eta + 1) / 2
            p = ((1-s)*(1-t)*face_corners[0] + s*(1-t)*face_corners[1]
                 + s*t*face_corners[2] + (1-s)*t*face_corners[3])
            grid[j, i] = normalize(p)
    return grid


def solve_expansion_ratio(delta_bl, n_bl, dr_int):
    N = n_bl
    def f(g):
        return dr_int * (g**N - 1) / (g**(N-1) * (g - 1)) - delta_bl
    g_lo, g_hi = 1.001, 5.0
    f_lo, f_hi = f(g_lo), f(g_hi)
    if f_lo * f_hi > 0:
        raise ValueError(f'Bisection failed for BL grading')
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
    if radial_dist == 'chebyshev':
        j = np.arange(nr + 1)
        nodes = -np.cos(np.pi * j / nr)
        return 0.5 * (ro + ri) + 0.5 * (ro - ri) * nodes

    if n_bl_cells == 0 or radial_dist == 'uniform':
        return np.linspace(ri, ro, nr + 1)

    L = ro - ri
    delta_bl = delta_factor * Ek**(1.0/3.0)
    if 2 * delta_bl >= L:
        return np.linspace(ri, ro, nr + 1)
    n_interior = nr - 2 * n_bl_cells
    if n_interior <= 0:
        raise ValueError(f'nr ({nr}) too small for BL grading')
    interior_width = L - 2 * delta_bl
    dr_int = interior_width / n_interior
    if delta_bl <= n_bl_cells * dr_int:
        return np.linspace(ri, ro, nr + 1)

    g = solve_expansion_ratio(delta_bl, n_bl_cells, dr_int)
    dr_wall = dr_int / g**(n_bl_cells - 1)
    radii = [ri]
    h = dr_wall
    for _ in range(n_bl_cells):
        radii.append(radii[-1] + h)
        h *= g
    for _ in range(n_interior):
        radii.append(radii[-1] + dr_int)
    h = dr_int
    for _ in range(n_bl_cells):
        h /= g
        radii.append(radii[-1] + h)
    radii[-1] = ro
    return np.array(radii)


# ============================================================================
# Boundary topography
# ============================================================================

def topo_displacement(direction, topo_params):
    """
    Compute radial topography displacement for a point on the unit sphere.

    The topography is a 2D super-Gaussian producing a bump with
    approximately rectangular cross-section on the sphere:

        h = A * exp(-(delta_ns / sigma_theta)^(2n))
              * exp(-(delta_ew / sigma_phi)^(2n))

    where delta_ns and delta_ew are angular displacements measured in a
    rotated coordinate system whose equator passes through the bump
    center (theta0, phi0).  This avoids the latitude-dependent metric
    distortion of (theta, phi) coordinates: near the rotated equator
    the metric is locally uniform, so sigma_theta and sigma_phi set
    the physical half-widths (in radians of great-circle arc) and
    the bump footprint is a true rectangle on the sphere.

    Parameters:
        direction: (3,) unit vector [x, y, z] on the unit sphere
        topo_params: dict with keys:
            'amplitude'   : float, radial displacement (same units as ri, ro)
            'theta0'      : float, center colatitude (radians)
            'phi0'        : float, center longitude (radians)
            'sigma_theta' : float, N-S half-width in arc-length (radians)
            'sigma_phi'   : float, E-W half-width in arc-length (radians)
            'n'           : int, super-Gaussian order (1=Gaussian, 4+=wall-like)

    Returns:
        h: float, radial displacement at this direction
    """
    if topo_params is None:
        return 0.0

    A = topo_params['amplitude']
    theta0 = topo_params['theta0']
    phi0 = topo_params['phi0']
    sig_th = topo_params['sigma_theta']
    sig_ph = topo_params['sigma_phi']
    n = topo_params['n']

    # --- Rotated coordinate system ---
    # Bump center on the unit sphere
    st0, ct0 = np.sin(theta0), np.cos(theta0)
    sp0, cp0 = np.sin(phi0), np.cos(phi0)
    # p0 = bump center direction
    p0 = np.array([st0 * cp0, st0 * sp0, ct0])
    # Local basis at bump center:
    #   e_theta = south (increasing colatitude)
    #   e_phi   = east  (increasing longitude)
    e_theta = np.array([ct0 * cp0, ct0 * sp0, -st0])
    e_phi   = np.array([-sp0, cp0, 0.0])

    # Rotation matrix R maps the original frame so that:
    #   p0      -> x-axis  (rotated equator at phi'=0)
    #   e_phi   -> y-axis  (east-west in rotated frame)
    #  -e_theta -> z-axis  (north in rotated frame)
    # Then the bump center is at (theta'=pi/2, phi'=0) in the rotated frame.

    # Rotated coordinates of the query point
    p = np.asarray(direction, dtype=float)
    px = np.dot(p0, p)       # radial at bump center
    py = np.dot(e_phi, p)    # east-west
    pz = np.dot(-e_theta, p) # north-south (toward pole)

    # Spherical coordinates in rotated frame
    theta_rot = np.arccos(np.clip(pz, -1.0, 1.0))
    phi_rot   = np.arctan2(py, px)

    # Displacements from bump center (theta'=pi/2, phi'=0)
    delta_ns = theta_rot - 0.5 * np.pi   # N-S displacement
    delta_ew = phi_rot                     # E-W displacement (already centered)

    h = A * np.exp(-(delta_ns / sig_th)**(2*n)) * np.exp(-(delta_ew / sig_ph)**(2*n))
    return h


# ============================================================================
# Hemisphere element selection
# ============================================================================

def compute_keep_set(face_grids, na):
    """
    Select angular elements for the y >= 0 hemisphere.

    Returns:
        keep_set: set of (iface_cube, j, i) tuples to keep
    """
    tol = -1e-12
    keep_set = set()
    for iface, grid in enumerate(face_grids):
        for j in range(na):
            for i in range(na):
                c = (grid[j,i] + grid[j,i+1] + grid[j+1,i+1] + grid[j+1,i]) / 4.0
                c = normalize(c)
                if c[1] >= tol:
                    keep_set.add((iface, j, i))
    return keep_set


# ============================================================================
# Build elements
# ============================================================================

def build_elements(face_grids, na, radii, nr, keep_set=None, topo_params=None):
    """
    Build hex element vertex coordinates for the spherical shell.

    If topo_params is provided, the outer boundary is deformed by a
    super-Gaussian bump and interior layers are linearly interpolated
    (zero displacement at ri, full displacement at ro).

    Returns:
        elements: list of dicts {'x': [8], 'y': [8], 'z': [8]}
        curved_sides: list of (elem_1based, face_1based, [cx,cy,cz,r,0], 's')
        elem_map: (iface, j, i, k) -> 1-based element ID
    """
    elements = []
    curved_sides = []
    elem_id = 0
    elem_map = {}

    ri = radii[0]
    ro = radii[-1]
    shell_gap = ro - ri

    for iface, grid in enumerate(face_grids):
        for j in range(na):
            for i in range(na):
                if keep_set is not None and (iface, j, i) not in keep_set:
                    continue

                corners = [grid[j,i], grid[j,i+1], grid[j+1,i+1], grid[j+1,i]]

                # Pre-compute topography displacement for each angular corner
                if topo_params is not None:
                    corner_h = [topo_displacement(c, topo_params) for c in corners]
                else:
                    corner_h = [0.0, 0.0, 0.0, 0.0]

                for k in range(nr):
                    r_in = radii[k]
                    r_out = radii[k + 1]
                    elem_id += 1

                    # Linear interpolation factor: 0 at inner boundary, 1 at outer
                    frac_in  = (r_in  - ri) / shell_gap
                    frac_out = (r_out - ri) / shell_gap

                    vx, vy, vz = np.zeros(8), np.zeros(8), np.zeros(8)
                    for iv, corner in enumerate(corners):
                        h = corner_h[iv]
                        r_in_def  = r_in  + h * frac_in
                        r_out_def = r_out + h * frac_out
                        vx[iv]     = corner[0] * r_in_def
                        vy[iv]     = corner[1] * r_in_def
                        vz[iv]     = corner[2] * r_in_def
                        vx[iv + 4] = corner[0] * r_out_def
                        vy[iv + 4] = corner[1] * r_out_def
                        vz[iv + 4] = corner[2] * r_out_def

                    elements.append({
                        'x': vx.tolist(),
                        'y': vy.tolist(),
                        'z': vz.tolist(),
                    })
                    elem_map[(iface, j, i, k)] = elem_id

                    # Curved sides for inner/outer spherical surfaces
                    # Inner boundary is always undeformed (spherical)
                    if k == 0:
                        curved_sides.append((elem_id, 5, [0,0,0,r_in,0], 's'))
                    # Outer boundary: use mean deformed radius for this element
                    # (exact for uniform topography across the face; approximate
                    #  when topography varies, but OK for small elements)
                    if k == nr - 1:
                        mean_h = np.mean(corner_h)
                        r_out_curved = r_out + mean_h * frac_out
                        curved_sides.append((elem_id, 6, [0,0,0,r_out_curved,0], 's'))

    return elements, curved_sides, elem_map


# ============================================================================
# Boundary conditions
# ============================================================================

def build_angular_neighbors(face_grids, na):
    """Build cross-face neighbor map for cube-face edges."""
    neighbor_map = {}

    def midpoint_key(p1, p2):
        mid = (p1 + p2) / 2.0
        return tuple(np.round(mid, decimals=10))

    for iface, grid in enumerate(face_grids):
        for idx in range(na):
            mid = midpoint_key(grid[0, idx], grid[0, idx+1])
            neighbor_map.setdefault(mid, []).append((iface, 0, idx, 1))
            mid = midpoint_key(grid[na, idx+1], grid[na, idx])
            neighbor_map.setdefault(mid, []).append((iface, na-1, idx, 3))
            mid = midpoint_key(grid[idx+1, 0], grid[idx, 0])
            neighbor_map.setdefault(mid, []).append((iface, idx, 0, 4))
            mid = midpoint_key(grid[idx, na], grid[idx+1, na])
            neighbor_map.setdefault(mid, []).append((iface, idx, na-1, 2))

    result = {}
    for mid, entries in neighbor_map.items():
        if len(entries) == 2:
            a, b = entries
            result[(a[0], a[1], a[2], a[3])] = (b[0], b[1], b[2])
            result[(b[0], b[1], b[2], b[3])] = (a[0], a[1], a[2])
    return result


def find_cut_plane_faces(face_grids, na, keep_set, elem_map, nr):
    """
    Find element faces on the y=0 cut plane and their periodic partners.

    Returns list of (elem_id, local_face, partner_elem_id, partner_face)
    for each face on the cut plane, across all radial layers.
    """
    neighbor_map = build_angular_neighbors(face_grids, na)

    # Find angular elements on the cut plane boundary
    cut_angular = []  # (iface, j, i, local_face, partner_iface, pj, pi, partner_local_face)

    face_vertex_offsets = {
        1: [(0, 0), (0, 1)],
        2: [(0, 1), (1, 1)],
        3: [(1, 1), (1, 0)],
        4: [(1, 0), (0, 0)],
    }

    # For each kept element, check if any angular neighbor is NOT in keep_set
    for (iface, j, i) in keep_set:
        for local_face, (di, dj) in enumerate([(0,-1),(1,0),(0,1),(-1,0)], start=1):
            ni, nj = i + di, j + dj
            if 0 <= ni < na and 0 <= nj < na:
                neighbor_key = (iface, nj, ni)
            else:
                nk = neighbor_map.get((iface, j, i, local_face))
                neighbor_key = nk if nk is not None else None

            if neighbor_key is None:
                continue

            if neighbor_key not in keep_set:
                # This face is on the cut plane — find its periodic partner
                # The partner is the element reached by 180° rotation
                grid = face_grids[iface]
                offsets = face_vertex_offsets[local_face]
                v0 = grid[j + offsets[0][0], i + offsets[0][1]]
                v1 = grid[j + offsets[1][0], i + offsets[1][1]]
                mid = normalize((v0 + v1) / 2.0)
                cut_angular.append((iface, j, i, local_face, mid))

    # Match periodic partners by rotating midpoint 180° around z
    n_cut = len(cut_angular)
    if n_cut == 0:
        return []

    mids = np.array([cf[4] for cf in cut_angular])
    matched = [False] * n_cut
    pairs = []

    for idx in range(n_cut):
        if matched[idx]:
            continue
        iface, j, i, lf, mid = cut_angular[idx]
        mid_rot = np.array([-mid[0], -mid[1], mid[2]])

        dists = np.linalg.norm(mids - mid_rot, axis=1)
        dists[idx] = 999.0
        for jj in range(n_cut):
            if matched[jj]:
                dists[jj] = 999.0
        best = np.argmin(dists)

        if dists[best] > 0.05:
            raise RuntimeError(f'No periodic partner for cut face ({iface},{j},{i})')

        pface, pj, pi, plf, _ = cut_angular[best]
        pairs.append((iface, j, i, lf, pface, pj, pi, plf))
        pairs.append((pface, pj, pi, plf, iface, j, i, lf))
        matched[idx] = True
        matched[best] = True

    if not all(matched):
        raise RuntimeError(f'{sum(1 for m in matched if not m)} cut faces unmatched')

    # Expand across radial layers
    periodic_bcs = []
    for (iface, j, i, lf, pface, pj, pi, plf) in pairs:
        for k in range(nr):
            eid = elem_map[(iface, j, i, k)]
            peid = elem_map[(pface, pj, pi, k)]
            periodic_bcs.append((eid, lf, peid, plf))

    return periodic_bcs


def build_bcs(elements, elem_map, keep_set, na, nr, face_grids,
              periodic_bcs, include_mhd=True):
    """Build BC lists for velocity, temperature, and optionally magnetic field."""
    bc_vel = []
    bc_temp = []
    bc_mag = []

    # Inner/outer sphere walls
    iter_set = keep_set if keep_set is not None else (
        (f, j, i) for f in range(6) for j in range(na) for i in range(na))
    for (iface, j, i) in iter_set:
        eid_inner = elem_map[(iface, j, i, 0)]
        eid_outer = elem_map[(iface, j, i, nr - 1)]
        bc_vel.append((eid_inner, 5, [0,0,0,0,0], 'W  '))
        bc_temp.append((eid_inner, 5, [0,0,0,0,0], 't  '))
        bc_vel.append((eid_outer, 6, [0,0,0,0,0], 'W  '))
        bc_temp.append((eid_outer, 6, [0,0,0,0,0], 't  '))
        if include_mhd:
            bc_mag.append((eid_inner, 5, [0,0,0,0,0], 'ON '))
            bc_mag.append((eid_outer, 6, [0,0,0,0,0], 'ON '))

    # Periodic BCs on cut plane
    for (eid, lf, peid, plf) in periodic_bcs:
        bc_vel.append((eid, lf, [peid, plf, 0, 0, 0], 'P  '))
        bc_temp.append((eid, lf, [peid, plf, 0, 0, 0], 'P  '))
        if include_mhd:
            bc_mag.append((eid, lf, [peid, plf, 0, 0, 0], 'P  '))

    bc_data = {'velocity': bc_vel, 'temperature': bc_temp}
    if include_mhd:
        bc_data['magnetic'] = bc_mag
    return bc_data


# ============================================================================
# .re2 writer (from gen_cubed_sphere.py)
# ============================================================================

def write_re2(filename, elements, curved_sides, bc_data, ndim=3, nelv=None):
    nelt = len(elements)
    if nelv is None:
        nelv = nelt

    with open(filename, 'wb') as f:
        hdr = f'#v002{nelt:9d}{ndim:3d}{nelv:9d} hemisphere'
        hdr = hdr.ljust(80)[:80]
        f.write(hdr.encode('ascii'))
        f.write(struct.pack('<f', 6.54321))

        for elem in elements:
            f.write(struct.pack('<d', 0.0))  # igroup
            for val in elem['x']:
                f.write(struct.pack('<d', val))
            for val in elem['y']:
                f.write(struct.pack('<d', val))
            for val in elem['z']:
                f.write(struct.pack('<d', val))

        ncurve = len(curved_sides)
        f.write(struct.pack('<d', float(ncurve)))
        for (eg, iface, curve_vals, ccurve) in curved_sides:
            f.write(struct.pack('<d', float(eg)))
            f.write(struct.pack('<d', float(iface)))
            for k in range(5):
                f.write(struct.pack('<d', curve_vals[k]))
            char_word = bytearray(8)
            char_word[0:1] = ccurve.encode('ascii')
            f.write(char_word)

        for field_name, bcs in bc_data.items():
            nbc = len(bcs)
            f.write(struct.pack('<d', float(nbc)))
            for (eg, iface, bc_params, cbc) in bcs:
                f.write(struct.pack('<d', float(eg)))
                f.write(struct.pack('<d', float(iface)))
                for k in range(5):
                    f.write(struct.pack('<d', float(bc_params[k])))
                char_word = bytearray(8)
                cbc_padded = cbc.ljust(3)[:3]
                char_word[0:3] = cbc_padded.encode('ascii')
                f.write(char_word)

    print(f'Wrote {filename}: nelt={nelt}, nelv={nelv}, {ncurve} curved sides')


# ============================================================================
# .ma2 writer — bypasses genmap for periodic connectivity
# ============================================================================

# Nek5000 hex face vertex indices (0-based element vertices)
FACE_VERTS = {
    1: [0, 1, 5, 4],   # j-min (y-min in local)
    2: [1, 2, 6, 5],   # i-max (x-max in local)
    3: [2, 3, 7, 6],   # j-max (y-max in local)
    4: [3, 0, 4, 7],   # i-min (x-min in local)
    5: [0, 3, 2, 1],   # k-min (z-min / inner sphere)
    6: [4, 5, 6, 7],   # k-max (z-max / outer sphere)
}


def compute_global_vertices(elements, periodic_bcs=None, merge_periodic=False):
    """
    Assign global vertex numbers based on physical position.

    If merge_periodic=True, periodic vertex pairs (under 180° rotation)
    are given the same global ID, enabling native Nek5000 P BC handling
    via dssum. fix_geom masks out P faces so coordinates are not averaged.

    Returns:
        cell: list of lists, cell[e] = [gv0, ..., gv7] in h2s order (1-based)
        nrnk: number of unique global vertices
    """
    nel = len(elements)
    nv = 8

    # h2s permutation: genmap reads .re2 vertices and reorders via h2s
    # cell slot v uses position at h2s[v] in .re2 element vertex ordering
    h2s = [0, 1, 3, 2, 4, 5, 7, 6]

    ROUND = 8  # decimal places for position matching
    pos_to_id = {}
    next_id = 1
    cell = []

    for e in range(nel):
        vids = []
        for v in range(nv):
            rv = h2s[v]  # .re2 vertex index for this cell slot
            key = (round(elements[e]['x'][rv], ROUND),
                   round(elements[e]['y'][rv], ROUND),
                   round(elements[e]['z'][rv], ROUND))
            if key not in pos_to_id:
                pos_to_id[key] = next_id
                next_id += 1
            vids.append(pos_to_id[key])
        cell.append(vids)

    nrnk_before = next_id - 1

    if merge_periodic and periodic_bcs is not None:
        # Union-find for merging periodic vertex pairs
        parent = list(range(nrnk_before + 1))  # 1-based

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra  # merge b into a

        # s2h: inverse of h2s, maps .re2 vertex index → cell slot
        # h2s is self-inverse: [0,1,3,2,4,5,7,6]
        s2h = [0, 1, 3, 2, 4, 5, 7, 6]

        # For each periodic face pair, match vertices under 180° rotation
        seen = set()
        for (eid, lf, peid, plf) in periodic_bcs:
            pair_key = (min(eid, peid), max(eid, peid))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            face_verts_a = FACE_VERTS[lf]  # .re2 vertex indices
            face_verts_b = FACE_VERTS[plf]
            ea = elements[eid - 1]
            eb = elements[peid - 1]

            for ia in range(4):
                va = face_verts_a[ia]  # .re2 vertex index
                # Rotated position under (x,y,z) -> (-x,-y,z)
                rx = -ea['x'][va]
                ry = -ea['y'][va]
                rz = ea['z'][va]
                # Find matching vertex on face B
                best_dist = 1e20
                best_ib = -1
                for ib in range(4):
                    vb = face_verts_b[ib]
                    d = ((eb['x'][vb] - rx)**2 +
                         (eb['y'][vb] - ry)**2 +
                         (eb['z'][vb] - rz)**2)
                    if d < best_dist:
                        best_dist = d
                        best_ib = ib
                if best_dist > 1e-12:
                    raise RuntimeError(
                        f'Periodic vertex match failed: dist={best_dist:.2e}')
                # Map .re2 vertex indices to cell slots via s2h
                gid_a = cell[eid - 1][s2h[face_verts_a[ia]]]
                gid_b = cell[peid - 1][s2h[face_verts_b[best_ib]]]
                union(gid_a, gid_b)

        # Renumber: compress union-find into contiguous IDs
        root_to_new = {}
        new_id = 1
        for e in range(nel):
            for v in range(nv):
                root = find(cell[e][v])
                if root not in root_to_new:
                    root_to_new[root] = new_id
                    new_id += 1
                cell[e][v] = root_to_new[root]

        nrnk = new_id - 1
        n_merged = nrnk_before - nrnk
        print(f'  Unique vertices (before merge): {nrnk_before}')
        print(f'  Unique vertices (after periodic merge): {nrnk}')
        print(f'  Merged {n_merged} periodic vertex pairs')
    else:
        nrnk = nrnk_before
        print(f'  Unique vertices (periodic NOT merged): {nrnk}')

    return cell, nrnk


def compute_periodic_permutations(elements, periodic_bcs):
    """
    For each periodic face pair, compute the vertex permutation.

    The permutation perm[0:4] means: vertex i of face_A maps to vertex
    perm[i] of face_B under 180° rotation (x,y,z) -> (-x,-y,z).

    Returns:
        list of (eid, lf, peid, plf, perm[4]) — one per UNIQUE pair
        (only the A->B direction; B->A is the inverse)
    """
    seen = set()
    pairs_with_perm = []

    for (eid, lf, peid, plf) in periodic_bcs:
        pair_key = (min(eid, peid), max(eid, peid))
        if pair_key in seen:
            continue
        seen.add(pair_key)

        face_verts_a = FACE_VERTS[lf]
        face_verts_b = FACE_VERTS[plf]

        ea = elements[eid - 1]
        eb = elements[peid - 1]

        coords_a = np.array([[ea['x'][v], ea['y'][v], ea['z'][v]]
                             for v in face_verts_a])
        coords_b = np.array([[eb['x'][v], eb['y'][v], eb['z'][v]]
                             for v in face_verts_b])

        # Under 180° rotation: (x,y,z) -> (-x,-y,z)
        coords_a_rot = coords_a.copy()
        coords_a_rot[:, 0] *= -1
        coords_a_rot[:, 1] *= -1

        perm = [-1, -1, -1, -1]
        for ia in range(4):
            dists = np.linalg.norm(coords_b - coords_a_rot[ia], axis=1)
            ib = np.argmin(dists)
            if dists[ib] > 1e-6:
                raise RuntimeError(
                    f'Periodic vertex match failed: dist={dists[ib]:.2e} '
                    f'elem {eid} face {lf} vert {ia}')
            perm[ia] = ib

        pairs_with_perm.append((eid, lf, peid, plf, perm))

    return pairs_with_perm


def write_periodic_pairs(filename, pairs_with_perm):
    """
    Write periodic_pairs.dat for the Fortran anti-periodic dssum module.

    Format:
        Line 1: npairs
        Lines 2+: elem_A  face_A  elem_B  face_B  perm0  perm1  perm2  perm3

    All indices are 1-based (Fortran convention). Perm values are 0-based
    face-local vertex indices.
    """
    with open(filename, 'w') as f:
        f.write(f'{len(pairs_with_perm)}\n')
        for (eid, lf, peid, plf, perm) in pairs_with_perm:
            f.write(f'{eid:8d} {lf:2d} {peid:8d} {plf:2d}'
                    f'  {perm[0]} {perm[1]} {perm[2]} {perm[3]}\n')

    print(f'Wrote {filename}: {len(pairs_with_perm)} unique periodic face pairs')


def recursive_bisect(elements, indices, depth=0):
    """
    Recursive coordinate bisection (RCB) of elements.

    Alternates splitting along x, y, z based on element centroids.
    Returns a reordered list of element indices that groups spatially
    adjacent elements together — critical for the coarse grid solver.

    Parameters:
        elements: list of element dicts with 'x', 'y', 'z'
        indices: list of element indices to partition
        depth: current recursion depth

    Returns:
        ordered_indices: list of element indices in RCB order
    """
    if len(indices) <= 1:
        return indices

    # Compute centroids
    centroids = np.array([
        [np.mean(elements[i]['x']),
         np.mean(elements[i]['y']),
         np.mean(elements[i]['z'])]
        for i in indices
    ])

    # Split along the axis with largest spread
    axis = depth % 3
    spreads = [centroids[:, k].max() - centroids[:, k].min() for k in range(3)]
    axis = int(np.argmax(spreads))

    # Sort by centroid along chosen axis
    sorted_pairs = sorted(zip(centroids[:, axis], indices))
    mid = len(sorted_pairs) // 2

    left_indices = [p[1] for p in sorted_pairs[:mid]]
    right_indices = [p[1] for p in sorted_pairs[mid:]]

    return (recursive_bisect(elements, left_indices, depth + 1) +
            recursive_bisect(elements, right_indices, depth + 1))


def write_ma2(filename, cell, nel, nrnk, elements=None):
    """
    Write a Nek5000 .ma2 binary connectivity file.

    If elements are provided, uses recursive coordinate bisection (RCB)
    for element ordering. Otherwise uses linear ordering.

    Format (from genmap.f dmp_mapfile):
      Header: 132 bytes ASCII + 4 bytes endian test
      Per element: ordering_id (int32) + 8 vertex IDs (int32 each)
    """
    nv = 8
    npts = nel * nv
    noutflow = 0
    nactive = nrnk - noutflow

    # Depth for RSB partitioning — ceil(log2(nel))
    depth = max(1, math.ceil(math.log2(nel))) if nel > 1 else 1
    d2 = 2 ** depth

    # Compute element ordering via RCB
    if elements is not None:
        print(f'  Computing RCB element ordering (depth={depth})...')
        rcb_order = recursive_bisect(elements, list(range(nel)))
        # rcb_order[i] = original element index that should be in position i
        # We need the inverse: for each original element, what's its RCB position
        order = [0] * nel
        for pos, orig_idx in enumerate(rcb_order):
            order[orig_idx] = pos
    else:
        order = list(range(nel))

    with open(filename, 'wb') as f:
        # Header: 132 bytes ASCII
        version = '#v001'
        hdr = f'{version}{nel:12d}{nactive:12d}{depth:12d}{d2:12d}{npts:12d}{nrnk:12d}{noutflow:12d}'
        hdr = hdr.ljust(132)[:132]
        f.write(hdr.encode('ascii'))

        # Endian test: 4-byte float
        f.write(struct.pack('<f', 6.54321))

        # Per element: RCB ordering position (int32) + nv vertex IDs (int32)
        # cell is already in h2s order (from compute_global_vertices)
        for e in range(nel):
            f.write(struct.pack('<i', order[e]))
            for v in range(nv):
                f.write(struct.pack('<i', cell[e][v]))

    print(f'Wrote {filename}: nel={nel}, nrnk={nrnk}, depth={depth}, RCB ordering')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate cubed-sphere hemisphere .re2 + .ma2 mesh')
    parser.add_argument('--na', type=int, default=8,
                        help='Elements per cube-face edge (must be even, default: 8)')
    parser.add_argument('--nr', type=int, default=8,
                        help='Radial layers (default: 8)')
    parser.add_argument('--ri', type=float, default=7.0/13.0,
                        help='Inner radius (default: 7/13)')
    parser.add_argument('--ro', type=float, default=20.0/13.0,
                        help='Outer radius (default: 20/13)')
    parser.add_argument('--Ek', type=float, default=1e-3,
                        help='Ekman number for BL grading (default: 1e-3)')
    parser.add_argument('--n_bl', type=int, default=0,
                        help='BL cells per wall (0=uniform, default: 0)')
    parser.add_argument('--delta_factor', type=float, default=5.0,
                        help='BL zone = delta_factor * Ek^(1/3) (default: 5)')
    parser.add_argument('--radial_dist', type=str, default='uniform',
                        choices=['uniform', 'chebyshev', 'geometric'],
                        help='Radial distribution (default: uniform)')
    parser.add_argument('--no-mhd', action='store_true',
                        help='Omit magnetic field BCs (hydro only)')
    parser.add_argument('--output', type=str, default='hemisphere',
                        help='Output base name (default: hemisphere)')

    # Topography parameters
    parser.add_argument('--topo_amp', type=float, default=0.0,
                        help='Topography amplitude (radial, same units as ri/ro)')
    parser.add_argument('--topo_theta0', type=float, default=90.0,
                        help='Topography center colatitude (degrees, default: 90)')
    parser.add_argument('--topo_phi0', type=float, default=90.0,
                        help='Topography center longitude (degrees, default: 90)')
    parser.add_argument('--topo_sigma_theta', type=float, default=15.0,
                        help='Topography half-width in colatitude (degrees, default: 15)')
    parser.add_argument('--topo_sigma_phi', type=float, default=15.0,
                        help='Topography half-width in longitude (degrees, default: 15)')
    parser.add_argument('--topo_n', type=int, default=4,
                        help='Super-Gaussian order (1=Gaussian, 4+=wall-like, default: 4)')
    args = parser.parse_args()

    na = args.na
    nr = args.nr
    ri = args.ri
    ro = args.ro

    if na % 2 != 0:
        raise ValueError(f'--na must be even (got {na})')

    # Generate cubed-sphere grids
    faces = cube_face_vertices()
    face_grids = [equiangular_grid(fc, na) for fc in faces]

    # Select hemisphere elements (y >= 0)
    keep_set = compute_keep_set(face_grids, na)
    nel_per_layer = len(keep_set)

    print(f'Generating cubed-sphere hemisphere mesh:')
    print(f'  Angular elements per layer: {nel_per_layer} (of {6*na*na} full sphere)')
    print(f'  Radial layers: {nr}')
    print(f'  ri = {ri:.8f}, ro = {ro:.8f}')

    nel = nel_per_layer * nr
    print(f'  Total elements: {nel}')

    # Radial layers
    radii = generate_radial_layers(ri, ro, nr, Ek=args.Ek,
                                   n_bl_cells=args.n_bl,
                                   delta_factor=args.delta_factor,
                                   radial_dist=args.radial_dist)

    # Build topography parameters (None if amplitude is zero)
    if args.topo_amp != 0.0:
        topo_params = {
            'amplitude':   args.topo_amp,
            'theta0':      np.radians(args.topo_theta0),
            'phi0':        np.radians(args.topo_phi0),
            'sigma_theta': np.radians(args.topo_sigma_theta),
            'sigma_phi':   np.radians(args.topo_sigma_phi),
            'n':           args.topo_n,
        }
        print(f'  Topography: super-Gaussian n={args.topo_n}')
        print(f'    amplitude = {args.topo_amp:.6f}')
        print(f'    center = ({args.topo_theta0:.1f}°, {args.topo_phi0:.1f}°)')
        print(f'    half-widths = ({args.topo_sigma_theta:.1f}°, {args.topo_sigma_phi:.1f}°)')
    else:
        topo_params = None

    # Build elements
    elements, curved_sides, elem_map = build_elements(
        face_grids, na, radii, nr, keep_set=keep_set, topo_params=topo_params)
    assert len(elements) == nel

    # Find periodic face pairs on the cut plane
    periodic_bcs = find_cut_plane_faces(
        face_grids, na, keep_set, elem_map, nr)
    print(f'  Periodic BC face pairs: {len(periodic_bcs)}')

    # Build BCs for .re2
    bc_data = build_bcs(elements, elem_map, keep_set, na, nr, face_grids,
                        periodic_bcs, include_mhd=not args.no_mhd)

    for name, bcs in bc_data.items():
        types = {}
        for _, _, _, cbc in bcs:
            t = cbc.strip()
            types[t] = types.get(t, 0) + 1
        print(f'  BCs ({name}): {types}')

    # Write .re2
    re2_file = args.output + '.re2'
    write_re2(re2_file, elements, curved_sides, bc_data)

    # Compute global vertex numbering (merge periodic vertices —
    # rotate_cyc handles anti-periodic velocity via face-local rotation)
    print(f'\nComputing connectivity (bypassing genmap):')
    cell, nrnk = compute_global_vertices(
        elements, periodic_bcs=periodic_bcs, merge_periodic=True)

    # Write .ma2
    ma2_file = args.output + '.ma2'
    write_ma2(ma2_file, cell, nel, nrnk, elements=elements)

    # Compute periodic vertex permutations and write periodic_pairs.dat
    print(f'\nComputing periodic face permutations:')
    pairs_with_perm = compute_periodic_permutations(elements, periodic_bcs)
    pairs_file = args.output + '_periodic_pairs.dat'
    write_periodic_pairs(pairs_file, pairs_with_perm)

    # Print SIZE recommendations
    lx1 = 8
    print(f'\nRecommended SIZE parameters:')
    print(f'  parameter (ldim=3)')
    print(f'  parameter (lx1={lx1})')
    print(f'  parameter (lxd={int(3*lx1/2)})')
    print(f'  parameter (lelg={nel})')
    print(f'  parameter (lelt={max(1, nel // 4 + 50)})')

    # Write mesh metadata
    info_file = args.output + '_mesh_info.txt'
    with open(info_file, 'w') as f:
        f.write(f'na = {na}\n')
        f.write(f'nr = {nr}\n')
        f.write(f'ri = {ri}\n')
        f.write(f'ro = {ro}\n')
        f.write(f'nelg = {nel}\n')
        f.write(f'nel_per_layer = {nel_per_layer}\n')
        f.write(f'radial_dist = {args.radial_dist}\n')
        f.write(f'Ek = {args.Ek}\n')
        f.write(f'n_bl = {args.n_bl}\n')
        f.write(f'delta_factor = {args.delta_factor}\n')
        f.write(f'lx1 = {lx1}\n')
        f.write(f'mhd = {not args.no_mhd}\n')
    print(f'\nWrote {info_file}')

    print(f'\nDone. Files: {re2_file}, {ma2_file}, {pairs_file}, {info_file}')
    print(f'NOTE: genmap is NOT needed — the .ma2 was written directly.')
    print(f'NOTE: Use {pairs_file} with anti_periodic.f for correct dssum.')


if __name__ == '__main__':
    main()
