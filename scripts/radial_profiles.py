#!/usr/bin/env python3
"""
Compute radial profiles of mean and RMS quantities on spherical surfaces.

Reads Nek5000 field files (pymech), converts velocity to spherical components
(u_r, u_theta, u_phi), bins GLL points by radius, and computes shell-averaged
mean and RMS values. Supports time-averaging over multiple snapshots.

Usage:
    python radial_profiles.py shell0.f00001
    python radial_profiles.py shell0.f0000[1-9]          # time-average
    python radial_profiles.py --mhd shell0.f00001        # include B field
    python radial_profiles.py --no-plot shell0.f00001    # CSV only
"""

import numpy as np
import argparse
import glob
import sys
from pymech.neksuite import readnek


def cartesian_to_spherical_vel(x, y, z, ux, uy, uz):
    """Convert Cartesian velocity to spherical (u_r, u_theta, u_phi).

    theta = colatitude (0 at north pole, pi at south pole)
    phi = longitude (azimuthal angle)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    rperp = np.sqrt(x**2 + y**2)

    # Avoid division by zero at origin and poles
    r_safe = np.where(r > 0, r, 1.0)
    rperp_safe = np.where(rperp > 0, rperp, 1.0)

    # Radial: e_r = (x, y, z) / r
    u_r = (x * ux + y * uy + z * uz) / r_safe

    # Colatitudinal: e_theta = (xz, yz, -(x^2+y^2)) / (r * rperp)
    u_theta = (x * z * ux + y * z * uy - rperp**2 * uz) / (r_safe * rperp_safe)

    # Azimuthal: e_phi = (-y, x, 0) / rperp
    u_phi = (-y * ux + x * uy) / rperp_safe

    # Zero out at degenerate points
    u_r = np.where(r > 0, u_r, 0.0)
    u_theta = np.where(rperp > 0, u_theta, 0.0)
    u_phi = np.where(rperp > 0, u_phi, 0.0)

    return u_r, u_theta, u_phi


def extract_fields(data, mhd=False):
    """Extract all GLL point positions and field values from a Nek5000 dataset."""
    nel = len(data.elem)
    lx, ly, lz = data.lr1

    # Pre-allocate
    npts = nel * lx * ly * lz
    pos = np.zeros((3, npts))
    vel = np.zeros((3, npts))
    temp = np.zeros(npts)
    if mhd:
        bfield = np.zeros((3, npts))

    idx = 0
    for el in data.elem:
        n = lx * ly * lz
        sl = slice(idx, idx + n)
        for d in range(3):
            pos[d, sl] = el.pos[d].ravel()
            vel[d, sl] = el.vel[d].ravel()
        if len(el.temp) > 0:
            temp[sl] = el.temp[0].ravel()
        if mhd and len(el.temp) >= 4:
            # B stored in temp fields 1,2,3 (temp[0] = T, temp[1:4] = Bx,By,Bz)
            for d in range(3):
                bfield[d, sl] = el.temp[d + 1].ravel()
        idx += n

    result = {'pos': pos, 'vel': vel, 'temp': temp}
    if mhd:
        result['bfield'] = bfield
    return result


def compute_radial_bins(r, data, n_bins=None):
    """Bin GLL points by radius into spherical shells.

    For a cubed-sphere mesh, GLL points at the same logical radial level have
    slightly different r values across cube faces. We use uniform bins whose
    count is auto-detected from the mesh: nr_elem * (lx - 1) + 1.

    Returns bin-center radii and bin index for each point.
    """
    if n_bins is None:
        # Auto-detect: count radial element layers from centroid radii
        lx = data.lr1[0]
        nel = len(data.elem)
        centroids_r = np.array([np.mean(np.sqrt(
            el.pos[0]**2 + el.pos[1]**2 + el.pos[2]**2)) for el in data.elem])
        # Number of unique radial layers = nr
        nr = len(np.unique(centroids_r.round(decimals=3)))
        n_bins = nr * (lx - 1) + 1

    r_min, r_max = r.min(), r.max()
    # Bin edges with small padding to include boundary points
    pad = (r_max - r_min) * 1e-8
    edges = np.linspace(r_min - pad, r_max + pad, n_bins + 1)
    bin_idx = np.digitize(r, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Bin centers
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    return bin_centers, bin_idx, n_bins


def shell_average(field, bin_idx, n_bins):
    """Compute mean over spherical shells."""
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(sums, bin_idx, field)
    np.add.at(counts, bin_idx, 1)
    counts = np.maximum(counts, 1)
    return sums / counts


def shell_rms(field, mean_profile, bin_idx, n_bins):
    """Compute RMS fluctuation over spherical shells."""
    fluct_sq = (field - mean_profile[bin_idx])**2
    mean_sq = shell_average(fluct_sq, bin_idx, n_bins)
    return np.sqrt(mean_sq)


def process_files(file_list, mhd=False):
    """Process one or more field files and return time-averaged radial profiles."""
    # First pass: read first file to set up radial bins
    print(f"Reading {file_list[0]}...")
    data0 = readnek(file_list[0])
    fields0 = extract_fields(data0, mhd=mhd)
    x, y, z = fields0['pos']
    r = np.sqrt(x**2 + y**2 + z**2)
    bin_centers, bin_idx, n_bins = compute_radial_bins(r, data0)
    print(f"  {len(data0.elem)} elements, lr1={data0.lr1}, {len(r)} GLL points")
    print(f"  {n_bins} radial shells, r in [{bin_centers[0]:.4f}, {bin_centers[-1]:.4f}]")
    unique_r = bin_centers

    # Accumulators for time averaging
    keys = ['ur', 'ut', 'up', 'T']
    if mhd:
        keys += ['br', 'bt', 'bp']

    sum_mean = {k: np.zeros(n_bins) for k in keys}
    sum_sq = {k: np.zeros(n_bins) for k in keys}
    n_files = 0

    for fname in file_list:
        if fname != file_list[0]:
            print(f"Reading {fname}...")
            data = readnek(fname)
            fields = extract_fields(data, mhd=mhd)
        else:
            fields = fields0

        ux, uy, uz = fields['vel']
        ur, ut, up = cartesian_to_spherical_vel(x, y, z, ux, uy, uz)

        # Shell means for this snapshot
        snap = {
            'ur': shell_average(ur, bin_idx, n_bins),
            'ut': shell_average(ut, bin_idx, n_bins),
            'up': shell_average(up, bin_idx, n_bins),
            'T': shell_average(fields['temp'], bin_idx, n_bins),
        }

        if mhd:
            bx, by, bz = fields['bfield']
            br, bt, bp = cartesian_to_spherical_vel(x, y, z, bx, by, bz)
            snap['br'] = shell_average(br, bin_idx, n_bins)
            snap['bt'] = shell_average(bt, bin_idx, n_bins)
            snap['bp'] = shell_average(bp, bin_idx, n_bins)

        # Accumulate means and squared values (for RMS)
        for k in keys:
            sum_mean[k] += snap[k]
            if k == 'ur':
                field_vals = ur
            elif k == 'ut':
                field_vals = ut
            elif k == 'up':
                field_vals = up
            elif k == 'T':
                field_vals = fields['temp']
            elif k == 'br':
                field_vals = br
            elif k == 'bt':
                field_vals = bt
            elif k == 'bp':
                field_vals = bp
            # RMS: accumulate <f^2> on shells
            sum_sq[k] += shell_average(field_vals**2, bin_idx, n_bins)

        n_files += 1

    # Finalize time averages
    mean = {k: sum_mean[k] / n_files for k in keys}
    rms = {}
    for k in keys:
        mean_sq = sum_sq[k] / n_files
        rms[k] = np.sqrt(np.maximum(0, mean_sq - mean[k]**2))

    return unique_r, mean, rms, keys


def save_csv(filename, r, mean, rms, keys):
    """Save profiles to CSV."""
    cols = [r]
    header_parts = ['r']
    for k in keys:
        cols.append(mean[k])
        header_parts.append(f'{k}_mean')
    for k in keys:
        cols.append(rms[k])
        header_parts.append(f'{k}_rms')

    output = np.column_stack(cols)
    np.savetxt(filename, output, header=', '.join(header_parts), delimiter=',')
    print(f"Saved {filename}")


def plot_profiles(r, mean, rms, keys, outfile='radial_profiles.png'):
    """Plot radial profiles of mean and RMS quantities."""
    import matplotlib.pyplot as plt

    has_mhd = 'br' in keys
    nrows = 2 if has_mhd else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows), squeeze=False)

    # Velocity
    ax = axes[0, 0]
    ax.plot(r, mean['ur'], label=r'$\langle u_r \rangle$')
    ax.plot(r, mean['ut'], label=r'$\langle u_\theta \rangle$')
    ax.plot(r, mean['up'], label=r'$\langle u_\phi \rangle$')
    ax.set_xlabel('r')
    ax.set_ylabel('Mean velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(r, rms['ur'], label=r"$u_r'$")
    ax.plot(r, rms['ut'], label=r"$u_\theta'$")
    ax.plot(r, rms['up'], label=r"$u_\phi'$")
    ax.plot(r, rms['T'], label=r"$T'$", color='k', linestyle='--')
    ax.set_xlabel('r')
    ax.set_ylabel('RMS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if has_mhd:
        ax = axes[1, 0]
        ax.plot(r, mean['br'], label=r'$\langle B_r \rangle$')
        ax.plot(r, mean['bt'], label=r'$\langle B_\theta \rangle$')
        ax.plot(r, mean['bp'], label=r'$\langle B_\phi \rangle$')
        ax.set_xlabel('r')
        ax.set_ylabel('Mean B')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(r, rms['br'], label=r"$B_r'$")
        ax.plot(r, rms['bt'], label=r"$B_\theta'$")
        ax.plot(r, rms['bp'], label=r"$B_\phi'$")
        ax.set_xlabel('r')
        ax.set_ylabel('RMS B')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser(description='Radial profiles from Nek5000 field files')
    parser.add_argument('files', nargs='+', help='Field file(s), e.g. shell0.f00001')
    parser.add_argument('--mhd', action='store_true', help='Include B field (stored in temp[1:4])')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('-o', '--output', default='radial_profiles', help='Output prefix')
    args = parser.parse_args()

    # Expand any glob patterns
    file_list = []
    for pattern in args.files:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            file_list.extend(expanded)
        else:
            file_list.append(pattern)

    if not file_list:
        print("No files found.")
        sys.exit(1)

    print(f"Processing {len(file_list)} file(s)...")
    r, mean, rms, keys = process_files(file_list, mhd=args.mhd)
    save_csv(f'{args.output}.csv', r, mean, rms, keys)

    if not args.no_plot:
        plot_profiles(r, mean, rms, keys, outfile=f'{args.output}.png')


if __name__ == '__main__':
    main()
