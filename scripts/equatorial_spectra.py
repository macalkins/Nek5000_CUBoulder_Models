#!/usr/bin/env python3
"""
Compute azimuthal energy spectra in the equatorial plane from Nek5000 field files.

Extracts all GLL points near the equatorial plane (z ~ 0), bins by radius,
interpolates onto a uniform phi grid per radial bin, and computes FFT-based
power spectra in the azimuthal direction.

For the 180-degree wedge hemisphere with anti-periodic BCs, the FFT mode k
in the half-domain corresponds to full-sphere azimuthal wavenumber m = 2k
(since the domain spans [0, pi)).

Usage:
    python equatorial_spectra.py shell0.f00001
    python equatorial_spectra.py shell0.f0000[1-9]          # time-average
    python equatorial_spectra.py --mhd shell0.f00001        # include B field
    python equatorial_spectra.py --nphi 128 shell0.f00001
"""

import numpy as np
import argparse
import glob
import sys
from pymech.neksuite import readnek


def extract_equatorial_points(data, z_tol=None, mhd=False):
    """Extract ALL GLL points near the equatorial plane (|z| < z_tol).

    Returns dict with 1D arrays: r, phi, ux, uy, uz, T, [bx, by, bz].
    """
    lx, ly, lz = data.lr1

    if z_tol is None:
        # Auto: half the typical element z-extent
        dz_list = [el.pos[2].max() - el.pos[2].min() for el in data.elem]
        z_tol = np.median(dz_list) * 0.6

    r_list, phi_list = [], []
    field_lists = {k: [] for k in ['ux', 'uy', 'uz', 'T']}
    if mhd:
        for k in ['bx', 'by', 'bz']:
            field_lists[k] = []

    for el in data.elem:
        z = el.pos[2].ravel()
        mask = np.abs(z) < z_tol
        if not np.any(mask):
            continue

        x = el.pos[0].ravel()[mask]
        y = el.pos[1].ravel()[mask]
        z_sel = z[mask]

        r = np.sqrt(x**2 + y**2 + z_sel**2)
        phi = np.arctan2(y, x)

        r_list.append(r)
        phi_list.append(phi)
        field_lists['ux'].append(el.vel[0].ravel()[mask])
        field_lists['uy'].append(el.vel[1].ravel()[mask])
        field_lists['uz'].append(el.vel[2].ravel()[mask])
        if len(el.temp) > 0:
            field_lists['T'].append(el.temp[0].ravel()[mask])
        if mhd and len(el.temp) >= 4:
            field_lists['bx'].append(el.temp[1].ravel()[mask])
            field_lists['by'].append(el.temp[2].ravel()[mask])
            field_lists['bz'].append(el.temp[3].ravel()[mask])

    result = {
        'r': np.concatenate(r_list),
        'phi': np.concatenate(phi_list),
    }
    for k, v in field_lists.items():
        if v:
            result[k] = np.concatenate(v)

    return result


def bin_and_interpolate(eq_data, nr=None, nphi=256, mhd=False):
    """Bin equatorial points by radius, then interpolate to uniform phi per bin.

    Returns r_grid, phi_grid, and dict of 2D field arrays (nr, nphi).
    """
    r_pts = eq_data['r']
    phi_pts = eq_data['phi']
    # Ensure phi in [0, pi]
    phi_pts = phi_pts % (2 * np.pi)

    r_min, r_max = r_pts.min(), r_pts.max()

    if nr is None:
        # Auto-detect from point density: use ~50 radial bins
        nr = 50

    # Radial bin edges
    r_edges = np.linspace(r_min, r_max, nr + 1)
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])
    phi_grid = np.linspace(0, np.pi, nphi, endpoint=False)

    field_keys = ['ux', 'uy', 'uz']
    if 'T' in eq_data:
        field_keys.append('T')
    if mhd:
        for k in ['bx', 'by', 'bz']:
            if k in eq_data:
                field_keys.append(k)

    fields = {k: np.zeros((nr, nphi)) for k in field_keys}

    bin_idx = np.digitize(r_pts, r_edges) - 1
    bin_idx = np.clip(bin_idx, 0, nr - 1)

    for ir in range(nr):
        mask = bin_idx == ir
        if np.sum(mask) < 3:
            continue

        phi_bin = phi_pts[mask]
        # Sort by phi for clean interpolation
        sort_idx = np.argsort(phi_bin)
        phi_sorted = phi_bin[sort_idx]

        for key in field_keys:
            vals_sorted = eq_data[key][mask][sort_idx]
            # Remove duplicate phi values (average them)
            phi_unique, inv = np.unique(phi_sorted.round(decimals=10), return_inverse=True)
            vals_unique = np.zeros(len(phi_unique))
            counts = np.zeros(len(phi_unique))
            np.add.at(vals_unique, inv, vals_sorted)
            np.add.at(counts, inv, 1)
            vals_unique /= counts

            # Interpolate to uniform phi grid
            fields[key][ir, :] = np.interp(phi_grid, phi_unique, vals_unique,
                                            period=np.pi)

    return r_grid, phi_grid, fields


def compute_spectra(r_grid, phi_grid, fields, mhd=False):
    """Compute azimuthal power spectra at each radius.

    FFT mode k in [0, pi) domain -> full sphere m = 2k.
    """
    nphi = len(phi_grid)
    nr = len(r_grid)
    nm = nphi // 2

    k = np.arange(nm)
    m = 2 * k

    E_kin = np.zeros((nr, nm))
    for comp in ['ux', 'uy', 'uz']:
        fhat = np.fft.fft(fields[comp], axis=1)
        E_kin += 0.5 * np.abs(fhat[:, :nm])**2 / nphi**2

    result = {'m': m, 'r': r_grid, 'E_kin': E_kin}

    if 'T' in fields:
        fhat = np.fft.fft(fields['T'], axis=1)
        result['E_T'] = 0.5 * np.abs(fhat[:, :nm])**2 / nphi**2

    if mhd and 'bx' in fields:
        E_mag = np.zeros((nr, nm))
        for comp in ['bx', 'by', 'bz']:
            fhat = np.fft.fft(fields[comp], axis=1)
            E_mag += 0.5 * np.abs(fhat[:, :nm])**2 / nphi**2
        result['E_mag'] = E_mag

    return result


def process_files(file_list, nr=None, nphi=256, mhd=False):
    """Process field files and return time-averaged spectra."""
    spectra_accum = None
    n_files = 0

    for fname in file_list:
        print(f"Reading {fname}...")
        data = readnek(fname)
        if n_files == 0:
            print(f"  {len(data.elem)} elements, lr1={data.lr1}")

        eq_data = extract_equatorial_points(data, mhd=mhd)
        print(f"  {len(eq_data['r'])} equatorial points extracted")

        r_grid, phi_grid, fields = bin_and_interpolate(eq_data, nr=nr, nphi=nphi, mhd=mhd)
        spec = compute_spectra(r_grid, phi_grid, fields, mhd=mhd)

        if spectra_accum is None:
            spectra_accum = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else v
                            for k, v in spec.items()}
            spectra_accum['m'] = spec['m']
            spectra_accum['r'] = spec['r']

        for k in spec:
            if isinstance(spec[k], np.ndarray) and k not in ('m', 'r'):
                spectra_accum[k] += spec[k]

        n_files += 1

    for k in spectra_accum:
        if isinstance(spectra_accum[k], np.ndarray) and k not in ('m', 'r'):
            spectra_accum[k] /= n_files

    print(f"Averaged {n_files} file(s)")
    return spectra_accum


def save_spectra(prefix, spectra):
    """Save spectra to CSV files."""
    m = spectra['m']
    r = spectra['r']
    nr = len(r)

    dr = np.gradient(r)
    weights = r**2 * dr

    E_kin_total = np.sum(spectra['E_kin'] * weights[:, None], axis=0)
    cols = [m, E_kin_total]
    header_parts = ['m', 'E_kin']

    if 'E_T' in spectra:
        E_T_total = np.sum(spectra['E_T'] * weights[:, None], axis=0)
        cols.append(E_T_total)
        header_parts.append('E_T')

    if 'E_mag' in spectra:
        E_mag_total = np.sum(spectra['E_mag'] * weights[:, None], axis=0)
        cols.append(E_mag_total)
        header_parts.append('E_mag')

    output = np.column_stack(cols)
    fname = f'{prefix}_total.csv'
    np.savetxt(fname, output, header=', '.join(header_parts), delimiter=',')
    print(f"Saved {fname}")

    # Per-radius spectra at representative radii
    n_select = min(5, nr)
    idx_select = np.linspace(0, nr - 1, n_select, dtype=int)

    cols = [m]
    header_parts = ['m']
    for i in idx_select:
        cols.append(spectra['E_kin'][i, :])
        header_parts.append(f'E_kin(r={r[i]:.4f})')
    output = np.column_stack(cols)
    fname = f'{prefix}_by_radius.csv'
    np.savetxt(fname, output, header=', '.join(header_parts), delimiter=',')
    print(f"Saved {fname}")


def plot_spectra(spectra, outfile='equatorial_spectra.png'):
    """Plot azimuthal energy spectra."""
    import matplotlib.pyplot as plt

    m = spectra['m']
    r = spectra['r']
    nr = len(r)

    has_mag = 'E_mag' in spectra
    has_T = 'E_T' in spectra
    ncols = 1 + int(has_mag) + int(has_T)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)

    dr = np.gradient(r)
    weights = r**2 * dr

    n_select = min(5, nr)
    idx_select = np.linspace(0, nr - 1, n_select, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_select))

    mask = m > 0

    # KE spectra
    ax = axes[0, 0]
    E_total = np.sum(spectra['E_kin'] * weights[:, None], axis=0)
    ax.semilogy(m[mask], E_total[mask], 'k-', linewidth=2, label='integrated')
    for ci, i in enumerate(idx_select):
        ax.semilogy(m[mask], spectra['E_kin'][i, mask], color=colors[ci],
                     alpha=0.7, label=f'r={r[i]:.3f}')
    ax.set_xlabel('m (full sphere)')
    ax.set_ylabel(r'$E_{kin}(m)$')
    ax.set_title('Kinetic energy spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    col = 1
    if has_T:
        ax = axes[0, col]
        E_total = np.sum(spectra['E_T'] * weights[:, None], axis=0)
        ax.semilogy(m[mask], E_total[mask], 'k-', linewidth=2, label='integrated')
        for ci, i in enumerate(idx_select):
            ax.semilogy(m[mask], spectra['E_T'][i, mask], color=colors[ci],
                         alpha=0.7, label=f'r={r[i]:.3f}')
        ax.set_xlabel('m (full sphere)')
        ax.set_ylabel(r'$E_T(m)$')
        ax.set_title('Temperature spectrum')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        col += 1

    if has_mag:
        ax = axes[0, col]
        E_total = np.sum(spectra['E_mag'] * weights[:, None], axis=0)
        ax.semilogy(m[mask], E_total[mask], 'k-', linewidth=2, label='integrated')
        for ci, i in enumerate(idx_select):
            ax.semilogy(m[mask], spectra['E_mag'][i, mask], color=colors[ci],
                         alpha=0.7, label=f'r={r[i]:.3f}')
        ax.set_xlabel('m (full sphere)')
        ax.set_ylabel(r'$E_{mag}(m)$')
        ax.set_title('Magnetic energy spectrum')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser(description='Equatorial spectra from Nek5000 field files')
    parser.add_argument('files', nargs='+', help='Field file(s)')
    parser.add_argument('--mhd', action='store_true', help='Include B field')
    parser.add_argument('--nr', type=int, default=None, help='Radial bins (auto: 50)')
    parser.add_argument('--nphi', type=int, default=256, help='Azimuthal grid points')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('-o', '--output', default='equatorial_spectra', help='Output prefix')
    args = parser.parse_args()

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
    spectra = process_files(file_list, nr=args.nr, nphi=args.nphi, mhd=args.mhd)
    save_spectra(args.output, spectra)

    if not args.no_plot:
        plot_spectra(spectra, outfile=f'{args.output}.png')


if __name__ == '__main__':
    main()
