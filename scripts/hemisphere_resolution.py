#!/usr/bin/env python3
"""
Resolution analysis for Nek5000 hemisphere (cubed-sphere) simulations.

Uses pysemtools Probes for GLL-accurate spectral interpolation onto
structured grids, then computes:

  1. Radial profiles of shell-averaged mean and RMS velocity (spherical
     components: u_r, u_theta, u_phi) and temperature
  2. Azimuthal Fourier spectra of velocity and temperature in the equatorial
     plane (using spherical velocity components — required for correct mode
     decomposition with anti-periodic BCs)
  3. Chebyshev radial spectra of RMS fields (interior, trimming BLs)
  4. Equatorial cross-section contours of fluctuating fields

For the 180-degree wedge with anti-periodic BCs, spherical velocity components
are periodic under phi -> phi+pi, so FFT mode k corresponds to full-sphere
azimuthal wavenumber m = 2k.

Must be run with MPI:
    mpirun -np N python hemisphere_resolution.py [field_file]

Requires pysemtools, mpi4py.
"""

from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import argparse
from numpy.polynomial.legendre import legder
from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
import pysemtools.interpolation.utils as interp_utils
from pysemtools.interpolation.probes import Probes


# ============================================================
# Utility functions
# ============================================================

def gll(n):
    """Compute n GLL points on [-1, 1]."""
    if n < 2:
        raise ValueError("Need at least 2 points.")
    coeffs = np.zeros(n)
    coeffs[-1] = 1
    dcoeffs = legder(coeffs)
    internal_roots = np.polynomial.legendre.legroots(dcoeffs)
    return np.sort(np.concatenate(([-1.0], internal_roots, [1.0])))


def cheb_spec(y):
    """Chebyshev coefficients via DCT-I."""
    N = len(y)
    padded = np.hstack((y, y[N - 2:0:-1]))
    dft = np.fft.fft(padded).real / (2 * N - 2)
    coeffs = dft[:N] * 2
    coeffs[0] /= 2
    coeffs[-1] /= 2
    return coeffs


def find_latest_field(prefix='shell0.f'):
    """Find the latest field file matching prefix."""
    files = sorted(glob.glob(prefix + '*'))
    if not files:
        raise FileNotFoundError(f"No files matching '{prefix}*'")
    return files[-1]


def detect_radial_structure(msh, comm, lx):
    """Auto-detect ri, ro, nr from mesh coordinates."""
    x = msh.x.ravel()
    y = msh.y.ravel()
    z = msh.z.ravel()
    r_all = np.sqrt(x**2 + y**2 + z**2)

    ri_local = float(r_all.min())
    ro_local = float(r_all.max())
    ri = comm.allreduce(ri_local, op=MPI.MIN)
    ro = comm.allreduce(ro_local, op=MPI.MAX)

    # Count radial element layers from centroid radii
    npts_per_elem = lx**3
    nel_local = len(x) // npts_per_elem
    centroid_r = np.array([
        np.mean(np.sqrt(
            x[i * npts_per_elem:(i + 1) * npts_per_elem]**2 +
            y[i * npts_per_elem:(i + 1) * npts_per_elem]**2 +
            z[i * npts_per_elem:(i + 1) * npts_per_elem]**2
        )) for i in range(nel_local)
    ])

    all_cr = comm.gather(centroid_r, root=0)
    rank = comm.Get_rank()
    if rank == 0:
        all_cr = np.concatenate(all_cr)
        cr_sorted = np.sort(all_cr)
        # Cluster centroids: two elements are in the same radial layer
        # if their centroids differ by less than tol (fraction of shell thickness)
        tol = (ro - ri) / 1000.0
        unique_r = [cr_sorted[0]]
        for c in cr_sorted[1:]:
            if c - unique_r[-1] > tol:
                unique_r.append(c)
        nr = len(unique_r)
    else:
        nr = None
    nr = comm.bcast(nr, root=0)

    return ri, ro, nr


def build_radial_grid(ri, ro, nr, lx, radial_dist='chebyshev'):
    """Build GLL radial grid: nr elements with lx GLL points each.

    Supports uniform or Chebyshev element spacing to match gen_hemisphere.py.
    Returns unique sorted radial positions.
    """
    xi = gll(lx)
    if radial_dist == 'chebyshev':
        # Chebyshev element edges (matches gen_hemisphere.py)
        j = np.arange(nr + 1)
        nodes = -np.cos(np.pi * j / nr)
        r_edges = 0.5 * (ro + ri) + 0.5 * (ro - ri) * nodes
    else:
        r_edges = np.linspace(ri, ro, nr + 1)
    r_grid = []
    for i in range(nr):
        r_shift = 0.5 * (r_edges[i + 1] - r_edges[i]) * (xi + 1.0) + r_edges[i]
        r_grid = np.append(r_grid, r_shift)
    return np.unique(r_grid)


# ============================================================
# Main
# ============================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser(
    description='Resolution analysis for Nek5000 hemisphere simulations')
parser.add_argument('field_file', nargs='?', default=None,
                    help='Field file (default: latest shell0.f*)')
parser.add_argument('--prefix', default='shell0.f', help='Field file prefix')
parser.add_argument('--ri', type=float, default=None, help='Inner radius (auto-detect)')
parser.add_argument('--ro', type=float, default=None, help='Outer radius (auto-detect)')
parser.add_argument('--nr', type=int, default=None, help='Radial elements (auto-detect)')
parser.add_argument('--lx', type=int, default=None, help='Polynomial order + 1 (auto-detect from mesh)')
parser.add_argument('--ntheta', type=int, default=32,
                    help='Colatitude points for shell averaging')
parser.add_argument('--nphi-shell', type=int, default=64,
                    help='Azimuthal points for shell averaging')
parser.add_argument('--nphi-eq', type=int, default=512,
                    help='Azimuthal points for equatorial spectra')
parser.add_argument('--Ek', type=float, default=1e-4,
                    help='Ekman number (for BL trimming in Chebyshev spectra)')
parser.add_argument('--radial-dist', default='chebyshev',
                    choices=['uniform', 'chebyshev'],
                    help='Radial element distribution (default: chebyshev)')
args = parser.parse_args()

# --- Find field file ---
if args.field_file is None:
    fname = find_latest_field(args.prefix)
else:
    fname = args.field_file

if rank == 0:
    print(f"Reading: {fname}")

# --- Read mesh and field ---
msh = Mesh(comm, create_connectivity=True)
fld = FieldRegistry(comm)
pynekread(fname, comm, data_dtype=np.double, msh=msh, fld=fld)

# --- Detect geometry ---
lx = args.lx if args.lx else int(msh.lx)
ri_det, ro_det, nr_det = detect_radial_structure(msh, comm, lx)
ri = args.ri if args.ri else ri_det
ro = args.ro if args.ro else ro_det
nr = args.nr if args.nr else nr_det

if rank == 0:
    print(f"Geometry: ri={ri:.4f}, ro={ro:.4f}, nr={nr}, lx={lx}")

# --- Build grids ---
eps = 1e-5
r_1d = build_radial_grid(ri + eps, ro - eps, nr, lx, radial_dist=args.radial_dist)
nr_pts = len(r_1d)

ntheta = args.ntheta
nphi_shell = args.nphi_shell
nphi_eq = args.nphi_eq

theta_1d = np.linspace(eps, np.pi - eps, ntheta)
phi_shell_1d = np.linspace(eps, np.pi - eps, nphi_shell)
# Equatorial probes: uniform grid on [0, pi) for clean FFT (no endpoint)
# Small offset avoids probes exactly on the periodic cut plane (y=0)
phi_eq_1d = np.linspace(0, np.pi, nphi_eq, endpoint=False) + np.pi / (4 * nphi_eq)

n_shell = ntheta * nphi_shell  # angular points per radius for shell avg
n_eq = nphi_eq                 # azimuthal points per radius for equatorial
ny_total = n_shell + n_eq

if rank == 0:
    print(f"Radial grid: {nr_pts} points")
    print(f"Shell avg: {ntheta} x {nphi_shell} = {n_shell} angular pts/radius")
    print(f"Equatorial: {nphi_eq} azimuthal pts/radius")
    print(f"Total probes: {nr_pts * ny_total}")

# --- Build probe coordinates (rank 0 only) ---
if rank == 0:
    x_probes = np.zeros((nr_pts, ny_total, 1))
    y_probes = np.zeros((nr_pts, ny_total, 1))
    z_probes = np.zeros((nr_pts, ny_total, 1))

    # Shell probes: (r, theta, phi) grid
    TH, PH = np.meshgrid(theta_1d, phi_shell_1d, indexing='ij')
    th_flat = TH.ravel()  # length ntheta*nphi_shell
    ph_flat = PH.ravel()

    for ir in range(nr_pts):
        r = r_1d[ir]
        # Shell part: indices [0, n_shell)
        x_probes[ir, :n_shell, 0] = r * np.sin(th_flat) * np.cos(ph_flat)
        y_probes[ir, :n_shell, 0] = r * np.sin(th_flat) * np.sin(ph_flat)
        z_probes[ir, :n_shell, 0] = r * np.cos(th_flat)

        # Equatorial part: indices [n_shell, ny_total)
        x_probes[ir, n_shell:, 0] = r * np.cos(phi_eq_1d)
        y_probes[ir, n_shell:, 0] = r * np.sin(phi_eq_1d)
        z_probes[ir, n_shell:, 0] = 0.0

    xyz = interp_utils.transform_from_array_to_list(
        nr_pts, ny_total, 1, [x_probes, y_probes, z_probes])
else:
    xyz = None

# --- Interpolate ---
if rank == 0:
    print("Setting up probes...")

probes = Probes(comm, probes=xyz, msh=msh,
                point_interpolator_type='multiple_point_legendre_numpy',
                max_pts=256, find_points_comm_pattern='point_to_point')

field_list = [fld.registry['u'], fld.registry['v'], fld.registry['w'],
              fld.registry['p'], fld.registry['t']]
probes.interpolate_from_field_list(0, field_list, comm, write_data=False)

if rank == 0:
    print("Interpolation done. Processing results...")

interp = interp_utils.transform_from_list_to_array(
    nr_pts, ny_total, 1, probes.interpolated_fields)

ux_all = interp[1][:, :, 0]  # shape (nr_pts, ny_total)
uy_all = interp[2][:, :, 0]
uz_all = interp[3][:, :, 0]
p_all = interp[4][:, :, 0]
T_all = interp[5][:, :, 0]

# ============================================================
# Process shell-averaged profiles
# ============================================================
if rank == 0:
    # Extract shell data
    ux_sh = ux_all[:, :n_shell]  # (nr_pts, n_shell)
    uy_sh = uy_all[:, :n_shell]
    uz_sh = uz_all[:, :n_shell]
    T_sh = T_all[:, :n_shell]

    # Probe positions for shell points
    x_sh = x_probes[:, :n_shell, 0]
    y_sh = y_probes[:, :n_shell, 0]
    z_sh = z_probes[:, :n_shell, 0]
    r_sh = np.sqrt(x_sh**2 + y_sh**2 + z_sh**2)
    rperp_sh = np.sqrt(x_sh**2 + y_sh**2)

    # Convert to spherical velocity
    ur_sh = (x_sh * ux_sh + y_sh * uy_sh + z_sh * uz_sh) / r_sh
    rp_safe = np.where(rperp_sh > 0, rperp_sh, 1.0)
    ut_sh = (x_sh * z_sh * ux_sh + y_sh * z_sh * uy_sh
             - rperp_sh**2 * uz_sh) / (r_sh * rp_safe)
    up_sh = (-y_sh * ux_sh + x_sh * uy_sh) / rp_safe
    ut_sh = np.where(rperp_sh > 0, ut_sh, 0.0)
    up_sh = np.where(rperp_sh > 0, up_sh, 0.0)

    # Shell averages (mean over angular points at each radius)
    ur_mean = np.mean(ur_sh, axis=1)
    ut_mean = np.mean(ut_sh, axis=1)
    up_mean = np.mean(up_sh, axis=1)
    T_mean = np.mean(T_sh, axis=1)

    # Shell RMS
    ur_rms = np.sqrt(np.mean((ur_sh - ur_mean[:, None])**2, axis=1))
    ut_rms = np.sqrt(np.mean((ut_sh - ut_mean[:, None])**2, axis=1))
    up_rms = np.sqrt(np.mean((up_sh - up_mean[:, None])**2, axis=1))
    T_rms = np.sqrt(np.mean((T_sh - T_mean[:, None])**2, axis=1))

    # Save CSV
    output = np.column_stack((
        r_1d, ur_mean, ut_mean, up_mean, T_mean,
        ur_rms, ut_rms, up_rms, T_rms))
    np.savetxt('radial_profiles.csv', output, delimiter=',',
               header='r, ur_mean, ut_mean, up_mean, T_mean, '
                      'ur_rms, ut_rms, up_rms, T_rms')
    print("Saved radial_profiles.csv")

    # ============================================================
    # Process equatorial spectra
    # ============================================================
    ux_eq = ux_all[:, n_shell:]  # (nr_pts, nphi_eq)
    uy_eq = uy_all[:, n_shell:]
    uz_eq = uz_all[:, n_shell:]
    T_eq = T_all[:, n_shell:]

    # Convert to spherical at equator (z=0):
    #   ur = ux*cos(phi) + uy*sin(phi)
    #   uphi = -ux*sin(phi) + uy*cos(phi)
    #   utheta = -uz
    cos_phi = np.cos(phi_eq_1d)[None, :]
    sin_phi = np.sin(phi_eq_1d)[None, :]

    ur_eq = ux_eq * cos_phi + uy_eq * sin_phi
    up_eq = -ux_eq * sin_phi + uy_eq * cos_phi
    ut_eq = -uz_eq

    # Azimuthal mean (zonal average)
    ur_eq_zonal = np.mean(ur_eq, axis=1, keepdims=True)
    ut_eq_zonal = np.mean(ut_eq, axis=1, keepdims=True)
    up_eq_zonal = np.mean(up_eq, axis=1, keepdims=True)
    T_eq_zonal = np.mean(T_eq, axis=1, keepdims=True)

    # Fluctuations
    ur_eq_p = ur_eq - ur_eq_zonal
    ut_eq_p = ut_eq - ut_eq_zonal
    up_eq_p = up_eq - up_eq_zonal
    T_eq_p = T_eq - T_eq_zonal

    # Fourier spectra: KE and temperature
    # FFT mode k -> full sphere wavenumber m = 2k
    ur_hat = np.fft.fft(ur_eq_p, axis=1)
    ut_hat = np.fft.fft(ut_eq_p, axis=1)
    up_hat = np.fft.fft(up_eq_p, axis=1)
    T_hat = np.fft.fft(T_eq_p, axis=1)

    nm = nphi_eq // 2
    k_modes = np.arange(nm)
    m_full = 2 * k_modes  # full-sphere wavenumber

    # Power spectra (per-radius): 0.5 * |hat|^2 / N^2
    E_kin = (0.5 / nphi_eq**2) * (np.abs(ur_hat[:, :nm])**2 +
                                    np.abs(ut_hat[:, :nm])**2 +
                                    np.abs(up_hat[:, :nm])**2)
    E_T = (0.5 / nphi_eq**2) * np.abs(T_hat[:, :nm])**2

    # Radially-integrated spectra (weight by r^2 dr)
    dr = np.gradient(r_1d)
    weights = r_1d**2 * dr
    E_kin_total = np.sum(E_kin * weights[:, None], axis=0)
    E_T_total = np.sum(E_T * weights[:, None], axis=0)

    # Save spectra CSV
    spec_out = np.column_stack((m_full, E_kin_total, E_T_total))
    np.savetxt('equatorial_spectra_total.csv', spec_out, delimiter=',',
               header='m, E_kin, E_T')
    print("Saved equatorial_spectra_total.csv")

    # Per-radius spectra at selected interior radii (skip BLs)
    trim = max(1, nr_pts // 10)  # skip ~10% near each wall
    n_select = min(5, nr_pts - 2 * trim)
    idx_sel = np.linspace(trim, nr_pts - 1 - trim, n_select, dtype=int)
    cols = [m_full]
    hdr = ['m']
    for i in idx_sel:
        cols.append(E_kin[i, :])
        hdr.append(f'E_kin(r={r_1d[i]:.4f})')
    np.savetxt('equatorial_spectra_by_radius.csv',
               np.column_stack(cols), delimiter=',', header=', '.join(hdr))
    print("Saved equatorial_spectra_by_radius.csv")

    # ============================================================
    # Chebyshev radial spectra (full domain including BLs)
    # ============================================================
    N_full = len(r_1d)

    if N_full > 10:
        x_cheb = np.cos(np.arange(N_full) * np.pi / (N_full - 1))
        r_cheb = 0.5 * (ro + ri) + 0.5 * (ro - ri) * (-x_cheb)

        # Speed RMS for Chebyshev spectrum
        speed_rms = np.sqrt(ur_rms**2 + ut_rms**2 + up_rms**2)
        vel_interior = np.interp(r_cheb, r_1d, speed_rms)
        temp_interior = np.interp(r_cheb, r_1d, T_rms)

        vel_cheb = cheb_spec(vel_interior)
        temp_cheb = cheb_spec(temp_interior)
        N_cheb = np.arange(len(vel_cheb))
        have_cheb = True
    else:
        have_cheb = False
        if rank == 0:
            print("Warning: too few radial points for Chebyshev spectra")

    # ============================================================
    # Plotting
    # ============================================================

    # 1. Shell-averaged radial profiles: mean velocity
    ms = 2  # marker size for GLL points
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.plot(r_1d, ur_mean, '-o', markersize=ms, label=r'$\langle u_r \rangle$')
    ax.plot(r_1d, ut_mean, '-o', markersize=ms, label=r'$\langle u_\theta \rangle$')
    ax.plot(r_1d, up_mean, '-o', markersize=ms, label=r'$\langle u_\phi \rangle$')
    ax.set_xlabel('r')
    ax.set_ylabel('Mean velocity')
    ax.set_title('Shell-averaged mean velocity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig('mean_velocity_r.png', dpi=150)
    plt.close(fig)

    # 2. RMS velocity profiles
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.plot(r_1d, ur_rms, 'r-o', markersize=ms, label=r"$u_r'$")
    ax.plot(r_1d, ut_rms, 'b-o', markersize=ms, label=r"$u_\theta'$")
    ax.plot(r_1d, up_rms, 'g-o', markersize=ms, label=r"$u_\phi'$")
    ax.set_xlabel('r')
    ax.set_ylabel('RMS velocity')
    ax.set_title('Shell-averaged RMS velocity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig('rms_velocity_r.png', dpi=150)
    plt.close(fig)

    # 3. RMS temperature profile
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.plot(r_1d, T_rms, 'k-o', markersize=ms)
    ax.set_xlabel('r')
    ax.set_ylabel(r"$T'_{rms}$")
    ax.set_title('Shell-averaged RMS temperature')
    ax.grid(True, alpha=0.3)
    fig.savefig('rms_temperature_r.png', dpi=150)
    plt.close(fig)

    # 3b. Ekman boundary layer zoom
    Ek = args.Ek
    delta_E = np.sqrt(Ek)
    zoom_width = 10 * delta_E  # show 10 Ekman depths from each wall

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(10, 4),
                                         tight_layout=True)
    for ax, r_wall, label, side in [
        (ax_in, ri, 'Inner', 'inner'),
        (ax_out, ro, 'Outer', 'outer'),
    ]:
        if side == 'inner':
            mask = r_1d <= ri + zoom_width
            dist = r_1d[mask] - ri
        else:
            mask = r_1d >= ro - zoom_width
            dist = ro - r_1d[mask]

        ax.plot(dist / delta_E, ur_rms[mask], 'r-o', markersize=ms,
                label=r"$u_r'$")
        ax.plot(dist / delta_E, ut_rms[mask], 'b-o', markersize=ms,
                label=r"$u_\theta'$")
        ax.plot(dist / delta_E, up_rms[mask], 'g-o', markersize=ms,
                label=r"$u_\phi'$")
        ax.axvline(1.0, color='k', ls=':', lw=1, alpha=0.5,
                   label=r'$\delta_E$')
        ax.set_xlabel(r'Distance from wall / $\delta_E$')
        ax.set_ylabel('Velocity')
        ax.set_title(f'{label} BL  (Ek={Ek:.0e})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.savefig('ekman_bl_zoom.png', dpi=150)
    plt.close(fig)

    # 4. Fourier spectrum: KE
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    mask = m_full > 0
    ax.loglog(m_full[mask], E_kin_total[mask], 'k-', linewidth=1.5,
              label='integrated')
    colors = plt.cm.viridis(np.linspace(0, 1, len(idx_sel)))
    for ci, i in enumerate(idx_sel):
        ax.loglog(m_full[mask], E_kin[i, mask], color=colors[ci],
                  alpha=0.7, label=f'r={r_1d[i]:.3f}')
    ax.set_xlabel('m (full sphere)')
    ax.set_ylabel(r'$E_{kin}(m)$')
    ax.set_title('KE spectrum (equatorial)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.savefig('fourier_spectrum_ke.png', dpi=150)
    plt.close(fig)

    # 5. Fourier spectrum: temperature
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)
    ax.loglog(m_full[mask], E_T_total[mask], 'k-', linewidth=1.5,
              label='integrated')
    for ci, i in enumerate(idx_sel):
        ax.loglog(m_full[mask], E_T[i, mask], color=colors[ci],
                  alpha=0.7, label=f'r={r_1d[i]:.3f}')
    ax.set_xlabel('m (full sphere)')
    ax.set_ylabel(r'$E_T(m)$')
    ax.set_title('Temperature spectrum (equatorial)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.savefig('fourier_spectrum_temp.png', dpi=150)
    plt.close(fig)

    # 6. Chebyshev spectra (radial)
    if have_cheb:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)

        max_v = np.max(np.abs(vel_cheb))
        min_v = np.abs(vel_cheb[-1]) if vel_cheb[-1] != 0 else 1e-30
        ratio_v = f"{max_v / min_v:.2e}"
        ax1.loglog(N_cheb + 1, np.abs(vel_cheb), 'ko-', markersize=3)
        ax1.set_title('Chebyshev: velocity (radial)')
        ax1.set_xlabel(r'$N_{cheb}$')
        ax1.set_ylabel(r'$|\hat{u}|$')
        ax1.text(0.98, 0.98, f"Max/min = {ratio_v}", transform=ax1.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7))
        ax1.grid(True, alpha=0.3)

        max_t = np.max(np.abs(temp_cheb))
        min_t = np.abs(temp_cheb[-1]) if temp_cheb[-1] != 0 else 1e-30
        ratio_t = f"{max_t / min_t:.2e}"
        ax2.loglog(N_cheb + 1, np.abs(temp_cheb), 'ko-', markersize=3)
        ax2.set_title('Chebyshev: temperature (radial)')
        ax2.set_xlabel(r'$N_{cheb}$')
        ax2.set_ylabel(r'$|\hat{T}|$')
        ax2.text(0.98, 0.98, f"Max/min = {ratio_t}", transform=ax2.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7))
        ax2.grid(True, alpha=0.3)

        fig.savefig('cheb_spectrum_r.png', dpi=150)
        plt.close(fig)

    # 7. Equatorial contour: fluctuating temperature
    R_eq, PHI_eq = np.meshgrid(r_1d, phi_eq_1d, indexing='ij')
    X_eq = R_eq * np.cos(PHI_eq)
    Y_eq = R_eq * np.sin(PHI_eq)

    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    vmax = np.max(np.abs(T_eq_p)) * 0.8
    c1 = ax.pcolormesh(X_eq, Y_eq, T_eq_p, cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, shading='gouraud')
    fig.colorbar(c1, ax=ax, label=r"$T'$")
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Fluctuating temperature (equatorial)")
    fig.savefig('t_prime_equatorial.png', dpi=150)
    plt.close(fig)

    # 8. Equatorial contour: fluctuating velocity magnitude
    speed_eq_p = np.sqrt(ur_eq_p**2 + ut_eq_p**2 + up_eq_p**2)
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
    c1 = ax.pcolormesh(X_eq, Y_eq, speed_eq_p, cmap='inferno',
                        shading='gouraud')
    fig.colorbar(c1, ax=ax, label=r"$|\mathbf{u}'|$")
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Fluctuating velocity (equatorial)")
    fig.savefig('u_prime_equatorial.png', dpi=150)
    plt.close(fig)

    print("\nSaved: mean_velocity_r.png, rms_velocity_r.png, rms_temperature_r.png,")
    print("       ekman_bl_zoom.png, fourier_spectrum_ke.png,")
    print("       fourier_spectrum_temp.png, cheb_spectrum_r.png,")
    print("       t_prime_equatorial.png, u_prime_equatorial.png")
