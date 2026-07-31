#!/usr/bin/env python3
"""
Plot Nek5000 spherical shell fields from shell0.f* files.

Produces three sets of plots:
  1. Equatorial plane (z=0): temperature, u_r, u_phi
  2. Meridional plane (y=0): temperature, u_r, u_y, u_z
  3. Azimuthally averaged helicity in the meridional (s, z) plane

Usage:
    python plot_shell.py [field_file]
    python plot_shell.py shell0.f00010

If no file given, reads the latest shell0.f* in the current directory.

Requires: pymech, numpy, matplotlib, scipy
"""
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pymech


def find_latest_field():
    files = sorted(glob.glob("shell0.f*"))
    files = [f for f in files if 'restart' not in f]
    if not files:
        print("No shell0.f* files found.")
        sys.exit(1)
    return files[-1]


def extract_slice(d, axis, value=0.0):
    lx, ly, lz = d.lr1
    arr_axis = {0: 2, 1: 1, 2: 0}[axis]
    n_along = [lz, ly, lx][arr_axis]

    cols = {k: [] for k in ['x', 'y', 'z', 'vx', 'vy', 'vz']}
    temp_list = []
    pres_list = []

    for e in d.elem:
        coords = e.pos[axis]
        if arr_axis == 0:
            vals = coords[:, 0, 0]
        elif arr_axis == 1:
            vals = coords[0, :, 0]
        else:
            vals = coords[0, 0, :]

        idx = np.argmin(np.abs(vals - value))
        spacing = np.abs(vals[1] - vals[0]) if n_along > 1 else 1.0
        if np.abs(vals[idx] - value) > 0.6 * spacing * n_along:
            continue

        def sl(arr):
            if arr_axis == 0:
                return arr[idx, :, :].ravel()
            elif arr_axis == 1:
                return arr[:, idx, :].ravel()
            else:
                return arr[:, :, idx].ravel()

        cols['x'].append(sl(e.pos[0]))
        cols['y'].append(sl(e.pos[1]))
        cols['z'].append(sl(e.pos[2]))
        cols['vx'].append(sl(e.vel[0]))
        cols['vy'].append(sl(e.vel[1]))
        cols['vz'].append(sl(e.vel[2]))
        if len(e.temp) > 0:
            temp_list.append(sl(e.temp[0]))
        if len(e.pres) > 0:
            pres_list.append(sl(e.pres[0]))

    result = {k: np.concatenate(v) for k, v in cols.items()}
    if temp_list:
        result['temp'] = np.concatenate(temp_list)
    if pres_list:
        result['pres'] = np.concatenate(pres_list)
    return result


def add_boundaries(ax, ri, ro):
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(ri*np.cos(theta), ri*np.sin(theta), 'k-', lw=0.8, alpha=0.5)
    ax.plot(ro*np.cos(theta), ro*np.sin(theta), 'k-', lw=0.8, alpha=0.5)


def plot_field(h, v, data, title, cbar_label, filename,
               cmap='RdBu_r', symmetric=True, ri=None, ro=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    if symmetric:
        lim = np.nanpercentile(np.abs(data), 99)
        if lim == 0:
            lim = 1e-12
        vmin, vmax = -lim, lim
    else:
        vmin = np.nanpercentile(data, 1)
        vmax = np.nanpercentile(data, 99)

    triang = tri.Triangulation(h, v)
    cntr = ax.tricontourf(triang, data, levels=60,
                          cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cntr, ax=ax, label=cbar_label, shrink=0.8)

    if ri is not None and ro is not None:
        add_boundaries(ax, ri, ro)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def _collect_gll(d):
    """Return all GLL point coordinates and velocities as flat arrays."""
    x_all, y_all, z_all, vx_all, vy_all, vz_all = [], [], [], [], [], []
    for e in d.elem:
        x_all.append(e.pos[0].ravel()); y_all.append(e.pos[1].ravel()); z_all.append(e.pos[2].ravel())
        vx_all.append(e.vel[0].ravel()); vy_all.append(e.vel[1].ravel()); vz_all.append(e.vel[2].ravel())
    return (np.concatenate(x_all), np.concatenate(y_all), np.concatenate(z_all),
            np.concatenate(vx_all), np.concatenate(vy_all), np.concatenate(vz_all))


def helicity_meridional_slice(d, phi=0.0, nx_g=200, nz_g=300):
    """
    Compute helicity H = u . curl(u) on a meridional slice at azimuthal angle phi (radians).
    NOTE: missing d/dy terms mean wx and wz are approximate; wy = dvx/dz - dvz/dx is exact.
    Use azimuthal_mean_helicity for a fully correct result.

    Returns: XI, ZI, H (2D arrays on regular grid), ri, ro
    """
    from scipy.interpolate import griddata
    x, y, z, vx, vy, vz = _collect_gll(d)
    r = np.sqrt(x**2 + y**2 + z**2)
    ri, ro = r.min(), r.max()

    cp, sp = np.cos(phi), np.sin(phi)
    dist = -x*sp + y*cp
    tol  = ro / nx_g
    mask = np.abs(dist) < tol

    s_plane = x[mask]*cp + y[mask]*sp
    zm      = z[mask]
    pts = np.column_stack([s_plane, zm])

    xi = np.linspace(-ro, ro, nx_g)
    zi = np.linspace(-ro, ro, nz_g)
    XI, ZI = np.meshgrid(xi, zi, indexing='ij')

    def interp(v):
        return griddata(pts, v[mask], (XI, ZI), method='linear')

    VX = interp(vx); VY = interp(vy); VZ = interp(vz)
    dx = xi[1] - xi[0]; dz = zi[1] - zi[0]

    WX = -np.gradient(VY, dz, axis=1)                               # approximate (no d/dy)
    WY =  np.gradient(VX, dz, axis=1) - np.gradient(VZ, dx, axis=0) # exact in-plane
    WZ =  np.gradient(VY, dx, axis=0)                               # approximate (no d/dy)

    H = VX*WX + VY*WY + VZ*WZ
    RG = np.sqrt(XI**2 + ZI**2)
    H = np.where((RG > ri*0.98) & (RG < ro*1.02), H, np.nan)
    return XI, ZI, H, ri, ro


def azimuthal_mean_helicity(d, ngrid=80):
    """
    Compute azimuthally averaged kinetic helicity H = <u . curl(u)>_phi
    in the meridional (s, z) plane.

    Interpolates all GLL points to a 3D Cartesian grid, computes the full
    3D vorticity via np.gradient (all derivatives correct), forms H = u.curl(u),
    then bins by (s, z) for the azimuthal average.

    ngrid: resolution of the 3D Cartesian grid (ngrid^3 points)

    Returns: SG, ZG, H (2D arrays, shape (ngrid//2, ngrid))
    """
    from scipy.interpolate import NearestNDInterpolator

    x, y, z, vx, vy, vz = _collect_gll(d)
    r = np.sqrt(x**2 + y**2 + z**2)
    ri, ro = r.min(), r.max()

    print(f"  Building 3D interpolator ({len(x):.2e} points) ...")
    pts = np.column_stack([x, y, z])
    interp_v = NearestNDInterpolator(pts, np.column_stack([vx, vy, vz]))

    # 3D Cartesian target grid
    g = np.linspace(-ro, ro, ngrid)
    XG, YG, ZG3 = np.meshgrid(g, g, g, indexing='ij')
    RG3 = np.sqrt(XG**2 + YG**2 + ZG3**2)
    shell = (RG3 > ri*0.98) & (RG3 < ro*1.02)

    # Interpolate only shell points (saves time and memory)
    print(f"  Interpolating to {shell.sum()} shell grid points ...")
    target_pts = np.column_stack([XG[shell], YG[shell], ZG3[shell]])
    vel = interp_v(target_pts)

    nan3 = np.full((ngrid, ngrid, ngrid), np.nan)
    VX3 = nan3.copy(); VY3 = nan3.copy(); VZ3 = nan3.copy()
    VX3[shell] = vel[:, 0]; VY3[shell] = vel[:, 1]; VZ3[shell] = vel[:, 2]

    # Full 3D vorticity: zero NaN for gradient, then re-mask
    def f(A):
        return np.where(np.isfinite(A), A, 0.0)

    dg = g[1] - g[0]
    WX = np.gradient(f(VZ3), dg, axis=1) - np.gradient(f(VY3), dg, axis=2)
    WY = np.gradient(f(VX3), dg, axis=2) - np.gradient(f(VZ3), dg, axis=0)
    WZ = np.gradient(f(VY3), dg, axis=0) - np.gradient(f(VX3), dg, axis=1)

    H3 = np.where(shell, VX3*WX + VY3*WY + VZ3*WZ, np.nan)

    # Azimuthal average: bin by (s, z)
    SG3 = np.sqrt(XG**2 + YG**2)
    ns_out = ngrid // 2
    s_edges = np.linspace(0,   ro, ns_out + 1)
    z_edges = np.linspace(-ro, ro, ngrid  + 1)
    s_cen   = 0.5 * (s_edges[:-1] + s_edges[1:])
    z_cen   = 0.5 * (z_edges[:-1] + z_edges[1:])

    valid = np.isfinite(H3)
    num, _, _ = np.histogram2d(SG3[valid].ravel(), ZG3[valid].ravel(),
                               bins=[s_edges, z_edges], weights=H3[valid].ravel())
    cnt, _, _ = np.histogram2d(SG3[valid].ravel(), ZG3[valid].ravel(),
                               bins=[s_edges, z_edges])
    H_mean = np.where(cnt > 0, num / np.where(cnt > 0, cnt, 1), np.nan)

    SG, ZG = np.meshgrid(s_cen, z_cen, indexing='ij')
    return SG, ZG, H_mean


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else find_latest_field()

    print(f"Reading {fname} ...")
    d = pymech.readnek(fname)
    print(f"  nel={d.nel}, lx1={d.lr1[0]}, time={d.time:.4e}")

    r_all = np.array([np.sqrt(e.pos[0]**2 + e.pos[1]**2 + e.pos[2]**2).min()
                      for e in d.elem])
    ri = r_all.min()
    ro = max(np.sqrt(e.pos[0]**2 + e.pos[1]**2 + e.pos[2]**2).max()
             for e in d.elem)
    print(f"  ri={ri:.4f}, ro={ro:.4f}")

    ts = f"{d.time:.4e}"

    # ------------------------------------------------------------------
    # 1. Equatorial plane (z=0)
    # ------------------------------------------------------------------
    print("Extracting equatorial plane (z=0)...")
    eq = extract_slice(d, axis=2, value=0.0)
    print(f"  {len(eq['x'])} points")

    if len(eq['x']) > 0:
        x, y = eq['x'], eq['y']
        r_eq = np.sqrt(x**2 + y**2)
        r_eq = np.where(r_eq < 1e-14, 1e-14, r_eq)

        ur  = (eq['vx']*x + eq['vy']*y) / r_eq
        uph = (-eq['vx']*y + eq['vy']*x) / r_eq

        if 'temp' in eq:
            plot_field(x, y, eq['temp'],
                       f"Temperature — equatorial plane\nt = {ts}",
                       'T', f"eq_T_{ts}.png",
                       cmap='RdYlBu_r', symmetric=False, ri=ri, ro=ro)

        plot_field(x, y, ur,
                   f"Radial velocity $u_r$ — equatorial plane\nt = {ts}",
                   '$u_r$', f"eq_ur_{ts}.png",
                   ri=ri, ro=ro)

        plot_field(x, y, uph,
                   f"Azimuthal velocity $u_\\phi$ — equatorial plane\nt = {ts}",
                   '$u_\\phi$', f"eq_uphi_{ts}.png",
                   ri=ri, ro=ro)

    # ------------------------------------------------------------------
    # 2. Meridional plane (y=0)
    # ------------------------------------------------------------------
    print("Extracting meridional plane (y=0)...")
    mr = extract_slice(d, axis=1, value=0.0)
    print(f"  {len(mr['x'])} points")

    if len(mr['x']) > 0:
        x, z = mr['x'], mr['z']

        if 'temp' in mr:
            plot_field(x, z, mr['temp'],
                       f"Temperature — meridional plane\nt = {ts}",
                       'T', f"mer_T_{ts}.png",
                       cmap='RdYlBu_r', symmetric=False, ri=ri, ro=ro)

        r_mr = np.sqrt(x**2 + z**2)
        r_mr = np.where(r_mr < 1e-14, 1e-14, r_mr)
        ur_mr = (mr['vx']*x + mr['vz']*z) / r_mr

        plot_field(x, z, ur_mr,
                   f"Radial velocity $u_r$ — meridional plane\nt = {ts}",
                   '$u_r$', f"mer_ur_{ts}.png",
                   ri=ri, ro=ro)

        plot_field(x, z, mr['vy'],
                   f"Azimuthal velocity $u_y$ — meridional plane\nt = {ts}",
                   '$u_y$', f"mer_uy_{ts}.png",
                   ri=ri, ro=ro)

        plot_field(x, z, mr['vz'],
                   f"Axial velocity $u_z$ — meridional plane\nt = {ts}",
                   '$u_z$', f"mer_uz_{ts}.png",
                   ri=ri, ro=ro)

    # ------------------------------------------------------------------
    # 3. Helicity in the meridional plane (y=0, no averaging)
    # ------------------------------------------------------------------
    print("Computing helicity in meridional slice (phi=0)...")
    XI, ZI, H_mer, ri_h, ro_h = helicity_meridional_slice(d, phi=0.0)
    if np.any(np.isfinite(H_mer)):
        fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
        lim = np.nanpercentile(np.abs(H_mer[np.isfinite(H_mer)]), 99)
        if lim == 0:
            lim = 1e-12
        cntr = ax.contourf(XI, ZI, H_mer, levels=60, cmap='RdBu_r', vmin=-lim, vmax=lim)
        fig.colorbar(cntr, ax=ax, label=r'$\mathbf{u} \cdot \boldsymbol{\omega}$', shrink=0.8)
        add_boundaries(ax, ri_h, ro_h)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'Helicity — meridional slice (phi=0)\nt = {ts}', fontsize=10)
        hmer_file = f"helicity_mer_{ts}.png"
        fig.savefig(hmer_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {hmer_file}")

    # ------------------------------------------------------------------
    # 4. Azimuthally averaged helicity in the meridional plane
    # ------------------------------------------------------------------
    print("Computing azimuthally averaged helicity (all phi)...")
    SG, ZG, H = azimuthal_mean_helicity(d)
    print(f"  H range: [{np.nanmin(H):.3e}, {np.nanmax(H):.3e}]")

    fig, ax = plt.subplots(figsize=(5, 8), tight_layout=True)
    lim = np.nanpercentile(np.abs(H[np.isfinite(H)]), 99) if np.any(np.isfinite(H)) else 1e-12
    if lim == 0:
        lim = 1e-12
    cntr = ax.contourf(SG, ZG, H, levels=60, cmap='RdBu_r', vmin=-lim, vmax=lim)
    fig.colorbar(cntr, ax=ax,
                 label=r'$\langle \mathbf{u} \cdot \boldsymbol{\omega} \rangle_\phi$',
                 shrink=0.8)
    theta = np.linspace(0, np.pi, 300)
    ax.plot(ri*np.sin(theta), ri*np.cos(theta), 'k-', lw=0.8, alpha=0.5)
    ax.plot(ro*np.sin(theta), ro*np.cos(theta), 'k-', lw=0.8, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('s')
    ax.set_ylabel('z')
    ax.set_title(f'Azimuthal mean helicity — meridional plane\nt = {ts}', fontsize=10)
    hfile = f"helicity_{ts}.png"
    fig.savefig(hfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {hfile}")


if __name__ == "__main__":
    main()
