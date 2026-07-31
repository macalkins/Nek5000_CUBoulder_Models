'''
    Write a gmsh .geo file for the Nek5000 rotating-cylinder model.

    The cylinder cross-section is meshed with an O-grid (butterfly): a central
    square core plus four surrounding side blocks that connect the square to the
    circular wall, giving an all-hex, singularity-free mesh after extrusion.

    Wall clustering uses Tilgner (1999) sine stretching, identical to the
    annulus and channel box generators:

        s = sin(beta*pi*xi/2) / sin(beta*pi/2)

    applied in two places so that BOTH the top/bottom Ekman layers and the
    outer Stewartson layer are resolved:

      * axial (z):  TWO-sided Tilgner, clustering at z=0 and z=h  (Ekman layers)
      * radial:     ONE-sided Tilgner, clustering at r=R          (Stewartson layer)

    The azimuthal direction is uniform. Physical parameters (Ek, aspect ratio)
    are read from the .par file, so the mesh always tracks the run parameters.

    Reference:
        Tilgner, A. (1999). "Spectral methods for the simulation of
        incompressible flows in spherical shells." Int. J. Numer. Meth.
        Fluids 30: 713-724.  (eq. 9)

    Usage:
        python make_cylinder_geo.py [Nc] [Nr] [Nz] [beta_z] [beta_r]

        Nc      azimuthal elements per quadrant side      (default 16)
        Nr      radial elements in the outer O-grid ring  (default 10)
        Nz      axial elements                            (default 32)
        beta_z  axial Tilgner parameter (Ekman)           (default 0.9)
        beta_r  radial Tilgner parameter (Stewartson)     (default 0.9)

    Then:
        gmsh -3 cylinder.geo -o cylinder.msh -format msh2 -bin 0
        gmsh2nek                       # -> rrbc_cylinder.re2
        genmap                         # -> rrbc_cylinder.map
'''

import sys
import numpy as np

pi = np.pi

# ------------------------------------------------------------------
# Command-line / defaults
# ------------------------------------------------------------------
Nc     = int(sys.argv[1]) if len(sys.argv) > 1 else 16
Nr     = int(sys.argv[2]) if len(sys.argv) > 2 else 10
Nz     = int(sys.argv[3]) if len(sys.argv) > 3 else 32
beta_z = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
beta_r = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9

PAR_FILE = 'rrbc_cylinder.par'
GEO_FILE = 'cylinder.geo'

# ------------------------------------------------------------------
# Read physical parameters from the .par (userParam02=Ek, 03=aspect)
# ------------------------------------------------------------------
def read_par(parfile):
    vals = {}
    with open(parfile) as f:
        for line in f:
            s = line.lstrip()
            if s.startswith('#') or '=' not in s:
                continue
            key = s.split('=')[0].strip().lower()
            if key.startswith('userparam'):
                rhs = s.split('=')[1].split('#')[0].strip()
                try:
                    vals[key] = float(rhs)
                except ValueError:
                    pass
    return vals

par    = read_par(PAR_FILE)
Ek     = par.get('userparam02', 1.73e-6)   # Ekman number, E = nu/(2 Omega H^2)
Gamma  = par.get('userparam03', 1.0)       # aspect ratio D/H

H = 1.0                                     # non-dimensional height
R = 0.5 * Gamma                             # radius, R = (D/H)/2 with H=1

# ------------------------------------------------------------------
# Boundary-layer thicknesses (non-dimensional, H=1) -- same rules as
# the annulus/channel generators
# ------------------------------------------------------------------
delta_E  = np.sqrt(Ek)          # Ekman layer thickness  ~ sqrt(Ek)
delta_S1 = Ek**(1.0/3.0)        # Stewartson E^{1/3} layer
delta_S2 = Ek**(1.0/4.0)        # Stewartson E^{1/4} layer

# ------------------------------------------------------------------
# Tilgner sine stretching (verbatim from write_box_file_annulus.py)
# ------------------------------------------------------------------
def tilgner_stretch(low, high, N, beta):
    '''N+1 edges between low..high, clustered at BOTH ends (Tilgner 1999).'''
    L_half = 0.5 * (high - low)
    center = 0.5 * (high + low)
    j  = np.arange(N + 1)
    xc = -np.cos(pi * j / N)                    # Chebyshev on [-1,1]
    if beta > 1e-10:
        xs = np.sin(beta * pi * xc / 2) / np.sin(beta * pi / 2)
    else:
        xs = xc
    return center + L_half * xs

def tilgner_onesided(low, high, N, beta):
    '''N+1 edges between low..high, clustered at the HIGH end only.

    Uses the same sine map on u in [0,1]; f(u)=sin(beta*pi*u/2)/sin(beta*pi/2)
    is monotone with small spacing near u=1 -> clustering at r=R.'''
    u = np.arange(N + 1) / N
    if beta > 1e-10:
        s = np.sin(beta * pi * u / 2) / np.sin(beta * pi / 2)
    else:
        s = u
    return low + (high - low) * s

# ------------------------------------------------------------------
# Geometry: butterfly core + concentric outer rings
# ------------------------------------------------------------------
core_frac = 0.35                # square half-width as a fraction of R
a  = core_frac * R              # square half-width
r0 = a * np.sqrt(2.0)           # radius of square corners (level 0)

# radial ring radii: level 0 = square corners, levels 1..Nr Tilgner->R
rho = tilgner_onesided(r0, R, Nr, beta_r)     # length Nr+1, rho[0]=r0, rho[-1]=R

# axial element-edge heights, two-sided Tilgner (Ekman top & bottom)
z_edges = tilgner_stretch(0.0, H, Nz, beta_z) # length Nz+1, 0..H
z_heights = z_edges[1:] / H                   # cumulative normalized tops (Nz)

# corner angles (deg): 45,135,225,315
ang = np.deg2rad([45.0, 135.0, 225.0, 315.0])

# ------------------------------------------------------------------
# Emit .geo
# ------------------------------------------------------------------
L = []
def w(s=''):
    L.append(s)

def P(tag, x, y, z=0.0):
    w(f'Point({tag}) = {{{x:.12f}, {y:.12f}, {z:.12f}}};')

# tag helpers
CENTER = 1
def cpt(k, i):   return 1000 + 10 * k + i          # corner point, level k, corner i
def ccrv(k, i):  return 2000 + 10 * k + i          # circumferential curve, level k, side i
def rcrv(k, i):  return 3000 + 10 * k + i          # radial curve, band k, corner i
def ssurf(k, i): return 5000 + 10 * k + i          # side-block surface, band k, side i
SQ_SURF = 500

w('// Auto-generated by make_cylinder_geo.py -- do not edit by hand.')
w(f'// Ek={Ek:.3e}  aspect(D/H)={Gamma}  R={R}  H={H}')
w(f'// Nc={Nc} Nr={Nr} Nz={Nz}  beta_z={beta_z} beta_r={beta_r}')
w('')
w('// ---- gmsh2nek / Nek settings ----')
w('Mesh.MshFileVersion = 2.2;')
w('Mesh.Binary = 0;')
w('Mesh.RecombineAll = 1;')
w('Mesh.ElementOrder = 2;')
w('Mesh.SecondOrderIncomplete = 1;')
w('')

# center
P(CENTER, 0.0, 0.0)

# corner points, all levels
for k in range(Nr + 1):
    for i in range(4):
        P(cpt(k, i), rho[k] * np.cos(ang[i]), rho[k] * np.sin(ang[i]))
w('')

# --- square core (level 0): straight sides ---
for i in range(4):
    j = (i + 1) % 4
    w(f'Line({ccrv(0, i)}) = {{{cpt(0, i)}, {cpt(0, j)}}};')
sq_sides = [ccrv(0, i) for i in range(4)]
w(f'Curve Loop({SQ_SURF}) = {{{", ".join(str(s) for s in sq_sides)}}};')
w(f'Plane Surface({SQ_SURF}) = {{{SQ_SURF}}};')
w(f'Transfinite Surface{{{SQ_SURF}}} = '
  f'{{{cpt(0,0)}, {cpt(0,1)}, {cpt(0,2)}, {cpt(0,3)}}};')
w('')

# --- outer rings (levels 1..Nr): circle arcs ---
for k in range(1, Nr + 1):
    for i in range(4):
        j = (i + 1) % 4
        w(f'Circle({ccrv(k, i)}) = {{{cpt(k, i)}, {CENTER}, {cpt(k, j)}}};')
w('')

# --- radial connectors between consecutive levels ---
for k in range(1, Nr + 1):
    for i in range(4):
        w(f'Line({rcrv(k, i)}) = {{{cpt(k-1, i)}, {cpt(k, i)}}};')
w('')

# --- side-block surfaces ---
for k in range(1, Nr + 1):
    for i in range(4):
        j = (i + 1) % 4
        loop = ssurf(k, i)
        # a=corner_i(k-1) b=corner_j(k-1) c=corner_j(k) d=corner_i(k)
        w(f'Curve Loop({loop}) = {{{ccrv(k-1, i)}, {rcrv(k, j)}, '
          f'-{ccrv(k, i)}, -{rcrv(k, i)}}};')
        w(f'Plane Surface({loop}) = {{{loop}}};')
        w(f'Transfinite Surface{{{loop}}} = '
          f'{{{cpt(k-1, i)}, {cpt(k-1, j)}, {cpt(k, j)}, {cpt(k, i)}}};')
w('')

# --- transfinite line constraints ---
all_circ = [ccrv(k, i) for k in range(Nr + 1) for i in range(4)]
all_rad  = [rcrv(k, i) for k in range(1, Nr + 1) for i in range(4)]
w(f'Transfinite Line{{{", ".join(str(c) for c in all_circ)}}} = {Nc + 1};')
w(f'Transfinite Line{{{", ".join(str(c) for c in all_rad)}}} = 2;')
w('Recombine Surface{:};')
w('')

# ------------------------------------------------------------------
# Axial extrusion with explicit Tilgner layer heights
# ------------------------------------------------------------------
nums    = ", ".join('1' for _ in range(Nz))
heights = ", ".join(f'{h:.12f}' for h in z_heights)
w('// Two-sided Tilgner layer distribution in z (Ekman layers top & bottom)')
w('base_surfaces[] = Surface{:};')
w(f'out[] = Extrude {{0, 0, {H}}} {{ Surface{{base_surfaces[]}}; '
  f'Layers{{ {{{nums}}}, {{{heights}}} }}; Recombine; }};')
w('')

# ------------------------------------------------------------------
# Physical groups for gmsh2nek
#   1 = WALL (insulating sidewall), 2 = WALL_BOTTOM, 3 = WALL_TOP
# ------------------------------------------------------------------
eps = 1e-6
w('eps = 1e-6;')
w(f'Physical Surface("WALL_BOTTOM", 2) = '
  f'Surface In BoundingBox{{{-R-eps}, {-R-eps}, {-eps}, {R+eps}, {R+eps}, {eps}}};')
w(f'Physical Surface("WALL_TOP", 3) = '
  f'Surface In BoundingBox{{{-R-eps}, {-R-eps}, {H-eps}, {R+eps}, {R+eps}, {H+eps}}};')
w('all_skin[] = CombinedBoundary{ Volume{:}; };')
w('Physical Surface("WALL", 1) = {all_skin[]};')
w('Physical Surface(1) -= { Physical Surface{2}, Physical Surface{3} };')
w('Physical Volume("FLUID", 1) = Volume{:};')
w('')
w('Mesh 3;')
w(f'Save "cylinder.msh";')

with open(GEO_FILE, 'w') as f:
    f.write("\n".join(L) + "\n")

# ------------------------------------------------------------------
# Diagnostics (GLL resolution within the boundary layers, lx1=8)
# ------------------------------------------------------------------
from numpy.polynomial.legendre import legder
def gll(n):
    c = np.zeros(n); c[-1] = 1
    roots = np.polynomial.legendre.legroots(legder(c))
    return np.sort(np.concatenate(([-1.0], roots, [1.0])))
xi = gll(8)

def gll_in_layer(edges, wall, thickness):
    pts = []
    for i in range(len(edges) - 1):
        aa, bb = edges[i], edges[i + 1]
        pts.extend((0.5 * (bb - aa) * (xi + 1) + aa).tolist())
    pts = np.unique(np.array(pts))
    return int(np.sum(np.abs(pts - wall) < thickness))

n_elem = (Nc * Nc + 4 * Nc * Nr) * Nz
dz = np.diff(z_edges)
dr = np.diff(rho)

print("=" * 64)
print(f"  Wrote {GEO_FILE}")
print(f"  Ek = {Ek:.3e}   aspect(D/H) = {Gamma}   R = {R}   H = {H}")
print(f"  Nc={Nc} Nr={Nr} Nz={Nz}   beta_z={beta_z} beta_r={beta_r}")
print(f"  Total elements: {n_elem}   (2D: {Nc*Nc + 4*Nc*Nr}  x  Nz={Nz})")
print("-" * 64)
print(f"  Z (Ekman):      delta_E  = sqrt(Ek) = {delta_E:.6f}")
print(f"                  dz_min   = {dz.min():.6f} ({dz.min()/delta_E:.2f} delta_E)")
print(f"                  dz_max/dz_min = {dz.max()/dz.min():.1f}")
print(f"                  GLL pts within delta_E of each plate: "
      f"{gll_in_layer(z_edges, 0.0, delta_E)}")
print(f"                  GLL pts within 5*delta_E:             "
      f"{gll_in_layer(z_edges, 0.0, 5*delta_E)}")
print("-" * 64)
print(f"  Radial (Stewartson): delta_S1 = E^1/3 = {delta_S1:.6f}")
print(f"                       delta_S2 = E^1/4 = {delta_S2:.6f}")
print(f"                       dr_min = {dr.min():.6f} ({dr.min()/delta_S1:.2f} delta_S1)")
print(f"                       dr_max/dr_min = {dr.max()/dr.min():.1f}")
print(f"                       GLL pts within delta_S1 of wall: "
      f"{gll_in_layer(rho, R, delta_S1)}")
print(f"                       GLL pts within delta_S2 of wall: "
      f"{gll_in_layer(rho, R, delta_S2)}")
print("=" * 64)
print("Next:  gmsh -3 cylinder.geo -o cylinder.msh -format msh2 -bin 0")
print("       gmsh2nek   &&   genmap")
