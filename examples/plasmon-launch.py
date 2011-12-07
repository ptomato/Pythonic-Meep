import numpy as N
import scipy as S
import scipy.constants as Const
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as P
import meep

# Parameters
a = 1e-6  # Length unit used in simulation = 1 micron
res = 50  # Pixels per length unit
size = (7.0, 3.0)  # Size of computational domain
wl = 0.83 # Source wavelength
timesteps = 150

def eV_to_meep_frequency(ev, a):
    """Convert a frequency @ev in eV to Meep units. @a is the length
    unit used in the Meep simulation."""
    freq = ev * Const.eV / Const.hbar
    return freq / (2.0 * N.pi * Const.c / a)

# Create material that simulates gold with dispersion
plasma_freq = 9.03 #eV
f_Au = N.array([0.760, 0.024, 0.010, 0.071, 0.601, 4.384]) # Oscillator strengths
gamma_Au = N.array([0.053, 0.241, 0.345, 0.870, 2.494, 2.214]) # Damping (eV)
omega_Au = N.array([0.000, 0.415, 0.830, 2.969, 4.304, 13.32]) # Resonance (eV)
# Convert those parameters to Meep units
omega_p_norm = eV_to_meep_frequency(plasma_freq, a)
gamma_norm = eV_to_meep_frequency(gamma_Au, a)
omega_norm = eV_to_meep_frequency(omega_Au, a)
omega_norm[0] = 1e-20 # Needs to be a small number but not zero
sigma_norm = f_Au * omega_p_norm ** 2.0 / omega_norm ** 2.0

gold = meep.Dielectric(epsilon=1.0)
for pol in zip(omega_norm, gamma_norm, sigma_norm):
    gold.add_polarizability(*pol)

# Geometry for no-slit case
geometry_noslit = meep.Geometry(size, res)
geometry_noslit.add_polygon(meep.Block(
    center=(0, 0.75),
    size=(N.inf, 1.5),
    material=gold))
geometry_noslit.add_source(meep.ContinuousSource(
    wavelength=wl,
    component='ex',
    size=(6, 0),
    center=(0, -0.5)))
geometry_noslit.add_pml(meep.PML(thickness=0.5))

# Geometry for slit case: the same, but with a slit in the gold
geometry_slit = meep.Geometry(size, res)
geometry_slit.add_polygon(meep.Block(
    center=(0, 0.75),
    size=(N.inf, 1.5),
    material=gold))
geometry_slit.add_polygon(meep.Block(
    center=(0, 0.75),
    size=(0.2, 1.5),
    material=meep.air))
geometry_slit.add_source(meep.ContinuousSource(
    wavelength=wl,
    component='ex',
    size=(6, 0),
    center=(0, -0.5)))
geometry_slit.add_pml(meep.PML(thickness=0.5))

# Run
simulation_noslit = meep.Simulation(geometry_noslit)
while simulation_noslit.time < 10.0:
    simulation_noslit.step()

simulation_slit = meep.Simulation(geometry_slit)
while simulation_slit.time < 10.0:
    simulation_slit.step()

efield = N.array(simulation_slit.efield) - N.array(simulation_noslit.efield)

fig = P.figure()
ax = fig.add_subplot(1, 2, 1)
im = ax.imshow(efield[..., 0].real, cmap=P.cm.RdBu)
fig.colorbar(im)
ax = fig.add_subplot(1, 2, 2)
im = ax.imshow(efield[..., 1].real, cmap=P.cm.RdBu)
fig.colorbar(im)
fig.savefig('slit-scattering.png')
