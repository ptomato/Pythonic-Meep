import numpy as N
import matplotlib.pyplot as P
import meep

res = 10  # points per length unit

# Define the computational grid
geometry = meep.Geometry((16, 8), res)
geometry.add_polygon(meep.Block(center=(0, 0), size=(N.inf, 1),
    material=meep.Dielectric(12)))
geometry.add_source(meep.ContinuousSource(component='Ez',
    frequency=0.15, center=(-7, 0), size=(0, 0)))
geometry.add_pml(meep.PML(1.0))

simulation = meep.Simulation(geometry)

# Obtain epsilon
epsilon = N.array(simulation.epsilon).real

# Run the simulation
while simulation.time < 200.0:
    simulation.step()

# Obtain the Ez field
ez = N.array(simulation.efield)[..., 2].real

# Display epsilon
from scipy.misc import imsave
imsave('../docs/static/tutorial-epsilon.png', -N.rot90(epsilon))

# Pick a nice limit for the color scale
limit = N.abs(ez).max()

# Display the Ez field overlaid with epsilon contours
fig = P.figure()
fig.gca().imshow(N.rot90(ez), vmin=-limit, vmax=limit, cmap=P.cm.RdBu_r)
fig.gca().contour(N.rot90(epsilon), 1, colors='gray', alpha=0.2)
P.show()
