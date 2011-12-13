Meep Tutorial
=============

.. |epsilon| replace:: *ε*
.. |times| replace:: ×
.. |infinity| replace:: ∞

This is taken from the first tutorial on the Meep website.
You'll see that the Pythonic interface to Meep allows easy access to and manipulation of the simulation data, while being less verbose and easier to read.

Make sure to import the packages you need at the top of your Python script. In most cases, that is Numpy and Meep::

    import numpy as N
    import meep

The Meep tutorial examines the field pattern excited by a localized CW source in a waveguide.
Our waveguide will have (non-dispersive) |epsilon| = 12 and width 1.
That is, we pick units of length so that the width is 1, and define everything in terms of that.

A straight waveguide
--------------------

First we have to define the computational cell.
We're going to put a source at one end and watch it propagate down the waveguide in the *x* direction, so let's use a cell of length 16 in the *x* direction to give it some distance to propagate.
In the *y* direction, we just need enough room so that the boundaries (below) don't affect the waveguide mode; let's give it a size of 8.

Meep will discretize this structure in space and time, and that is specified by
a single parameter, the resolution, that gives the number of pixels per distance unit.
We'll set this resolution to 10, which corresponds to around 67 pixels/wavelength, or around 20 pixels/wavelength in the high-dielectric material.
(In general, at least 8 pixels/wavelength in the highest dielectric is a good idea.)
This will give us a 160 |times| 80 cell.

We now create a :class:`~meep.Geometry` using these sizes in our Python script::

    geometry = meep.Geometry(size=(16, 8), resolution=10)

By specifying a size vector with a length of 2, we have also specified that the
simulation is to be two-dimensional.

Now, we can add the waveguide.
Most commonly, the structure is specified by adding :class:`~meep.Polygon` objects to the geometry object.
Here, we do::

    geometry.add_polygon(meep.Block(center=(0, 0), size=(N.inf, 1),
        material=meep.Dielectric(12)))

.. figure:: static/tutorial-epsilon.png
   :figwidth: 160px
   :align: right
   :target: _images/tutorial-epsilon.png

   Dielectric function (black = high, white = air), for straight waveguide simulation.

The waveguide is specified by a :class:`~meep.Block` (rectangle) of size |infinity| |times| 1, centered at (0, 0) (the center of the computational cell).
It is made of a :class:`~meep.Dielectric` with |epsilon| = 12.
By default, any place where there are no objects there is air (|epsilon| = 1). The resulting structure is shown at right.

.. , although this can be changed by setting the default-material variable.

Now that we have the structure, we need to specify the current sources, which are represented by various subclasses of :class:`~meep.Source`.
The simplest thing is to add a point source :math:`J_z`::

    geometry.add_source(meep.ContinuousSource(component='Ez',
        frequency=0.15, center=(-7, 0), size=(0, 0)))

Here, we gave the source a frequency of 0.15, and specified a :class:`~meep.ContinuousSource` which is just a fixed-frequency sinusoid :math:`e^{-i\omega t}` that (by default) is turned on at *t* = 0.
Recall that, in Meep units, frequency is specified in units of :math:`2\pi c`, which is equivalent to the inverse of vacuum wavelength. Thus, 0.15 corresponds to a vacuum wavelength of about 1 / 0.15 = 6.67, or a wavelength of about 2 in the |epsilon| = 12 material — thus, our waveguide is half a wavelength wide, which should hopefully make it single-mode.
(In fact, the cutoff for single-mode behavior in this waveguide is analytically solvable, and corresponds to a frequency of 1/2√11 or roughly 0.15076.) Note also that to specify a :math:`J_z`, we specify a component ``Ez``.
The current is located at (−7, 0), which is 1 unit to the right of the left edge of the cell — we always want to leave a little space between sources and the cell boundaries, to keep the boundary conditions from interfering with them.

.. (e.g. if we wanted a magnetic current, we would specify ``Hx``, ``Hy``, or ``Hz``).

Speaking of boundary conditions, we want to add absorbing boundaries around our cell.
Absorbing boundaries in Meep are handled by perfectly matched layers (:class:`~meep.PML`) — which aren't really a boundary condition at all, but rather a fictitious absorbing material added around the edges of the cell.
To add an absorbing layer of thickness 1 around all sides of the cell, we do::

    geometry.add_pml(meep.PML(1.0))

.. You may have more than one pml object if you want PML layers only on certain sides of the cell, e.g. (make pml (thickness 1.0) (direction X) (side High)) specifies a PML layer on only the + x side.

Now, we note an important point: the PML layer is *inside* the cell, overlapping whatever objects you have there.
So, in this case our PML overlaps our waveguide, which is what we want so that it will properly absorb waveguide modes.
The finite thickness of the PML is important to reduce numerical reflections.

.. see perfectly matched layers for more information.

Now, we are ready to run the simulation!
We first create a :class:`~meep.Simulation` object::

    simulation = meep.Simulation(geometry)

Then, we can obtain the dielectric function |epsilon| from the simulation.
It doesn't change during the simulation, of course, so we may as well retrieve it before the simulation starts::

    epsilon = N.array(simulation.epsilon).real

Then, we run the simulation in a while-loop using the :meth:`~meep.Simulation.step` method::

    while simulation.time < 200.0:
        simulation.step()

It should complete in a few seconds.

Outputting PNG images with SciPy
--------------------------------

SciPy provides a quick, rudimentary way to output data to a PNG image.
We can visualize the dielectric function as a grayscale image like so::

    from scipy.misc import imsave
    imsave('epsilon.png', -N.rot90(epsilon))

The minus sign is to reverse the color scale, so that low values are white and high values are black.
In fact, precisely this command is what created the dielectric image above.

Visualizing data with Matplotlib
--------------------------------

Much more interesting, however, are the fields.
We would also like a bit more flexibility in our visualization than just outputting a grayscale PNG file.
Matplotlib is an ideal tool with which to visualize the fields.
There are several ways to import it, but here we will use::

    from matplotlib import pyplot as P

We store the electric field in an array, and get the maximum of its absolute value which will serve as limits for the color scale in our plot::

    ez = N.array(simulation.efield)[..., 2].real
    limit = N.abs(ez).max()

Note that :attr:`~meep.Simulation.efield` is a 160 |times| 80 |times| 3 array, where the third dimension is the *x*, *y*, and *z* components.

Then we plot::

    fig = P.figure()
    fig.gca().imshow(N.rot90(ez), vmin=-limit, vmax=limit, cmap=P.cm.RdBu_r)
    fig.gca().contour(N.rot90(epsilon), 1, colors='gray', alpha=0.2)
    P.show()

Briefly, the ``cmap=P.cm.RdBu_r`` makes the color scale go from dark blue (negative) to white (zero) to dark red (positive), the ``vmin`` and ``vmax`` arguments make sure the color scale is symmetric, and the call to :func:`~matplotlib.Axes.contour` overlays the dielectric function as light gray contours. This results in the image:

.. image:: static/tutorial-ez.png
   :align: center
   :width: 500px
   :target: _images/tutorial-ez.png

Here, we see that the the source has excited the waveguide mode, but has also excited radiating fields propagating away from the waveguide. At the boundaries, the field quickly goes to zero due to the PML layers. If we look carefully (click on the image to see a larger view), we see something else — the image is "speckled" towards the right side. This is because, by turning on the current abruptly at *t* = 0, we have excited high-frequency components (very high order modes), and we have not waited long enough for them to die away; we'll eliminate these in the next section by turning on the source more smoothly.
