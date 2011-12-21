Dispersion and Plasmonics
=========================

.. |epsilon| replace:: *ε*
.. |gamma| replace:: *Γ*
.. |omega| replace:: *ω*

This tutorial explains how to use Meep to simulate dispersive materials like metals, in order to demonstrate surface plasmon polaritons.
There is little help available on how to do it with the regular Meep interface, but the Python interface makes it easy.

Make sure to import the necessary packages at the top of your script::

    import numpy as N
    import scipy as S
    import scipy.constants as Const
    import matplotlib.pyplot as P
    import meep

A dispersive material
---------------------

In this tutorial, we will examine a plane wave incident on a gold surface.
If there is a corrugation, such as a slit, in the surface, then there will be a small amount of scattering into a surface plasmon mode.
By simulating the system both with and without a slit, we can show the difference between the two cases.

We start out by defining some parameters for our simulation::

    a = 1e-6           # Length unit used in simulation = 1 micron
    res = 50           # Pixels per length unit
    size = (7.0, 3.0)  # Size of computational domain
    wl = 0.83          # Source wavelength

We will define our own :class:`~meep.Material` to represent gold.
The dielectric function of gold can be approximated quite faithfully for energies of 0.1 to 6 eV by Drude-Lorentz resonances [#Rakic1998]_.
These energies correspond to wavelengths of about 200 to 1200 nm.
We will illuminate our gold surface with a wavelength of 830 nm, as shown above.

The Drude-Lorentz model in the abovementioned paper is given as a sum of resonances:

.. math:: \epsilon(\omega) = 1 - \frac{f_0 \omega_p^2}{\omega(\omega - i\Gamma_0)}
    + \sum_{j=1}^k \frac{f_j \omega_p^2}{(\omega_j^2 - \omega^2) + i\omega\Gamma_j}

The first term is the dielectric function far away from the resonances, i.e. at high frequency (:math:`\epsilon_\infty = 1`).
The second is the Drude conductivity model, and the following terms are the additional resonances.
The plasma frequency of the metal is :math:`\omega_p`, *f* is the resonance strength, |gamma| is the damping frequency of the resonance, and :math:`\omega_j` is the frequency at which the resonance occurs.
This can be consolidated into an expression more similar to Meep's internal model of dispersion, if we take :math:`\omega_0 = 0`:

.. math:: \epsilon(\omega) = 1 + \sum_{j=0}^k \frac{f_j \omega_p^2}{\omega_j^2 - \omega(\omega - i\Gamma_j)}

::

    plasma_freq = 9.03 #eV
    # Oscillator strengths
    f_Au = N.array([0.760, 0.024, 0.010, 0.071, 0.601, 4.384])
    # Damping frequency (eV)
    gamma_Au = N.array([0.053, 0.241, 0.345, 0.870, 2.494, 2.214])
    # Resonance frequency (eV) 
    omega_Au = N.array([0.000, 0.415, 0.830, 2.969, 4.304, 13.32])

.. [#Rakic1998] Rakić, A. D., Djurišić, A. B., Elazar, J. M., & Majewski, M. L. (1998). Optical properties of metallic films for vertical-cavity optoelectronic devices. *Applied Optics 37* (22), pp. 5271–5283. *This paper contains good approximations for the dielectric functions of several metals, including gold and silver.*