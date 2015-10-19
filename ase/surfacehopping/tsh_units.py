import numpy as np

from ase import units


hbar = units._hbar * units.J * units.s
coulombs = 1/(4*np.pi*units._eps0)
coulombs = coulombs*units.kg*units.m*units.m*units.m \
           /units.s/units.s/units.C/units.C
c = units._c*units.m/units.s
#hbar = 1.0
