import numpy as np

from math import exp, sqrt

from ase.calculators.calculator import Calculator


class MorsePotential(Calculator):
    """Morse potential.

    Default values chosen to be similar as Lennard-Jones.
    """

    implemented_properties = ['energy', 'forces']
    default_parameters = {'epsilon': 1.0,
                          'rho0': 6.0,
                          'r0': 1.0}
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges','magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = self.parameters.epsilon
        rho0 = self.parameters.rho0
        r0 = self.parameters.r0
        positions = self.atoms.get_positions()
        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        preF = 2 * epsilon * rho0 / r0
        for i1, p1 in enumerate(positions):
            for i2, p2 in enumerate(positions[:i1]):
                diff = p2 - p1
                r = sqrt(np.dot(diff, diff))
                expf = exp(rho0 * (1.0 - r / r0))
                energy += epsilon * expf * (expf - 2)
                F = preF * expf * (expf - 1) * diff / r
                forces[i1] -= F
                forces[i2] += F
        self.results['energy'] = energy
        self.results['forces'] = forces

class MorsePotential2:
    """Morse potential2.

    Default values chosen to be similar as Lennard-Jones.
    """
    def __init__(self, a=6.0, D=1.0, r0=1.0):
        self.D = D
        self.a = a
        self.r0 = r0
        self.positions = None

    def update(self, atoms):
        assert not atoms.get_pbc().any()
        if (self.positions is None or
            (self.positions != atoms.get_positions()).any()):
            self.calculate(atoms)

    def get_potential_energy(self, atoms):
        self.update(atoms)
        return self.energy

    def get_forces(self, atoms):
        self.update(atoms)
        return self._forces

    def get_stress(self, atoms):
        return np.zeros((3, 3))

    def calculate(self, atoms):
        positions = atoms.get_positions()
        self.energy = 0.0
        F = 0.0
        self._forces = np.zeros((len(atoms), 3))
        if self.a > 0:
            r = positions[1][0]
        else:
            r = -positions[1][0]
        expf = exp(- self.a * (r - self.r0))
        self.energy += self.D * (1 - expf)**2
        F =  -2 * self.a * self.energy * expf * np.sign(r)
        self._forces[1][0] = F
        self.positions = positions.copy()

        #positions = atoms.get_positions()
        #self.energy = 0.0
        #F = 0.0
        #self._forces = np.zeros((len(atoms), 3))
        #for i1, p1 in enumerate(positions):
            #for i2, p2 in enumerate(positions[:i1]):
                #diff = p2 - p1
                #r = sqrt(np.dot(diff, diff))
                #expf = exp(- self.a * (r - self.r0))
                #self.energy += self.D * (1 - expf)**2
                #F =  -2 * self.a * self.energy * expf * (diff / r)
                #self._forces[i1] -= F
                #self._forces[i2] += F
        #self.positions = positions.copy()
