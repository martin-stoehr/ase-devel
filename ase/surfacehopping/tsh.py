"""Surface Hopping MD"""

from copy import deepcopy
import warnings
import numpy as np
#from math import factorial as fact

from ase.optimize.optimize import Dynamics
from ase.data import atomic_masses
from ase.md.logger import MDLogger

class SurfaceHoppingMD(Dynamics):
    """Base-class for all Surface Hopping MD classes.

       SurfaceHoppingMD(atoms,...)

       atoms : Atoms object
           The Atoms object under study having been assigned the groundstate calculator
       timestep : int
           Time step for the integration of the EOM
       NAC_method : str
           Method of choice for the non-adiabatic coupling elements
           Options:
                  otf : on-the-fly calculation of non-adiabatic couplings.
                        Supplied by the calculator

                  precalc : necessitates a precalculated form of the NACs

    """
    def __init__(self, atoms, timestep, trajectory, logfile=None,
                 loginterval=1, nstates=1):
        Dynamics.__init__(self, atoms, logfile=None, trajectory=trajectory)

        #NAC_types = ['otf', 'precalc']

        self.dt = timestep
        self.masses = self.atoms.get_masses()
        #number of elec. states
        self.nstates = nstates
        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')
        self.masses.shape = (-1, 1)
        if logfile:
            self.attach(MDLogger(dyn=self, atoms=atoms, logfile=logfile),
                        interval=loginterval)

    def run(self, steps=50):
        """Integrate equation of motion."""

        #calculate relevant initial properties
        if self.nstates>1:
            e = self.atoms.get_potential_energy()[:self.nstates]
            f = self.atoms.get_forces()[:,:,:self.nstates]
            d = self.atoms.calc.get_couplings(self.atoms)[:,:,:fact(self.nstates)]
        else:
            e = [self.atoms.get_potential_energy()]
            f = self.atoms.get_forces()
            d = self.atoms.calc.get_couplings(self.atoms)

    def get_time(self):
        return self.nsteps * self.dt
