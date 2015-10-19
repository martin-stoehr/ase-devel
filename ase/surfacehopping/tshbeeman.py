import numpy as np
import ase.units as units
from ase.surfacehopping.tshverlet import SurfaceHoppingASEVerlet
from ase.surfacehopping.tsh_units import hbar

#NOT YET MODIFIED FROM VelocityVerlet

class SurfaceHoppingBeeman(SurfaceHoppingASEVerlet):
    """VelocityVerlet with Tully's surface hopping fewest switches method"""

    def __init__(self, atoms, dt, dmsteps=None, nstates=1, init_densmat=None ,
                 NAC_method=None, trajectory=None, logfile=None,
                 loginterval=1):
        SurfaceHoppingASEVerlet.__init__(self, atoms, dt=dt, dmsteps=dmsteps, nstates=nstates,
                 init_densmat=init_densmat ,NAC_method=NAC_method, 
                 trajectory=trajectory, logfile=logfile, loginterval=loginterval)
        
        self.f_very_old = None


    def integrate(self, f):

        if self.nsteps == 0:
            p = self.atoms.get_momenta()
            self.atoms.set_positions(self.atoms.get_positions() +
                self.dt * p / self.atoms.get_masses()[:,np.newaxis]+self.dt*self.dt * 
                f / (2*self.atoms.get_masses()[:,np.newaxis]) )
                
            ##once evaluating new stuff
            f_old = f
            f = self.atoms.get_forces()
            
            self.atoms.set_velocities(self.atoms.get_velocities() + 
                self.dt * (f_old + f) / (2 * self.atoms.get_masses()[:,np.newaxis]))
                
            self.f_very_old = f_old
        ### -------------------------------------------- #
        
        ### Beemann algorithm
        if self.nsteps > 0:
            p = self.atoms.get_momenta()
            self.atoms.set_positions(self.atoms.get_positions() +
                self.dt * p / self.atoms.get_masses()[:,np.newaxis]+self.dt*self.dt * 
                (4*f-self.f_very_old) / (6*self.atoms.get_masses()[:,np.newaxis]) )
                
            #once evaluating new stuff
            f_old = f
            f = self.atoms.get_forces()
            
            self.atoms.set_velocities(self.atoms.get_velocities() + 
                self.dt * (5*f_old + 2*f - self.f_very_old) / (6 * self.atoms.get_masses()[:,np.newaxis]))
                
            self.f_very_old = f_old


            return f
