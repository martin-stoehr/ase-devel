import os
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree




def return_one(x): return 1.
def return_zero(x): return 0.


class ZBL(Calculator):
    """
    An ASE compatible calculator for the Ziegler-Biersack-Littmark screened
    Coulomb potential (originally used to describe heavy element collisions)
    [J.F. Ziegler, J.P. Biersack, U. Littmark, "The Stopping and Ranges of
        Ions in Solids", vol. 1. Pergamon Press, New York (1985).]
    
    In atomic units the ZBL potential is given by
    
      V_ZBL = \sum_{A<B} f_cut(R_AB) * Z_A*Z_B / R_AB *
                      \sum_{i=1}^4 c_i * exp(-b_i * (Z_A^p + Z_B^p) * R_AB/x)
        
    where R_AB is the interatomic distance between atoms A and B and
    Z denotes the respective atomic numbers.
    
    Arguments:
    ==========
        . c1,c2,c3,c4    [float] prefactors in sum of exponentials
        . b1,b2,b3,b4    [float] scaling factors in individual exponentials
        . power          [float] exponent of atomic numbers in exponentials
                           (denoted as p in above formula)
        . radial_factor  [float] scaling factor to define reduced radius
                           (denoted as x in above formula)
        . cutoff         [callable/function, optional] function to cutoff the
                         contributions from increased distances, default: None
                           (denoted f_cut in above formula)
        . dcutoff_dR     [callable/function, optional] derivative of above
                         cutoff function with respect to interatomic distance
    
    """
    
    default_parameters = {
                          'c1':0.1818, 'c2':0.5099, 'c3':0.2802, 'c4':0.0281,
                          'b1':3.2000, 'b2':0.9423, 'b3':0.4028, 'b4':0.2016,
                          'power':0.23, 'radial_factor':0.8854,
                          'cutoff':None, 'dcutoff_dR':None,
                         }
    
    implemented_properties = ['energy', 'forces']

    valid_args = ['c1','c2','c3','c4','b1','b2','b3','b4',\
                  'power', 'radial_factor', 'cutoff', 'dcutoff_dR']
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        
        ## set default arguments
        for arg, val in self.default_parameters.items():
            setattr(self, arg, val)
        
        ## set or overwrite any additional keyword arguments provided
        for arg, val in kwargs.items():
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s": not in %s'
                                   %(arg, self.valid_args))
        
        if ( (self.cutoff is None) and (self.dcutoff_dR is None) ):
            self.cutoff = return_one
            self.dcutoff_dR = return_zero
        
        call_check = ( callable(self.cutoff), callable(self.dcutoff_dR) )
        if not all(call_check):
            msg  = "Please provide 'cutoff' as well as its derivative "
            msg += "'dcutoff_dR' as callable function instance!"
            raise RuntimeError(msg)
        
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        
    
    def get_potential_energy(self, atoms=None):
        self.calculate(atoms)
        return self.results['energy']
        
    
    def get_forces(self, atoms=None):
        self.calculate(atoms)
        return self.results['forces']
        
    
    def update_system(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.nAtoms = len(atoms)
            self.c_vec = np.array([self.c1, self.c2, self.c3, self.c4])
            self.b_vec = np.array([self.b1, self.b2, self.b3, self.b4])
        
    
    def calculate(self, atoms, properties=['energy', 'forces'],
                  system_changes=all_changes):
        self.update_system(atoms)
        pos = atoms.positions / Bohr
        distances = atoms.get_all_distances() / Bohr
        Z = atoms.get_atomic_numbers()
        E = 0.
        F = np.zeros((self.nAtoms, 3))
        for iAtom, Z_i in enumerate(Z):
            for jAtom in range(iAtom+1, self.nAtoms):
                d = distances[iAtom,jAtom]
                coul = Z_i * Z[jAtom] / d
                dred = d / self.radial_factor
                ZAB = Z_i**self.power + Z[jAtom]**self.power
                args = self.b_vec * ZAB * dred
                f_screen = self.c_vec * np.exp(-1. * args)
                Eij = coul * np.sum(f_screen)
                E += Eij
                
                e_bond = (pos[iAtom] - pos[jAtom]) / d
                f_screen = f_screen * (args + 1./self.radial_factor)
                dEij_dR = coul * e_bond/d * np.sum(f_screen) / 2.
                Fij = self.cutoff(d) * dEij_dR + self.dcutoff_dR(d) * Eij
                F[iAtom] += Fij
                F[jAtom] -= Fij
            
        self.results['energy'] = E * Hartree
        self.results['forces'] = F * Hartree / Bohr
        
        return
        
    


#--EOF--#
