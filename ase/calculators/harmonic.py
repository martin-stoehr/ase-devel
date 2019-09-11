# Evaluates the energy and force based on a simple harmonic potential between nearest neighbors
# Subsequent atoms in Atoms object are considered to be "nearest neighbors" in that regard
# Only simple structures supported so far: Can be either a chain or a ring of atoms.

import os
import numpy as np
from ase.calculators.calculator import Calculator


## force_constant for harmonic bond in eV/A^2
force_constant = {}
force_constant['C-C']   = 39.57
force_constant['Si-Si'] = 18.6
for symsym, k in force_constant.items():
    sym1, sym2 = symsym.split('-')
    force_constant[sym2+'-'+sym1] = k

## parameters for A*exp(-gamma*R)-type repulsion
repulsion_parameters = {}
repulsion_parameters['C-C'] = {'A':1878.381, 'gamma':5.4875}
for symsym, pars in repulsion_parameters.items():
    sym1, sym2 = symsym.split('-')
    repulsion_parameters[sym2+'-'+sym1] = pars


## equilibrium distances in A
equilibrium_distance = {'Ac-Ac':4.3,  'Ag-Ag':2.9,  'Al-Al':2.42,
          'Am-Am':3.6,  'Ar-Ar':2.12, 'As-As':2.38, 'At-At':3.0,
          'Au-Au':2.72, 'B-B': 1.68,  'Ba-Ba':4.3,  'Be-Be':1.92,
          'Bi-Bi':2.96, 'Br-Br':2.4,  'C-C':1.52,   'Ca-Ca':3.52,
          'Cd-Cd':2.88, 'Ce-Ce':4.08, 'Cl-Cl':2.04, 'Cm-Cm':3.38,
          'Co-Co':2.52, 'Cr-Cr':2.78, 'Cs-Cs':4.88, 'Cu-Cu':2.64,
          'Dy-Dy':3.84, 'Er-Er':3.78, 'Eu-Eu':3.96, 'F-F':1.14,
          'Fe-Fe':2.64, 'Fr-Fr':5.2,  'Ga-Ga':2.44, 'Gd-Gd':3.92,
          'Ge-Ge':2.4,  'H-H':0.62,   'He-He':0.56, 'Hf-Hf':3.5,
          'Hg-Hg':2.64, 'Ho-Ho':3.84, 'I-I':2.78,   'In-In':2.84,
          'Ir-Ir':2.82, 'K-K':4.06,   'Kr-Kr':2.32, 'La-La':4.14,
          'Li-Li':2.56, 'Lu-Lu':3.74, 'Mg-Mg':2.82, 'Mn-Mn':2.78,
          'Mo-Mo':3.08, 'N-N':1.42,   'Na-Na':3.32, 'Nb-Nb':3.28,
          'Nd-Nd':4.02, 'Ne-Ne':1.16, 'Ni-Ni':2.48, 'Np-Np':3.8,
          'O-O':1.32,   'Os-Os':2.88, 'P-P':2.14,   'Pa-Pa':4.0,
          'Pb-Pb':2.92, 'Pd-Pd':2.78, 'Pm-Pm':3.98, 'Po-Po':2.8,
          'Pr-Pr':4.06, 'Pt-Pt':2.72, 'Pu-Pu':3.74, 'Ra-Ra':4.42,
          'Rb-Rb':4.4,  'Re-Re':3.02, 'Rh-Rh':2.84, 'Rn-Rn':3.0,
          'Ru-Ru':2.92, 'S-S':2.1,    'Sb-Sb':2.78, 'Sc-Sc':3.4,
          'Se-Se':2.4,  'Si-Si':2.22, 'Sm-Sm':3.96, 'Sn-Sn':2.78,
          'Sr-Sr':3.9,  'Ta-Ta':3.4,  'Tb-Tb':3.88, 'Tc-Tc':2.94,
          'Te-Te':2.76, 'Th-Th':4.12, 'Ti-Ti':3.2,  'Tl-Tl':2.9,
          'Tm-Tm':3.8,  'U-U':3.92,   'V-V':3.06,   'W-W':3.24,
          'Xe-Xe':2.8,  'Y-Y':3.8,    'Yb-Yb':3.74, 'Zn-Zn':2.44,
          'Zr-Zr':3.5}

for symsym, R in equilibrium_distance.items():
    sym1, sym2 = symsym.split('-')
    equilibrium_distance[sym2+'-'+sym1] = R


def get_force_constant(sym1, sym2):
    """ Returns force constant for harmonic bond between <sym1> and <sym2>. """
    return force_constant[sym1+'-'+sym2]
    

def get_equilibrium_distance(sym1, sym2):
    """ Returns equilibrium distance for harmonic bond between <sym1> and <sym2>. """
    return equilibrium_distance[sym1+'-'+sym2]
    

default_avg_a_div_a0 = {   #TODO: adapt values
'H':0.4,                                                                                    'He':0.9, \
'Li':0.35, 'Be':0.55,                'B':0.69,  'C':0.72,  'N':0.76,  'O':0.88,  'F':0.95,  'Ne':0.9, \
'Na':0.40, 'Mg':0.57,                'Al':0.72, 'Si':0.78, 'P':0.92,  'S':0.89,  'Cl':0.93, 'Ar':0.9, \
}




default_parameters = {
                      'mode':'chain',
                      'k':39.57,                # all Carbon
                      'R0':1.52,                # all Carbon
                      'shift':-1035.2,          # all Carbon
                      'with_repulsion':True,    # add exponential repulsion for non-NN
                      'Arep':1878.38,           # all Carbon
                      'gamma':5.49,             # all Carbon
                     }


class harmonic_potential(Calculator):
    """
    Class of simple harmonic potential for homonuclear chains and rings of atoms.
    
    Arguments:
    ==========
        . mode            treat 'chain' (default) or 'ring' of atoms
        . k               force constant of harmonic potential ( Vij = k/2 |xi - xj|^2 )
                          float: value for all bonds,
                          ndarray: individual k per bond (between atoms i and i+1: k[i])
        . R0              equilibrium distance of harmonic potential between atoms
                          float: value for all bonds,
                          ndarray: individual R0 per bond
                          (between atoms i and i+1: R0[i])
        . shift           binding energy at equilibrium distance
                          float: value for all bonds
                          ndarray: individual shift per bond
                          (between atoms i and i+1: shift[i])
        . with_repulsion  add exponential repulsion between non-nearest neighbors
                          bool, default: True
                          V = Arep * exp(-gamma * |Ri - Rj|)
        . Arep            prefactor of exponential repulsion, see definition above.
                          float: value for all pairs,
                          ndarray: individual Arep per pair of atoms
                          (between atoms i and j: Arep[i,j])
        . gamma           factor in exponential repulsion, see definition above.
                          float: value for all pairs,
                          ndarray: individual gamma per pair of atoms
                          (between atoms i and j: gamma[i,j])

    
    """
    
    implemented_properties = ['energy', 'forces']

    valid_args = ['mode', 'k', 'R0', 'shift', 'with_repulsion', 'Arep', 'gamma']
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        
        ## set default arguments
        for arg, val in default_parameters.items(): setattr(self, arg, val)
        
        ## set or overwrite any additional keyword arguments provided
        for arg, val in kwargs.items():
            if arg == 'mode' and val not in ['ring', 'chain']:
                raise ValueError("Available values of 'mode' are 'chain' and 'ring'")
            
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s": not in %s'
                                   %(arg, self.valid_args))
        
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        
    
    def get_potential_energy(self, atoms=None):
        self.update_properties(atoms)
        return self.energy
        
    
    def get_forces(self, atoms=None):
        self.update_properties(atoms)
        return self.forces
        
    
    def update_properties(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.symbols = atoms.get_chemical_symbols()
            for par in ["k", "R0", "shift"]:
                self.check_vector_params(atoms, to_check=par)
            if self.with_repulsion:
                for par in ["Arep", "gamma"]:
                    self.check_tensor_params(atoms, to_check=par)
            
            self.calculate(atoms)
        
    
    def calculate(self, atoms):
        nAtoms = len(atoms)
        pos = atoms.positions
        
        ## harmonic potential
        E = 0.
        for iAtom in range(nAtoms-1):
            R = np.linalg.norm(pos[iAtom]-pos[iAtom+1])
            dR = R - self.R0[iAtom]
            dR2 = dR * dR
            E += self.k[iAtom] * dR2 / 2. + self.shift[iAtom]

        if (self.mode == 'ring'):   # add end to end interaction in ring
            dR = np.linalg.norm(pos[-1]-pos[0]) - self.R0[-1]
            dR2 = dR * dR
            E += self.k[-1] * dR2 / 2. + self.shift[-1]
        
        F = np.zeros((nAtoms, 3))
        for iAtom in range(nAtoms-1):
            Rij = pos[iAtom] - pos[iAtom+1]
            Nij = np.linalg.norm(Rij)
            Fij = self.k[iAtom] * Rij * (self.R0[iAtom] / Nij - 1.)
            F[iAtom] += Fij
            F[iAtom+1] -= Fij
        
        if (self.mode == 'ring'):   # add end to end interaction in ring
            Rij = pos[-1] - pos[0]
            Nij = np.linalg.norm(Rij)
            Fij = self.k[-1] * Rij * (self.R0[-1] / Nij - 1.)
            F[-1] += Fij
            F[0] -= Fij
        
        
        ## add exponential repulsion between non-nearest neighbors
        if not self.with_repulsion:
            self.energy = E
            self.forces = F
            return
        
        ## first and last atom in a ring are nearest neighbors => no repulsion
        idx_end = nAtoms if (self.mode == 'chain') else nAtoms-1
        for iAtom in range(nAtoms-2):
            for jAtom in range(iAtom+2,idx_end):
                Rij = pos[iAtom]-pos[jAtom]
                R = np.linalg.norm(Rij)
                eij = Rij / R
                gR = self.gamma[iAtom,jAtom] * R
                AexpgR = self.Arep[iAtom,jAtom] * np.exp(-gR)
                AgexpgReij = self.gamma[iAtom,jAtom] * AexpgR * eij
                E += AexpgR
                F[iAtom] += AgexpgReij
                F[jAtom] -= AgexpgReij
        
        self.energy = E
        self.forces = F
        
    
    def check_vector_params(self, atoms, to_check="k"):
        nAtoms = len(atoms)
        try:
            shape_par = eval("np.shape(np.asarray(self."+to_check+", dtype=float))")
        except ValueError:
            errtxt  = "Parameter '"+to_check+"' (force constants) should be of type "
            errtxt += "float or list/array of floats"
            raise ValueError(errtxt)
        
        nbpar_req = nAtoms if (self.mode == 'ring') else nAtoms-1
        if shape_par == ():
            exec("self."+to_check+" *= np.ones(nbpar_req, dtype=float)")
        elif ( shape_par != (nbpar_req,) ):
            errtxt  = "Incorrect number of force constants.\n"
            errtxt += "Parameter '"+to_check+"' should be float or list/array with "
            errtxt += "length "+str(nbpar_req)+" for a "+self.mode+" of atoms!"
            raise ValueError(errtxt)
        
    
    def check_tensor_params(self, atoms, to_check="k"):
        nAtoms = len(atoms)
        try:
            shape_par = eval("np.shape(np.asarray(self."+to_check+", dtype=float))")
        except ValueError:
            errtxt  = "Parameter '"+to_check+"' (force constants) should be of type "
            errtxt += "float or list/array of floats"
            raise ValueError(errtxt)

        shape_req = (nAtoms,Atoms) if (self.mode == 'chain') else (nAtoms-1,nAtoms-1)
        if shape_par == ():
            exec("self."+to_check+" *= np.ones(shape_req, dtype=float)")
        elif ( shape_par != shape_req ):
            errtxt  = "Incorrect number of force constants.\n"
            errtxt += "Parameter '"+to_check+"' should be float or list/array with "
            errtxt += "shape "+repr(shape_req)+" for a "+self.mode+" of atoms!"
            raise ValueError(errtxt)


    def get_hirsh_volrat(self):
        return [default_avg_a_div_a0[sym] for sym in self.symbols]
        
    

#--EOF--#
