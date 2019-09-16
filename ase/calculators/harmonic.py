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




class harmonic_potential(Calculator):
    """
    Calculator for model systems with a simple harmonic potential for covalent bonds and
    and optional exponential repulsion as non-covalent interaction.
    
    Arguments:
    ==========
        . mode            treat 'chain' (default) or 'ring' of atoms or custom definition
                          of bonded neighbors that interact via harmonic potential.
                          If using mode='lattice', no bonds and repulsion between atoms
                          will be added. Only the (harmonic) restraint from reference_geom.
                          In 'lattice' mode, reference_geom is a required argument.
        . k               force constant of harmonic potential ( Vij = k/2 |xi - xj|^2 )
                          float: value for all bonds,
                          ndarray: individual k per bond (between atoms i and i+1: k[i])
                          if using custom bonds, dimension: (nAtom, nAtoms)
        . R0              equilibrium distance of harmonic potential between atoms
                          float: value for all bonds,
                          ndarray: individual R0 per bond
                          (between atoms i and i+1: R0[i])
                          if using custom bonds, dimension: (nAtom, nAtoms)
        . shift           binding energy at equilibrium distance
                          float: value for all bonds
                          ndarray: individual shift per bond
                          (between atoms i and i+1: shift[i])
                          if using custom bonds, dimension: (nAtom, nAtoms)
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
        . neighborlist    list of nearest neighbors to interact via harmonic potential.
                          ndarray, dimension (n,2) [OPTIONAL IF USING 'RING' OR 'CHAIN']
        . restrain_axis   list of coordinate axes along which the atoms should be restrained,
                          elements can be a combination of 'x', 'y', and/or 'z'.
                          Default: no restrain
        . restrain_level  strength (force constant) of restraining potential
                          V_r = restrain_level/2 * <restrain_axis>^2.
                          Can also be per atom restraint, default: 2 for all atoms.
        . reference_geom  Reference structure to which to restrain system to.
                          If none is given, the code will use 0. as reference along the specified
                          restrain_axis. This argument is is required when mode='lattice'
    
    """
    
    default_parameters = {
                          'mode':'chain',           # treat atoms as chain
                          'k':39.57,                # all Carbon
                          'R0':1.52,                # all Carbon
                          'shift':-1035.2,          # all Carbon
                          'with_repulsion':True,    # add exponential repulsion for non-NN
                          'Arep':1878.38,           # all Carbon
                          'gamma':5.49,             # all Carbon
                          'restrain_axis':[],       # restrain atoms along axis by k_r/2*<axis>^2
                          'restrain_level':2.,      # restrain atoms with k_r = 2
                         }
    
    implemented_properties = ['energy', 'forces']

    valid_args = ['mode', 'k', 'R0', 'shift', \
                  'with_repulsion', 'Arep', 'gamma', \
                  'neighborlist', \
                  'restrain_axis', 'restrain_level', 'reference_geom']
    valid_modes = ['ring', 'chain', 'custom', 'lattice']
    
    # classify kwargs
    bool_args = ['with_repulsion']
    string_args = ['mode']
    list_args = ['restrain_axis', 'restrain_level']
    tensor_args = ['k', 'R0', 'shift', 'Arep', 'gamma']
    N3tensor_args = ['reference_geom']
    
    # documentation-type specification for kwargs
    par2arg = {'k':'force constant', 'R0':'equilibrium distance', \
               'shift':'potential minimum', 'Arep':'repulsion prefactor', \
               'gamma':'repulsion scaling'}
    ax2dim = {'x':0, 'y':1, 'z':2}
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        
        ## set default arguments
        for arg, val in self.default_parameters.items(): setattr(self, arg, val)
        
        ## set or overwrite any additional keyword arguments provided
        for arg, val in kwargs.items():
            if arg == 'mode' and val not in self.valid_modes:
                errtxt  = "Available values of 'mode' are '"
                errtxt += "', '".join(self.valid_modes)+"'"
                raise ValueError(errtxt)
            
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s": not in %s'
                                   %(arg, self.valid_args))
        
        if self.mode == 'custom' and not hasattr(self, 'neighborlist'):
            raise RuntimeError("Please, provide 'neighborlist' for mode='custom'.")
        
        restraint_OK1 = all([axis in ['x','y','z'] for axis in self.restrain_axis])
        restraint_OK2 = all([axis in [0, 1, 2] for axis in self.restrain_axis])
        if restraint_OK1:
            self.restrain_axis = [self.ax2dim[ax] for ax in self.restrain_axis]
        elif restraint_OK2:
            pass
        else:
            errtxt  = "Elements of 'restrain_axis' have to be from "
            errtxt += "['x', 'y', 'z'] or [0, 1, 2]"
            raise RuntimeError(errtxt)
        
        if self.mode == 'lattice':
            if not hasattr(self, 'reference_geom'):
                errtxt  = "When choosing mode='lattice', you have to specify a reference geometry "
                errtxt += "via the argument 'reference_geom'."
                raise RuntimeError(errtxt)
            else:
                self.k, self.with_repulsion = 0., False
                self.restrain_axis = [0, 1, 2]
            
        self.with_restraint = (len(self.restrain_axis) > 0)
        
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
            self.nAtoms = len(atoms)
            self.symbols = atoms.get_chemical_symbols()
            self.build_interaction_lists()
            for par in self.tensor_args:
                self.check_interaction_params(atoms, to_check=par)
    
            self.calculate(atoms)
        
    
    def calculate(self, atoms):
        pos = atoms.positions
        if self.mode != 'lattice':
            distances = atoms.get_all_distances()
            bond_vec = np.zeros((self.nAtoms,self.nAtoms,3))
            for ipos, pos_i in enumerate(pos):
                for jpos in range(ipos+1, self.nAtoms):
                    bond = pos_i-pos[jpos]
                    bond_vec[ipos,jpos] = bond
                    bond_vec[jpos,ipos] = -bond
        
        ## harmonic potential for neighbors
        E = 0.
        F = np.zeros((self.nAtoms, 3))
        for [iAtom, jAtom] in self.neighborlist:
            dR = distances[iAtom,jAtom] - self.R0[iAtom,jAtom]
            dR2 = dR * dR
            E += self.k[iAtom,jAtom] * dR2 / 2. + self.shift[iAtom,jAtom]
            
            Fij = self.R0[iAtom,jAtom] / distances[iAtom,jAtom] - 1.
            Fij = Fij * self.k[iAtom,jAtom] * bond_vec[iAtom,jAtom]
            F[iAtom] += Fij
            F[jAtom] -= Fij
        
        ## add any restraint along axes
        if self.with_restraint:
            for ax in self.restrain_axis:
                dpos = pos[:,ax] - self.reference_geom[:,ax]
                krax = self.restrain_level * dpos
                krdax = np.dot(krax, dpos)
                F[:,ax] += -self.restrain_level * dpos
                E += krdax / 2.
        
        ## add exponential repulsion between non-nearest neighbors
        if not self.with_repulsion:
            self.energy = E
            self.forces = F
            return
        
        ## add exponential repulsion between non-nearest neighbors
        for iAtom in range(self.nAtoms-1):
            for jAtom in range(iAtom+1,self.nAtoms):
                if not self.repulsion_pair[iAtom,jAtom]: continue
                gR = self.gamma[iAtom,jAtom] * distances[iAtom,jAtom]
                AexpgR = self.Arep[iAtom,jAtom] * np.exp(-gR)
                E += AexpgR
                
                eij = bond_vec[iAtom,jAtom] / distances[iAtom,jAtom]
                Fij = self.gamma[iAtom,jAtom] * AexpgR * eij
                F[iAtom] += Fij
                F[jAtom] -= Fij
        
        self.energy = E
        self.forces = F
        
    
    def check_interaction_params(self, atoms, to_check="k"):
        try:
            shape_par = eval("np.shape(np.asarray(self."+to_check+", dtype=float))")
            ndim_par = eval("np.ndim(np.asarray(self."+to_check+", dtype=float))")
        except ValueError:
            errtxt  = "Parameter '"+to_check+"' should be of type "
            errtxt += "float or list/ndarray of floats"
            raise ValueError(errtxt)
        
        shape_req = (self.nAtoms,self.nAtoms)
        wrong_dim = False
        if ndim_par == 0:
            exec("self."+to_check+" *= np.ones(shape_req, dtype=float)")
        elif ndim_par == 1:
            if shape_par[0] == self.nAtoms:
                cmdtxt  = "np.diag(self."+to_check+"[:-1], k=1) + "
                cmdtxt += "np.diag(self."+to_check+"[:-1], k=-1)"
                tensorfromlist = eval(cmdtxt)
                exec("tensorfromlist[0,-1] = self."+to_check+"[-1]")
                exec("tensorfromlist[-1,0] = self."+to_check+"[-1]")
                exec("self."+to_check+" = tensorfromlist")
            else:
                wrong_dim = True
        elif ( shape_par != shape_req ):
            wrong_dim = True
        
        if wrong_dim:
            errtxt  = "\nIncorrect number of parameters defined in '"+to_check+"'.\n"
            errtxt += "Should be: float -> same "+self.par2arg[to_check]+" for all or\n"
            errtxt += "           list of length "+str(self.nAtoms)+" -> "
            errtxt += self.par2arg[to_check]+" for all pairs of subsequent atoms or\n"
            errtxt += "           ndarray shape "+repr(shape_req)+" -> individual "
            errtxt += self.par2arg[to_check]+" for each pair of atoms."
            raise ValueError(errtxt)
        
    
    def build_interaction_lists(self):
        if self.mode in ['ring', 'chain']:
            self.neighborlist = []
            for iAtom in range(self.nAtoms-1): self.neighborlist.append([iAtom,iAtom+1])
            if self.mode == 'ring': self.neighborlist.append([0,self.nAtoms-1])
        elif self.mode == 'lattice':
            self.neighborlist = []
            return
        
        self.repulsion_pair = np.ones((self.nAtoms,self.nAtoms), dtype=bool)
        self.repulsion_pair[np.diag_indices(self.nAtoms)] = False
        for [iAtom, jAtom] in self.neighborlist: self.repulsion_pair[iAtom, jAtom] = False
        if not hasattr(self, 'reference_geom') and self.with_restraint:
            self.reference_geom = np.zeros((self.nAtoms, 3))
        
    
    def get_hirsh_volrat(self):
        return [default_avg_a_div_a0[sym] for sym in self.symbols]
        
    


#--EOF--#
