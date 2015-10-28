#######################################################
#                                                     #
##  This script is meant to postprocess AIMS output  ##
#    and calculate (approximate) rescaling ratios     #
##    for atomic polarizabilities as obtained by     ##
#            overlap population analysis              #
##                                                   ##
#   by Martin Stoehr (martin.stoehr@tum.de),          #
##  Technische universitaet Muenchen Oct/28/2015     ##
#                                                     #
#######################################################

import numpy as np
from OPA_recode import opa_recode as OPA
from box.data import data

class ext_OPA_wrapper:
    """
    get rescaling ratios
    """
    def __init__(self, atoms, basisfile="basis-indices.out", eigvfile='wvfn.dat'):
        """
        Parse aims basis-indices files to get information on basis functions.
        adapted from 'assignAO.py' by Christoph Schober.
    
        parameters:
        ===========
            atoms :     ASE atoms object
            basisfile : filename of basis indices,
                        str (optional)
            eigvfile :  filename of restart file containing eigenvectors,
                        str (optional)
        """
        with open(basisfile) as f:
            tmp = f.readlines()[2:]
        
        self.atoms = atoms.copy()
        self.nAtoms = self.atoms.get_number_of_atoms()
        self.Nel_val = np.array([data[sym]['valence_number'] \
                       for sym in self.atoms.get_chemical_symbols()],dtype=int)
        
        self.filename = eigvfile
        self.nOrbs = len(tmp)
        self.Orb2Atom = np.zeros(self.nOrbs, dtype=int)
        for OrbLine in tmp:
            idx, basistype, atom, n, l, m = OrbLine.split()
            self.Orb2Atom[int(idx)-1] = int(atom)
        
    
    def get_ONOP(self):
        """
        return (approximate) rescaling ratios for effective atomic
        polarizabilities as obtained by overlap population.
        """
        
        ONOP = OPA.get_wvfn_info(self.filename, self.nOrbs, self.nAtoms, self.Orb2Atom)
        return np.array(ONOP)/self.Nel_val
        
    

#--EOF--#
