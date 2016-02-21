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
from os import listdir
from box.data import data


class ext_OPA_wrapper:
    """
    get rescaling ratios from overlap population analysis
    """
    
    def __init__(self, atoms, output_file='aims.out', basisfile='basis-indices.out', eigvfile='wvfn.dat'):
        """
        Parse aims basis-indices files to get information on basis functions.
        adapted from 'assignAO.py' by Christoph Schober/Georg Michelitsch (TUM)
        and extended by Martin Stoehr (TUM).
    
        parameters:
        ===========
            atoms :       ASE atoms object
            output_file:  filename of aims output,
                          str(optional)
            basisfile :   filename of basis indices,
                          str (optional)
            eigvfile :    filename of restart file containing eigenvectors,
                          str (optional)
        """
        
        with open(basisfile) as f:
            tmp = f.readlines()[2:]
        
        self.atoms = atoms.copy()
        if np.any(self.atoms.pbc):
            self.pbc = 1
            self._get_k_point_weightings(output_file)
            with open(output_file) as f:
                tmp2 = f.readlines()[2:]
            for iline, line in enumerate(tmp2):
                if 'Internal wall clock time zero' in line:
                    ntaskline = tmp2[iline+2]
                    assert(('Using' in ntaskline) and ('parallel tasks.' in ntaskline))
                    self.nFiles = int(ntaskline.split()[1])
                elif '  The eigenvectors in the calculations are' in line:
                    self.evectype = line.split()[-1][0]
        else:
            self.pbc, self.nFiles = 0, 1
            self.evectype = 'R'
            self.nk, self.wk = 1, np.array([1.])
        
        assert (self.evectype in ['R','C'])
        self.nAtoms = self.atoms.get_number_of_atoms()
        symbols = self.atoms.get_chemical_symbols()
        self.ZAtoms = self.atoms.get_atomic_numbers()
        
        self.nOrbs = len(tmp)
        self.Orb2Atom = np.zeros(self.nOrbs, dtype=int)
        for OrbLine in tmp:
            idx, basistype, atom, n, l, m = OrbLine.split()
            self.Orb2Atom[int(idx)-1] = int(atom)
        
    
    def _get_k_point_weightings(self, outfile):
        """
        extract k-point weightings and total number of k-points used
        """
        
        lines = open(outfile,'r').readlines()
        self.wk = []
        for iline, line in enumerate(lines):
            if 'K-points in task  ' in line:
                self.nk = int(line.split()[-1])
                n_line = iline + 1
                
                while 'K-points in task  ' in lines[n_line]:
                    self.nk += int(lines[n_line].split()[-1])
                    n_line += 1
                    
                for ik in range(self.nk):
                    self.wk.append(float(lines[n_line].split()[9]))
                    n_line += 1
                break
            
        self.wk = np.array(self.wk)
        
    
    def get_ONOP(self):
        """
        return (approximate) rescaling ratios for effective atomic
        polarizabilities as obtained by overlap population.
        """
        
        ONOP = OPA.get_wvfn_info(self.nFiles, self.nk, self.nOrbs, self.nAtoms, \
                                 self.Orb2Atom, self.wk, self.pbc, self.evectype)
        return np.array(ONOP)/self.ZAtoms
        
    

#--EOF--#
