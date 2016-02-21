import numpy as np


class OverlapPopulationVolumeAnalysis:
    """
    class for approximate analysis of volume ratios (effective / free)
    as used in vdW(TS), MBD, or dDsC based on density matrix elements
    and overlap integrals as available in DFTB.
    
    approaches:  'ONOP':    ratio = 1 + ( ONOP - n_{el}^{(neutral)} )/Z = Tr{D}/Z,
                 
         where:  ONOP  = sum_k w_k sum_a f[k,a] sum_{i in A} |c[k,a,i]|^2 ,
                         (diagonal elements of Mulliken Orbital Overlap Population)
                 
                 n_{el}^{(neutral)} = number of (valence) electrons in free, neutral atom
                 D = density matrix (incl. k-point weightings and occupations)
                 Z = atomic number
        (only valence electrons considered in DFTB => renormalization to total number of electrons)
    
    """
    
    def __init__(self, coeff, f, wk, n_el_atom, atomic_numbers, Orb2Atom):
        """
        initialize arrays, get (approximate) volume ratios.
        
        parameters:
        ===========
            . coeff           = LCAO-coefficients, numpy-array shape(n_kpoints, n_States, n_Orbitals)
            . f               = fillings, numpy array shape(n_kpoints, n_States)
            . wk              = k-point weighting factors, numpy-array shape(n_kpoints,)
            . n_el_atom       = number of electrons in free, neutral atom, numpy-array shape(n_Atoms,)
            . atomic_numbers  = atomic numbers, nd-array shape(n_Atoms,)
            . Orb2Atom        = list of atom orbital i is located, np.array shape(n_Orbitals)
              (e.g. H2O: [0,0,0,0,1,2]: orbitals[0:4] on atom 0, orbital 4 on atom 1, and orbital 5 on atom 2)
        """
        
        ## LCAO coefficients, fillings, k-point weightings
        self.coeff, self.f, self.wk = coeff, f, wk
        ## number of electrons of neutral, free atoms, overlap matrix
        self.n_el_neutral = n_el_atom
        ## orbital i located at atom Orb2Atom[i]
        self.Orb2Atom = Orb2Atom
        ## number of atoms, k-points, and orbitals (= number of states)
        self.nAtoms = len(self.n_el_neutral)
        self.ZAtoms = atomic_numbers
        self.nk = len(self.wk)
        self.nOrbs = self.coeff.shape[-1]
        ## result arrays (OFFOP approaches not implemented yet.)
        self.ONOP = np.zeros(self.nAtoms)
        
        ## calculate!
        self._get_volrats()
        
    
    def _get_volrats(self):
        """
        calculate (approximate!) volume ratios 
        """
        
        ONOP = np.zeros(self.nAtoms)
        for ik in xrange(self.nk):
            for a in xrange(self.nOrbs):
                wkfka = self.wk[ik]*self.f[ik,a]
                for iOrb in xrange(self.nOrbs):
                    ONOP[self.Orb2Atom[iOrb]] += wkfka*abs( self.coeff[ik,a,iOrb] )**2
            
        self.ONOP = 1. + (ONOP - self.n_el_neutral)/self.ZAtoms
        
    
    def get_hvr(self):
        """  return volume ratios as obtained by ONOP approach.  """
        return self.ONOP
        
    

#--EOF--#
