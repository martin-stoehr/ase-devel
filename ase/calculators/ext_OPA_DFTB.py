import numpy as np


class OverlapPopulationVolumeAnalysis:
    """
    class for approximate analysis of volume ratios (effective / free)
    as used in vdW(TS), MBD, or dDsC based on density matrix elements
    and overlap integrals as available in DFTB.
    
    approaches:  'ONOP':    ratio = ONOP / ( 2 n_{el}^{(neutral)} ) ,
                 'OFFOP1':  ratio = 1 - ( OFFOP / n_{el}^{(neutral)} ) ,
                 'OFFOP2':  ratio = 1 - ( OFFOP / ONOP ) ,
         #FIXME! 'OFFOP3':  ratio = 1 - ( OFFOP / sumCC )  (yields nonsymmetric results, disabled)
                 
         where:  ONOP  = sum_k w_k sum_a f[k,a] 2 sum_{i in A} |c[k,a,i]|^2 ,
                         (diagonal elements of Mulliken Orbital Overlap Population)
                 
                 n_{el}^{(neutral)} = number of (valence) electrons in free, neutral atom
                 
                 OFFOP = sum_k w_k sum_a f[k,a] sum_{i in A, j in B} (c[k,a,i]* S[k,i,j] c[k,a,j] + c[k,a,i] S[k,i,j]* c[k,a,j]*) ,
                         (off-diagonal elements of Mulliken Orbital Overlap Population)
                 
                 sumCC = sum_k w_k sum_a f[k,a] sum_{i in A, j in B} (c[k,a,i]* c[k,a,j] + c[k,a,i] c[k,a,j]*) .
    
    """
    
    def __init__(self, coeff, f, wk, n_el_atom, Atom2Orbs):
#    def __init__(self, coeff, f, wk, n_el_atom, S_matrix, Atom2Orbs):
        """
        initialize arrays, get (approximate) volume ratios.
        
        parameters:
        ===========
            . coeff     = LCAO-coefficients, numpy-array shape(n_kpoints, n_States, n_Orbitals)
            . f         = fillings, numpy array shape(n_kpoints, n_States)
            . wk        = k-point weighting factors, numpy-array shape(n_kpoints,)
            . n_el_atom = number of electrons in free, neutral atom, numpy-array shape(n_Atoms,)
           [ . S_matrix  = overlap integral matrix, numpy-array shape(n_kpoints, n_Orbitals, n_Orbitals) ]
            . Atom2Orbs = list of lists containing orbitals located at atom iAtom
                          (e.g. [[0,1,2,3],[4]]: orbitals 0,1,2,3 on atom 0, orbital 4 on atom 1)
        """
        
        ## LCAO coefficients, fillings, k-point weightings
        self.coeff, self.f, self.wk = coeff, f, wk
        ## number of electrons of neutral, free atoms, overlap matrix
        self.n_el_neutral = n_el_atom
#        self.S = S_matrix
        ## orbitals on atom i
        self.Atom2Orbs = Atom2Orbs
        ## number of atoms, k-points, and orbitals (= number of states)
        self.nAtoms = len(self.n_el_neutral)
        self.nk = len(self.wk)
        self.nOrbs = self.coeff.shape[-1]
        ## result arrays (OFFOP approaches not implemented yet.)
        self.ONOP = np.zeros(self.nAtoms)
#        self.OFFOP1 = np.zeros(self.nAtoms)
#        self.OFFOP2 = np.zeros(self.nAtoms)
#        self.OFFOP3 = np.zeros(self.nAtoms)
        
        ## calculate!
        self._get_volrats()
        
    
    def _get_volrats(self):
        """
        calculate (approximate!) volume ratios 
        """
        
        OFFOP, ONOP = np.zeros(self.nAtoms), np.zeros(self.nAtoms)
#        sumCC = np.zeros(self.nAtoms)
        for iAtom in xrange(self.nAtoms):
            for ik in xrange(self.nk):
                for a in xrange(self.nOrbs):
                    ## cSc[i] = c[i]*sum_{j in B != A} S[i,j] c[j] = c[i]*(S-1).dot(c)
#                    cSc = np.conjugate(self.coeff[ik,a])*( (self.Skij[ik] - np.eye(self.nOrbs)).dot(self.coeff[ik,a]) )
                    wkfka = self.wk[ik]*self.f[ik,a]
                    for iOrb in self.Atom2Orbs[iAtom]:
#                        OFFOP[iAtom] += wkfka*( cSc[iOrb] + np.conjugate(cSc[iOrb]) ).real
                        ONOP[iAtom] += wkfka*abs( self.coeff[ik,a,iOrb] )**2
#                        for jOrb in xrange(self.nOrbs):
#                            cicj = self.coeff[ik,a,iOrb].conjugate()*self.coeff[ik,a,jOrb]
#                            sumCC[iAtom] += wkfka*(( cicj + np.conjugate(cicj) ).real)
#                        sumCC[iAtom] -= 2.*wkfka*abs( self.coeff[ik,a,iOrb] )**2
        
            
        self.ONOP = ONOP/(self.n_el_neutral)
#        self.OFFOP1 = 1. - (OFFOP / self.n_el_neutral)
#        self.OFFOP2 = 1. - (OFFOP / ONOP)
#        self.OFFOP3 = 1. - (OFFOP / sumCC)
        
    
    def get_hvr(self, approach='ONOP'):
        """ 
        return volume ratios as obtained by approach <approach> 
        
        input:
        =====
            approach:  'ONOP', 'OFFOP1', or 'OFFOP2'.
                       (see class doc-string for further details)
        """
        if (approach == 'ONOP'):
            return self.ONOP
#        elif (approach == 'OFFOP1'):
#            return self.OFFOP1
#        elif (approach == 'OFFOP2'):
#            return self.OFFOP2
#        elif (approach == 'OFFOP3'):
#            return self.OFFOP3
        else:
            raise NotImplementedError("Sorry, I don't know about an approach called '"+str(approach)+"'.")
        
    

#--EOF--#
