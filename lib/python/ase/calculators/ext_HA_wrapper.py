from ase.units import Bohr
import numpy as np
from copy import copy
from hotbit.analysis.DFT_free_atom import KSAllElectron
from box.data import R_conf, data, ValOccs_lm_free
import random
from string import digits
from os import listdir,system
from scipy.io import FortranFile
from HA_recode import ha_recode as HA


class HirshfeldWrapper:
    """
    Wrapper for HA_recode.so
        ~ calculate radial parts of free/confined atomic wavefunctions
        ~ save in temporary (*.unf) files (to be read-in by F90-Routine)
    """
    #TODO!
    def __init__(self, species, positions, At2Orbs, dr=0.2, nThetas=36, nPhis=72, cutoff=3.,conf='default'):
        """
        initialize basis functions and grid parameters
                    
        parameters:
        ===========
            dr(opt):       radial step width in Angstroms, default: 0.1 Angstroms
            nThetas(opt):  number of discrete azemuthal angles, default: 72
            nPhis(opt):    number of discrete polar angles, default: 144
            cutoff(opt):   cutoff radius for partial grid in Angstroms, default: 3 Angstroms
            conf(opt):     confinement for basis functions in density construction in Angstroms,
                           default: 'default' = (R_conf[symbol] .or. 5*R_cov[symbol] from box.data).
                           Alternative:
                             . 'None': use free radial wave functions throughout,
                             . 'Both': use confined radial wave functions throughout,
                             . list of confinement radii to use per atom.
        """
        self.nAtoms = len(species)
#        self.atoms = self.calc.get_atoms()
        self.species = species
        self.species_set = list(set(self.species))
        self.at2orbs = At2Orbs
        
        if cutoff == 0.:
            print '\033[91m'+'WARNING: input cutoff 0. \
Defaulting to 3 Angstroms.'+'\033[0m'
            cutoff = 3.
        self.nr = round(cutoff/dr)
        self.dr = dr/Bohr
        self.nThetas = int(nThetas)
        self.nPhis = int(nPhis)
        
        ## create random ID for Rnl data
        Rnl_dump_id = ''.join(random.choice(digits) for _ in xrange(16))
        ## if ID exists (minimal chance though) create new random ID
        while Rnl_dump_id+'_' in listdir('./'):
            Rnl_dump_id = ''.join(random.choice(digits) for _ in xrange(16))
        
        f = open(Rnl_dump_id+'_','w')
        f.close()
        self.Rnl_id = Rnl_dump_id
        
        self.occ_free, self.len_r = [], []
        self.orb2at, self.otypes = [], []
        self.rmins = []
        self.rgrid, self.free_occs = {}, {}
        for sym in self.species_set:
            atom = KSAllElectron(sym)
            self.rgrid[sym] = atom.grid.get_grid()
            try:
                occs = ValOccs_lm_free[sym]
                self.free_occs[sym] = occs
            except KeyError:
                print('Occupation for {0: 2s} not available. Update "ValOccs_lm_free" in "box.data"!'.format(sym))
                raise
        
        for iAtom in xrange(self.nAtoms):
            atom = KSAllElectron(self.species[iAtom])
            self.len_r.append( atom.grid.get_N() )
            self.rmins.append( atom.rmin )

        orb2nr = ['s','px','py','pz','dxy','dyz','dzx','dx2-y2','d3z2-r2','f3yx2-y3','fxyz','fyz2','fxz2','fzx2-zy2','fx3-3xy2','fz3']
        for iOrb in xrange(self.nOrbs):
            sym = self.species[self.Orb2Atom[iOrb]]
            self.orb2at.append(atom_idx+1)
            self.otypes.append(orb2nr.index(orb))
            self.occ_free.append(self.free_occs[sym][orb[0]])
        
        if conf == 'Both':
            self._get_radial_conf()
        elif conf == 'None':
            self._get_radial_free()
        else:
            self._get_radial_free()
            self._get_radial_conf()
        
        self._write_radial_free(conf=conf)
        self._write_radial_conf(conf=conf)
        
        self.otypes = np.array(self.otypes, dtype=int)
        self.orb2at = np.array(self.orb2at, dtype=int)
        self.occ_free = np.array(self.occ_free)
        self.at2orbs = self.at2orbs.transpose()
        self.len_r = np.array(self.len_r)
        self.rmins = np.array(self.rmins)
        
    
    def _get_radial_free(self):
        """
        Calcutate radial parts of free atomic wave functions
        """
        Rnl_free, self.Rnls_free = {}, {}
        for sym in self.species_set:
            atom = KSAllElectron(sym)
            Rnl_free[sym] = {}
            atom.run()
            for iOrb in atom.get_valence_orbitals():
                Rnl_free[sym][iOrb[1]] = atom.Rnl_fct[iOrb]
        
        for iAtom in xrange(self.nAtoms):
            sym = self.species[iAtom]
            self.Rnls_free[str(iAtom)] = Rnl_free[sym]
        
    
    def _write_radial_free(self, conf='default'):
        """
        Write radial part of valence orbitals for isolated atoms
        on radial grid to file as obtained by DFT calculation.
        
        writes: <Rnl_id>_Ofree_<orb_idx>.unf (for all orbital indices)
        """
        if conf == 'Both':
            Rnls_free = self.Rnls_conf
        else:
            Rnls_free = self.Rnls_free
        for idx, OrbInfo in enumerate(self.calc.el.orb):
            orb = OrbInfo['orbital']
            sym = OrbInfo['symbol']
            iAtom = OrbInfo['atom']
            f = FortranFile(self.Rnl_id+"_Ofree_{0:d}.unf".format(idx+1), 'w')
            f.write_record( Rnls_free[str(iAtom)][orb[0]](self.rgrid[sym]) )
            f.close()
        
    
    def _get_radial_conf(self, conf='default'):
        """
        Calculate radial parts of atomic wave functions
        for all atoms in additional harmonic confinement potential.
        """
        self.Rnls_conf = {}
        for iAtom in xrange(self.nAtoms):
            sym = self.species[iAtom]
            self.Rnls_conf[str(iAtom)] = {}
#            if conf == 'default':
#                try:
#                    r_conf = R_conf[sym]/Bohr
#                except KeyError:
#                    print 'Confinement radius of '+sym+' not in box.data.R_conf, using 5*R_cov.'
#                    r_conf = 5.*data[sym]['R_cov']/Bohr
#            else:
#                r_conf = conf[iAtom]/Bohr
            r_conf = 2.*data[sym]['R_cov']/Bohr
            atom = KSAllElectron(sym, confinement={'mode':'quadratic','r0':r_conf})
            atom.run()
            for iOrb in atom.get_valence_orbitals():
                self.Rnls_conf[str(iAtom)][iOrb[1]] = atom.Rnl_fct[iOrb]
        
    
    def _write_radial_conf(self,conf='default'):
        """
        Write radial part of valence orbitals for confined atoms 
        on radial grid to file as obtained by DFT calculation.
        
        writes
          . <Rnl_id>_Oconf_<orb_idx>.unf: Rnl for all orbitals (according to ordering in calc.st.wf[k,a])
        """
        if conf == 'None':
            Rnls_conf = self.Rnls_free
        else:
            Rnls_conf = self.Rnls_conf
        
        for idx, OrbInfo in enumerate(self.calc.el.orb):
            iAtom = OrbInfo['atom']
            orb = OrbInfo['orbital']
            sym = OrbInfo['symbol']
            f = FortranFile(self.Rnl_id+"_Oconf_{0:d}.unf".format(idx+1), 'w')
            f.write_record( Rnls_conf[str(iAtom)][orb[0]](self.rgrid[sym]) )
            f.close()
        
    
    def get_hirsh_volrat(self):
        nOrbs = self.calc.el.norb
        positions = (self.atoms.positions/Bohr).transpose()
        c = self.calc.st.wf.transpose()
        wk = self.calc.st.wk.transpose()
        f = self.calc.st.f.transpose()
        nkpts = len(wk)
        hvr = HA.hirshfeld_main(self.nAtoms,nkpts,nOrbs,self.nThetas,self.nPhis,positions,\
                                c,wk,f,max(self.len_r),self.dr,self.nr,self.len_r,self.rmins,\
                                self.Rnl_id,self.occ_free,self.otypes,self.orb2at,self.at2orbs)
                                
        self.remove_files()
        return hvr
        
    
    def remove_files(self):
        """ Remove dump directory containing Rnl data """
        
        system("rm "+self.Rnl_id+"_*")
        
    

#--EOF--#
