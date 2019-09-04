from ase.units import Bohr
import numpy as np
from copy import copy
from ase.calculators.DFT_free_atom import KSAllElectron
from ase.calculators.hb_box_data import data, ValOccs_lm_free
import random
from string import digits
from os import listdir,system
from scipy.io import FortranFile
from ase.calculators.HA_recode import ha_recode as HA


class HirshfeldWrapper:
    """
    Wrapper for HA_recode.so
        ~ calculate radial parts of free/confined atomic wavefunctions
        ~ save in temporary (*.unf) files (to be read-in by F90-Routine)
    """
    def __init__(self, atoms, wk, wf, f, otypes, Atom2Orbs, Orb2Atom, \
                 dr=0.2, nThetas=36, nPhis=72, cutoff=3.,conf='Both'):
        """
        initialize basis functions and grid parameters
                    
        parameters:
        ===========
            atoms:         ASE-atoms object, len = n_Atoms
            wk:            (normalized) k-point weighting factors, len = n_kpoints
            wf:            LCAO-coefficients, shape = (n_kpoints, n_States, n_Orbitals)
            f:             occupations of states, shape = (n_kpoints, n_States)
            otypes:        type of basis orbitals, len = n_Orbitals
                           (e.g. H2O: otypes = ['s','px','py','pz','s','s'])
            Atom2Orbs:     first and (last+1) orbital index located at atom i in FORTRAN convention,
                           shape = (n_Atoms, 2)    --    (e.g. H2O: Atom2Orbs = [[1,4],[5,5],[6,6]])
            Orb2Atom:      atom index of orbital i, len = n_Orbitals
                           (e.g. H2O: Orb2Atom = [0,0,0,0,1,2])
            dr(opt):       radial step width in Angstroms, default: 0.1 Angstroms
            nThetas(opt):  number of discrete azemuthal angles, default: 36
            nPhis(opt):    number of discrete polar angles, default: 72
            cutoff(opt):   cutoff radius for partial grid in Angstroms, default: 3 Angstroms
            conf(opt):     confinement for basis functions in density construction in Angstroms,
                             . 'None': use free radial wave functions throughout,
                             . 'Both': use confined radial wave functions throughout,
                             . list of confinement radii n Angstroms to use per atom.
        """
        self.atoms = atoms.copy()
        self.nAtoms = len(self.atoms)
        self.species = self.atoms.get_chemical_symbols()
        self.species_set = []
        for sym in self.species:
            if sym not in self.species_set:
                self.species_set.append(sym)
        
        self.wk = np.array(wk)
        self.wf = np.array(wf).transpose()
        self.f = np.array(f).transpose()
        self.otypes = np.array(otypes)
        self.Atom2Orbs = np.array(Atom2Orbs, dtype=int)#.transpose()
        self.Orb2Atom = np.array(Orb2Atom, dtype=int)
        self.nOrbs = len(self.Orb2Atom)
        
        if cutoff == 0.:
            print('\033[91m'+'WARNING: input cutoff 0. \
Defaulting to 3 Angstroms.'+'\033[0m')
            cutoff = 3.
        self.nr = int( round(cutoff/dr) )
        self.dr = dr/Bohr
        self.nThetas = int(nThetas)
        self.nPhis = int(nPhis)
        
        ## create random ID for Rnl data
        Rnl_dump_id = ''.join(random.choice(digits) for _ in range(16))
        ## if ID exists (minimal chance though) create new random ID
        while Rnl_dump_id+'_' in listdir('./'):
            Rnl_dump_id = ''.join(random.choice(digits) for _ in range(16))
        
        f = open(Rnl_dump_id+'_','w')
        f.close()
        self.Rnl_id = Rnl_dump_id
        
        self.occ_free, self.len_r = [], []
        self.rmins = []
        self.rgrid, self.free_occs = {}, {}
        for sym in self.species_set:
            atom = KSAllElectron(sym)
            self.rgrid[sym] = atom.grid.get_grid()
            try:
                occs = ValOccs_lm_free[sym]
                self.free_occs[sym] = occs
            except KeyError:
                print('Occupation for {0: 2s} not available. Update "ValOccs_lm_free" in "hb_box_data"!'.format(sym))
                raise
        
        for iAtom in range(self.nAtoms):
            atom = KSAllElectron(self.species[iAtom])
            self.len_r.append( atom.grid.get_N() )
            self.rmins.append( atom.rmin )
        
        self.otypes_num = np.zeros(self.nOrbs)
        orb2nr = ['s','px','py','pz','dxy','dyz','dzx','dx2-y2','d3z2-r2','f3yx2-y3','fxyz','fyz2','fxz2','fzx2-zy2','fx3-3xy2','fz3']
        for iOrb in range(self.nOrbs):
            sym = self.species[self.Orb2Atom[iOrb]]
            self.otypes_num[iOrb] = orb2nr.index(self.otypes[iOrb])
            self.occ_free.append(self.free_occs[sym][self.otypes[iOrb][0]])
        
        if conf == 'Both':
            self._get_radial_conf()
        elif conf == 'None':
            self._get_radial_free()
        else:
            self._get_radial_free()
            self._get_radial_conf()
        
        self._write_radial_free(conf=conf)
        self._write_radial_conf(conf=conf)
        
        self.otypes_num = np.array(self.otypes_num, dtype=int)
        self.occ_free = np.array(self.occ_free)
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
        
        for iAtom in range(self.nAtoms):
            sym = self.species[iAtom]
            self.Rnls_free[str(iAtom)] = Rnl_free[sym]
        
    
    def _write_radial_free(self, conf='Both'):
        """
        Write radial part of valence orbitals for isolated atoms
        on radial grid to file as obtained by DFT calculation.
        
        writes: <Rnl_id>_Ofree_<orb_idx>.unf (for all orbital indices)
        """
        if conf == 'Both':
            Rnls_free = self.Rnls_conf
        else:
            Rnls_free = self.Rnls_free
        for iOrb in range(self.nOrbs):
            iAtom = self.Orb2Atom[iOrb]
            sym = self.species[iAtom]
            f = FortranFile(self.Rnl_id+"_Ofree_{0:d}.unf".format(iOrb+1), 'w')
            f.write_record( Rnls_free[str(iAtom)][self.otypes[iOrb][0]](self.rgrid[sym]) )
            f.close()
        
    
    def _get_radial_conf(self, conf='Both'):
        """
        Calculate radial parts of atomic wave functions
        for all atoms in additional harmonic confinement potential.
        """
        self.Rnls_conf = {}
        for iAtom in range(self.nAtoms):
            sym = self.species[iAtom]
            self.Rnls_conf[str(iAtom)] = {}
            if (conf == 'Both'):
                r_conf = 2.*data[sym]['R_cov']/Bohr
            else:
                r_conf = conf[iAtom]/Bohr
            
            atom = KSAllElectron(sym, confinement={'mode':'quadratic','r0':r_conf})
            atom.run()
            for iOrb in atom.get_valence_orbitals():
                self.Rnls_conf[str(iAtom)][iOrb[1]] = atom.Rnl_fct[iOrb]
        
    
    def _write_radial_conf(self,conf='Both'):
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
        for iOrb in range(self.nOrbs):
            iAtom = self.Orb2Atom[iOrb]
            sym = self.species[iAtom]
            f = FortranFile(self.Rnl_id+"_Oconf_{0:d}.unf".format(iOrb+1), 'w')
            f.write_record( Rnls_conf[str(iAtom)][self.otypes[iOrb][0]](self.rgrid[sym]) )
            f.close()
        
    
    def get_hvr(self):
        positions = (self.atoms.positions/Bohr).transpose()
        nkpts = len(self.wk)
        Orb2AtomF = (self.Orb2Atom + 1)
        hvr = np.array( HA.hirshfeld_main(self.nAtoms,nkpts,self.nOrbs,self.nThetas,self.nPhis,positions, \
                                          self.wf,self.wk,self.f,max(self.len_r),self.dr,self.nr,self.len_r, \
                                          self.rmins,self.Rnl_id,self.occ_free,self.otypes_num,Orb2AtomF, \
                                          self.Atom2Orbs) )
                                
        self.remove_files()
        return hvr
        
    
    def remove_files(self):
        """ Remove dump files containing Rnl data """
        
        system("rm -rf "+self.Rnl_id+"_*")
        
    

#--EOF--#
