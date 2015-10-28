"""This module defines an ASE interface to DftbPlus

http://http://www.dftb-plus.info//
http://www.dftb.org/

markus.kaukonen@iki.fi

The file 'geom.out.gen' contains the input and output geometry
and it will be updated during the dftb calculations.

If restart == None
                   it is assumed that a new input file 'dftb_hsd.in'
                   will be written by ase using default keywords
                   and the ones given by the user.

If restart != None
                   it is assumed that keywords are in file restart

The keywords are given, for instance, as follows::

    Hamiltonian_SCC ='YES',
    Hamiltonian_SCCTolerance = 1.0E-008,
    Hamiltonian_MaxAngularMomentum = '',
    Hamiltonian_MaxAngularMomentum_O = '"p"',
    Hamiltonian_MaxAngularMomentum_H = '"s"',
    Hamiltonian_InitialCharges_ = '',
    Hamiltonian_InitialCharges_AllAtomCharges_ = '',
    Hamiltonian_InitialCharges_AllAtomCharges_1 = -0.88081627,
    Hamiltonian_InitialCharges_AllAtomCharges_2 = 0.44040813,
    Hamiltonian_InitialCharges_AllAtomCharges_3 = 0.44040813,

"""

import os

import numpy as np

from ase.calculators.calculator import FileIOCalculator, kpts2mp

## analysis of atomic polarizabilities via Hirshfeld volume ratios (M.S. 19/Oct/15)
from ase.io import read
from box.data import data
from ase.calculators.ext_OPA_DFTB import OverlapPopulationVolumeAnalysis
from ase.calculators.ext_HA_wrapper import HirshfeldWrapper


class Dftb(FileIOCalculator):
    """ A dftb+ calculator with ase-FileIOCalculator nomenclature
    """
    if 'DFTB_COMMAND' in os.environ:
        command = os.environ['DFTB_COMMAND'] + ' > PREFIX.out'
    else:
        command = 'dftb+ > PREFIX.out'

    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='dftb', atoms=None, kpts=None,
                 **kwargs):
        """Construct a DFTB+ calculator.
        """

        from ase.dft.kpoints import monkhorst_pack

        if 'DFTB_PREFIX' in os.environ:
            slako_dir = os.environ['DFTB_PREFIX']
        else:
            slako_dir = './'

        self.default_parameters = dict(
            Hamiltonian_='DFTB',
            Driver_='ConjugateGradient',
            Driver_MaxForceComponent='1E-4',
            Driver_MaxSteps=0,
            Hamiltonian_SlaterKosterFiles_='Type2FileNames',
            Hamiltonian_SlaterKosterFiles_Prefix=slako_dir,
            Hamiltonian_SlaterKosterFiles_Separator='"-"',
            Hamiltonian_SlaterKosterFiles_Suffix='".skf"'
            )

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        self.kpts = kpts
        # kpoint stuff by ase
        if self.kpts != None:
            mpgrid = kpts2mp(atoms, self.kpts)
            mp = monkhorst_pack(mpgrid)
            initkey = 'Hamiltonian_KPointsAndWeights'
            self.parameters[initkey + '_'] = ''
            for i, imp in enumerate(mp):
                key = initkey + '_empty' + str(i)
                self.parameters[key] = str(mp[i]).strip('[]') + ' 1.0'

        #the input file written only once
        if restart == None:
            self.write_dftb_in()
        else:
            if os.path.exists(restart):
                os.system('cp ' + restart + ' dftb_in.hsd')
            if not os.path.exists('dftb_in.hsd'):
                raise IOError('No file "dftb_in.hsd", use restart=None')

        #indexes for the result file
        self.first_time = True
        self.index_energy = None
        self.index_force_begin = None
        self.index_force_end = None
        
        ## default approach to Hirshfeld rescaling ratios (Martin Stoehr)
        self.hvr_approach = 'OPA'
        
    
    def write_dftb_in(self):
        """ Write the innput file for the dftb+ calculation.
            Geometry is taken always from the file 'geo_end.gen'.
        """

        outfile = open('dftb_in.hsd', 'w')
        outfile.write('Geometry = GenFormat { \n')
        outfile.write('    <<< "geo_end.gen" \n')
        outfile.write('} \n')
        outfile.write(' \n')

        #--------MAIN KEYWORDS-------
        previous_key = 'dummy_'
        myspace = ' '
        for key, value in sorted(self.parameters.items()):
            current_depth = key.rstrip('_').count('_')
            previous_depth = previous_key.rstrip('_').count('_')
            for my_backsclash in reversed(\
                range(previous_depth - current_depth)):
                outfile.write(3 * (1 + my_backsclash) * myspace + '} \n')
            outfile.write(3 * current_depth * myspace)
            if key.endswith('_'):
                outfile.write(key.rstrip('_').rsplit('_')[-1] + \
                                  ' = ' + str(value) + '{ \n')
            elif key.count('_empty') == 1:
                outfile.write(str(value) + ' \n')
            else:
                outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
            previous_key = key
        current_depth = key.rstrip('_').count('_')
        for my_backsclash in reversed(range(current_depth)):
            outfile.write(3 * my_backsclash * myspace + '} \n')
        #output to 'results.tag' file (which has proper formatting)
        outfile.write('Options { \n')
        outfile.write('   WriteResultsTag = Yes  \n')
        outfile.write('   WriteEigenvectors = Yes  \n')
        outfile.write('} \n')
        outfile.close()

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
            self.write_dftb_in()

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        return system_changes

    def write_input(self, atoms, properties=None, system_changes=None):
        from ase.io import write
        FileIOCalculator.write_input(\
            self, atoms, properties, system_changes)
        self.write_dftb_in()
        write('geo_end.gen', atoms)

    def read_results(self):
        """ all results are read from results.tag file
            It will be destroyed after it is read to avoid
            reading it once again after some runtime error """
        from ase.io import read
        from os import remove

        myfile = open('results.tag', 'r')
        self.lines = myfile.readlines()
        myfile.close()
        if self.first_time:
            self.first_time = False
            # Energy line index
            for iline, line in enumerate(self.lines):
                estring = 'total_energy'
                if line.find(estring) >= 0:
                    self.index_energy = iline + 1
                    break
            # Force line indexes
            for iline, line in enumerate(self.lines):
                fstring = 'forces   '
                if line.find(fstring) >= 0:
                    self.index_force_begin = iline + 1
                    line1 = line.replace(':', ',')
                    self.index_force_end = iline + 1 + int(line1.split(',')[-1])
                    break
                
            ## get number of orbitals and k-points (Martin Stoehr)
            for line in self.lines:
                if 'eigenvalues  ' in line:
                    line1 = line.replace(':',',')
                    self.nk = int(line1.split(',')[-2])
                    self.nOrbs = int(line1.split(',')[-3])
                    break
            ## read further information (Martin Stoehr)
            self.read_additional_info()
            
            try:
                myfile = open('eigenvec.out','r')
                self.evlines = myfile.readlines()
                myfile.close()
                ## read-in LCAO-coefficients (Martin Stoehr)
                self.read_eigenvectors()
            except IOError:
                print("No file name 'eigenvec.out', set WriteEigenvectors = Yes,\n \
                       if you wish to use additional electronic structure properties.\n \
                       For density dependent dispersion corrections, for instance!")
                
        self.read_energy()
        # read geometry from file in case dftb+ has done steps
        # to move atoms, in that case forces are not read
        #if int(self.parameters['Driver_MaxSteps']) > 0:
            #self.atoms = read('geo_end.gen')
            #self.results['forces'] = np.zeros([len(self.atoms), 3])
        #else:
            #self.read_forces()
        self.read_forces()
        os.remove('results.tag')
        
    
    def read_additional_info(self):
        """
        Read additional info, i.e. nAtoms, positions, fillings, etc.
        by Martin Stoehr, martin.stoehr@tum.de (Oct/20/2015)
        """
        orbitalslist = {0:['s'], 1:['py','pz','px'], 2:['dxy','dyz','dz2','dxz','dx2-y2'], \
                       3:['f3yx2-y3','fxyz','fyz2','fz3','fxz2','fzx2-zy2','fx3-3xy2']}
        self.atoms = read('geo_end.xyz')
        self.nAtoms = len(self.atoms)
        self.atomic_charges = np.zeros(self.nAtoms)
        self.Orb2Atom = np.zeros(self.nOrbs)
        self.otypes = []
        ## read 'detailed.out' in lines
        myfile = open('detailed.out', 'r')
        linesdet = myfile.readlines()
        myfile.close()
        ## net atomic charges and basis orbital types
        for iline, line in enumerate(linesdet):
            if 'Net atomic charges (e)' in line:
                for iAtom in xrange(self.nAtoms):
                    self.atomic_charges[iAtom] = float(linesdet[iline+2+iAtom].split()[-1])
                iline = iline+2+self.nAtoms
            if 'Orbital populations (up)' in line:
                for iOrb in xrange(self.nOrbs):
                    self.Orb2Atom[iOrb] = int( linesdet[iline+2+iOrb].split()[0] ) - 1
                    l = int(linesdet[iline+2+iOrb].split()[2])
                    m = int(linesdet[iline+2+iOrb].split()[3])
                    self.otypes.append(orbitalslist[l][l+m])
            
        self.Atom2Orbs = np.zeros((self.nAtoms, 2))
        for iAtom in xrange(self.nAtoms):
            startOrb = list(self.Orb2Atom).index(iAtom) + 1
            nOrbsiAtom = len(self.Orb2Atom) - np.count_nonzero(self.Orb2Atom - iAtom) - 1
            self.Atom2Orbs[iAtom] = np.array([startOrb, startOrb + nOrbsiAtom])
            
        ## Fillings 
        myfile = open('detailed.out','r')
        textdet = myfile.read()   ## read 'detailed.out' as string
        myfile.close()
        textoccs = (textdet.split('Fillings')[1]).split('\n \n')[0]
        textoccs = np.array(textoccs.split(), dtype=float)  ## 1D array of occupations
        if len(textoccs) != (self.nOrbs*self.nk):
            print('Error in reading occupations (~> length). Skipping.')
            pass
        else:
            ## reshape array into shape = (n_kpoints, n_Orbitals)
            self.f = textoccs.reshape((self.nOrbs, self.nk)).T
        
        ## k-point weighting
        myfile = open('dftb_in.hsd','r')
        linesin = myfile.readlines()
        myfile.close()
        self.pbc = False
        self.wk = np.ones(self.nk)
        self.kpts = np.zeros((self.nk,3))
        for iline, line in enumerate(linesin):
            if 'KPointsAndWeights =' in line:
                self.pbc = True
                for ik in xrange(self.nk):
                    self.wk[ik] = float(linesin[iline+1+ik].split()[3])
                    self.kpts[ik,:] = np.array(linesin[iline+1+ik].split()[:3], dtype=float)
                break
        ## normalize k-point weighting factors (sum_k wk = 1)
        self.wk /= np.sum(self.wk)
        
    
    def read_eigenvectors(self):
        """
        Read LCAO-coefficients to self.wf, shape = (n_kpoints, n_States, n_Orbitals)
        by Martin Stoehr, martin.stoehr@tum.de (Oct/20/2015)
        """
        ## LCAO-coefficients
        self.wf = np.zeros((self.nk, self.nOrbs, self.nOrbs))
        c = []
        if self.pbc:
            for line in self.evlines[1:]:
                if (line.split() != []) and ('Eigenvector: ' not in line):
                    line1 = line.replace(',','').replace(')','')
                    c.append(float(line1.split()[-3]) + float(line1.split()[-2])*1.j)
            self.wf = np.array(c, dtype=complex).reshape((self.nk, self.nOrbs, self.nOrbs))
        else:
            for line in self.evlines[1:]:
                if (line.split() != []) and ('Eigenvector: ' not in line):
                    c.append(float(line.split()[1]))
            self.wf[0] = np.array(c).reshape((self.nk, self.nOrbs, self.nOrbs))
        
    
    def read_energy(self):
        """Read Energy from dftb output file (results.tag)."""
        from ase.units import Hartree
        
        # Energy:
        try:
            energy = float(self.lines[self.index_energy].split()[0]) * Hartree
            self.results['energy'] = energy
        except:
            raise RuntimeError('Problem in reading energy')

    def read_forces(self):
        """Read Forces from dftb output file (results.tag)."""
        from ase.units import Hartree, Bohr

        try:
            gradients = []
            for j in range(self.index_force_begin, self.index_force_end):
                word = self.lines[j].split()
                gradients.append([float(word[k]) for k in range(0, 3)])

            self.results['forces'] = np.array(gradients) * Hartree / Bohr

        except:
            raise RuntimeError('Problem in reading forces')
        
    
    #-------------------------------------------------#
    #  Approach(es) to atomic polarizabilities        #
    #  by Martin Stoehr, martin.stoehr@tum.de (M.S.)  #
    #-------------------------------------------------#
    
    def set_hvr_approach(self, approach):
        """ set approach for obtaining (approximate) Hirshfeld ratios """
        valid_approaches = ['const', 'OPA', 'HA']
        
        if approach in valid_approaches:
            self.hvr_approach = approach
        else:
            print('\033[91m'+"WARNING: '"+str(approach)+"' is not a valid identifier. \
                   Defaulting to constant ratios of 1 instead..."+'\033[0m')
            self.hvr_approach = 'const'
        
    
    def get_hirsh_volrat(self):
        """
        Return Hirshfeld volume ratios using method <approach>
        
        parameters (hard-coded so far):
        ===============================
            approach:  . 'HA'     actual Hirshfeld analysis using confined basis confinement
                                  (see ext_HA_DFTB.py)
                       . 'OPA'    approx. ratios as obtained by overlap population analysis
                                  (see ext_OPA_DFTB.py)
                       . 'const'  return constant ratios of 1.
        """
        if self.hvr_approach == 'const':
            self.hirsh_volrat = self.get_hvr_const()
        elif self.hvr_approach == 'HA':
            self.hirsh_volrat = self.get_hvr_HA()
        elif self.hvr_approach == 'OPA':
            self.hirsh_volrat = self.get_hvr_OPA()
        else:
            raise NotImplementedError("Sorry dude, I don't know about a scheme called '"+approach+"'.")
        return self.hirsh_volrat
        
    
    def get_hvr_HA(self, dr=0.2, nThetas=36, nPhis=72, cutoff=3.,conf='Both'):
        """
        Return Hirsfeld volume ratios as obtained by density partitioning
        
        parameters:
        ===========
            dr(opt):       radial step width in Angstroms, default: 0.2 Angstroms
            nThetas(opt):  number of discrete azemuthal angles, default: 36
            nPhis(opt):    number of discrete polar angles, default: 72
            cutoff(opt):   cutoff radius for partial grid in Angstroms, default: 3 Angstroms
            conf(opt):     confinement for basis functions in density construction in Angstroms,
                           default: 'default' = (R_conf[symbol] .or. 5*R_cov[symbol] from box.data).
                           Alternative:
                             . 'None': use free radial wave functions throughout,
                             . 'Both': use confined radial wave functions throughout,
                             . list of confinement radii in Angstroms to use per atom.
        """
        Atom2OrbsF = np.array(self.Atom2Orbs, dtype=int).transpose()
        
        HA = HirshfeldWrapper(self.atoms, self.wk, self.wf, self.f, self.otypes, Atom2OrbsF, self.Orb2Atom, \
                              dr=dr, nThetas=nThetas, nPhis=nPhis, cutoff=cutoff,conf=conf)
        return HA.get_hvr()
        
    
    def get_hvr_OPA(self):
        """ Return (approximate) Hirshfeld ratios as obtained by on-site overlap population """
        syms = self.atoms.get_chemical_symbols()
        n_el_atom = np.zeros(self.nAtoms)
#        n_el_core = np.zeros(self.nAtoms)
        for iAtom in xrange(self.nAtoms):
#            Z = data[syms[iAtom]]['Z']
            n_el_atom[iAtom] = data[syms[iAtom]]['valence_number'] #Z
#            n_el_core[iAtom] = Z - data[syms[iAtom]]['valence_number']
        
        OPA = OverlapPopulationVolumeAnalysis(self.wf, self.f, self.wk, n_el_atom, self.Orb2Atom)
        return OPA.get_hvr()
        
    

#--EOF--#
