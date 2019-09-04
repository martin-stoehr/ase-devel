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
from ase.calculators.hb_box_data import data
from ase.calculators.ext_CPA_DFTB import ChargePopulationAnalysis
from ase.calculators.ext_HA_wrapper import HirshfeldWrapper


## default value of maximal angular momentum to be included in DFTB calculation (ease calculator init, MS)
DefaultMaxAngMom = { 'H':'"s"',                                                                  'He':'"s"', \
                    'Li':'"p"','Be':'"p"', 'B':'"p"', 'C':'"p"', 'N':'"p"', 'O':'"p"', 'F':'"p"','Ne':'"p"', \
                    'Na':'"p"','Mg':'"p"','Al':'"p"','Si':'"p"', 'P':'"p"', 'S':'"d"','Cl':'"d"','Ar':'"p"', \
                     'K':'"p"','Ca':'"p"','Sc':'"d"','Ti':'"d"', 'V':'"d"','Cr':'"d"','Mn':'"d"','Fe':'"d"', \
                    'Co':'"d"','Ni':'"d"','Cu':'"d"','Zn':'"d"','Ga':'"p"','Ge':'"p"','As':'"p"','Se':'"p"', \
                    'Br':'"d"','Kr':'"p"','Rb':'"p"','Sr':'"p"', 'Y':'"d"','Zr':'"d"','Nb':'"d"','Mo':'"d"', \
                    'Tc':'"d"','Ru':'"d"','Rh':'"d"','Pd':'"d"','Ag':'"d"','Cd':'"d"','In':'"p"','Sn':'"p"', \
                    'Sb':'"p"','Te':'"p"', 'I':'"d"','Xe':'"p"','Cs':'"p"','Ba':'"p"','Lu':'"d"','Hf':'"d"', \
                    'Ta':'"d"', 'W':'"d"','Re':'"d"','Os':'"d"','Ir':'"d"','Pt':'"d"','Au':'"d"','Hg':'"d"', \
                    'Tl':'"p"','Pb':'"p"','Bi':'"p"','Po':'"p"','As':'"p"','Rn':'"p"' }

# calculated reference values for Hubbard Derivatives (required for DFTB3 calculations)
DefaultdU = { \
            # from DFTB.ORG (http://www.dftb.org/parameters/download/3ob/3ob-3-1-cc/):
            'H':-0.1857, \
            'C':-0.1492, 'N':-0.1535, 'O':-0.1575, 'F':-0.1623, \
            'Na':-0.0454, 'Mg':-0.02, 'P':-0.14, 'S':-0.11, 'Cl':-0.0697, \
            'K':-0.0339, 'Ca':-0.0340, 'Zn':-0.03, 'Br':-0.0573, \
            'I':-0.0433, \
            }


class Dftb(FileIOCalculator):
    """  A dftb+ calculator with ase-FileIOCalculator nomenclature  """
    if 'DFTB_COMMAND' in os.environ:
        command = os.environ['DFTB_COMMAND'] + ' > PREFIX.out'
    else:
        command = 'dftb+ > PREFIX.out'
    
    implemented_properties = ['energy', 'forces', 'charges']
    
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='dftb', atoms=None, kpts=None, **kwargs):
        """  Construct a DFTB+ calculator.  """
        
        from ase.dft.kpoints import monkhorst_pack
        from os.path import exists as pexists
        
        
        do_3rd_o = kwargs.get('Hamiltonian_ThirdOrder', 'No')
        if (do_3rd_o.lower() == 'yes'):
            print("REMARK: You chose ThirdOrder = 'Yes'. This only corrects on-site terms.")
            print("        For full 3rd order DFTB, please use ThirdOrderFull = 'Yes'.")
        
        do_3rd_f = kwargs.get('Hamiltonian_ThirdOrderFull', 'No')
        do_3rd_order = any(np.asarray([do_3rd_o.lower(), do_3rd_f.lower()]) =='yes' )
        if 'DFTB_PREFIX' in os.environ:
            slako_dir = os.environ['DFTB_PREFIX']
            if ( do_3rd_order and (not pexists(slako_dir+'3rd_order/')) ):
                print("WARNING: You chose ThirdOrder(Full), but I didn't find the default directory")
                print("         '"+slako_dir+"3rd_order/' for .skf files")
                print("         Please, make sure they are in the working directory or specified otherwise!")
            elif do_3rd_order: slako_dir += '3rd_order/'
        else:
            slako_dir = './'
        
        self.default_parameters = dict(
            Options_='',
            Options_WriteResultsTag='Yes',
#            Options_WriteEigenvectors='No',
#            Options_WriteCPA='No',
#            Options_CalculateForces='Yes',
            Options_MinimiseMemoryUsage='No',
            Hamiltonian_='DFTB',
#            Hamiltonian_SCCTolerance = 1.0E-005,
            Hamiltonian_SlaterKosterFiles_='Type2FileNames',
            Hamiltonian_SlaterKosterFiles_Prefix=slako_dir,
            Hamiltonian_SlaterKosterFiles_Separator='"-"',
            Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
            Hamiltonian_MaxAngularMomentum_ = ''
            )
        
        ## set default maximum angular momentum to consider
        for species in list(set(atoms.get_chemical_symbols())):
            self.default_parameters['Hamiltonian_MaxAngularMomentum_'+species] = DefaultMaxAngMom[species]
        
        self.pbc = np.any(atoms.pbc)
        ## control whether DFTB+ should calculate forces or enable singe-point calculations (large systems!)
        calc_forces = kwargs.get('Options_CalculateForces', 'Yes')
        self.calculate_forces = ( calc_forces.lower()=='yes' )
        if self.calculate_forces:
            self.default_parameters['Driver_']='ConjugateGradient'
            self.default_parameters['Driver_MaxForceComponent']='1E-4'
            self.default_parameters['Driver_MaxSteps']=0
        else:
            self.default_parameters['Driver']='{}'
        
        minmem = kwargs.get('Options_MinimiseMemoryUsage', 'No')
        if minmem.lower() == 'yes':
            defaultsolver = 'Standard{}'
        else:
            defaultsolver = 'DivideAndConquer{}'
        
        dftbsolver = kwargs.get('Hamiltonian_Eigensolver', defaultsolver)
        if not dftbsolver.endswith('{}'): dftbsolver += '{}'
        if (dftbsolver in ['QR{}','qr{}']): dftbsolver = 'Standard{}'
        if not dftbsolver in ['Standard{}', 'DivideAndConquer{}']:
            print("Eigensolver '"+dftbsolver+"' not known. Defaulting to '"+defaultsolver+"'.")
            dftbsolver = defaultsolver
        
        kwargs['Hamiltonian_Eigensolver'] = dftbsolver
        
        if do_3rd_order:
            self.default_parameters['Hamiltonian_DampXH'] = 'Yes'
            self.default_parameters['Hamiltonian_DampXHExponent'] = '4.00'
            self.default_parameters['Hamiltonian_HubbardDerivs_'] = ''
            for species in list(set(atoms.get_chemical_symbols())):
                input_dU = kwargs.get('Hamiltonian_HubbardDerivs_'+species, 'inputdoesntlooklikethis')
                if (not (input_dU=='inputdoesntlooklikethis')):
                    self.default_parameters['Hamiltonian_HubbardDerivs_'+species] = input_dU
                else:
                    try:
                        self.default_parameters['Hamiltonian_HubbardDerivs_'+species] = DefaultdU[species]
                    except KeyError:
                        raise NotImplementedError("Hubbard Derivative for '"+species+"' not found. Please specify on input or implement.")
        
        ## default approach to Hirshfeld rescaling ratios (Martin Stoehr)
        ## 'CPA' native in newest DFTB+ versions
        self.hvr_approach = 'CPA'
        
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
        
        self.atoms = atoms
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
        
    
    def write_dftb_in(self):
        """
        Write the input file for the dftb+ calculation.
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
        if (self.hvr_approach == 'HA'):
            self.parameters['Options_WriteEigenvectors'] = 'Yes'

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
            self.solved_hvr = {'HA':False, 'CPA':False}
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
            
            ## read further information for SCC calculations (Martin Stoehr)
            ## where is this information for non-SCC calculations
            hSCC = 'Hamiltonian_SCC'
            if self.parameters.has_key(hSCC):
                if (self.parameters[hSCC].lower() == 'yes'):
                    self.read_additional_info()
                else:
                    print('You started a non-SCC calculation. No additional Information available.')
            
            try:
                myfile = open('eigenvec.out','r')
                self.evlines = myfile.readlines()
                myfile.close()
                ## read-in LCAO-coefficients (Martin Stoehr)
                self.read_eigenvectors()
                self.eigenvectors_missing = False
            except IOError:
                self.eigenvectors_missing = True
            
        self.read_energy()
        # read geometry from file in case dftb+ has done steps
        # to move atoms, in that case forces are not read
        #if int(self.parameters['Driver_MaxSteps']) > 0:
            #self.atoms = read('geo_end.gen')
            #self.results['forces'] = np.zeros([len(self.atoms), 3])
        #else:
            #self.read_forces()
        if self.calculate_forces:
            self.read_forces()
        
        os.remove('results.tag')
        
    
    def read_additional_info(self):
        """
        Read additional info, i.e. nAtoms, positions, fillings, etc.
        by Martin Stoehr, martin.stoehr@tum.de (Oct/20/2015)
        """
        orbitalslist = {0:['s'], 1:['py','pz','px'], 2:['dxy','dyz','dz2','dxz','dx2-y2'], \
                       3:['f3yx2-y3','fxyz','fyz2','fz3','fxz2','fzx2-zy2','fx3-3xy2']}
        try:
            atoms_end = read('geo_end.xyz')
        except IOError:
            atoms_end = self.atoms
        
        self.atoms = atoms_end
        self.nAtoms = len(self.atoms)
        self.charges = np.zeros(self.nAtoms)
        self.Orb2Atom = np.zeros(self.nOrbs)
        self.otypes = []
        ## read 'detailed.out' in lines
        myfile = open('detailed.out', 'r')
        linesdet = myfile.readlines()
        myfile.close()
        ## net atomic charges and basis orbital types
        for iline, line in enumerate(linesdet):
            if 'Net atomic charges (e)' in line:
                for iAtom in range(self.nAtoms):
                    self.charges[iAtom] = float(linesdet[iline+2+iAtom].split()[-1])
                iline = iline+2+self.nAtoms
            if 'Orbital populations (up)' in line:
                for iOrb in range(self.nOrbs):
                    self.Orb2Atom[iOrb] = int( linesdet[iline+2+iOrb].split()[0] ) - 1
                    l = int(linesdet[iline+2+iOrb].split()[2])
                    m = int(linesdet[iline+2+iOrb].split()[3])
                    self.otypes.append(orbitalslist[l][l+m])
            
        self.results['charges'] = self.charges
        self.Atom2Orbs = np.zeros((self.nAtoms, 2))
        for iAtom in range(self.nAtoms):
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
        
        ## electronic energy: move this out of read_additional_info!!
        textelec = textdet.split('Total Electronic energy:')[1].split('\n')[0]
        self.electronic_energy = float(textelec.split()[-2])
        
        ## dispersion energy (MBD/TS): move this out of read_additional_info!!
        try:
            vdWmode = self.parameters['Hamiltonian_ManyBodyDispersion_']
            oldvdW = True
        except KeyError:
            try:
                vdWmode = self.parameters['Hamiltonian_Dispersion_']
                olfvdW = False
            except KeyError:
                vdWmode = 'none'
        
        if ( (vdWmode == 'MBD') or (vdWmode == 'TS') ):
            if oldvdW:
                textdisp = textdet.split('MBD/TS energy:')[1].split('\n')[0]
            else:
                if (vdWmode == 'MBD'):
                    textdisp = textdet.split('Many-body dispersion energy:')[1].split('\n')[0]
                elif (vdWmode == 'TS'):
                    textdisp = textdet.split('Dispersion energy:')[1].split('\n')[0]
            
            self.dispersion_energy = float(textdisp.split()[-2])
#        else:
#            print('No dispersion model defined.')
        
        ## k-point weighting
        myfile = open('dftb_in.hsd','r')
        linesin = myfile.readlines()
        myfile.close()
        self.wk = np.ones(self.nk)
        self.kpts = np.zeros((self.nk,3))
        for iline, line in enumerate(linesin):
            if 'KPointsAndWeights =' in line:
                for ik in range(self.nk):
                    self.wk[ik] = float(linesin[iline+1+ik].split()[3])
                    self.kpts[ik,:] = np.array(linesin[iline+1+ik].split()[:3], dtype=float)
                break
        ## normalize k-point weighting factors (sum_k wk = 1)
        ## In principle, this should not be neccessary. Anyway, ...
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
        
    
    def read_CPA(self):
        """
        Read CPA ratios as returned by DFTB+, length = nAtoms
        by Martin Stoehr, martin.stoehr@tum.de (Oct/20/2015)
        """
        try:
            fCPA = open('CPA_ratios.out','r')
            lines = fCPA.readlines()[1:]
            fCPA.close()
            self.hvr_CPA = np.zeros(self.nAtoms)
            for iAtom in range(self.nAtoms):
                self.hvr_CPA[iAtom] = float(lines[iAtom].split()[-1])
            
        except IOError:
            raise RuntimeError("No file 'CPA_ratios.out'. Something went terribly wrong.")
        
    
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
        
    
    #-------------------------------------------#
    #  Approach(es) to atomic polarizabilities  #
    #  by Martin Stoehr, martin.stoehr@uni.lu   #
    #-------------------------------------------#
    
    def set_hvr_approach(self, approach):
        """ set approach for obtaining (approximate) Hirshfeld ratios """
        valid_approaches = ['const', 'CPA', 'HA']
        
        if approach in valid_approaches:
            self.hvr_approach = approach
        else:
            print('\033[91m'+"WARNING: '"+str(approach)+"' is not a valid identifier. \
                   Defaulting to ratios as obtained by charge population approach instead..."+'\033[0m')
            self.hvr_approach = 'CPA'
        
    
    def get_hirsh_volrat(self):
        """
        Return Hirshfeld volume ratios using method <self.approach>
        """
        if self.hvr_approach == 'const':
            self.rescaling = self.get_hvr_const()
        elif self.hvr_approach == 'HA':
            self.rescaling = self.get_hvr_HA()
        elif self.hvr_approach == 'CPA':
            self.rescaling = self.get_hvr_CPA()
        else:
            raise NotImplementedError("Sorry dude, I don't know about a scheme called '"+self.hvr_approach+"'.")
        return self.rescaling
        
    
    def get_hvr_const(self):
        """  Return constant rescaling (use free atom parameters).  """
        
        return np.ones(self.nAtoms)
        
    
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
                             . 'None':     use free radial wave functions throughout,
                             . 'Both':     use confined radial wave functions throughout (DEFAULT),
                             . list of confinement radii in Angstroms to use per atom.
        """
        if self.eigenvectors_missing:
            raise ValueError('No eigenvectors available. Please set WriteEigenvectors = True for DFTB calculation.')
        
        Atom2OrbsF = np.array(self.Atom2Orbs, dtype=int).transpose()
        HA = HirshfeldWrapper(self.atoms, self.wk, self.wf, self.f, self.otypes, Atom2OrbsF, self.Orb2Atom, \
                              dr=dr, nThetas=nThetas, nPhis=nPhis, cutoff=cutoff,conf=conf)
        
        self.hvr_HA = HA.get_hvr()
        
        return self.hvr_HA
        
    
    def get_hvr_CPA(self):
        """  Return rescaling ratios as obtained by charge population approach.  """
        doCPA1 = (self.parameters['Options_WriteCPA'].lower() == 'yes')
        doCPA2 = (self.parameters['Options_WriteCPA'].lower() == 'true')
        if (doCPA1 or doCPA2):
            self.read_CPA()
        else:
            raise ValueError("Sorry, you need to set Options_WriteCPA = 'Yes' for this feature.")
        
        return self.hvr_CPA
        
    
    def get_dispersion_energy(self):
        """"
        Return van der Waals dispersion energy as obtained by MBD/TS scheme.
        """
        try:
            vdWmode = self.parameters['Hamiltonian_ManyBodyDispersion_']
        except KeyError:
            try:
                vdWmode = self.parameters['Hamiltonian_Dispersion_']
            except KeyError:
                vdWmode = 'none'
        
        if ( (vdWmode == 'MBD') or (vdWmode == 'TS') ):
            return self.dispersion_energy
        else:
            raise ValueError('You did not specify a dispersion model.')
        
    
    def get_electronic_energy(self):
        """
        Return electronic energy = Etot - EBS - EC (- E3rd) in eV.
        """
        return self.electronic_energy
        
    

#--EOF--#
