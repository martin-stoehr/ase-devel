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
from ase.calculators.calculator import PropertyNotImplementedError


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
    
    implemented_properties = ['energy', 'forces', 'stress', 'charges']
    
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='dftb', atoms=None, kpts=None, **kwargs):
        """  Construct a DFTB+ calculator.  """
        
        from ase.dft.kpoints import monkhorst_pack
        from os.path import exists as pexists
        
        
        do_3rd_o = kwargs.get('Hamiltonian_ThirdOrder', 'No')
        do_3rd_f = kwargs.get('Hamiltonian_ThirdOrderFull', 'No')
        do_3rd_order = any(np.asarray([do_3rd_o.lower(), do_3rd_f.lower()]) =='yes' )
        
        default_beta_MBD = 0.89 if do_3rd_order else 0.95
        default_sR_TS = 1.03 if do_3rd_order else 1.06
        
        if 'DFTB_PREFIX' in os.environ:
            slako_dir = os.environ['DFTB_PREFIX']
            if not slako_dir.endswith('/'): slako_dir += '/'
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
            Options_WriteCPA='No',
            Options_WriteDetailedOut='Yes',
            Analysis_='',
            Analysis_CalculateForces='Yes',
            Hamiltonian_='DFTB',
            Hamiltonian_Scc='Yes',
            Hamiltonian_SccTolerance = 1.0E-005,
            Hamiltonian_SlaterKosterFiles_='Type2FileNames',
            Hamiltonian_SlaterKosterFiles_Prefix=slako_dir,
            Hamiltonian_SlaterKosterFiles_Separator='"-"',
            Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
            Hamiltonian_MaxAngularMomentum_ = ''
            )
        
        ## set default maximum angular momentum to consider
        maxang = 'Hamiltonian_MaxAngularMomentum_'
        for s in list(set(atoms.get_chemical_symbols())):
            self.default_parameters[maxang+s] = DefaultMaxAngMom[s]
        
        self.pbc = np.any(atoms.pbc)
        self.default_parameters['Driver']='{}'
        
        if do_3rd_order:
            hcorr = 'Hamiltonian_HCorrection_'
            hubder = 'Hamiltonian_HubbardDerivs_'
            self.default_parameters[hcorr] = 'Damping'
            self.default_parameters[hcorr+'Exponent'] = '4.05'
            self.default_parameters[hubder] = ''
            for s in list(set(atoms.get_chemical_symbols())):
                idU = kwargs.get(hubder+s, 'inputdoesntlooklikethis')
                if (not (idU=='inputdoesntlooklikethis')):
                    self.default_parameters[hubder+s] = idU
                else:
                    try:
                        self.default_parameters[hubder+s] = DefaultdU[s]
                    except KeyError:
                        msg  = "Hubbard Derivative for '"+s+"' not found. "
                        msg += "Please specify on input or implement."
                        raise NotImplementedError(msg)
        
        
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
        
        dispkey = 'Hamiltonian_Dispersion_'
        vdWmethod = self.parameters.get(dispkey, 'No_vdW_method_defined')
        doMBD = ( vdWmethod.lower()=='mbd' )
        doTS = ( vdWmethod.lower()=='ts' )
        self.CPAavail = (self.parameters['Options_WriteCPA'].lower() == 'yes')
        self.CPAavail = self.CPAavail and (doMBD or doTS)
        self.do_vdW = dispkey in self.parameters.keys()
        if doMBD:
            if not (dispkey+'Beta' in self.parameters.keys()):
                self.parameters[dispkey+'Beta'] = default_beta_MBD
        elif doTS:
            if not (dispkey+'RangeSeparation' in self.parameters.keys()):
                self.parameters[dispkey+'RangeSeparation'] = default_sR_TS
        if self.pbc and (doMBD or doTS):
            if not (dispkey+'KGrid' in self.parameters.keys()):
                if self.kpts is None:
                    kpts_vdW = '1 1 1'
                else:
                    kpts_vdW = ' '.join([str(k) for k in self.kpts])
                
                self.parameters[dispkey+'KGrid'] = kpts_vdW
        
        _calc_forces = self.parameters['Analysis_CalculateForces']
        self.calculate_forces = ( _calc_forces.lower()=='yes' )
        _writeDet = self.parameters['Options_WriteDetailedOut']
        self.writeDetOut = (_writeDet.lower() == 'yes')
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
        self.index_stress_begin = None
        self.index_stress_end = None
        self.index_charges_begin = None
        self.index_charges_end = None
        
    
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
        from os import remove
        
        myfile = open('results.tag', 'r')
        self.lines = myfile.readlines()
        myfile.close()
        if self.first_time:
            self.first_time = False
            # line indices
            estring, fstring = 'total_energy', 'forces   '
            sstring, qstring = 'stress   ', 'gross_atomic_charges'
            found_indices = [False,False,False,False]
            for iline, line in enumerate(self.lines):
                if line.find(estring) >= 0:
                    self.index_energy = iline + 1
                    found_indices[0] = True
                    continue
                if line.find(fstring) >= 0:
                    self.index_force_begin = iline + 1
                    line1 = line.replace(':', ',')
                    self.index_force_end = iline+1+int(line1.split(',')[-1])
                    found_indices[1] = True
                    continue
                if line.find(sstring) >= 0:
                    self.index_stress_begin = iline + 1
                    self.index_stress_end = iline + 4
                    found_indices[2] = True
                    continue
                if line.find(qstring) >= 0:
                    self.index_charges_begin = iline + 1
                    line1 = line.replace(':', ',')
                    nqlines = (int(line1.split(',')[-1]) - 1)//3 + 1
                    self.index_charges_end = iline + 1 + nqlines
                    found_indices[3] = True
                    continue
                if all(found_indices): break
            
        self.read_energy()
        if self.calculate_forces:
            self.read_forces()
            if self.pbc: self.read_stress()
        
        self.read_charges()
        if self.writeDetOut: self.read_DetailedOut()
        if self.CPAavail: self.read_CPA()
        
        os.remove('results.tag')
        
    
    def read_energy(self):
        """Read Energy from dftb output file (results.tag)."""
        from ase.units import Hartree
        
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
        
    
    def read_stress(self):
        """Read Stress from dftb output file (results.tag)."""
        from ase.units import Hartree, Bohr
        try:
            _stress = []
            for j in range(self.index_stress_begin, self.index_stress_end):
                word = self.lines[j].split()
                _stress.append([float(word[k]) for k in range(0, 3)])

            # Convert 3x3 stress tensor to Voigt form (xx,yy,zz,yz,xz,xy)
            # as required by ASE
            _stress = np.array(_stress).flat[[0, 4, 8, 5, 2, 1]]
            self.results['stress'] =  _stress * Hartree / Bohr
        except:
            raise RuntimeError('Problem in reading stress')
        
    
    def read_charges(self):
        """Read Charges from dftb output file (results.tag)."""
        try:
            _qstart = self.index_charges_begin
            _qend = self.index_charges_end
            _qtxt = " ".join(self.lines[_qstart:_qend])
            _charges = np.array(_qtxt.split(), dtype=float)
            self.results['charges'] = _charges
        except:
            raise RuntimeError('Problem in reading charges')
        
    
    def read_DetailedOut(self):
        """
        Read energy components from detailed.out.
        """
        try:
            fdet = open('detailed.out', 'r')
            lines = fdet.readlines()
            fdet.close()
        except IOError:
            msg = "No file 'detailed.out'. Something went terribly wrong."
            raise RuntimeError(msg)
        
        read_properties = [False, False, False]
        for line in lines:
            if line.find('Total Electronic energy: ') >= 0:
                self.E_el = float(line.split()[-2])
                read_properties[0] = True
                continue
            if line.find('Repulsive energy:        ') >= 0:
                self.E_rep = float(line.split()[-2])
                read_properties[1] = True
                continue
            if line.find('Dispersion energy:       ') >= 0:
                self.E_vdW = float(line.split()[-2])
                read_properties[2] = True
                continue
            if all(read_properties): return
        
    
    def read_CPA(self):
        """
        Read CPA ratios as returned by DFTB+, length = nAtoms
        by Martin Stoehr, martin.stoehr@tum.de (Oct/20/2015)
        """
        try:
            fCPA = open('CPA_ratios.out','r')
            lines = fCPA.readlines()[1:]
            fCPA.close()
            CPA = np.zeros(len(self.atoms))
            for iAtom in range(len(self.atoms)):
                CPA[iAtom] = float(lines[iAtom].split()[-1])
            self.CPA_ratios = CPA
            return
        except IOError:
            msg = "No file 'CPA_ratios.out'. Something went terribly wrong."
            raise RuntimeError(msg)
        
    
    def get_hirsh_volrat(self):
        """
        Return rescaling ratios for atomic polarizabilities (CPA ratios)
        """
        if hasattr(self, 'CPA_ratios'):
            return self.CPA_ratios
        else:
            msg  = "Could not obtain CPA ratios. You  need to specify the "
            msg += "MBD or TS dispersion model and set "
            msg += "Options_WriteCPA = 'Yes'"
            raise PropertyNotImplementedError(msg)
        
    
    def get_stress(self, atoms):
        if self.calculate_forces and self.pbc:
            return FileIOCalculator.get_stress(self, atoms)
        else:
            raise PropertyNotImplementedError
        
    
    def get_dispersion_energy(self):
        """"
        Return van der Waals dispersion energy as obtained by MBD/TS scheme.
        """
        if not self.do_vdW:
            raise ValueError('You did not specify a dispersion model.')
        elif self.writeDetOut:
            return self.E_vdW
        else:
            msg  = "Need to enable output of 'detailed.out' to get "
            msg += 'dispersion energy'
            raise ValueError(msg)
       
    
    def get_electronic_energy(self):
        """
        Return electronic energy = EBS + EC (+ E3rd) in eV.
        """
        if self.writeDetOut:
            return self.E_el
        else:
            msg  = "Need to enable output of 'detailed.out' to get "
            msg += 'electronic energy'
            raise ValueError(msg)
        
    
    def get_repulsive_energy(self):
        """
        Return repulsive energy = Etot - EBS - EC (- E3rd) in eV.
        """
        if self.writeDetOut:
            return self.E_rep
        else:
            msg  = "Need to enable output of 'detailed.out' to get "
            msg += 'repulsive energy'
            raise ValueError(msg)
        
    

#--EOF--#
