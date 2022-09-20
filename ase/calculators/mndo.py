"""This module defines an ASE interface to MNDO (developed for MNDO2020)

https://mndo.kofo.mpg.de

written by Martin Stoehr (martin.stoehr@stanford.edu)
Sep 20, 2022.

"""

import os

import numpy as np

from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError


## list of available MNDO keywords  # equiv. human-readable keyword
mndo_kwargs = [
    'iform',      ## THIS IS FIXED TO 1 (UNFORMATTED)! ##
    'limit',      # maxtime
    'iop',        # method
    'jop',        # calc_type
    'igeom',      # coord_type
    'mplib',      # parallel_mode
    'ief',        # optimizer
    'idiis',      # diis_mode
    'inrefd',     # ref_data
    'iparok',     # custom_param
    'mminp',      # external_pts
    'nmr',        # compute_nmr
    'nsav7',      # write_full_input
    'nsav8',      # output_mopac
    'nsav9',      # output_pdb
    'nsav13',     # output_aux
    'nsav15',     # output_each
    'nsav16',     # output_sybyl
    'immok',      # peptide_correction
    'ihbond',     # hbond_dmax
    'ifld1',      # external_field
    'ifld2',      # field_strength
    'ifld3',      # ext_field_scf
    'icuts',      # cutoff_3center_ovlp
    'icutg',      # cutoff_3center_grad
    'iexbas',     # polarized_basis
    'icosmo',     # cosmo_mode
    'ipsana',     # analyt_grad
    'immdp',      # vdw_corr
    'inac',       # nac_mode
    'maxend',     # max_scf_opt
    'maxlin',     # max_scf_ls
    'maxrtl',     # max_scf_tot
    'iscf',       # ene_threshold
    'iplscf',     # rho_threshold
    'middle',     # continuation
    'iprint',     # write_opt
    'kprint',     # write_force_const
    'lprint',     # write_vib
    'mprint',     # write_gradients
    'jprint',     # write_input
    'iprec',      # opt_fmax
    'iconv',      # opt_fmax_type
    'ihess',      # init_hessian
    'idfp',       # update_type_hessian
    'nrepet',     # special_convergence
    'linitg',     # check_f_init
    'lconvg',     # check_f_norm
    'lgdum',      # check_f_requested
    'ihdlc1',     # hdlc_coords
    'ihdlc2',     # hdlc_coords_core
    'ihdlc3',     # hdlc_extra
    'ingeom',     # special_geom_in
    'intdir',     # direct_scf
    'lindms',     # cgdms
    'lindia',     # cgdms_diag
    'linfrg',     # guess_from_frag
    'inpfrg',     # extra_inp_frag
    'inp21',      # extra_in21
    'inp22',      # extra_in22
    'inp23',      # extra_in23
    'inp24',      # extra_in24
    'inp25',      # extra_in25
    'iaterg',     # atomization_type
## general
    'kharge',     # charge
    'imult',      # multiplicity
    'ktrial',     # rdm_init
    'kgeom',      # geometry_grid
    'ipubo',      # save_scf
    'iuhf',       # scf_type
    'kitscf',     # max_scf
    'nprint',     # output_lvl
    'ifast',      # pseudodiag_type
    'idiag',      # diag_type
    'ksym',       # symmetry
    'numsym',     # symmetry_num
    'kci',        # ci_type
    'nstart',     # extrapol_1st_scf
    'nstep',      # extrapol_scf_step
    'ktitle',     # title
## optimizer and force constant calcs
    'nrst',       # reset_hessian
    'ldrop',      # restart_opt_thresh
    'ldell',      # updated_restart
    'lsub',       # which_ls
    'lalpha',     # ls_dinit
    'lconv',      # ls_thresh_step
    'ltolf',      # ls_thresh_ene
    'lmaxst',     # ls_max_step
    'igrad',      # special_grad
    'lpoint',     # num_grad_pts
    'lfac',       # num_grad_step
    'kpoint',     # num_vib_pts
    'kfac',       # num_vib_step
    'kmass',      # vib_masses
    'ntemp',      # temperatures_n
    'ntemp1',     # temperatures_min
    'ntemp2',     # temepratures_max
]


## list of corresponding human readable keywords
kwargs_all = [
    'maxtime', 'method', 'calc_type', 'coord_type', 'parallel_mode',
    'optimizer', 'diis_mode', 'ref_data', 'custom_param', 'external_pts',
    'compute_nmr', 'write_full_input', 'output_mopac', 'output_pdb',
    'output_aux', 'output_each', 'output_sybyl', 'peptide_correction',
    'hbond_dmax', 'external_field', 'field_strength', 'ext_field_scf',
    'cutoff_3center_ovlp', 'cutoff_3center_grad', 'polarized_basis',
    'cosmo_mode', 'analyt_grad', 'vdw_corr', 'nac_mode', 'max_scf_opt',
    'max_scf_ls', 'max_scf_tot', 'ene_threshold', 'rho_threshold',
    'continuation', 'write_opt', 'write_force_const', 'write_vib',
    'write_gradients', 'write_input', 'opt_fmax', 'opt_fmax_type',
    'init_hessian', 'update_type_hessian', 'special_convergence',
    'check_f_init', 'check_f_norm', 'check_f_requested', 'hdlc_coords',
    'hdlc_coords_core', 'hdlc_extra', 'special_geom_in', 'direct_scf',
    'cgdms', 'cgdms_diag', 'guess_from_frag', 'extra_inp_frag', 
    'extra_in21', 'extra_in22', 'extra_in23', 'extra_in24', 'extra_in25',
    'atomization_type', 'charge', 'multiplicity', 'rdm_init', 
    'geometry_grid', 'save_scf', 'scf_type', 'max_scf', 'output_lvl',
    'pseudodiag_type', 'diag_type', 'symmetry', 'symmetry_num', 'ci_type',
    'extrapol_1st_scf', 'extrapol_scf_step', 'title', 'reset_hessian', 
    'restart_opt_thresh', 'updated_restart', 'which_ls', 'ls_dinit', 
    'ls_thresh_step', 'ls_thresh_ene', 'ls_max_step', 'special_grad', 
    'num_grad_pts', 'num_grad_step', 'num_vib_pts', 'num_vib_step', 
    'vib_masses', 'temperatures_n', 'temperatures_min', 'temepratures_max',
    'opt_mask']

## create dictionary of human-readable -> mndo kwargs
h2mndo_key = dict(zip(kwargs_all, mndo_kwargs))

## convert human-readable values of kwargs to mndo values
def h2mndo_val(hkey, hval):
    raise NotImplementedError("Need to implement this function!")
    return mndo_val

## fix input format to unformatted
iform = 1


class Mndo(FileIOCalculator):
    """  A MNDO calculator with ase-FileIOCalculator nomenclature  """
    if 'MNDO_COMMAND' in os.environ:
        command = os.environ['MNDO_COMMAND'] + ' < asejob_mndo.inp > PREFIX.out'
    else:
        command = 'mndo < asejob_mndo.inp > PREFIX.out'
    
    implemented_properties = ['energy', 'forces', 'charges', 'dipole']
    
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='mndo_ase', atoms=None, kpts=None, **kwargs):
        """  Construct a MNDO calculator.  """
        
        from os.path import exists as pexists
        
        
        self.default_parameters = dict(
            igeom=1,    # use Cartesian coordinates
        )
        
        ## translate human-readable to mndo kwargs and pop
        for key in kwargs.keys():
            if key in kwargs_all:
                kwargs[h2mndo_key[key]] = h2mndo_val(kwargs.pop(key))
        
        for key, val in self.default_parameters.items():
            if not key in kwargs.keys(): kwargs[key] = val
        
        kwargs['iform'] = iform
        self.opt_mask = kwargs.pop('opt_mask', None)
        
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        
        self.first_time = True
        self.label = label
        
    
    def set_atoms(self, atoms):
        self.atoms = atoms
    
    def write_mndo_in(self):
        """
        Write the input file for the dftb+ calculation.
        Geometry is taken always from the file 'geo_end.gen'.
        """

        self.nAtoms = len(self.atoms)
        self.pbc = np.any(self.atoms.pbc)
        if self.pbc: raise NotImplementedError('PBC not supported')
        
        arg_str = ''
        title = self.parameters.pop('ktitle', ' MNDO calculation from ASE')
        if self.opt_mask is None: self.opt_mask = np.zeros((self.nAtoms,3))
        for key, val in self.parameters.items():
            if key not in mndo_kwargs: continue
            test_len = len(key)+len(str(val))+3
            if (len(arg_str) + test_len > 78): arg_str += ' +\n'
            arg_str += key+'='+str(val)+' '
        
        if arg_str.count('+\n')>10: raise ValueError('Too many inputs!')
        inp = open('asejob_mndo.inp', 'w')
        inp.write(arg_str+'\n')
        inp.write('{0:76s}'.format(title.replace('\n',' ')))
        inp.write('\n#Generated by ASE\n')
        
        ## write geometry
        mol = self.get_molecule_array()
        gen  = '{0: >2.0f}     {1: >13.9f}   {2:1.0f}   {3: >13.9f}   '
        gen += '{4:1.0f}   {5: >13.9f}   {6:1.0f}\n'
        for iAtom in range(self.nAtoms): inp.write(gen.format(*mol[iAtom]))
        inp.write("0")
        
        
        ## write symmetry data
        #TODO, but do we need this to run?
        
        inp.close()
        
    def get_molecule_array(self):
        """
        Convert atomic numbers, positions and optimization mask into input
        array as used by MNDO:
        
        Cartesian:
        ==========
            Z[0]  x[0] opt?  y[0] opt?  z[0] opt?
            Z[1]  x[1] opt?  y[1] opt?  z[1] opt?
            ...
        
        internal:
        =========
            Z[0]
            Z[1]  d1   opt?                         i for d1=1--i
            Z[2]  d2   opt?  a2   opt?              i for d2=2--i   j for a2=2-i-j
            Z[3]  d3   opt?  a3   opt?   t3  opt?   i for d3=3--i   j for a3=3-i-j   k for t3=3-i-j-k
            ...
        
        """
        Z = self.atoms.get_atomic_numbers()
        xyz = self.atoms.positions
        
        if self.parameters['igeom'] == 1: # Cartesian
            res = np.insert(xyz, 0, Z, axis=1)
            res = np.insert(res, (2,3,4), self.opt_mask, axis=1)
        elif self.parameters['igeom'] == 0: # internal
            msg = "Using internal coordinates not supported yet!"
            raise NotImplementedError(msg)
        else:
            msg  = "'coord_type' has to be 'Cartesian' or 'internal'"
            msg += " (or equivalently 'igeom' has to be 1 or 0)!"
            raise ValueError(msg)
        return res
        
        
    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        return changed_parameters
    
    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        return system_changes
    
    def write_input(self, atoms, properties=None, system_changes=None):
        self.write_mndo_in()
    
    def read_results(self):
        myfile = open(self.label+'.out', 'r')
        outlines = myfile.readlines()
        myfile.close()
        if str(self.parameters['jop']) == '-2':
            myfile = open('fort.15', 'r')
            extra_lines = myfile.readlines()
            myfile.close()
            get_forces = True
        else:
            get_forces = False
        
        if self.first_time:
            self.first_time = False
            estring = ' SCF TOTAL ENERGY '
            fstring = ' CARTESIAN GRADIENT'
            qstring = 'NET ATOMIC CHARGES AND'
            dstring = ' DIPOLE     '
            found_indices = [False,False,False,False]
            for i, line in enumerate(outlines):
                if estring in line: lenergy = line; found_indices[0] = True
                if qstring in line:
                    lcharges = outlines[i+4:i+4+2*self.nAtoms]
                    found_indices[2] = True
                if dstring in line:
                    ldipole = outlines[i+2:i+10]; found_indices[3] = True
            
            if get_forces:
                for i, line in enumerate(extra_lines):
                    if fstring in line:
                        lforces = extra_lines[i+1:i+1+self.nAtoms]
                        found_indices[1] = True
                
            for i, prop in enumerate(['energy','forces','charges','dipole']):
                if found_indices[i]: exec('self.read_'+prop+'(l'+prop+')')
        
        
    
    def read_energy(self, output_line):
        """Read energy from line of MNDO output"""
        try:
            self.results['energy'] = float(output_line.split()[-2])
        except:
            raise RuntimeError('Problem reading energy')
    
    def read_forces(self, output_lines):
        """Read forces from lines of MNDO output"""
        from ase.units import kcal, mol
        kcalmol = kcal/mol
        
        try:
            gradients = []
            for j in range(self.nAtoms):
                my_grad = output_lines[j].split()[2:5]
                gradients.append([my_grad[k] for k in range(3)])
            
            ## TODO: CHECK IF WE NEED A (-1) HERE!
            self.results['forces'] = np.array(gradients, dtype=float) / kcalmol
        except:
            raise RuntimeError('Problem reading forces')
    
    def read_charges(self, output_lines):
        """Read net atomic charges from lines of MNDO output"""
        try:
            _charges = []
            for line in output_lines:
                if len(line.split())==0: break
                if line.split()[0] not in self.atoms.get_chemical_symbols():
                    continue
                _charges.append(line.split()[2])
            
            assert len(_charges)==self.nAtoms, 'Problem reading atomic charges'
            self.results['charges'] = np.array(_charges, dtype=float)
        except:
            raise RuntimeError('Problem reading atomic charges')
        
    def read_dipole(self, output_lines):
        """Read molecular dipole from lines of MNDO output"""
        try:
            for line in output_lines:
                if line.split()[0] == 'SUM':
                    _dip = line.split()[1:4]
                    self.results['dipole'] = np.array(_dip, dtype=float)
                    return
        except:
            raise RuntimeError('Problem reading dipole')
        
    
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
        
    

#--EOF--#
