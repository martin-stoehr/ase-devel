"""This module defines an ASE interface to MNDO (developed for MNDO2020)

https://mndo.kofo.mpg.de

written by Martin Stoehr (martin.stoehr@stanford.edu)

The keywords are given, for instance, as follows::

???

"""

import os

import numpy as np

from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError


## list of available MNDO keywords  # equiv. human-readable keyword
mndo_kwargs = [
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

## fix input format to formatted
iform = 1


class Mndo(FileIOCalculator):
    """  A MNDO calculator with ase-FileIOCalculator nomenclature  """
    if 'MNDO_COMMAND' in os.environ:
        command = os.environ['MNDO_COMMAND'] + ' mndo_ase.inp > PREFIX.out'
    else:
        command = 'mndo mndo_ase.inp > PREFIX.out'
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='mndo', atoms=None, kpts=None, **kwargs):
        """  Construct a MNDO calculator.  """
        
        from os.path import exists as pexists
        
        
        self.default_parameters = dict(
            igeom=1,    # use Cartesian coordinates
        )
        
        ## translate human-readable to mndo kwargs and pop
        for key in kwargs.keys():
            if key in kwargs_all:
                kwargs[h2mndo_key[key]] = h2mndo_val(kwargs.pop(key))
        
        kwargs['iform'] = iform
        self.pbc = np.any(atoms.pbc)
        
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        
        self.atoms, self.nAtoms = atoms, len(atoms)
        
        if not hasattr(self, 'opt_mask'):
            self.opt_mask = np.zeros_like(atoms.positions)
        
        if restart == None:
            self.write_mndo_inp()
        else:
            if os.path.exists(restart):
                os.system('cp ' + restart + ' mndo_ase.inp')
            if not os.path.exists('mndo_ase.inp'):
                raise IOError('No file "mndo_ase.inp", use restart=None')
        
        if self.pbc: raise NotImplementedError('PBC not supported')
        
        ## indexes for the result file
        self.first_time = True
        self.index_energy = None
#        self.index_force_begin = None
#        self.index_force_end = None
        
    
    def write_mndo_in(self):
        """
        Write the input file for the dftb+ calculation.
        Geometry is taken always from the file 'geo_end.gen'.
        """

        outfile = open('mndo_ase.inp', 'w')
        ## write kwargs
        in_str = ''
        title = self.parameters.pop('ktitle', ' MNDO calculations from ASE')
        for key, val in self.parameters.items():
            if key == 'ktitle': continue
            test_len = len(key)+len(str(val))+3
            if (len(in_str) + test_len > 78): in_str += ' +\n'
            in_str += key+'='+str(val)+' '
        
        if in_str.count('+\n')>10: raise ValueError('Too many inputs!')
        outfile.write(in_str)
        
        ## write geometry
        mol = self.get_molecule_array()
        gen  = '{0: >2d}    {1: >14.10f}  {2:1d}   {3: >14.10f}  {4:1d}   '
        gen += '{5: >14.10f}  {6:1d}'
        for iAtom in range(self.nAtoms): f.write(gen.format(mol[iAtom]))
        
        ## write symmetry data
        
        
        outfile.close()
        
    def get_molecule_array(self):
        """
        Convert atomic numbers, positions and ptimization mask into input
        array as used by MNDO.
        """
        Z = self.atoms.get_atomic_numbers()
        xyz = self.atoms.positions
        
        if self.igeom == 1: # Cartesian
            res = np.insert(np.insert(xyz, 0, Z, axis=1), (2,3,4), self.opt_mask)
        elif self.igeom == 0: # internal
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
            self.write_mndo_in()
    
    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        return system_changes
    
    def write_input(self, atoms, properties=None, system_changes=None):
        pass
    
    def read_results(self):
        myfile = open('PREFIX.out', 'r')
        self.lines = myfile.readlines()
        myfile.close()
        myfile = open('fort.15', 'r')
        self.extra_lines = myfile.readlines()
        myfile.close()
        if self.first_time:
            self.first_time = False
            # line indices
            estring, fstring = 'SCF TOTAL ENERGY', ' CARTESIAN GRADIENT'
            dstring, qstring = '', 'NET ATOMIC CHARGES AND'
            found_indices = [False,False,False,False]
            for iline, line in enumerate(self.lines):
            
            
        self.read_energy()
        if self.calculate_forces:
            self.read_forces()
        
        
    
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
