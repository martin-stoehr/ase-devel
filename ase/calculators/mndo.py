"""This module defines an ASE interface to MNDO (developed for MNDO2020)

https://mndo.kofo.mpg.de

written by Martin Stoehr (martin.stoehr@stanford.edu)
Sep 20, 2022.

"""

import os

import numpy as np
from math import log10, floor

from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError

from ase.units import kcal, mol, Bohr
kcalmol = kcal/mol


## list of available MNDO keywords with list of possible values
## equiv. human-readable keyword given in comment
mndo_kwargs = {
    'iform' :[0,1],  ## THIS IS FIXED TO 1 (UNFORMATTED)! ##
    'limit' :'numerical',  # maxtime
    'iop'   :[-23,-22,-13,-10,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,5,6],  # method
    'jop'   :[-3,-2,-1,0,1,2,3,4,5,6],  # calc_type
    'igeom' :[0,1],  # coord_type
    'mplib' :[-2,-1,0,1,2],  # parallel_mode
    'ief'   :[-3,-1,0,1,2,3],  # optimizer
    'idiis' :'numerical',  # diis_mode
    'inrefd':'numerical',  # ref_data
    'iparok':[-1,0,1,2,3,4,5,6,7,8],  # custom_param
    'mminp' :[0,1,2],  # external_pts
    'nmr'   :[-2,-1,0,1,2,11,12],  # compute_nmr
    'nsav7' :[0,1,2,3,4,5,6,7,8],  # write_full_input
    'nsav8' :[0,1],  # output_mopac
    'nsav9' :[0,1],  # output_pdb
    'nsav13':[0,1,2],  # output_aux
    'nsav15':[0,1,2,3,4,9],  # output_each_lvl
    'nsav16':[0,1,2],  # output_sybyl
    'immok' :[0,1],  # peptide_correction
    'ihbond':'numerical',  # hbond_dmax
    'ifld1' :[0,1],  # external_field
    'ifld2' :'numerical',  # field_strength
    'ifld3' :'numerical',  # ext_field_scf
    'icuts' :'numerical',  # cutoff_3center_ovlp
    'icutg' :'numerical',  # cutoff_3center_grad
    'iexbas':[0,1],  # polarized_basis
    'icosmo':[-4,-3,-2,-1,0,1,2,3,4],  # cosmo_mode
    'ipsana':[-2,-1,0,1,2],  # analyt_grad
    'immdp' :[-3,-1,0,1,2],  # vdw_corr
    'inac'  :'numerical',  # nac_mode
    'maxend':'numerical',  # max_scf_opt
    'maxlin':'numerical',  # max_scf_ls
    'maxrtl':'numerical',  # max_scf_tot
    'iscf'  :'numerical',  # ene_thresh
    'iplscf':'numerical',  # rho_thresh
    'middle':[-1,0,1,2],   # continuation
    'iprint':[-5,-1,0,1,5],  # output_lvl_opt
    'kprint':[-5,-1,0,1,5],  # output_lvl_force_const
    'lprint':[-5,-2,-1,0,1,5],  # output_lvl_vib
    'mprint':[-1,0,1,2,5],  # output_lvl_gradients
    'jprint':[-1,0,1,2,5,6,7],  # output_lvl_input
    'iprec' :'numerical',  # opt_lvl
    'iconv' :[0,1,2,3,4],  # opt_converge_type
    'ihess' :[0,1,2,3],  # init_hessian
    'idfp'  :[0,1],  # update_type_hessian
    'nrepet':'numerical',  # special_convergence
    'linitg':'numerical',  # check_f_init
    'lconvg':'numerical',  # check_f_norm
    'ihdlc1':[0,1],  # hdlc_coords
    'ihdlc2':[0,1,2,3,4],  # hdlc_coords_core
    'ihdlc3':[0,1,2],  # hdlc_extra
    'ingeom':[-1,0,1],  # special_geom_in
    'intdir':[0,1,2,3],  # direct_scf
    'lindms':[0,1,2,3,4,5],  # cgdms
    'lindia':[0,1],  # cgdms_fulldiag
    'linfrg':'numerical',  # guess_from_frag
    'inpfrg':[-1,0],  # extra_inp_frag
    'inp21' :[0,1],  # extra_in21
    'inp22' :[0,1],  # extra_in22
    'inp23' :[0,1],  # extra_in23
    'inp24' :[0,1,2],  # extra_in24
    'inp25' :[0,1],  # extra_in25
    'iaterg':[-1,0,1],  # atomization_type
## general
    'kharge':'numerical',  # charge
    'imult' :'numerical',  # multiplicity
    'ktrial':[0,1,2,11,12,13,20,21,22,23,24,25,26,27,
              30,31,32,33,34,35,36,37,38,39,41],  # rdm_init
    'kgeom' :[-1,0,1,2,3,4],  # geometry_grid
    'ipubo' :[0,1,2,3],  # save_scf
    'iuhf'  :[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5],  # scf_type
    'kitscf':'numerical',  # max_scf
    'nprint':[-5,-1,0,1,2,5],  # output_lvl
    'ifast' :[0,1,2],  # pseudodiagonalization
    'idiag' :[0,1,2,3,4,5,6,7,8,9],  # diagonalizer
    'ksym'  :[0,1],  # symmetry
    'numsym':[0,1,2,3,4,5,6,8,10,12,24],  # symmetry_num
    'kci'   :[-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8],  # post_hf
    'nstart':'numerical',  # extrapol_1st_scf
    'nstep' :'numerical',  # extrapol_scf_step
    'ktitle':'numerical',  # title
    'ifermi':'numerical',  # electronic_temp
    'imode' :'numerical',  # two_e_int_storage
    'ihcorr':[0,1],  # hbond_corr
    'domega':'numerical',  # fomo_width
    'nfloat':'numerical',  # fomo_n_orbs
## optimizer and force constant calcs
    'nrst'  :'numerical',  # reset_hessian_step
    'ldrop' :'numerical',  # restart_opt_thresh
    'ldell' :'numerical',  # update_d_restart
    'lsub'  :[0,1,2],  # which_ls
    'lalpha':[0,1,2],  # ls_dinit
    'lconv' :'numerical',  # ls_thresh_step
    'ltolf' :'numerical',  # ls_thresh_ene
    'lmaxst':'numerical',  # ls_max_step
    'igrad' :[-3,0,1],  # special_grad
    'lfac'  :'numerical',  # num_grad_step
    'kpoint':[0,1],  # avg_fd_vib
    'kfac'  :'numerical',  # num_vib_step
    'kmass' :'numerical',  # vib_masses
    'ntemp' :'numerical',  # temperatures_n
    'ntemp1':'numerical',  # temperatures_min
    'ntemp2':'numerical',  # temepratures_max
}


## list of corresponding human readable keywords
kwargs_all = {
    'format'                :['formatted','unformatted'],
    'maxtime'               :'numerical', 
    'method'                :['odm3','odm2','mndo/dh','mndo/d','om3','pm3',
                              'om2','om1','mndocustom','mndo/h','am1','mndoc',
                              'mndo','mindo/3','cndo/2','dftb','dftbjorgensen'],
    'calc_type'             :['bomd','gradient','singlepoint','optimization',
                              'force_constants','minimize2force_constant',
                              'ts2force_constant','force_constant2minimize',
                              'force_constant2ts'], 
    'coord_type'            :['internal','cartesian'], 
    'parallel_mode'         :['coarsepvm','coarsempi','sequential','fullmpi',
                              'fullpvm'],
    'optimizer'             :['hdlcexpert','hdlc','default','evec_follow',
                              'evec_follow_Newton','evec_followexpert'], 
    'diis_mode'             :'numerical',
    'ref_data'              :'numerical', 
    'custom_param'          :['mndowheremissing',False,'mopacformat','format1',
                              'format2','original_mndo_si_s','mndo89',
                              'fullerene_c','recomputemndo','pddg'], 
    'external_pts'          :[None,'computeatpoints','embedding'],
    'compute_nmr'           :['standardparamexpert','standardparam',False,
                              'nmrparama','nmrparamb'], 
    'write_full_input'      :[False,'stdin_formatted','stdin_unformatted',
                              'kwarg_formatted','kwarg_unformatted','mopac',
                              'stdin_unformatted_forcecartesian',
                              'kwarg_unformatted_forcecartesian',
                              'mopac_forcecartesian'], 
    'output_mopac'          :[False,True], 
    'output_pdb'            :[False,True],
    'output_aux'            :[False,True,'molden'], 
    'output_each_lvl'       :[0,1,2,3,4,5], 
    'output_sybyl'          :[False,True,'stdout_only'], 
    'peptide_correction'    :[False,True],
    'hbond_dmax'            :'numerical', 
    'external_field'        :[False,True],
    'field_strength'        :'numerical', 
    'ext_field_scf'         :'numerical',
    'cutoff_3center_ovlp'   :'ftexp',
    'cutoff_3center_grad'   :'ftexp',
    'polarized_basis'       :[False,True],
    'cosmo_mode'            :['singlepoint_expertplus','singlepoint_expert',
                              'singlepoint_plus','singlepoint_default', False,
                              'default', 'plus', 'expert', 'expertplus'], 
    'analyt_grad'           :['expertfalse',False,'default',True,'experttrue'], 
    'vdw_corr'              :['d3bj_atm',False,'default',True,'d2','d3bj'], 
    'nac_mode'              :'numerical',
    'max_scf_opt'           :'numerical',
    'max_scf_ls'            :'numerical', 
    'max_scf_tot'           :'numerical', 
    'ene_thresh'            :'exp', 
    'rho_thresh'            :'exp',
    'continuation'          :[False,'default',True,'newsettings'], 
    'output_lvl_opt'        :[0,1,2,3,4],
    'output_lvl_force_const':[0,1,2,3,4],
    'output_lvl_vib'        :[0,1,2,3,4,5],
    'output_lvl_gradients'  :[0,1,2,3,4],
    'output_lvl_input'      :[0,1,2,3,4,5,6],
    'opt_lvl'               :'numerical',
    'opt_converge_type'     :[0,1,2,3,4],  ## TODO: translate!
    'init_hessian'          :['read4path','finite_diff','read','identity'], 
    'update_type_hessian'   :['bfgs','dfp'], 
    'special_convergence'   :'numerical',
    'check_f_init'          :'ftnum',
    'check_f_norm'          :'ftnum',
    'hdlc_coords'           :['internal','cartesian'],
    'hdlc_coords_core'      :['primitive','totalconnection','cartesian',
                              'dlc_primitive','dlc_totalconnetion'], 
    'hdlc_extra'            :['custom_input','default_input',False], 
    'special_geom_in'       :['skip_input',False,'pdb'], 
    'direct_scf'            :[False,'full','thresh','threshexpert'],
    'cgdms'                 :[False,'square_full','square_thresh',
                              'square_threshexpert','sparese_thresh',
                              'sparse_threshexpert'], 
    'cgdms_fulldiag'        :[False,True], 
    'guess_from_frag'       :'numerical',
    'extra_inp_frag'        :[False,True], 
    'extra_in21'            :[False,True], 
    'extra_in22'            :[False,True], 
    'extra_in23'            :[False,True], 
    'extra_in24'            :[False,'oneline','twoline'], 
    'extra_in25'            :[False,True],
    'atomization_type'      :['needszpve','default','haszpve'], 
    'charge'                :'numerical', 
    'multiplicity'          :'numerical', 
    'rdm_init'              :['default_diag','simple_diag','mod_diag',
                              'read_rho','read_evec','read_rho_rhf',
                              'from_simple_scf','from_simple_mndoc',
                              'from_simple_am1','from_simple_mndo/h',
                              'from_simple_mndo','from_simple_om1',
                              'from_simple_om2','from_simple_pm3',
                              'diag_from_frag', 'diag_from_frag_scf1',
                              'diag_from_frag_scf2', 'diag_from_frag_scf3',
                              'diag_from_frag_scf4', 'diag_from_frag_scf5',
                              'diag_from_frag_scf6', 'diag_from_frag_scf7',
                              'diag_from_frag_scf8', 'diag_from_frag_scf9',
                              'sccdftb_from_charge_file'], 
    'geometry_grid'         :['only_create_geoms','default','add_info_path',
                              'add_info_grid1''add_info_grid2',
                              'add_info_interp'], 
    'save_scf'              :[False,'rdm','evecs','rdm_evecs'], 
    'scf_type'              :['fomo-rhf','smear-rhf','fo-rhf','read_irrep-rhf',
                              'expl_occ-rhf','default-rhf','default',
                              'default-uhf','expl_occ-uhf','read_irrep-uhf',
                              'fo-uhf','smear-uhf'], 
    'max_scf'               :'numerical', 
    'output_lvl_scf'        :[0,1,2,3,4,5],
    'pseudodiagonalization' :[True,'post_full',False], 
    'diagonalizer'          :['default','eispack_lin','eispack_square',
                              'lapack_lin','lapack_linx','lapack_square',
                              'lapack_squarex','lapack_lind','lapack_packed',
                              'eispack-based'], 
    'symmetry'              :[False,True], 
    'symmetry_num'          :['auto','c1,ci,cs,c0v','c2,c2v,c2h,d0h',
                              'c3,c3v,c3h,s6','c4,c4v,c4h,d2,d2d,d2h',
                              'c5,c5v,c5h','c6,c6v,c6h,d3,d3d,d3h',
                              'd4,d4d,d4h','d5,d5d,d5h','d6,d6d,d6h,t,td',
                              'oh'], 
    'post_hf'               :['singlepoint_rpa','singlepoint_sasfcis',
                              'singlepoint_sfcis','singlepoint_gugaci',
                              'singlepoint_bwen2','singlepoint_bwen1',
                              'singlepoint_bwen','singlepoint_minci',False,
                              'minci','bwen','bwen1','bwen2','gugaci',
                              'sfcis','sasfcis','rpa'],
    'extrapol_1st_scf'      :'ftnum', 
    'extrapol_scf_step'     :'numerical',
    'title'                 :'numerical', 
    'electronic_temp'       :'numerical',
    'two_e_int_storage'     :'numerical',
    'hbond_corr'            :[False, True],
    'fomo_width'            :'numerical',
    'fomo_n_orbs'           :'numerical',
    'reset_hessian_step'    :'numerical', 
    'restart_opt_thresh'    :1/kcalmol,
    'update_d_restart'      :0.001, 
    'which_ls'              :['quadratic_fstmin','quadratic_locmin','cubic'], 
    'ls_dinit'              :['default','full','previous'], 
    'ls_thresh_step'        :'numerical', 
    'ls_thresh_ene'         :'numerical', 
    'ls_max_step'           :0.1, 
    'special_grad'          :['fd_fock','default','fd_all'], 
    'num_grad_step'         :0.00001, 
    'avg_fd_vib'            :[True,False], 
    'num_vib_step'          :1/0.00001/Bohr, 
    'vib_masses'            :'numerical', 
    'temperatures_n'        :'numerical', 
    'temperatures_min'      :'numerical', 
    'temperatures_max'      :'numerical',
}


## create dictionary of human-readable -> mndo kwargs
h2mndo_key = dict(zip(kwargs_all.keys(), mndo_kwargs.keys()))

## convert human-readable values of kwargs to mndo values
def h2mndo_val(hkey, hval):
    hkey, hval = hkey.lower(), hval.lower()
    hval_list = kwargs_all[hkey]
    mkey = h2mndo_key[hkey]
    mval_list = mndo_kwargs[mkey]
    if type(mval_list) == list:
        if hkey in ['output_lvl_opt','output_lvl_force_const']:
            if hval == 'default': hval = 2
        elif hkey == 'output_lvl_vib':
            if hval == 'default': hval = 3
        elif hkey in ['output_lvl_gradients','output_lvl_input']:
            if hval == 'default': hval = 1
        
        return mval_list[hval_list.index(hval)]
    elif mval_list == 'numerical':
        if type(hval_list) == float:
            return hval*hval_list
        elif hval_list == 'ftnum':
            if type(hval) == float:
                return hval
            elif type(hval) == bool:
                return 0 if hval else -1
        elif hval_list == 'ftexp':
            if type(hval) == float:
                return -1*round(log10(hval))
            elif type(hval) == bool:
                return 0 if hval else -1
        elif hval_list == 'numerical':
            if hkey == 'extrapol_scf_step':
                if hval == 'default':
                    hval = 0
                elif str(hval).startswith('damp'):
                    hval = -int(hval[4:])
            elif hkey == 'guess_from_frag':
                if type(hval) == bool:
                    hval = 1 if hval else 0
            elif hkey in ['nac_mode', 'fomo_n_orbs']:
                if hval == 'default': hval = 0
            elif hkey == 'electronic_temp':
                if hval == 'default': hval = 20000
            elif hkey == 'two_e_int_storage':
                if hval == 'default':
                    hval = 0
                elif hval == 'in_memory_mat_fock':
                    hval = -2
                elif hval == 'in_memory_mat_fockx':
                    hval = -1
                elif hval == 'in_memory_lin':
                    hval = 1
                elif hval.startswith('in_memory_lim'):
                    try:
                        lm6lim = int(hval.replace('in_memory_lim',''))
                        hval = -lm6lim
                    except:
                        msg = "Unrecognized option for 'two_e_int_storage'."
                        raise ValueError(msg)
                elif hval.startswith('on_disk_size'):
                    try:
                        iosize = int(hval.replace('on_disk_size',''))
                        hval = iosize
                    except:
                        msg = "Unrecognized option for 'two_e_int_storage'."
                        raise ValueError(msg)
                elif type(hval) == int:
                    hval = hval
            elif hkey == 'fomo_width':
                #                             = 0.2 Ha
                if hval == 'default': hval = 5.442277204873449
            elif hkey == 'diis_mode':
                if type(hval) == bool:
                    hval = 1 if hval else -1
                elif hval == 'forconvergence':
                    hval = 0
            elif hkey == 'ext_field_scf':
                hval = -1*round(log10(hval))
            return hval



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
        
        self.default_parameters = dict(
            igeom=1,    # use Cartesian coordinates
        )
        
        ## handle complicated definition of field strength for external field
        ext_field = kwargs.pop('external_field', False)
        if ext_field:
            field = kwargs.pop('field_strength', 0)
            if field != 0:
                kwargs['ifld1'] = round(a/10**floor(log10(a)))
                kwargs['ifld2'] = -1*floor(log10(field))
            elif field == 'default':
                kwargs['ifld1'] = 1
                kwargs['ifld2'] = 0
        
        ## translate human-readable to mndo kwargs and pop
        input_keys = list(kwargs.keys())
        for key in input_keys:
            if key in kwargs_all.keys():
                kwargs[h2mndo_key[key]] = h2mndo_val(key, kwargs.pop(key))
        
        for key, val in self.default_parameters.items():
            if not key in kwargs.keys(): kwargs[key] = val
        
        kwargs['iform'] = iform
        self.opt_mask = kwargs.pop('opt_mask', None)
        
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        
        self.first_time = True
        self.label = label
        self.calc_info = {}
        self.extra_properties = {}
        
    
    def set_atoms(self, atoms):
        self.atoms = atoms
    
    def write_mndo_in(self):
        """
        Write the input file for the dftb+ calculation.
        Geometry is taken always from the file 'geo_end.gen'.
        """

        self.nAtoms = len(self.atoms)
        pbc = np.any(self.atoms.pbc)
        if pbc: raise NotImplementedError('PBC not supported')
        
        arg_str = ''
        title = self.parameters.pop('ktitle', ' MNDO calculation from ASE')
        if self.opt_mask is None: self.opt_mask = np.zeros((self.nAtoms,3))
        for key, val in self.parameters.items():
            if key not in mndo_kwargs: continue
            test_len = len(key)+len(str(val))
            if (len(arg_str) + test_len > 75): arg_str += ' +\n'
            arg_str += key+'='+str(val)+' '
        
        if arg_str.count('+\n')>10: raise ValueError('Too many inputs!')
        inp = open('asejob_mndo.inp', 'w')
        inp.write(arg_str+'\n')
        inp.write('{0:76s}'.format(title.replace('\n',' ')))
        inp.write('\n## Generated by ASE\n')
        
        ## write geometry
        arr = self.get_molecule_array()
        gen  = '{0: >2.0f}     {1: >13.9f}   {2:1.0f}   {3: >13.9f}   '
        gen += '{4:1.0f}   {5: >13.9f}   {6:1.0f}\n'
        for iAtom in range(self.nAtoms): inp.write(gen.format(*arr[iAtom]))
        inp.write("0")
        
        
        ## write symmetry data
        ## TODO!
        
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
        if changed_parameters: self.reset()
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
        if self.first_time:
            self.first_time = False
            estring = ' SCF TOTAL ENERGY '
            fstring = '      GRADIENTS (KCAL'
            qstring = 'NET ATOMIC CHARGES AND'
            dstring = ' DIPOLE     '
            heat_of_formation_str = ['SCF HEAT OF FORMATION   ']
            electronic_energy_str = ['ELECTRONIC ENERGY       ']
            nuclear_energy_str    = ['NUCLEAR ENERGY          ']
            ionization_energy_str = ['IONIZATION ENERGY       ']
            point_group_str = ['   POINT GROUP ', ' ASSIGNED.']
            scf_cycles_str  = ' SCF CYCLES           '
            timing_ene_str  = ' TIME FOR ENERGY EVALUATION    '
            timing_grad_str = ' TIME FOR GRADIENT EVALUATION  '
            timing_tot_str  = 'COMPUTATION TIME      '
            
            found = {'energy':False, 'forces':False, 'charges':False,
                     'dipole':False}
            found_extra = {'heat_of_formation':False, 'electronic_energy':False,
                           'nuclear_energy':False,    'ionization_energy':False, 
                           'point_group':False}
            found_info = {'scf_cycles':False,        'timing_ene':False, 
                          'timing_grad':False,       'timing_tot':False}
            
            for i, line in enumerate(outlines):
                if estring in line:
                    lenergy = line
                    found['energy'] = True
                    continue
                ## molecule might contain ghost atoms -> read twice the lines
                if fstring in line:
                    lforces = outlines[i+4:i+4+2*self.nAtoms]
                    found['forces'] = True
                    continue
                ## molecule might contain ghost atoms -> read twice the lines
                if qstring in line:
                    lcharges = outlines[i+4:i+4+2*self.nAtoms]
                    found['charges'] = True
                    continue
                ## depending on settings MNDO might output more details here
                ## -> read more lines and find "SUM" entry
                if dstring in line:
                    ldipole = outlines[i+2:i+10]
                    found['dipole'] = True
                    continue
                ## read extra properties
                for prop in found_extra.keys():
                    search_str = eval(prop+'_str')
                    if all(i in line for i in search_str):
                        exec(prop+'_line = line')
                        found_extra[prop] = True
                ## read info
                for prop in found_info.keys():
                    search_str = eval(prop+'_str')
                    if search_str in line:
                        exec(prop+'_line = line')
                        found_info[prop] = True
                            
            for prop in ['energy','forces','charges','dipole']:
                if found[prop]: exec('self.read_'+prop+'(l'+prop+')')
            
            indices_extra = {'heat_of_formation':-2, 'electronic_energy':-2,
                             'nuclear_energy':-2,    'ionization_energy':-2,
                             'point_group':-2}
            type_extra = {'heat_of_formation':'float', 'electronic_energy':'float',
                          'nuclear_energy':'float',    'ionization_energy':'float',
                          'point_group':'str'}
            
            for prop in found_extra.keys():
                if found_extra[prop]:
                    idx = str(indices_extra[prop])
                    lin = eval(prop+'_line')
                    self.read_extra(prop, lin, idx, type_extra[prop])
            
            indices_info = {'scf_cycles':-1,        'timing_ene':-2,
                            'timing_grad':-2,       'timing_tot':-2}
            type_info = {'scf_cycles':'int',          'timing_ene':'float',
                         'timing_grad':'float',       'timing_tot':'float'}
            
            for prop in found_info.keys():
                if found_info[prop]:
                    idx = str(indices_info[prop])
                    lin = eval(prop+'_line')
                    self.read_info(prop, lin, idx, type_info[prop])
        
        
    def read_energy(self, output_line):
        """Read energy from line of MNDO output"""
        try:
            self.results['energy'] = float(output_line.split()[-2])
        except:
            raise RuntimeError('Problem reading energy')
    
    def read_forces(self, output_lines):
        """Read forces from lines of MNDO output"""
        try:
            gradients = []
            for line in output_lines:
                if len(line.split())==0: break
                if int(line.split()[1]) not in self.atoms.get_atomic_numbers():
                    continue
                my_grad = line.split()[-3:]
                gradients.append([float(my_grad[k]) for k in range(3)])
            
            assert len(gradients)==self.nAtoms, 'Problem reading forces'
            self.results['forces'] = -1 * np.array(gradients) * kcalmol
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
        
    
    def read_extra(self, prop, prop_line, idx, prop_type):
        """ Read extra property from one-line output line """
        try:
            res = eval(prop_type+'(prop_line.split()['+idx+'])')
            if prop == 'heat_of_formation': res *= kcalmol
            self.extra_properties[prop] = res
        except:
            raise RuntimeError('Proplem reading '+prop)
        
    def read_info(self, info, info_line, idx, info_type):
        """ Read info from one-line output line """
        try:
            self.calc_info[info] = eval(info_type+'(info_line.split()['+idx+'])')
        except:
            raise RuntimeError('Proplem reading '+info)
        
    
    def get_info(self, name):
        """
        Return extra info on calculation including <name>=
            . scf_cycles:  number of SCF cycles
            . timing_tot:  total computation time in seconds
            . timing_ene:  computation time for energy in seconds
            . timing_grad: computation time for gradients in seconds
        """
        info = self.calc_info.get(name.lower(), None)
        if info is None:
            raise NotImplementedError("Sorry, '"+name+"' is not available.")
        return info
        
    def get_extra_property(self, name):
        """
        Return extra quantity available from calculation, such as <name>=
            . point_group:       Point group assigned to system by MNDO
            . heat_of_formation: heat of formation in eV
            . electronic_energy: electronic part of total energy in eV
            . nuclear_energy:    nuclear part of total energy in eV
            . ionization_energy: ionization potential in eV
        """
        prop = self.extra_properties.get(name.lower(), None)
        if prop is None:
            raise NotImplementedError("Sorry, '"+name+"' is not available.")
        return prop
    
    def get_hirsh_volrat(self):
        """
        Return rescaling ratios for atomic polarizabilities (CPA ratios)
        """
        if hasattr(self, 'CPA_ratios'):
            return self.CPA_ratios
        else:
            msg  = "Could not obtain CPA ratios. "
            msg += "This might be implemented in a future version."
            raise NotImplementedError(msg)
        
    
    def get_stress(self, atoms):
        msg = 'PBC and thus stress not available (yet).'
        raise NotImplementedError(msg)
        
    

#--EOF--#
