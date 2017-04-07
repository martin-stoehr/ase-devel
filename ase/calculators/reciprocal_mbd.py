import numpy as np
import os
from mbd_scalapack import mbd_scalapack as mbd_s
#from mbd import mbd
from pymbd import get_free_atom_data, get_damping
from ase import atoms
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator


default_parameters = {'xc':'PBE',
                      'n_omega_SCS':20,
                      'Coulomb_SCS':'fermi,dip,gg',
                      'Coulomb_CFDM':'fermi,dip',
                      'TS_accuracy':1E-6,
                      'TS_cutoff':30./Bohr,
                      'ewald_real_cutoff_scaling':0.9,
                      'ewald_rec_cutoff_scaling':1.,
                      'k_grid_shift':0.5,
                      'vacuum_axis':(True,True,True),
                      'max_nbody_MBD':3,
                      'do_reciprocal':True,
                      'Ggrid':(3,3,3),
                      'do_supercell':False,
                      'supercell':(2,2,2),
                      'do_TS':False,
                      'do_SCS':True,
                      'rsSCS':True,
                      'do_TSSCS':False,
                      'use_scalapack':True,
                      'eigensolver':'qr',
                      'get_MBD_eigenvalues':False,
                      'get_MBD_eigenvectors':False,
                      'set_negative_eigenvalues_zero':True,
                      }


try:
    from mpi4py import MPI
    default_parameters['ntasks'] = MPI.COMM_WORLD.Get_size()
    default_parameters['myid'] = MPI.COMM_WORLD.Get_rank()
except ImportError:
    sys.stderr.write('warning: Install mpi4py for MPI support\n')
    default_parameters['ntasks'] = 1
    default_parameters['myid'] = 0


class kSpace_MBD_calculator(Calculator):
    """
    Many-Body Dispersion calculator class for reciprocal space formulation.
    written by Martin Stoehr (martin.stoehr@uni.lu), Feb 2016.
    """
    
    
    implemented_properties = ['energy', 'forces']
    
    valid_args = ['xc', \
                  'n_omega_SCS', \
                  'Ggrid', \
                  'Coulomb_SCS', \
                  'Coulomb_CFDM', \
                  'TS_accuracy', \
                  'TS_cutoff', \
                  'ewald_real_cutoff_scaling', \
                  'ewald_rec_cutoff_scaling', \
                  'k_grid_shift', \
                  'vacuum_axis', \
                  'max_nbody_MBD', \
                  'do_reciprocal', \
                  'do_supercell', \
                  'supercell', \
                  'do_TS', \
                  'do_SCS', \
                  'rsSCS', \
                  'do_TSSCS', \
                  'use_scalapack', \
                  'eigensolver', \
                  'get_MBD_eigenvalues', \
                  'get_MBD_eigenvectors', \
                  'set_negative_eigenvalues_zero']
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        
        ## set default arguments
        for arg, val in default_parameters.iteritems():
            setattr(self, arg, val)
        
        ## set or overwrite any additional keyword arguments provided
        for arg, val in kwargs.items():
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s" : not in %s'
                                   % (arg, self.valid_args))
        
        if ( self.do_supercell and self.do_reciprocal ):
            if ( 'do_reciprocal' in kwargs.iterkeys() ):
                print("You specified 'do_supercell'=True and 'do_reciprocal'=True.")
                print("This is not available at the moment. Please chose ONE of the above.")
                print("Defaulting to 'do_supercell'=False and 'do_reciprocal'=True.")
                self.do_supercell = False
            else:
                self.do_reciprocal = False
        
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        
    
    def get_potential_energy(self, atoms=None):
        """ Return dispersion energy as obtained by MBD calculation. """
        self.update_properties(atoms)
        return self.E_MBD
        
    
    def get_forces(self, atoms=None):
        return NotImplementedError
        
    
    def update_properties(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.calculate(atoms)
        
    
    def calculate(self, atoms):
        self.atoms = atoms.copy()
        self.init_MBD(atoms=self.atoms)
        
    
    def _get_TS_damping_params(self):
        """
        get damping parameters for vdW(TS) with current settings.
        """
        
        damp_dict = get_damping(self.xc)
        self.ts_d = damp_dict['ts_d']
        self.ts_s_r = damp_dict['ts_s_r']
        
    
    def _get_SCS_and_MBD_damping_params(self):
        """
        get damping parameters for MBD and SCS (they're the same) with current settings.
        """
        
        damp_dict = get_damping(self.xc)
        if self.do_SCS:
            if self.rsSCS:
                self.damp_par_a = damp_dict['mbd_rsscs_a']
            else:
                self.damp_par_a = damp_dict['mbd_scs_a']
        else:
            self.damp_par_a = damp_dict['mbd_ts_a']
        
        if self.do_SCS:
            if self.rsSCS:
                self.damp_par_beta = damp_dict['mbd_rsscs_beta']
            else:
                self.damp_par_beta = damp_dict['mbd_scs_beta']
        elif ('erf' in self.Coulomb_CFDM):
                self.damp_par_beta = damp_dict['mbd_ts_erf_beta']
        elif ('fermi' in self.Coulomb_CFDM):
                self.damp_par_beta = damp_dict['mbd_ts_fermi_beta']
        else:
            print("No default parameters for current settings. Using standard values of arguably applicability...")
            self.damp_par_beta = 1.
        
    
    def init_MBD(self, atoms=None):
        """
        Initialize MBD calculation and evaluate properties.
        """
#        if (self.use_scalapack):
        mbd_mod = mbd_s
#        else:
#            mbd_mod = mbd
        
        mbd_mod.init_grid(self.n_omega_SCS)
        mbd_mod.param_ts_energy_accuracy = self.TS_accuracy
        mbd_mod.param_ts_cutoff_radius = self.TS_cutoff
        mbd_mod.param_ewald_real_cutoff_scaling = self.ewald_real_cutoff_scaling
        mbd_mod.param_ewald_rec_cutoff_scaling = self.ewald_rec_cutoff_scaling
        mbd_mod.param_k_grid_shift = self.k_grid_shift
        mbd_mod.param_mbd_nbody_max = self.max_nbody_MBD
        mbd_mod.param_vacuum_axis = self.vacuum_axis
        mbd_mod.param_zero_negative_eigs = self.set_negative_eigenvalues_zero
        mbd_mod.my_task, mbd_mod.n_tasks = self.myid, self.ntasks
        if self.use_scalapack:
            solver_bak = self.eigensolver
            self.eigensolver = self.eigensolver.strip()[:5].ljust(5)
            if self.eigensolver not in ['qr   ', 'mrrr ', 'dandc']:
                print("The specified eigensolver '"+solver_bak.strip()+"' is not known (yet).")
                print("Using default solver 'qr' instead...")
                self.eigensolver = 'qr   '

            mbd_mod.eigensolver = self.eigensolver
        
        assert hasattr(self, 'a_div_a0'), \
        "Please provide rescaling to obtain initial dispersion parameters from accurate free atom reference data via 'set_rescaling(rescaling)'!"
        
        self.modus = ''
        if (self.ntasks > 1): self.modus += 'P'
        if np.any(self.vacuum_axis == atoms.pbc):
            print('Your specification for vacuum_axis is not in line with the boundary conditions of the atoms object!')
        
        if (not np.all(self.vacuum_axis)):
            self.modus += 'C'
            if not (self.do_reciprocal or self.do_supercell):
                raise ValueError("You chose periodic boundary condition via vacuum_axis, but did not specify how to handle it (do_reciprocal or do_supercell)!")
        
        self.pos = atoms.positions/Bohr
        self.UC = atoms.get_cell()/Bohr
        symbols = atoms.get_chemical_symbols()
        self.alpha_f, self.C6_f, self.RvdW_f = get_free_atom_data(symbols)
        self.alpha_0_TS = self.alpha_f*self.a_div_a0
        self.C6_TS = self.C6_f*self.a_div_a0*self.a_div_a0
        self.RvdW_TS = self.RvdW_f*self.a_div_a0**(1./3.)
        self.omega_TS = mbd_mod.omega_eff(self.C6_TS, self.alpha_0_TS)
        
        self._get_SCS_and_MBD_damping_params()
        if self.do_SCS:
            self._run_electrostatic_screening()
        
        if self.do_reciprocal: self.modus += 'R' #.replace('C','R')
        if self.get_MBD_eigenvalues: self.modus += 'E'
        if self.get_MBD_eigenvectors: self.modus += 'V'
        
        self._get_mbd_energy()
        if self.do_TS: self._get_TS_energy1()
        if self.do_TSSCS:
            if not self.do_SCS:
                print('You requested TS energy with screened (SCS) polarizabilities,')
                print("but didn't set do_SCS to True. Skipping...")
            else:
                self._get_TSSCS_energy1()
        
        mbd_mod.destroy_grid()
#       if self.use_scalapack:
#            # finalize should be handled by mpi4py, I think...
#            mbd_mod.exit_blacs_and_finalize(1)
        
    
    def _run_electrostatic_screening(self):
#        if (self.use_scalapack):
        mbd_mod = mbd_s
#        else:
#            mbd_mod = mbd
        
        self.alpha_dyn_TS = mbd_mod.alpha_dynamic_ts_all('C', \
                                                 mbd_mod.n_grid_omega, \
                                                 self.alpha_0_TS, \
                                                 c6=self.C6_TS)
        if self.use_scalapack: # use ScaLAPACK?
            self.alpha_dyn_SCS = mbd_mod.run_scs_s(self.modus, \
                                              self.Coulomb_SCS, \
                                              self.pos, \
                                              self.alpha_dyn_TS, \
                                              r_vdw=self.RvdW_TS, \
                                              beta=self.damp_par_beta, \
                                              a=self.damp_par_a, \
                                              unit_cell=self.UC)
        else:
            self.alpha_dyn_SCS = mbd_mod.run_scs(self.modus, \
                                            self.Coulomb_SCS, \
                                            self.pos, \
                                            self.alpha_dyn_TS, \
                                            r_vdw=self.RvdW_TS, \
                                            beta=self.damp_par_beta, \
                                            a=self.damp_par_a, \
                                            unit_cell=self.UC)
        
        self.alpha_0_SCS = self.alpha_dyn_SCS[0]
        self.C6_SCS = mbd_mod.get_c6_from_alpha(self.alpha_dyn_SCS)
        self.RvdW_SCS = self.RvdW_TS*(self.alpha_0_SCS/self.alpha_0_TS)**(1./3.)
        self.omega_SCS = mbd_mod.omega_eff(self.C6_SCS, self.alpha_0_SCS)
        
    
    def _get_mbd_energy(self):
#        if (self.use_scalapack):
        mbd_mod = mbd_s
#        else:
#            mbd_mod = mbd
        
        kgrid = mbd_mod.make_k_grid(mbd_mod.make_g_grid(*self.Ggrid), self.UC)
        
        if self.do_SCS:     # do MBD@(rs)SCS?
            alph, om, rvdwAB = self.alpha_0_SCS, self.omega_SCS, self.RvdW_SCS
        else:               # do MBD@TS
            alph, om, rvdwAB = self.alpha_0_TS, self.omega_TS, self.RvdW_TS
        
        if self.use_scalapack:
            self.E_MBD = mbd_mod.get_mbd_energy_s(self.modus, \
                                             self.Coulomb_CFDM, \
                                             self.pos, alph, om, \
                                             supercell=self.supercell, \
                                             k_grid=kgrid, \
                                             unit_cell=self.UC, \
                                             r_vdw=rvdwAB, \
                                             beta=self.damp_par_beta, \
                                             a=self.damp_par_a)
        else:
            self.E_MBD = mbd_mod.get_mbd_energy(self.modus, \
                                             self.Coulomb_CFDM, \
                                             self.pos, alph, om, \
                                             supercell=self.supercell, \
                                             k_grid=kgrid, \
                                             unit_cell=self.UC, \
                                             r_vdw=rvdwAB, \
                                             beta=self.damp_par_beta, \
                                             a=self.damp_par_a)
        
        self.E_MBD *= Hartree
        
    
    def _get_TS_energy1(self):
        self._get_TS_damping_params()
#        if (self.use_scalapack):
        mbd_mod = mbd_s
#        else:
#            mbd_mod = mbd
        
        if self.use_scalapack:
            self.E_TS = mbd_mod.get_ts_energy_lowmem(self.modus, \
                                                 'fermi', \
                                                 self.pos, self.C6_TS, \
                                                 self.alpha_0_TS, \
                                                 r_vdw=self.RvdW_TS, \
                                                 s_r=self.ts_s_r, \
                                                 d=self.ts_d, \
                                                 unit_cell=self.UC)
        else:
            self.E_TS = mbd_mod.get_ts_energy(self.modus, 'fermi', \
                                             self.pos, self.C6_TS, \
                                             self.alpha_0_TS, \
                                             r_vdw=self.RvdW_TS, \
                                             s_r=self.ts_s_r, \
                                             d=self.ts_d, \
                                             unit_cell=self.UC)
        
        self.E_TS *= Hartree
        
    
    def _get_TSSCS_energy1(self):
        self._get_TS_damping_params()
#        if (self.use_scalapack):
        mbd_mod = mbd_s
#        else:
#            mbd_mod = mbd
        
        if self.use_scalapack:
            self.E_TS_SCS = mbd_mod.get_ts_energy_lowmem(self.modus, \
                                                'fermi2', \
                                                self.pos, self.C6_SCS, \
                                                self.alpha_0_SCS, \
                                                r_vdw=self.RvdW_SCS, \
                                                s_r=self.ts_s_r, \
                                                d=self.ts_d, \
                                                unit_cell=self.UC)
        else:
            self.E_TS_SCS = mbd_mod.get_ts_energy(self.modus, \
                                                'fermi2', \
                                                self.pos, self.C6_SCS, \
                                                self.alpha_0_SCS, \
                                                r_vdw=self.RvdW_SCS, \
                                                s_r=self.ts_s_r, \
                                                d=self.ts_d, \
                                                unit_cell=self.UC)
        
        self.E_TS_SCS *= Hartree
        
    
    def set_a_div_a0(self, rescaling):
        self.a_div_a0 = np.array(rescaling)
        
    
    def get_frequency_dependent_alpha_TS(self):
        return self.alpha_dyn_TS
        
    
    def get_static_alpha_SCS(self):
        if self.do_SCS:
            return self.alpha_0_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get alpha0 after SCS")
        
    
    def get_frequency_dependent_alpha_SCS(self):
        if self.do_SCS:
            return self.alpha_dyn_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get dynamic alpha after SCS")
        
    
    def get_C6_TS(self):
        return self.C6_TS
        
    
    def get_C6_SCS(self):
        if self.do_SCS:
            return self.C6_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get C6_SCS")
        
    
    def get_RvdW_TS(self):
        return self.RvdW_TS
        
    
    def get_RvdW_SCS(self):
        if self.do_SCS:
            return self.RvdW_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get RvdW_SCS")
        
    
    def get_omega_TS(self):
        return self.omega_TS
        
    
    def get_omega_SCS(self):
        if self.do_SCS:
            return self.omega_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get omega_SCS")
        
    
    def get_TS_energy(self):
        if self.do_TS:
            return self.E_TS
        else:
            raise ValueError("Please specify 'do_TS = True' in order to get TS energy")
        
    
    def get_TSSCS_energy(self):
        if self.do_TSSCS:
            return self.E_TS_SCS
        else:
            raise ValueError("Please specify 'do_TSSCS = True' in order to get TS+SCS energy")
        
    

#--EOF--#
