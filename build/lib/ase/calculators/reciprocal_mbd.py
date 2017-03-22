import numpy as np
import os
from mbd_scalapack import mbd_scalapack as mbd
from pymbd import get_free_atom_data
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
                      'k_grid_shift':0.,
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
                      'get_MBD_eigenvalues':False,
                      'get_MBD_eigenvectors':False,
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
                  'get_MBD_eigenvalues', \
                  'get_MBD_eigenvectors']
    
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
        
        self.ts_d = 20.
        ts_s_r_dict = {'PBE':0.94, 'pbe':0.94, \
                       'PBE0':0.96, 'pbe0':0.96, \
                       'HSE':0.96, 'hse':0.96, \
                       'BLYP':0.62, 'blyp':0.62, \
                       'B3LYP':0.84, 'b3lyp':0.84, \
                       'revPBE':0.60, 'revpbe':0.60, \
                       'AM05':0.84, 'am05':0.84}
        
        if self.xc in ts_s_r_dict.iterkeys():
            self.ts_s_r = ts_s_r_dict[self.xc]
        else:
            print("No default parameters for '"+self.xc+"'. Using standard values of arguably applicability...")
            self.ts_s_r = 1.
        
    
    def _get_SCS_and_MBD_damping_params(self):
        """
        get damping parameters for MBD and SCS (they're the same) with current settings.
        """
        
        mbd_scs_d_dict = {'PBE':2.56, 'pbe':2.56, \
                          'PBE0':2.53, 'pbe0':2.53, \
                          'HSE':2.53, 'hse':2.53}
        
        mbd_rsscs_beta_dict = {'PBE':0.83, 'pbe':0.83, \
                               'PBE0':0.85, 'pbe0':0.85, \
                               'HSE':0.85, 'hse':0.85}
        
        mbd_ts_erf_beta_dict = {'PBE':2.56, 'pbe':2.56, \
                                'PBE0':2.53, 'pbe0':2.53, \
                                'HSE':2.53, 'hse':2.53}
        
        mbd_ts_fermi_beta_dict = {'PBE':2.56, 'pbe':2.56, \
                                  'PBE0':2.53, 'pbe0':2.53, \
                                  'HSE':2.53, 'hse':2.53}
        
        if (self.do_SCS and (not self.rsSCS)):
            if self.xc in mbd_scs_d_dict.iterkeys():
                self.damp_par_d = mbd_scs_d_dict[self.xc]
            else:
                print("No default parameters for '"+self.xc+"'. Using standard values of arguably applicability...")
                self.damp_par_d = 2.
        else:
            self.damp_par_d = 6.
        
        if self.do_SCS:
            if self.rsSCS:
                if self.xc in mbd_rsscs_beta_dict.iterkeys():
                    self.damp_par_beta = mbd_rsscs_beta_dict[self.xc]
                else:
                    print("No default parameters for '"+self.xc+"'. Using standard values of arguably applicability...")
                    self.damp_par_beta = 1.
            else:
                print("Default parameters for MBD@SCS (without range separation) not defined.")
                print("Using standard parameters, implement this otherwise (also needs different damping).")
                self.damp_par_beta = 1.
        
        elif ('erf' in self.Coulomb_CFDM):
            if self.xc in mbd_ts_erf_beta_dict.iterkeys():
                self.damp_par_beta = mbd_ts_erf_beta_dict[self.xc]
            else:
                print("No default parameters for '"+self.xc+"'. Using standard values of arguably applicability...")
                self.damp_par_beta = 1.
        elif ('fermi' in self.Coulomb_CFDM):
            if self.xc in mbd_ts_fermi_beta_dict.iterkeys():
                self.damp_par_beta = mbd_ts_fermi_beta_dict[self.xc]
            else:
                print("No default parameters for '"+self.xc+"'. Using standard values of arguably applicability...")
                self.damp_par_beta = 1.
        else:
            print("No default parameters for current settings. Using standard values of arguably applicability...")
            self.damp_par_beta = 1.
        
    
    def init_MBD(self, atoms=None):
        """
        Initialize MBD calculation and evaluate properties.
        """
        mbd.init_grid(self.n_omega_SCS)
        mbd.param_ts_energy_accuracy = self.TS_accuracy
        mbd.param_ts_cutoff_radius = self.TS_cutoff
        mbd.param_ewald_real_cutoff_scaling = self.ewald_real_cutoff_scaling
        mbd.param_ewald_rec_cutoff_scaling = self.ewald_rec_cutoff_scaling
        mbd.param_k_grid_shift = self.k_grid_shift
        mbd.param_mbd_nbody_max = self.max_nbody_MBD
        mbd.param_vacuum_axis = self.vacuum_axis
        mbd.my_task, mbd.n_tasks = self.myid, self.ntasks
        
        assert hasattr(self, 'a_div_a0'), \
        "Please provide rescaling to obtain initial dispersion parameters from accurate free atom reference data via 'set_rescaling(rescaling)'!"
        
        if np.any(self.vacuum_axis == atoms.pbc):
            print('Your specification for vacuum_axis is not in line with the boundary conditions of the atoms object!')
        
        if np.all(self.vacuum_axis):
            self.modus = ''
        else:
            self.modus = 'C'
            if not (self.do_reciprocal or self.do_supercell):
                raise ValueError("You chose periodic boundary condition via vacuum_axis, but did not specify how to handle it (do_reciprocal or do_supercell)!")
        
        if self.do_reciprocal:
            self.modus += 'R'
        if (self.ntasks > 1):
            self.modus += 'P'
        if self.get_MBD_eigenvalues:
            self.modus += 'E'
        if self.get_MBD_eigenvectors:
            self.modus += 'V'
        
        self.pos = atoms.positions/Bohr
        self.UC = atoms.get_cell()/Bohr
        symbols = atoms.get_chemical_symbols()
        self.alpha_f, self.C6_f, self.RvdW_f = get_free_atom_data(symbols)
        self.alpha_0_TS = self.alpha_f*self.a_div_a0
        self.C6_TS = self.C6_f*self.a_div_a0*self.a_div_a0
        self.RvdW_TS = self.RvdW_f*self.a_div_a0**(1./3.)
        self.omega_TS = mbd.omega_eff(self.C6_TS, self.alpha_0_TS)
        
        if ((self.do_TSSCS) and (not self.do_SCS)):
            print('You requested TS energy with screened (SCS) polarizabilities,')
            print("but didn't set do_SCS to True! I will correct this for you...")
            self.do_SCS = True
        
        self._get_SCS_and_MBD_damping_params()
        if self.do_SCS:
            self._run_electrostatic_screening()
        
        self._get_mbd_energy()
        if self.do_TS:
            self._get_TS_energy1()
        if self.do_TSSCS:
            self._get_TSSCS_energy1()
        
        mbd.destroy_grid()
#        # finalize should be handled by mpi4py, I think...
#        mbd.exit_blacs_and_finalize(1)
        
    
    def _run_electrostatic_screening(self):
        self.alpha_dyn_TS = mbd.alpha_dynamic_ts_all('C', \
                                                     mbd.n_grid_omega, \
                                                     self.alpha_0_TS, \
                                                     c6=self.C6_TS)
        modus_scs = self.modus.replace('R','')
        if self.use_scalapack:
            self.alpha_dyn_SCS = mbd.run_scs_s(modus_scs, \
                                               self.Coulomb_SCS, \
                                               self.pos, \
                                               self.alpha_dyn_TS, \
                                               r_vdw=self.RvdW_TS, \
                                               beta=self.damp_par_beta, \
                                               a=self.damp_par_d, \
                                               unit_cell=self.UC)
        else:
            self.alpha_dyn_SCS = mbd.run_scs(modus_scs, \
                                             self.Coulomb_SCS, \
                                             self.pos, \
                                             self.alpha_dyn_TS, \
                                             r_vdw=self.RvdW_TS, \
                                             beta=self.damp_par_beta, \
                                             a=self.damp_par_d, \
                                             unit_cell=self.UC)
        
        self.alpha_0_SCS = self.alpha_dyn_SCS[0]
        self.C6_SCS = mbd.get_c6_from_alpha(self.alpha_dyn_SCS)
        self.RvdW_SCS = self.RvdW_TS*(self.alpha_0_SCS/self.alpha_0_TS)**(1./3.)
        self.omega_SCS = mbd.omega_eff(self.C6_SCS, self.alpha_0_SCS)
        
    
    def _get_mbd_energy(self):
        g_grid = mbd.make_g_grid(self.Ggrid[0],self.Ggrid[1],self.Ggrid[2])
        kgrid = mbd.make_k_grid(g_grid, self.UC)
        
        if self.do_SCS:     # do MBD@(rs)SCS?
            alph, om, rvdwAB = self.alpha_0_SCS, self.omega_SCS, self.RvdW_SCS
        else:               # do MBD@TS
            alph, om, rvdwAB = self.alpha_0_TS, self.omega_TS, self.RvdW_TS
        
        if self.use_scalapack:
            self.E_MBD = mbd.get_mbd_energy_s(self.modus, \
                                              self.Coulomb_CFDM, \
                                              self.pos, alph, om, \
                                              supercell=self.supercell, \
                                              k_grid=kgrid, \
                                              unit_cell=self.UC, \
                                              r_vdw=rvdwAB, \
                                              beta=self.damp_par_beta, \
                                              a=self.damp_par_d)
        else:
            self.E_MBD = mbd.get_mbd_energy(self.modus, \
                                            self.Coulomb_CFDM, \
                                            self.pos, alph, om, \
                                            supercell=self.supercell, \
                                            k_grid=kgrid, \
                                            unit_cell=self.UC, \
                                            r_vdw=rvdwAB, \
                                            beta=self.damp_par_beta, \
                                            a=self.damp_par_d)
        
        self.E_MBD *= Hartree
        
    
    def _get_TS_energy1(self):
        self._get_TS_damping_params()
        if self.use_scalapack:
            self.E_TS = mbd.get_ts_energy_lowmem(self.modus, 'fermi2', \
                                                 self.pos, self.C6_TS, \
                                                 self.alpha_0_TS, \
                                                 r_vdw=self.RvdW_TS, \
                                                 s_r=self.ts_s_r, \
                                                 d=self.ts_d, \
                                                 unit_cell=self.UC)
        else:
            self.E_TS = mbd.get_ts_energy(self.modus, 'fermi2', \
                                          self.pos, self.C6_TS, \
                                          self.alpha_0_TS, \
                                          r_vdw=self.RvdW_TS, \
                                          s_r=self.ts_s_r, \
                                          d=self.ts_d, \
                                          unit_cell=self.UC)
        
        self.E_TS *= Hartree
        
    
    def _get_TSSCS_energy1(self):
        self._get_TS_damping_params()
        if self.use_scalapack:
            self.E_TS_SCS = mbd.get_ts_energy_lowmem(self.modus, 'fermi2', \
                                                     self.pos, self.C6_SCS, \
                                                     self.alpha_0_SCS, \
                                                     r_vdw=self.RvdW_SCS, \
                                                     s_r=self.ts_s_r, \
                                                     d=self.ts_d, \
                                                     unit_cell=self.UC)
        else:
            self.E_TS_SCS = mbd.get_ts_energy(self.modus, 'fermi2', \
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
