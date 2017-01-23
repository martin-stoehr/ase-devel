import numpy as np
import os
from mbd import mbd
from pymbd import get_free_atom_data
from ase import atoms
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator


## range-seperation parameter for given xc functional or method
xc2beta = {'PBE':0.83, 'DFTB':1.1}

default_parameters = {'xc':'PBE',
                      'n_omega':20,
                      'Ggrid':(3,3,3),
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
                      'do_TS':False,
                      'do_SCS':True,
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
                  'n_omega', \
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
                  'do_TS', \
                  'do_SCS', \
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
        self.init_ewaldMBD(atoms=self.atoms)
        
    
    def init_ewaldMBD(self, atoms=None):
        """
        Initialize MBD calculation and evaluate properties.
        """
        mbd.init_grid(self.n_omega)
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
            raise Warning('Your specification for vacuum_axis is not in line with the boundary conditions of the atoms object!')
        
        if np.any(atoms.pbc):
            modus = 'C'
        else:
            modus = ''
        
        if self.do_reciprocal:
            modus += 'R'
        if (self.ntasks > 1):
            modus += 'P'
        if self.get_MBD_eigenvalues:
            modus += 'E'
        if self.get_MBD_eigenvectors:
            modus += 'V'
        
        self.pos = atoms.positions/Bohr
        self.UC = atoms.get_cell()/Bohr
        symbols = atoms.get_chemical_symbols()
        self.alpha_f, self.C6_f, self.RvdW_f = get_free_atom_data(symbols)
        self.alpha_0_TS = self.alpha_f*self.a_div_a0
        self.C6_TS = self.C6_f*self.a_div_a0*self.a_div_a0
        self.RvdW_TS = self.RvdW_f*self.a_div_a0**(1./3.)
        self.omega_TS = mbd.omega_eff(self.C6_TS, self.alpha_0_TS)
        self.beta = xc2beta[self.xc]
        
        if self.do_SCS:
            self._run_electrostatic_screening(mode='C')
        
        self._get_reciprocal_space_mbd_energy(mode=modus)
        if self.do_TS:
            self._get_TS_energy1()
            if self.do_SCS:
                self._get_TSSCS_energy1()
        
        mbd.destroy_grid()
        
    
    def _run_electrostatic_screening(self, mode=''):
        self.alpha_dyn_TS = mbd.alpha_dynamic_ts_all(mode, mbd.n_grid_omega, \
                                                     self.alpha_0_TS, c6=self.C6_TS)
        
        self.alpha_dyn_SCS = mbd.run_scs(mode, self.Coulomb_SCS, self.pos, \
                                         self.alpha_dyn_TS, r_vdw=self.RvdW_TS, \
                                         beta=self.beta, a=6, unit_cell=self.UC)
        
        self.alpha_0_SCS = self.alpha_dyn_SCS[0]
        self.C6_SCS = mbd.get_c6_from_alpha(self.alpha_dyn_SCS)
        self.RvdW_SCS = self.RvdW_TS*(self.alpha_0_SCS/self.alpha_0_TS)**(1./3.)
        self.omega_SCS = mbd.omega_eff(self.C6_SCS, self.alpha_0_SCS)
        
    
    def _get_reciprocal_space_mbd_energy(self, mode=''):
        g_grid = mbd.make_g_grid(self.Ggrid[0],self.Ggrid[1],self.Ggrid[2])
        kgrid = mbd.make_k_grid(g_grid, self.UC)
        
        if self.do_SCS:
            alph, om, rvdwAB = self.alpha_0_SCS, self.omega_SCS, self.RvdW_SCS
        else:
            alph, om, rvdwAB = self.alpha_0_TS, self.omega_TS, self.RvdW_TS
        
        self.E_MBD = mbd.get_mbd_energy(mode, self.Coulomb_CFDM, self.pos, \
                                        alph, om, k_grid=kgrid, unit_cell=self.UC, \
                                        r_vdw=rvdwAB, beta=self.beta, a=6.)
        self.E_MBD *= Hartree
        
    
    def _get_TS_energy1(self):
        self.E_TS = mbd.get_ts_energy('C', 'fermi2', self.pos, self.alpha_0_TS, \
                                      r_vdW=self.RvdW_TS, s_r=1., d=6., unit_cell=self.UC)
        
    
    def _get_TSSCS_energy1(self):
        self.E_TS_SCS = mbd.get_ts_energy('C', 'fermi2', self.pos, self.alpha_0_SCS, \
                                          r_vdw=self.RvdW_SCS, s_r=1., d=6., unit_cell=self.UC)
        
    
    def set_a_div_a0(self, rescaling):
        self.a_div_a0 = np.array(rescaling)
        
    
    def get_frequency_dependent_alpha_TS(self):
        return self.alpha_dyn_TS
        
    
    def get_static_alpha_SCS(self):
        return self.alpha_0_SCS
        
    
    def get_frequency_dependent_alpha_SCS(self):
        return self.alpha_dyn_SCS
        
    
    def get_C6_TS(self):
        return self.C6_TS
        
    
    def get_C6_SCS(self):
        return self.C6_SCS
        
    
    def get_RvdW_TS(self):
        return self.RvdW_TS
        
    
    def get_RvdW_SCS(self):
        return self.RvdW_SCS
        
    
    def get_omega_TS(self):
        return self.omega_TS
        
    
    def get_omega_SCS(self):
        return self.omega_SCS
        
    
    def get_TS_energy(self):
        if self.do_TS:
            return self.E_TS
        else:
            raise ValueError("Please specify 'do_TS = True' in order to get TS energy")
        
    
    def get_TSSCS_energy(self):
        if self.do_TS:
            return self.E_TS_SCS
        else:
            raise ValueError("Please specify 'do_TS = True' in order to get TS+SCS energy")
        
    

#--EOF--#
