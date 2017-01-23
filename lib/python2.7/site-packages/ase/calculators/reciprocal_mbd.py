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
                      'grid_size':25,
                      'Ggrid':(3,3,3),
                      'Coulomb_SCS':'fermi,dip,gg',
                      'Coulomb_CFDM':'fermi,dip'}


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
                  'grid_size', \
                  'Ggrid', \
                  'Coulomb_SCS', \
                  'Coulomb_CFDM']
    
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
        mbd.init_grid(self.grid_size)
        mbd.my_task, mbd.n_tasks = self.myid, self.ntasks
        
        assert hasattr(self, 'a_div_a0'), \
        "Please provide rescaling to obtain initial dispersion parameters from accurate free atom reference data via 'set_rescaling(rescaling)'!"
        
        self.pos = atoms.positions/Bohr
        self.UC = atoms.get_cell()/Bohr
        symbols = atoms.get_chemical_symbols()
        self.alpha_f, self.C6_f, self.RvdW_f = get_free_atom_data(symbols)
        self.alpha_TS = self.alpha_f*self.a_div_a0
        self.C6_TS = self.C6_f*self.a_div_a0**2
        self.RvdW_TS = self.RvdW_f*self.a_div_a0**(1./3.)
        self.omega_TS = mbd.omega_eff(self.C6_TS, self.alpha_TS)
        self.beta = xc2beta[self.xc]
        
        print('STARTING SCS SUBROUTINE')
        self.run_electrostatic_screening(mode='C')
        print('SCS SUBROUTINE DONE')
        print('STARTING EVALUATION')
        self.get_reciprocal_space_mbd_energy()
        print('EVALUATION DONE')
        mbd.destroy_grid()
        
    
    def run_electrostatic_screening(self, mode='C'):
        self.alpha_dyn_TS = mbd.alpha_dynamic_ts_all(mode, mbd.n_grid_omega, \
                                                     self.alpha_TS, c6=self.C6_TS)
        print('START SCS')
        self.alpha_dyn_SCS = mbd.run_scs(mode, self.Coulomb_SCS, self.pos, \
                                         self.alpha_dyn_TS, r_vdw=self.RvdW_TS, \
                                         beta=self.beta, a=6, unit_cell=self.UC)
        print('SCS DONE')
        self.alpha_0_SCS = self.alpha_dyn_SCS[0]
        self.C6_SCS = mbd.get_c6_from_alpha(self.alpha_dyn_SCS)
        self.RvdW_SCS = self.RvdW_TS*(self.alpha_0_SCS/self.alpha_TS)**(1./3.)
        self.omega_SCS = mbd.omega_eff(self.C6_SCS, self.alpha_0_SCS)
        
    
    def get_reciprocal_space_mbd_energy(self):
        g_grid = mbd.make_g_grid(self.Ggrid[0],self.Ggrid[1],self.Ggrid[2])
        k_grid = mbd.make_k_grid(g_grid, self.UC)
        
        self.E_MBD = mbd.get_reciprocal_mbd_energy('R', self.Coulomb_CFDM, self.pos, \
                                    self.alpha_0_SCS, self.omega_SCS, k_grid, \
                                    self.UC, r_vdw=self.RvdW_SCS, a=6, beta=self.beta)[0]
        self.E_MBD *= Hartree
        
    
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
        
    

#--EOF--#
