import numpy as np
import os
from warnings import warn
from pymbd import mbd_energy as MBDcalc_Py, from_volumes
from ase import atoms
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator

modes_avail = ['python']
try:
    from pymbd.fortran import MBDGeom as MBDcalc_F
    modes_avail.append('fortran')
except ImportError:
    warn("Failed to import FORTRAN module.")

try:
    from pymbd.tensorflow import MBDEvaluator as MBDcalc_TF
    modes_avail.append('tensorflow')
except ImportError:
    warn("Failed to import TensorFlow module.")


beta_parameters = {"pbe":0.83,   "pbe0":0.85, "hse":0.85}
sR_parameters   = {"pbe":0.94,   "pbe0":0.96, "hse":0.96, 
                   "b3lyp":0.84, "am05":0.84, "blyp":0.62, 
                   "revpbe":0.60}

def beta_from_xc(xcf):
    try:
        return beta_parameters[xcf.lower()]
    except KeyError:
        warn("beta-parameter for "+xc+" functional not known. Using 1.0")
        return 1.0


def sR_from_xc(xcf):
    try:
        return sR_parameters[xcf.lower()]
    except KeyError:
        warn("s_R-parameter for "+xc+" functional not known. Using 1.0")
        return 1.0



default_parameters = {
                      'xc':'pbe',
                      'n_omega_SCS':15,
                      'kgrid':(3,3,3),
                      'get_MBD_spectrum':False,
                      'mode':'fortran',
                      'calc_forces':True,
                      'calc_hessian':False,
                     }



class MBD(Calculator):
    """
    
    Many-Body Dispersion calculator class
    
    Interface to libmbd (https://github.com/jhrmnn/libmbd.git),
    featuring various implementations:
        . FORTRAN: MPI/OpenMP/(Sca)LAPACK support with access to
            energy, forces, stress, eigenfrequencies, eigenmodes
            with and without periodic boundary conditions
        . TensorFlow: access to forces, hessian (non-periodic)
        . Python: access to energy, (numerical) forces and stress
            with and without periodic boundary conditions
    
    by Martin Stoehr (martin.stoehr@uni.lu), Aug 2019.
    
    """
    
    implemented_properties = ['energy', 'forces', 'stress', 'hessian']
    
    valid_args = ['xc', \
                  'n_omega_SCS', \
                  'kgrid', \
                  'get_MBD_spectrum', \
                  'mode', \
                  'calc_forces', \
                  'calc_hessian', \
                  'custom_beta', \
                 ]
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        
        ## set default arguments
        for arg, val in default_parameters.items():
            setattr(self, arg, val)
        
        ## set or overwrite any additional keyword arguments provided
        for arg, val in kwargs.items():
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s" : not in %s'
                                   % (arg, self.valid_args))
        
        if self.mode.lower() in ['f90', 'fortran']:
            self.mode = 'fortran'
        elif self.mode.lower() in ['tf', 'tensorflow']:
            self.mode = 'tensorflow'
        elif self.mode.lower() in ['py', 'python']:
            self.mode = 'python'
        else:
            msg = "'mode' has to be in ['fortran', 'tensorflow', 'python']."
            raise ValueError(msg)
        
        if self.mode not in modes_avail:
            msg =  self.mode.title()+" implementation not available "
            msg += "(for error log, see above)."
            raise ValueError(msg)
        
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        
    
    def get_potential_energy(self, atoms=None):
        """ Return dispersion energy as obtained by MBD calculation. """
        self.update_properties(atoms)
        return self.E_MBD
        
    
    def get_forces(self, atoms=None):
        if not self.calc_forces:
            raise ValueError("Please, specify 'calc_forces=True'.")
        
        self.update_properties(atoms)
        return self.F_MBD
        
    
    def get_stress(self, atoms=None):
        if not self.calc_forces:
            raise ValueError("Please, specify 'calc_forces=True'.")
        
        self.update_properties(atoms)
        return self.F_MBD_latt
        
    
    def get_hessian(self, atoms=None):
        if self.mode != 'tensorflow':
            msg = "Hessian only available in Tensorflow mode for now."
            raise NotImplementedError(msg)
        
        if not self.calc_hessian:
            raise ValueError("Please, specify 'calc_hessian=True'.")
        
        self.update_properties(atoms)
        return self.H_MBD
        
    
    def update_properties(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.calculate(atoms)
        
    
    def calculate(self, atoms):
        self.atoms = atoms.copy()
        self._set_vdw_parameters(atoms=self.atoms)
        self._set_coordinates(atoms=self.atoms)
        if self.mode == 'fortran':
            self._run_MBD_f()
        elif self.mode == 'tensorflow':
            if self.periodic:
                msg =  "Periodic boundary conditions not supported by "
                msg += "TensorFlow implementation.\n"
                msg += "Please, use mode='fortran'."
                raise NotImplementedError(msg)
            else:
                self._run_MBD_tf()
        elif self.mode == 'python':
            self._run_MBD_py()
        
    
    def _set_vdw_parameters(self, atoms=None):
        #TODO: add functionality:
        #        . fractional ionic
        #        . RvdW from alpha
        
        if hasattr(self, 'custom_beta'):
            self.beta = self.custom_beta
        else:
            self.beta = beta_from_xc(self.xc)
        
        self.alpha0_TS, self.C6_TS, self.RvdW_TS = \
                from_volumes(atoms.get_chemical_symbols(), self.a_div_a0)
        
    
    def _set_coordinates(self, atoms=None):
        self.xyz = atoms.positions/Bohr
        if any(atoms.pbc):
            [a, b, c] = atoms.get_cell()
            V = abs( np.dot(np.cross(a, b), c) )
            if V < 1e-2: warn("Cell volume < 0.01 A^3")
            self.UC = atoms.get_cell()/Bohr
            self.periodic = True
        else:
            self.UC, self.kgrid, self.periodic = None, None, False
        
    
    def _run_MBD_f(self):
        """
        Run MBD calculation via FORTRAN implementation.
        """
        
        MBD = MBDcalc_F(self.xyz, lattice=self.UC, k_grid=self.kgrid,
                 n_freq=self.n_omega_SCS, get_spectrum=self.get_MBD_spectrum)
        
        res = MBD.mbd_energy(self.alpha0_TS, self.C6_TS, self.RvdW_TS, 
                 self.beta, force=self.calc_forces)
        
        if self.periodic and self.get_MBD_spectrum and self.calc_forces:   # all
            ((self.E_MBD, self.MBDevals, self.MBDmodes), \
                  self.F_MBD, self.F_MBD_latt) = res
            self.F_MBD *= -1.*Hartree/Bohr
            self.F_MBD_latt *= -1.*Hartree/Bohr
        elif self.periodic and self.get_MBD_spectrum:   # no forces
            (self.E_MBD, self.MBDevals, self.MBDmodes) = res
        elif self.periodic and self.calc_forces:   # no spectrum
            (self.E_MBD, self.F_MBD, self.F_MBD_latt) = res
            self.F_MBD *= -1.*Hartree/Bohr
            self.F_MBD_latt *= -1.*Hartree/Bohr
        elif self.get_MBD_spectrum and self.calc_forces:   # no PBC
            ((self.E_MBD, self.MBDevals, self.MBDmodes), self.F_MBD) = res
            self.F_MBD *= -1.*Hartree/Bohr
        elif self.get_MBD_spectrum:   # no forces and no PBC
            (self.E_MBD, self.MBDevals, self.MBDmodes) = res
        elif self.calc_forces:   # no spectrum and no PBC
            (self.E_MBD, self.F_MBD) = res
            self.F_MBD *= -1.*Hartree/Bohr
        else:   # only energy (with and without PBC)
            self.E_MBD = res
        
        self.E_MBD *= Hartree
        
    
    def _run_MBD_tf(self):
        """
        Run MBD calculation via TensorFlow implementation.
        """
        
        MBD = MBDcalc_TF(gradients=self.calc_forces, 
                         hessian=self.calc_hessian, 
                         nfreq=self.n_omega_SCS)
        
        res = MBD(coords=self.xyz, alpha_0=self.alpha0_TS, C6=self.C6_TS, 
                  R_vdw=self.RvdW_TS, beta=self.beta)
        
        if self.calc_forces and self.calc_hessian:
            self.E_MBD, self.F_MBD, self.H_MBD = res
            self.F_MBD *= -1.*Hartree/Bohr
            self.H_MBD *= -1.*Hartree/Bohr/Bohr
        elif self.calc_forces:
            self.E_MBD, self.F_MBD = res
            self.F_MBD *= -1.*Hartree/Bohr
        elif self.calc_hessian:
            self.E_MBD, self.H_MBD = res
            self.H_MBD *= -1.*Hartree/Bohr/Bohr
        else:
            self.E_MBD = res
        
        self.E_MBD *= Hartree
        
    
    def _run_MBD_py(self):
        """
        Run MBD calculation via pure Python implementation.
        """
        
        self.E_MBD = MBDcalc_Py(self.xyz, self.alpha0_TS, self.C6_TS, 
                                self.RvdW_TS, self.beta, lattice=self.UC, 
                                k_grid=self.kgrid, nfreq=self.n_omega_SCS)
        self.E_MBD *= Hartree
        
        #TODO: numerical forces ?and stress?
        
    
    def set_a_div_a0(self, rescaling):
        """
        Set ratio of effective polarizability to polarizability of atom in vacuo.
        This is typically the Hishfeld-Volume-Ratio.
        """
        
        self.a_div_a0 = np.array(rescaling)
        
    
    def get_MBD_frequencies(self, atoms=None):
        """
        Returns the spectrum of MBD (eigen)frequencies in a.u.
        """
        if self.mode != 'fortran':
            msg = "MBD frequencies are only available in 'fortran' mode for now."
            raise NotImplementedError(msg)
        elif not self.get_MBD_spectrum:
            msg = "Please, specify 'get_MBD_spectrum=True' when initializing the calculator."
            raise ValueError(msg)
        elif not hasattr(self, 'MBDevals'):
            if not hasattr(self, 'atoms') and atoms is None:
                msg = "Please, specify atoms on input or run get_potential_energy() first"
                raise ValueError(msg)
            
            self.update_properties(atoms)
            
        return self.MBDevals
        
    
    def get_MBD_modes(self):
        """
        Returns the MBD (eigen)modes
        """
        if self.mode != 'fortran':
            msg = "MBD modes are only available in 'fortran' mode for now."
            raise NotImplementedError(msg)
        elif not self.get_MBD_spectrum:
            msg = "Please, specify 'get_MBD_spectrum=True' when initializing the calculator."
            raise ValueError(msg)
        elif not hasattr(self, 'MBDmodes'):
            if not hasattr(self, 'atoms') and atoms is None:
                msg = "Please, specify atoms on input or run get_potential_energy() first"
                raise ValueError(msg)
            
            self.update_properties(atoms)
            
        return self.MBDmodes
        
    

#--EOF--#
