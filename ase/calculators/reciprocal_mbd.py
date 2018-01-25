import numpy as np
import os
from mbd_scalapack import mbd_scalapack as mbd_mod
from pymbd import get_free_atom_data, get_damping
from ase import atoms
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from sys import stderr


default_parameters = {
                      'xc':'pbe',
                      'n_omega_SCS':15,
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
                      'kgrid':(3,3,3),
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
                      'use_MBDrsSCS_damping':False, 
                      'use_fractional_ionic_approach':False,
                     }


try:
    from mpi4py import MPI
    default_parameters['ntasks'] = MPI.COMM_WORLD.Get_size()
    default_parameters['myid'] = MPI.COMM_WORLD.Get_rank()
except ImportError:
    stderr.write('warning: Install mpi4py for MPI support\n')
    default_parameters['ntasks'] = 1
    default_parameters['myid'] = 0


class kSpace_MBD_calculator(Calculator):
    """
    Many-Body Dispersion calculator class
    
    by Martin Stoehr (martin.stoehr@uni.lu), Jan 2017.
    """
    
    
    implemented_properties = ['energy', 'forces']
    
    valid_args = ['xc', \
                  'n_omega_SCS', \
                  'kgrid', \
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
                  'set_negative_eigenvalues_zero', \
                  'use_MBDrsSCS_damping', \
                  'use_fractional_ionic_approach', \
                  'alpha_model', \
                  'atomic_charges']
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        
        use_FI = kwargs.get('use_fractional_ionic_approach', False)
        if use_FI:
            default_parameters['alpha_model'] = 'PGG_Scaled'
            valid_models = ['Mixed_Scaled', 'PGG', 'PGG_Scaled', 'RXH_Scaled']
            if ( kwargs.has_key('alpha_model') and \
                 kwargs['alpha_model'] not in valid_models ):
                print("WARNING: alpha_model '"+kwargs['alpha_model']+"' not known. Using 'PGG_Scaled'")
                del kwargs['alpha_model']
        
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
        return float(self.E_MBD)
        
    
    def get_forces(self, atoms=None):
        return NotImplementedError
        
    
    def update_properties(self, atoms, do_MBD=True):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.calculate(atoms, do_MBD=do_MBD)
        
    
    def calculate(self, atoms, do_MBD=True):
        self.atoms = atoms.copy()
        self.init_MBD(atoms=self.atoms, do_MBD=do_MBD)
        
    
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
        if self.use_MBDrsSCS_damping:
            self.damp_par_a = damp_dict['mbd_rsscs_a']
        elif self.do_SCS:
            if self.rsSCS:
                self.damp_par_a = damp_dict['mbd_rsscs_a']
            else:
                self.damp_par_a = damp_dict['mbd_scs_a']
        else:
            self.damp_par_a = damp_dict['mbd_ts_a']
        
        if self.use_MBDrsSCS_damping:
            self.damp_par_beta = damp_dict['mbd_rsscs_beta']
        elif self.do_SCS:
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
        
    
    def init_MBD(self, atoms=None, do_MBD=True):
        """
        Initialize MBD calculation and evaluate properties.
        """
        from numpy.linalg import eigvals
        
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
        solver_bak = self.eigensolver
        self.eigensolver = self.eigensolver.strip()[:5].ljust(5)
        if self.eigensolver not in ['qr   ', 'mrrr ', 'dandc']:
            self.eigensolver = 'qr   '
            if (self.myid == 0):
                print("The specified eigensolver '"+solver_bak.strip()+"' is not known (yet).")
                print("Using default solver 'qr' instead...")
        
        mbd_mod.eigensolver = self.eigensolver
        
        assert hasattr(self, 'a_div_a0'), \
        "Please provide rescaling to obtain initial dispersion parameters from accurate free atom reference data via 'set_rescaling(rescaling)'!"
        
        self.modus = ''
        if (self.ntasks > 1): self.modus += 'P'
        if np.any(self.vacuum_axis == atoms.pbc):
            if (self.myid == 0):
                msg = '\n WARNING: Your specification for vacuum_axis is not in line with '
                msg += 'the boundary conditions of the atoms object!\n'
                print(msg)
        
        if (not np.all(self.vacuum_axis)):
            self.modus += 'C'
            if not (self.do_reciprocal or self.do_supercell):
                raise ValueError("You chose periodic boundary condition via vacuum_axis, but did not specify how to handle it (do_reciprocal or do_supercell)!")
        
        self.n_atoms = len(atoms)
        self.pos = atoms.positions/Bohr
        if np.product(eigvals(atoms.get_cell())) < 1e-2:
            if not np.all(self.vacuum_axis):
                raise ValueError("Volume < 0.01 \AA^3. Please, define reasonable unit cell for periodic calculations.")
            else:
                atoms.set_cell([[1e6,0.,0.],[0.,1e6,0.],[0.,0.,1e6]])
        
        self.UC = atoms.get_cell()/Bohr
        symbols = atoms.get_chemical_symbols()
        self._get_dispersion_params(atoms)
        self._get_SCS_and_MBD_damping_params()
        
        if self.do_SCS:
            self._run_electrostatic_screening()
        
        if self.do_reciprocal: self.modus = self.modus.replace('C','CR')
        if self.get_MBD_eigenvalues: self.modus += 'E'
        if self.get_MBD_eigenvectors: self.modus += 'V'
        
        if do_MBD: self._get_mbd_energy()
        if self.do_TS: self._get_TS_energy1()
        if self.do_TSSCS:
            if not self.do_SCS:
                print('You requested TS energy with screened (SCS) polarizabilities,')
                print("but didn't set do_SCS to True. Skipping...")
            else:
                self._get_TSSCS_energy1()
        
        mbd_mod.destroy_grid()
        
    
    def _get_dispersion_params(self, atoms):
        self.alpha_ref, self.C6_ref, self.RvdW_ref = get_free_atom_data(\
                                            atoms.get_chemical_symbols() )
        if self.use_fractional_ionic_approach:
            from alpha_FI.AlphaModel import AlphaModel
            
            Z = atoms.get_atomic_numbers()
            if not hasattr(self, 'atomic_charges'):
                from ase.calculators.calculator import PropertyNotImplementedError
                try:
                    q = atoms.get_charges()
                except RuntimeError, PropertyNotImplementedError:
                    msg  = 'WARNING: Cannot get charges for fractional ionic approach from atoms object.\n'
                    msg += '         Please, provide argument atomic_charges. Using neutral atoms for now...'
                    print(msg)
                    q = np.zeros_like(Z)
            else:
                q = np.asarray(self.atomic_charges)
            
            Npop_l, Npop_u = int(Z - q), int(Z - q + 1)
            f_FI = Z - q - Npop_l
            a_FI = AlphaModel(filename='Model'+self.alpha_model+'.dat')
            for i in xrange(self.n_atoms):
                self.alpha_ref[i] =   f_FI   * a_FI.GetAlpha((Z[i],Npop_u[i])) + \
                                   (1.-f_FI) * a_FI.GetAlpha((Z[i],Npop_l[i]))
                self.C6_ref[i] =   f_FI   * a_FI.GetC6((Z[i],Npop_u[i])) + \
                                (1.-f_FI) * a_FI.GetC6((Z[i],Npop_l[i]))
            if self.do_SCS:
                a_dyn = np.zeros((self.n_omega_SCS+1, self.n_atoms))
                for i in xrange(self.n_atoms):
                    a_dyn[:,i] =    f_FI   * a_FI.GetAlpha((Z[i],Npop_u[i]), \
                                                     omega=mbd_mod.omega_grid) + \
                                 (1.-f_FI) * a_FI.GetAlpha((Z[i],Npop_l[i]), \
                                                     omega=mbd_mod.omega_grid)
            
            self.a_div_a0 *= Z/(Z - q)
            self.alpha_dyn_TS = a_dyn*self.a_div_a0
        
        self.alpha_0_TS = self.alpha_ref*self.a_div_a0
        self.C6_TS = self.C6_ref*self.a_div_a0*self.a_div_a0
        if ( self.do_SCS and not self.use_fractional_ionic_approach):
            self.alpha_dyn_TS = mbd_mod.alpha_dynamic_ts_all('C', self.n_omega_SCS, \
                                                      self.alpha_0_TS, c6=self.C6_TS)
        
        self.RvdW_TS = self.RvdW_ref*self.a_div_a0**(1./3.)
        self.omega_TS = mbd_mod.omega_eff(self.C6_TS, self.alpha_0_TS)
        
    
    def _run_electrostatic_screening(self):
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
        kgrid = mbd_mod.make_k_grid(mbd_mod.make_g_grid(*self.kgrid), self.UC)
        
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
        
    
    def get_frequency_dependent_alpha_TS(self, atoms=None):
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        if not hasattr(self, 'alpha_dyn_TS'):
            self.update_properties(atoms, do_MBD=False)
        
        return self.alpha_dyn_TS
        
    
    def get_static_alpha_SCS(self, atoms=None):
        if self.do_SCS:
            if (not hasattr(self, 'atoms')) and (atoms is None):
                raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
            
            if not hasattr(self, 'alpha_0_SCS'):
                self.update_properties(atoms, do_MBD=False)
            
            return self.alpha_0_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get alpha0 after SCS")
        
    
    def get_frequency_dependent_alpha_SCS(self, atoms=None):
        if self.do_SCS:
            if (not hasattr(self, 'atoms')) and (atoms is None):
                raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
            
            if not hasattr(self, 'alpha_dyn_SCS'):
                self.update_properties(atoms, do_MBD=False)
            
            return self.alpha_dyn_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get dynamic alpha after SCS")
        
    
    def get_C6_TS(self, atoms=None):
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        if not hasattr(self, 'C6_TS'):
            self.update_properties(atoms, do_MBD=False)
        
        return self.C6_TS
        
    
    def get_C6_SCS(self, atoms=None):
        if self.do_SCS:
            if (not hasattr(self, 'atoms')) and (atoms is None):
                raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
            
            if not hasattr(self, 'C6_SCS'):
                self.update_properties(atoms, do_MBD=False)
            
            return self.C6_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get C6_SCS")
        
    
    def get_RvdW_TS(self, atoms=None):
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        if not hasattr(self, 'RvdW_TS'):
            self.update_properties(atoms, do_MBD=False)
        
        return self.RvdW_TS
        
    
    def get_RvdW_SCS(self, atoms=None):
        if self.do_SCS:
            if (not hasattr(self, 'atoms')) and (atoms is None):
                raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
            
            if not hasattr(self, 'RvdW_SCS'):
                self.update_properties(atoms, do_MBD=False)
            
            return self.RvdW_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get RvdW_SCS")
        
    
    def get_omega_TS(self, atoms=None):
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        if not hasattr(self, 'omega_TS'):
            self.update_properties(atoms, do_MBD=False)
        
        return self.omega_TS
        
    
    def get_omega_SCS(self, atoms=None):
        if self.do_SCS:
            if (not hasattr(self, 'atoms')) and (atoms is None):
                raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
            
            if not hasattr(self, 'omega_SCS'):
                self.update_properties(atoms, do_MBD=False)
            
            return self.omega_SCS
        else:
            raise ValueError("Please specify 'do_SCS = True' in order to get omega_SCS")
        
    
    def get_TS_energy(self, atoms=None):
        if self.do_TS:
            if (not hasattr(self, 'atoms')) and (atoms is None):
                raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
            
            if not hasattr(self, 'E_TS'):
                self.update_properties(atoms, do_MBD=False)
            
            return self.E_TS
        else:
            raise ValueError("Please specify 'do_TS = True' in order to get TS energy")
        
    
    def get_TSSCS_energy(self):
        if self.do_TSSCS:
            if (not hasattr(self, 'atoms')) and (atoms is None):
                raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
            
            if not hasattr(self, 'E_TS_SCS'):
                self.update_properties(atoms, do_MBD=False)
            
            return self.E_TS_SCS
        else:
            raise ValueError("Please specify 'do_TSSCS = True' in order to get TS+SCS energy")
    
    def get_effective_masses(self, charges=None, atoms=None):
        """
        return effective mass of oscillators according to
        m[iAtom] = charges[iAtom]/(alpha_0[iAtom] * omega_0[iAtom]**2)
        
        parameters:
        ===========
            . charges  (ndarray) charges of pseudo-particles [default 1]
            . atoms    (ASE obj) Atoms-object [define if no calculation is performed before]
        
        """
        
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        if self.do_SCS:
            has_alph = hasattr(self, 'alpha_0_SCS')
            has_om = hasattr(self, 'omega_SCS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            alphas, omega_0 = self.alpha_0_SCS, self.omega_SCS
        else:
            has_alph = hasattr(self, 'alpha_0_TS')
            has_om = hasattr(self, 'omega_TS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            alphas, omega_0 = self.alpha_0_TS, self.omega_TS
        
        if charges is None: charges = np.ones(len(self.atoms))
        m_eff = charges/(alphas*omega_0*omega_0)
        return m_eff
        
    
    def get_mbd_density(self, grid, charges, evals, \
                        fname_modes="mbd_eigenmodes.out", atoms=None):
        """
        calculates MBD density of current system based on custom grid.
        
        parameters:
        ===========
            grid:        (ndarray)   [:,3] array of grid points to evaluate density in \AA
            charges:     (ndarray)   charges of pseudoelectrons [a.u.]
            evals:       (ndarray)   eigenenergies of MBD Hamiltonian [a.u.]
            fname_modes: (str)       filename containing (binary) eigenmodes
            atoms:       (atoms obj) specify only in case MBD energy is not needed
                                     otherwise run get_potential_energy() first.
        
        """
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        try:
            nAtoms = len(self.atoms)
        except (AttributeError, TypeError):
            nAtoms = len(atoms)
        
        if (np.size(evals) != 3*nAtoms):
            errstr = "\n ValueError: The number of MBD eigenvalues does not match the specified system.\n"
            errstr += " Please provide appropriate eigenvalues.\n"
            if (self.myid == 0): print errstr
            exit()
        
        if self.do_SCS:
            has_alph = hasattr(self, 'alpha_0_SCS')
            has_om = hasattr(self, 'omega_SCS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            alphas, omega_0 = self.alpha_0_SCS, self.omega_SCS
        else:
            has_alph = hasattr(self, 'alpha_0_TS')
            has_om = hasattr(self, 'omega_TS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            alphas, omega_0 = self.alpha_0_TS, self.omega_TS
            
        m_eff = charges/(alphas*omega_0*omega_0)
        omega_int = np.asarray(evals)*np.asarray(evals)
        rho = mbd_mod.eval_mbd_int_density_io(grid/Bohr, self.pos, charges, \
                                              m_eff, omega_int, fname_modes)
        
        return rho
        
    
    def get_mbd_density_cell(self, cell, origin, n_gridpoints, charges, evals, \
                       return_density=False, fname_modes="mbd_eigenmodes.out", \
                       atoms=None, write_cube=True, cube_name="mbd_density.cube"):
        """
        calculates MBD density of current system based on a regular
        cell-shaped grid and optionally writes .CUBE file.
        
        parameters:
        ===========
            cell:           (ndarray) [a,b,c] cell vectors in \AA
            origin:         (ndarray) origin of volumetric data in \AA
            n_gridpoints:   (ndarray) [Na,Nb,Nc] number of gridpoints along cell vectors
            charges:        (ndarray) charges of pseudoelectrons [a.u.]
            evals:          (ndarray) eigenenergies of MBD Hamiltonian [a.u.]
            return_density: (boolean) function call returns density as ndarray (default: False)
            fname_modes:    (string)  filename containing (FORTRAN binary) eigenmodes
            atoms:          (ASE obj) atoms object (required if no calculation is done before)
            write_cube:     (boolean) writes .cube file if True [default: True]
            cube_name:      (string)  density will be written to cube file named <cube_name>
        
        """
        
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        try:
            nAtoms = len(self.atoms)
        except (AttributeError, TypeError):
            nAtoms = len(atoms)
        
        if (np.size(evals) != 3*nAtoms):
            errstr = "\n ValueError: The number of MBD eigenvalues does not match the specified system.\n"
            errstr += " Please provide appropriate eigenvalues.\n"
            if (self.myid == 0): print errstr
            exit()
        
        if not any([return_density, write_cube]):
            if self.myid == 0:
                print " "
                print " You did not specify any way to output the density (on return or .CUBE output file)."
                print " I will write the data to a .CUBE file named "+cube_name+"for you..."
                print " "
                write_cube = True
        
        if ( not hasattr(self, 'grid') ):
            construct_grid = True
        elif ( (np.size(self.grid) != np.prod(n_gridpoints)) or \
               any(self.grid[0] != origin) ):
            construct_grid = True
        else:
            construct_grid = False
        
        if construct_grid: self._build_regular_grid(cell, origin, n_gridpoints)
        rho = self.get_mbd_density(self.grid, charges, evals, \
                                   fname_modes=fname_modes, atoms=atoms)
        
        if ( write_cube and (self.myid == 0) ):
            from ase.utils.write_data import write_cubefile
            write_cubefile(origin, cell, n_gridpoints, self.atoms.positions, \
                           self.atoms.get_atomic_numbers(), rho, \
                           file_name=cube_name)
        
        if (return_density): return rho
        
    
    def get_noninteracting_density(self, grid, charges, atoms=None):
        """
        calculates MBD density of current system based on custom grid.
        
        parameters:
        ===========
            grid:        (ndarray)   [:,3] array of grid points to evaluate density in \AA
            charges:     (ndarray)   charges of pseudoelectrons [a.u.]
            atoms:       (atoms obj) specify only in case MBD energy is not needed
                                     otherwise run get_potential_energy() first.
        
        """
        
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        if self.do_SCS:
            has_alph = hasattr(self, 'alpha_0_SCS')
            has_om = hasattr(self, 'omega_SCS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            omega_0, alphas_0 = self.omega_SCS, self.alpha_0_SCS
        else:
            has_alph = hasattr(self, 'alpha_0_TS')
            has_om = hasattr(self, 'omega_TS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            omega_0, alphas_0 = self.omega_TS, self.alpha_0_TS
        
        m_eff = charges/(alphas_0*omega_0*omega_0)
        rho = mbd_mod.eval_mbd_nonint_density(grid/Bohr, self.pos, charges, \
                                              m_eff, omega_0)
        
        return rho
        
    
    def get_noninteracting_density_cell(self, cell, origin, n_gridpoints, charges, \
                                return_density=False, atoms=None, write_cube=True, \
                                cube_name="mbd_nonint_density.cube"):
        """
        calculates MBD density of current system based on a regular
        cell-shaped grid and optionally writes .CUBE file.
        
        parameters:
        ===========
            cell:           (ndarray) [a,b,c] cell vectors in \AA
            origin:         (ndarray) origin of volumetric data in \AA
            n_gridpoints:   (ndarray) [Na,Nb,Nc] number of gridpoints along cell vectors
            charges:        (ndarray) charges of pseudoelectrons [a.u.]
            return_density: (boolean) function call returns density as ndarray (default: False)
            atoms:          (ASE obj) atoms object (required if no calculation is done before)
            write_cube:     (boolean) writes .cube file if True [default: True]
            cube_name:      (string)  density will be written to cube file named <cube_name>
        
        """
        
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        if not any([return_density, write_cube]):
            if self.myid == 0:
                print " "
                print " You did not specify any way to output the density (on return or .CUBE output file)."
                print " I will write the data to a .CUBE file named "+cube_name+"for you..."
                print " "
                write_cube = True
            
        if ( not hasattr(self, 'grid') ):
            construct_grid = True
        elif ( (np.size(self.grid) != np.prod(n_gridpoints)) or \
               any(self.grid[0] != origin) ):
            construct_grid = True
        else:
            construct_grid = False
        
        if construct_grid: self._build_regular_grid(cell, origin, n_gridpoints)
        rho = self.get_noninteracting_density(self.grid, charges, atoms=atoms)
        
        if ( write_cube and (self.myid == 0) ):
            from ase.utils.write_data import write_cubefile
            write_cubefile(origin, cell, n_gridpoints, self.atoms.positions, \
                           self.atoms.get_atomic_numbers(), rho, \
                           file_name=cube_name)
        
        if (return_density): return rho
        
    
    def get_drho_int_nonint(self, grid, charges, evals, fname_modes="mbd_eigenmodes.out", \
                            atoms=None):
        """
        returns difference in MBD density between fully and non-interacting system.
        
        parameters:
        ===========
            grid:           (ndarray) (Npoints, 3)-shaped coordinates of grid points in \AA
            charges:        (ndarray) charges of pseudoelectrons [a.u.]
            evals:          (ndarray) eigenenergies of MBD Hamiltonian [a.u.]
            fname_modes:    (string)  filename containing (FORTRAN binary) eigenmodes
            atoms:          (ASE obj) atoms object (required if no calculation is done before)
        
        """
        
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        try:
            nAtoms = len(self.atoms)
        except (AttributeError, TypeError):
            nAtoms = len(atoms)
        
        if (np.size(evals) != 3*nAtoms):
            errstr = "\n ValueError: The number of MBD eigenvalues does not match the specified system.\n"
            errstr += " Please provide appropriate eigenvalues.\n"
            if (self.myid == 0): print errstr
            exit()
        
        if self.do_SCS:
            has_alph = hasattr(self, 'alpha_0_SCS')
            has_om = hasattr(self, 'omega_SCS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            omega_0, alphas_0 = self.omega_SCS, self.alpha_0_SCS
        else:
            has_alph = hasattr(self, 'alpha_0_TS')
            has_om = hasattr(self, 'omega_TS')
            if not (has_alph and has_om):
                self.update_properties(atoms, do_MBD=False)
            
            omega_0, alphas_0 = self.omega_TS, self.alpha_0_TS
        
        m_eff = charges/(alphas_0*omega_0*omega_0)
        omega_int = np.asarray(evals)*np.asarray(evals)
        drho = mbd_mod.eval_mbd_drho_int_nonint_io(grid/Bohr, self.pos, charges, \
                                                   m_eff, omega_int, omega_0, \
                                                   fname_modes)
        return drho
        
    
    def get_drho_int_nonint_cell(self, cell, origin, n_gridpoints, charges, evals, \
                                 return_drho=False, fname_modes="mbd_eigenmodes.out", \
                                 atoms=None, write_cube=True, write_cube_int=False, \
                                 cube_name="mbd_density_difference.cube", \
                                 cube_name_int="mbd_density.cube"):
        """
        calculates MBD density difference between fully and non-interacting system
        on a regular cell-shaped grid and optionally writes cube files of the densities.
        
        parameters:
        ===========
            cell:           (ndarray) [a,b,c] cell vectors in \AA
            origin:         (ndarray) origin of volumetric data in \AA
            n_gridpoints:   (ndarray) [Na,Nb,Nc] number of gridpoints along cell vectors
            charges:        (ndarray) charges of pseudoelectrons [a.u.]
            evals:          (ndarray) eigenenergies of MBD Hamiltonian [a.u.]
            return_drho:    (boolean) function call returns density as ndarray (default: False)
            fname_modes:    (string)  filename containing (FORTRAN binary) eigenmodes
            atoms:          (ASE obj) atoms object (required if no calculation is done before)
            write_cube:     (boolean) write .cube file of density difference (default: True)
            write_cube_int: (boolean) write .cube file of interacting density (default: False)
            cube_name:      (string)  filename for density difference .cube file
            cube_name_int:  (string)  filename for interacting density .cube file
        
        """
        
        if (not hasattr(self, 'atoms')) and (atoms is None):
            raise ValueError("Please specify atoms object on input or run get_potential_energy() first!")
        
        try:
            nAtoms = len(self.atoms)
        except (AttributeError, TypeError):
            nAtoms = len(atoms)
        
        if (np.size(evals) != 3*nAtoms):
            errstr = "\n ValueError: The number of MBD eigenvalues does not match the specified system.\n"
            errstr += " Please provide appropriate eigenvalues.\n"
            if (self.myid == 0): print errstr
            exit()
        
        if not any([return_drho, write_cube]):
            if self.myid == 0:
                print " "
                print " No specification of how to output density difference (return or .CUBE file)."
                print " I will write the data to a .CUBE file named "+cube_name_drho+"for you..."
                print " "
                write_cube_drho = True
        
        if ( not hasattr(self, 'grid') ):
            construct_grid = True
        elif ( (np.size(self.grid) != np.prod(n_gridpoints)) or \
               any(self.grid[0] != origin) ):
            construct_grid = True
        else:
            construct_grid = False
        
        if construct_grid: self._build_regular_grid(cell, origin, n_gridpoints)
        if (write_cube_int):
            rho_int = self.get_mbd_density(self.grid, charges, evals, \
                                           fname_modes=fname_modes, \
                                           atoms=atoms)
            if (self.myid == 0):
                from ase.utils.write_data import write_cubefile
                write_cubefile(origin, cell, n_gridpoints, self.atoms.positions, \
                               self.atoms.get_atomic_numbers(), rho_int, \
                               file_name=cube_name_int)
            del(rho_int)
        
        drho = self.get_drho_int_nonint(self.grid, charges, evals, \
                                        fname_modes=fname_modes, \
                                        atoms=atoms)
        
        if ( write_cube and (self.myid == 0) ):
            from ase.utils.write_data import write_cubefile
            write_cubefile(origin, cell, n_gridpoints, self.atoms.positions, \
                           self.atoms.get_atomic_numbers(), drho, file_name=cube_name)
        
        if (return_drho): return drho
        
    
    def _build_regular_grid(self, cell, origin, n_gridpts):
        """ build regular grid for given cell, origin, and number of grid points. """
        n_gridpts = np.asarray(n_gridpts, dtype=int)
        duc = ( np.asarray(cell).T/(n_gridpts-1) ).T
        self.grid, i = np.zeros((np.prod(n_gridpts),3)), 0
        for fx_i in xrange(n_gridpts[0]):
            for fy_i in xrange(n_gridpts[1]):
                for fz_i in xrange(n_gridpts[2]):
                    lattpt = np.asarray([fx_i,fy_i,fz_i]).dot(duc)
                    self.grid[i] = np.asarray(origin) + lattpt
                    i += 1
        
        
    

#--EOF--#
