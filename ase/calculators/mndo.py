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


## PUT SOME DEFAULTS HERE


class Dftb(FileIOCalculator):
    """  A MNDO calculator with ase-FileIOCalculator nomenclature  """
    if 'MNDO_COMMAND' in os.environ:
        command = os.environ['MNDO_COMMAND'] + ' > PREFIX.out'
    else:
        command = 'mndo > PREFIX.out'
    
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='mndo', atoms=None, kpts=None, **kwargs):
        """  Construct a MNDO calculator.  """
        
        from os.path import exists as pexists
        
        
        self.default_parameters = dict(
#            ???
        )
        
        self.pbc = np.any(atoms.pbc)
        
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        
        self.atoms = atoms
        if restart == None:
            self.write_mndo_inp()
        else:
            if os.path.exists(restart):
                os.system('cp ' + restart + ' mndo_ase.inp')
            if not os.path.exists('mndo_ase.inp'):
                raise IOError('No file "mndo_ase.inp", use restart=None')
        
        #indexes for the result file
        self.first_time = True
        self.index_energy = None
        self.index_force_begin = None
        self.index_force_end = None
        
    
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
