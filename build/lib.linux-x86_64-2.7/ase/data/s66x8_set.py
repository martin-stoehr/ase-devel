##########################################################
#                                                        #
##  This module contains the S66X8 set by Hobza et al.  ##
#                                                        #
##  Incl. reference interaction energies obtained by    ##
#        CCSD(T)/CBS, PBE+MBD, and PBE+vdW(TS)           #
##                                                      ##
#   by Martin Stoehr (martin.stoehr@tum.de)              #
##  Oct/29/2015                                         ##
#                                                        #
##########################################################


class s66x8:
    """
    Class for S66X8 data set (Hobza).
    Implemented Properties:
        names:          list of names for systems in data set, list len = 528
                        return via s66x8.get_names()
        systems:        from dictionary of ASE atoms objects corresponding to name
                        return via s66x8.create_s66x8_system(<name>)
        monomer atoms:  dictionary of number of monomer atoms
                        return via s66x8.get_number_monomer_atoms(<name>)
        
        interaction energies (IAE)
        ==========================
            . CCSD(T) reference using complete basis set interpolation, IAE in eV
              return via s66x8.get_interaction_energy_CC_s66x8(<name>)
            . PBE+MBD reference: PBE-DFT with many-body dispersion correction, IAE in eV
              (taken from <reference to MBD paper>)
              return via s66x8.get_interaction_energy_PBE_MBD_s66x8(<name>)
            . PBE+TS reference: PBE-DFT with sempi-empirical dispersion correction 'TS', IAE in eV
              (taken from <reference to MBD paper>)
              return via s66x8.get_interaction_energy_PBE_TS_s66x8(<name>)
    
    TODO: implement systems and reference values from external file into module!
    """
    
    def __init__(self):
        """ read reference data from external binary file """
        import cPickle as pick
        from ase.atoms import Atoms
        
        
        inp = open('S66X8_set.dat','rb')
        s66x8data = pick.load(inp)
        inp.close()
        
        self.names = s66x8data['s66x8_names']
	self.systems = s66x8data['s66x8_dimers']
        self.n_mono_atoms = s66x8data['s66x8__number_of_atoms_monomer']
        self.IAE_CC = s66x8data['s66x8_interaction_energies']
        self.IAE_PBE_MBD = s66x8data['s66x8_interaction_energies_PBE_MBD']
        self.IAE_PBE_TS = s66x8data['s66x8_interaction_energies_PBE_TS']
        
    
    def get_names(self):
        """ return list of names for systems contained in S66X8 set. """
        
        return self.names
        
    
    def create_s66x8_system(self, name_system):
        """ return ASE atoms object of system in S66X8 set named <name_system>. """
        
        return self.systems[name_system]
        
    
    def get_number_monomer_atoms(self, name_system):
        """
        return number of atoms in monomers for system in S66X8 set named <name_system>.
        
        NOTE:
        =====
            Monomer1 = S66x8_dimer[:n_mono_atoms[0]]
            Monomer2 = S66x8_dimer[n_mono_atoms[0]:]
        """
        import numpy as np
        
        return np.array(self.n_mono_atoms[name_system])
        
    
    def get_interaction_energy_CC_s66x8(self, name_system):
        """
        return interaction energy for S66X8 dimer <name_system>
        in eV as obtained by CCSD(T)/CBS.
        """
        
        return self.IAE_CC[name_system]
        
    
    def get_interaction_energy_PBE_MBD_s66x8(self, name_system):
        """
        return interaction energy for S66X8 dimer <name_system>
        in eV as obtained by PBE-DFT with many-body dispersion.
        """
        
        return self.IAE_PBE_MBD[name_system]
        
    
    def get_interaction_energy_PBE_TS_s66x8(self, name_system):
        """
        return interaction energy for S66X8 dimer <name_system>
        in eV as obtained by PBE-DFT with semi-empirical dispersion
        correction 'TS' (Tkatchenko, Scheffler, et al.).
        """
        
        return self.IAE_PBE_TS[name_system]
        
    

#--EOF--#
