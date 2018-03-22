##########################################################
#                                                        #
##   This class wraps data for the S66X8 set (Hobza)    ##
#                                                        #
##  Incl. reference interaction energies obtained by    ##
#        CCSD(T)/CBS, PBE+MBD, and PBE+vdW(TS)           #
##         see module s66x8_data for details            ##
#                                                        #
##  by Martin Stoehr (martin.stoehr@tum.de), Oct 2015   ##
#                                                        #
##########################################################

import numpy as np


"""
Implemented Properties:
    names:          list of names for systems in data set, list len = 528
                    return via get_names()
    systems:        from dictionary of ASE atoms objects corresponding to name
                    return via create_s66x8_system(<name>)
    monomer atoms:  dictionary of number of monomer atoms
                    return via get_number_monomer_atoms(<name>)
    
    interaction energies (IAE)
    ==========================
        . CCSD(T) reference using complete basis set interpolation, IAE in eV
          return via get_interaction_energy_CC(<name>)
        . PBE+MBD reference: PBE-DFT with many-body dispersion correction, IAE in eV
          (taken from <reference to MBD paper>)
          return via get_interaction_energy_PBE_MBD(<name>)
        . PBE+TS reference: PBE-DFT with sempi-empirical dispersion correction 'TS', IAE in eV
          (taken from <reference to MBD paper>)
          return via get_interaction_energy_PBE_TS(<name>)

"""

from s66x8_data import *


def get_names():
    """ return list of names for systems contained in S66X8 set. """
    
    return s66x8_names
    

def create_s66x8_system(name_system):
    """ return ASE atoms object of system in S66X8 set named <name_system>. """
    
    return s66x8_systems[name_system]
    

def get_number_monomer_atoms(name_system):
    """
    return number of atoms in monomers for system in S66X8 set named <name_system>.
    
    NOTE:
    =====
        Monomer1 = s66x8_systems[<name_system>][:n_mono_atoms[0]]
        Monomer2 = s66x8_systems[<name_system>][n_mono_atoms[0]:]
    """
    
    return np.array(s66x8_nAtoms_monomers[name_system])
    

def get_interaction_energy_CC(name_system):
    """
    return interaction energy for S66X8 dimer <name_system>
    in eV as obtained by CCSD(T)/CBS.
    """
    
    return s66x8_IAE_CC[name_system]
    

def get_interaction_energy_PBE_MBD(name_system):
    """
    return interaction energy for S66X8 dimer <name_system>
    in eV as obtained by PBE-DFT with many-body dispersion.
    """
    
    return s66x8_IAE_PBE_MBD[name_system]
    

def get_interaction_energy_PBE_TS(name_system):
    """
    return interaction energy for S66X8 dimer <name_system>
    in eV as obtained by PBE-DFT with semi-empirical dispersion
    correction 'TS' (Tkatchenko, Scheffler, et al.).
    """
    
    return s66x8_IAE_PBE_TS[name_system]
    


#--EOF--#
