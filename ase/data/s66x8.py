##########################################################
#                                                        #
##   This class wraps data for the S66X8 set (Hobza)    ##
#                                                        #
##  Incl. reference interaction energies obtained by    ##
#        CCSD(T)/CBS, PBE+MBD, and PBE+vdW(TS)           #
##         see module s66x8_data for details            ##
#                                                        #
##  by Martin Stoehr (martin.stoehr@uni.lu), Aug 2018   ##
#                                                        #
##########################################################


from s66x8_data import *


class s66x8_class:
    """
    Implemented Properties:
        names:      list of names for systems in data set, list len = 528
                    return via s66x8.get_names()
        systems:    from dictionary of ASE atoms objects corresponding to name
                    return via s66x8.create_s66x8_system(<name>)
        monomers:   ASE atoms object for monomer1 and monomer2
                    return via s66x8.create_s66x8_monomer1/2(<name>)
        
        interaction energies (IAE)
        ==========================
            . CCSD(T) reference using complete basis set interpolation, IAE in eV
              return via s66x8.get_interaction_energy_CC(<name>)
            . PBE+MBD reference: PBE-DFT with many-body dispersion correction, IAE in eV
              (taken from <reference to MBD paper>)
              return via s66x8.get_interaction_energy_PBE_MBD(<name>)
            . PBE+TS reference: PBE-DFT with sempi-empirical dispersion correction 'TS', IAE in eV
              (taken from <reference to MBD paper>)
              return via s66x8.get_interaction_energy_PBE_TS(<name>)
    
    """
    
    def __init__(self):
        pass
        
    
    def get_names(self):
        """ return list of names for systems contained in S66X8 set. """
        
        return s66x8_names
        
    
    def create_s66x8_system(self, name):
        """ return ASE atoms object of system in S66X8 set named <name>. """
        
        return s66x8_systems[name]
        
    
    def create_s66x8_monomer1(self, name):
        """ return first monomer for system in S66X8 set named <name>. """
        
        return s66x8_systems[name][:s66x8_nAtoms_monomers[name][0]]
        
    
    def create_s66x8_monomer2(self, name):
        """ return second monomer for system in S66X8 set named <name>. """
        
        return s66x8_systems[name][s66x8_nAtoms_monomers[name][0]:]
        
    
    def get_interaction_energy_CC(self, name):
        """
        return interaction energy for S66X8 dimer <name>
        in eV as obtained by CCSD(T)/CBS.
        """
        
        return s66x8_IAE_CC[name]
        
    
    def get_interaction_energy_PBE_MBD(self, name):
        """
        return interaction energy for S66X8 dimer <name>
        in eV as obtained by PBE-DFT with many-body dispersion.
        """
        
        return s66x8_IAE_PBE_MBD[name]
        
    
    def get_interaction_energy_PBE_TS(self, name):
        """
        return interaction energy for S66X8 dimer <name>
        in eV as obtained by PBE-DFT with semi-empirical dispersion
        correction 'TS' (Tkatchenko, Scheffler, et al.).
        """
        
        return s66x8_IAE_PBE_TS[name]
        
    

s66x8 = s66x8_class()


#--EOF--#
