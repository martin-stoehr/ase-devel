##############################################################
#                                                            #
##      X16 data set as used by Tkatchenko, Scheffler       ##
#          set contains organic molecular crystals           #
##      unit cells optimized using FHI-AIMS code with       ##
#         semi-empirical dispersion correction 'TS'          #
##                                                          ##
#   Experimental lattice energies derived from sublimation   #
##    enthalpies taken from Otero-de-la-Roza and Johnson    ##
#                                                            #
##     Module by Martin Stoehr (martin.stoehr@tum.de),      ##
#      Technische Universitaet Muenchen,                     #
##     Nov/2015                                             ##
#                                                            #
##############################################################


from ase.io import read
from ase.units import kJ, mol
from x16_geometries import *
import numpy as np


######################  DATA  ######################

x16_names = ['naph', 'ethcar', 'cytosine', 'uracil', 'benzene', 'oxaca', 'urea', \
             'formamide', 'hexdio', 'oxacb', 'cyanamide', 'anthracene', 'adaman', \
             'acetic', 'imdazole', 'ammonia']

x16_long_names = {'naph':'Naphtalene', 'ethcar':'Ethylcarbamat', 'cytosine':'Cytosine', \
                  'uracil':'Uracil', 'benzene':'Benzene', 'oxaca':'alpha-Oxalic acid', \
                  'urea':'Urea', 'formamide':'Formamide', 'hexdio':'1,4-Cyclohexanedione', \
                  'oxacb':'beta-Oxalic acid', 'cyanamide':'Cyanamide', 'anthracene':'Anthracene', \
                  'adaman':'Adamantane', 'acetic':'Acetic acid', 'imdazole':'Imidazole', 'ammonia':'Ammonia'}

## unit cells as obtained by optimization using PBE+TS
x16_PBE_UC = {}
x16_PBE_UC['naph']       = np.array([[8.108790, 0.000000, 0.043741],   [0.000000, 5.883793, 0.000000],  [-4.887810, 0.000000, 7.148529]])
x16_PBE_UC['ethcar']     = np.array([[5.008629, 0.018505, -0.070215],  [1.630244, 6.759721, 0.064532],  [-1.987199, -0.981878, 7.079604]])
x16_PBE_UC['cytosine']   = np.array([[12.959561, 0.000000, 0.000000],  [0.000000, 9.466517, 0.000000],  [0.000000, 0.000000, 3.763064]])
x16_PBE_UC['uracil']     = np.array([[11.895469, 0.000000, -0.083440], [0.000000, 12.266420, 0.000000], [-1.892330, 0.000000, 3.112045]])
x16_PBE_UC['benzene']    = np.array([[7.368440, 0.000000, 0.000000],   [0.000000, 9.225186, 0.000000],  [0.000000, 0.000000, 6.839004]])
x16_PBE_UC['oxaca']      = np.array([[6.663948, 0.000000, 0.000000],   [0.000000, 7.543432, 0.000000],  [0.000000, 0.000000, 6.244516]])
x16_PBE_UC['urea']       = np.array([[5.558909, 0.000000, 0.000000],   [0.000000, 5.558909, 0.000000],  [0.000000, 0.000000, 4.683049]])
x16_PBE_UC['formamide']  = np.array([[3.580823, 0.000000, 0.015223],   [0.000000, 9.000957, 0.000000],  [-1.237311, 0.000000, 6.841324]])
x16_PBE_UC['hexdio']     = np.array([[6.547967, 0.000000, 0.041715],   [0.000000, 6.324863, 0.000000],  [-1.111261, 0.000000, 6.665794]])
x16_PBE_UC['oxacb']      = np.array([[5.270530, 0.000000, 0.043493],   [0.000000, 6.164417, 0.000000],  [-2.302125, 0.000000, 4.852535]])
x16_PBE_UC['cyanamide']  = np.array([[6.878666, 0.000000, 0.000000],   [0.000000, 6.898291, 0.000000],  [0.000000, 0.000000, 8.838231]])
x16_PBE_UC['anthracene'] = np.array([[8.398539, 0.000000, -0.006493],  [0.000000, 5.943255, 0.000000],  [-6.405302, 0.000000, 9.090420]])
x16_PBE_UC['adaman']     = np.array([[6.452893, 0.000000, 0.000000],   [0.000000, 6.452893, 0.000000],  [0.000000, 0.000000, 8.960618]])
x16_PBE_UC['acetic']     = np.array([[13.156960, 0.000000, 0.000000],  [0.000000, 3.907239, 0.000000],  [0.000000, 0.000000, 5.712377]])
x16_PBE_UC['imdazole']   = np.array([[7.476138, 0.000000, 0.039309],   [0.000000, 5.349700, 0.000000],  [-4.632668, 0.000000, 8.523872]])
x16_PBE_UC['ammonia']    = np.array([[4.962164, 0.000000, 0.000000],   [0.000000, 4.962164, 0.000000],  [0.000000, 0.000000, 4.962164]])

## experimental lattice energies derived from sublimation enthalpies in kJ/mol
x16_exp_lattice_energy = {}
x16_exp_lattice_energy['naph']       =  -81.7
x16_exp_lattice_energy['ethcar']     =  -86.3
x16_exp_lattice_energy['cytosine']   = -169.8
x16_exp_lattice_energy['uracil']     = -135.7
x16_exp_lattice_energy['benzene']    =  -51.7
x16_exp_lattice_energy['oxaca']      =  -96.3
x16_exp_lattice_energy['urea']       = -102.5
x16_exp_lattice_energy['formamide']  =  -79.2
x16_exp_lattice_energy['hexdio']     =  -88.6
x16_exp_lattice_energy['oxacb']      =  -96.1
x16_exp_lattice_energy['cyanamide']  =  -79.7
x16_exp_lattice_energy['anthracene'] = -112.7
x16_exp_lattice_energy['adaman']     =  -69.4
x16_exp_lattice_energy['acetic']     =  -72.8
x16_exp_lattice_energy['imdazole']   =  -86.8
x16_exp_lattice_energy['ammonia']    =  -37.2

## number of monomer units in unit cell for bulk systems
x16_num_mono_UC = {'naph':2, 'ethcar':2, 'cytosine':4, 'uracil':4, 'benzene':4, 'oxaca':4, \
                  'urea':2, 'formamide':4, 'hexdio':2, 'oxacb':2, 'cyanamide':8, 'anthracene':2, \
                  'adaman':2, 'acetic':4, 'imdazole':4, 'ammonia':4}


######################  FUNCTIONS  ######################

def get_names():
    """ return names for X16 crystal data set """
    
    return x16_names
    

def get_number_of_monomers_in_cell(name):
    """ return number of monomer units per unit cell """
    
    return x16_num_mono_UC[name]
    

def get_unit_cell(name):
    """ return unic cell for <name> as obtained by PBE+TS """
    
    return x16_PBE_UC[name]
    

def get_lattice_energy_x16_experiment(name):
    """ return experimental lattice energy for X16 crystal <name> in eV """
    
    return x16_exp_lattice_energy[name]*kJ/mol
    

def create_x16_crystal(name):
    """ create ASE atoms object for X16 crystal <name> as obtained by PBE+TS """
    from os import listdir
    
    try:
        atoms = x16_crystals[name]
    except OSError:
        print("Something went wrong with reading geometry from x16_geometries.py for '"+name+"' crystal")
        raise
    
    atoms.set_cell(x16_PBE_UC[name])
    atoms.set_pbc([True, True, True])
    
    return atoms
    

def create_x16_monomer(name):
    """ create ASE atoms object of X16 monomer <name> as obtained by PBE+TS """
    
    try:
        atoms = x16_monomers[name]
    except OSError:
        print("Something went wrong with reading geometry from x16_geometries.py for '"+name+"' molecule")
        raise
    
    atoms.set_cell(np.array([[100.,0.,0.], [0.,100.,0.], [0.,0.,100.]]))
    atoms.set_pbc([True, True, True])
    
    return atoms
    


#--EOF--#
