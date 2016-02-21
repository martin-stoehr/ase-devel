##############################################################
#                                                            #
##      X23 data set as used by Tkatchenko, Scheffler       ##
#          set contains organic molecular crystals           #
##      unit cells optimized using FHI-AIMS code with       ##
#         semi-empirical dispersion correction 'TS'          #
##                                                          ##
#   Experimental lattice energies derived from sublimation   #
##    enthalpies taken from Otero-de-la-Roza and Johnson    ##
#                                                            #
##     Module by Martin Stoehr (martin.stoehr@tum.de),      ##
#      Technische Universitaet Muenchen,                     #
##     Dec/2015                                             ##
#                                                            #
##############################################################


from ase.units import kJ, mol
from x23_geometries import *
import numpy as np


######################  DATA  ######################

x23_names = ['naph', 'ethcar', 'cytosine', 'uracil', 'benzene', 'oxaca', 'urea', \
             'formamide', 'hexdio', 'oxacb', 'cyanamide', 'anthracene', 'adaman', \
             'acetic', 'imdazole', 'ammonia', 'CO2', 'hexamine', 'pyrazine', \
             'pyrazole', 'succinic', 'triazine', 'trioxane']

x23_long_names = {'naph':'Naphtalene', 'ethcar':'Ethylcarbamat', 'cytosine':'Cytosine', \
                  'uracil':'Uracil', 'benzene':'Benzene', 'oxaca':'alpha-Oxalic acid', \
                  'urea':'Urea', 'formamide':'Formamide', 'hexdio':'1,4-Cyclohexanedione', \
                  'oxacb':'beta-Oxalic acid', 'cyanamide':'Cyanamide', 'anthracene':'Anthracene', \
                  'adaman':'Adamantane', 'acetic':'Acetic acid', 'imdazole':'Imidazole', \
                  'ammonia':'Ammonia', 'CO2':'carbon dioxide', 'hexamine':'hexamethylenetetramine', \
                  'pyrazine':'pyrazine', 'pyrazole':'pyrazole', 'succinic':'succinic acid', \
                  'triazine':'1,3,5-triazine', 'trioxane':'1,3,5-trioxane'}

## unit cells as obtained by optimization using PBE+TS
x23_PBE_UC = {}
x23_PBE_UC['naph']       = np.array([[8.108790, 0.000000, 0.043741],   [0.000000, 5.883793, 0.000000],  [-4.887810, 0.000000, 7.148529]])
x23_PBE_UC['ethcar']     = np.array([[5.008629, 0.018505, -0.070215],  [1.630244, 6.759721, 0.064532],  [-1.987199, -0.981878, 7.079604]])
x23_PBE_UC['cytosine']   = np.array([[12.959561, 0.000000, 0.000000],  [0.000000, 9.466517, 0.000000],  [0.000000, 0.000000, 3.763064]])
x23_PBE_UC['uracil']     = np.array([[11.895469, 0.000000, -0.083440], [0.000000, 12.266420, 0.000000], [-1.892330, 0.000000, 3.112045]])
x23_PBE_UC['benzene']    = np.array([[7.368440, 0.000000, 0.000000],   [0.000000, 9.225186, 0.000000],  [0.000000, 0.000000, 6.839004]])
x23_PBE_UC['oxaca']      = np.array([[6.663948, 0.000000, 0.000000],   [0.000000, 7.543432, 0.000000],  [0.000000, 0.000000, 6.244516]])
x23_PBE_UC['urea']       = np.array([[5.558909, 0.000000, 0.000000],   [0.000000, 5.558909, 0.000000],  [0.000000, 0.000000, 4.683049]])
x23_PBE_UC['formamide']  = np.array([[3.580823, 0.000000, 0.015223],   [0.000000, 9.000957, 0.000000],  [-1.237311, 0.000000, 6.841324]])
x23_PBE_UC['hexdio']     = np.array([[6.547967, 0.000000, 0.041715],   [0.000000, 6.324863, 0.000000],  [-1.111261, 0.000000, 6.665794]])
x23_PBE_UC['oxacb']      = np.array([[5.270530, 0.000000, 0.043493],   [0.000000, 6.164417, 0.000000],  [-2.302125, 0.000000, 4.852535]])
x23_PBE_UC['cyanamide']  = np.array([[6.878666, 0.000000, 0.000000],   [0.000000, 6.898291, 0.000000],  [0.000000, 0.000000, 8.838231]])
x23_PBE_UC['anthracene'] = np.array([[8.398539, 0.000000, -0.006493],  [0.000000, 5.943255, 0.000000],  [-6.405302, 0.000000, 9.090420]])
x23_PBE_UC['adaman']     = np.array([[6.452893, 0.000000, 0.000000],   [0.000000, 6.452893, 0.000000],  [0.000000, 0.000000, 8.960618]])
x23_PBE_UC['acetic']     = np.array([[13.156960, 0.000000, 0.000000],  [0.000000, 3.907239, 0.000000],  [0.000000, 0.000000, 5.712377]])
x23_PBE_UC['imdazole']   = np.array([[7.476138, 0.000000, 0.039309],   [0.000000, 5.349700, 0.000000],  [-4.632668, 0.000000, 8.523872]])
x23_PBE_UC['ammonia']    = np.array([[4.962164, 0.000000, 0.000000],   [0.000000, 4.962164, 0.000000],  [0.000000, 0.000000, 4.962164]])
x23_PBE_UC['CO2']        = np.array([[5.790345, 0.000000, 0.000000],   [0.000000, 5.790345, 0.000000],  [0.000000, 0.000000, 5.790345]])
x23_PBE_UC['hexamine']   = np.array([[-3.476325, 3.476325, 3.476325],  [3.476325, -3.476325, 3.476325], [3.476325, 3.476325, -3.476325]])
x23_PBE_UC['pyrazine']   = np.array([[9.292369, 0.000000, 0.000000],   [0.000000, 5.716558, 0.000000],  [0.000000, 0.000000, 3.683419]])
x23_PBE_UC['pyrazole']   = np.array([[7.985102, 0.000000, 0.000000],   [0.000000, 12.928258, 0.000000], [0.000000, 0.000000, 6.907363]])
x23_PBE_UC['succinic']   = np.array([[5.465888, 0.000000, -0.006464],  [0.000000, 8.740231, 0.000000],  [-0.146148, 0.000000, 5.105696]])
x23_PBE_UC['triazine']   = np.array([[4.811744, -8.334186, 0.000000],  [4.811744, 8.334186, 0.000000],  [0.000000, 0.000000, 6.934483]])
x23_PBE_UC['trioxane']   = np.array([[4.674866, -8.097106, 0.000000],  [4.674866, 8.097106, 0.000000],  [0.000000, 0.000000, 8.004250]])


## experimental lattice energies derived from sublimation enthalpies in kJ/mol
x23_exp_lattice_energy = {}
x23_exp_lattice_energy['naph']       =  -81.7
x23_exp_lattice_energy['ethcar']     =  -86.3
x23_exp_lattice_energy['cytosine']   = -169.8
x23_exp_lattice_energy['uracil']     = -135.7
x23_exp_lattice_energy['benzene']    =  -51.7
x23_exp_lattice_energy['oxaca']      =  -96.3
x23_exp_lattice_energy['urea']       = -102.5
x23_exp_lattice_energy['formamide']  =  -79.2
x23_exp_lattice_energy['hexdio']     =  -88.6
x23_exp_lattice_energy['oxacb']      =  -96.1
x23_exp_lattice_energy['cyanamide']  =  -79.7
x23_exp_lattice_energy['anthracene'] = -112.7
x23_exp_lattice_energy['adaman']     =  -69.4
x23_exp_lattice_energy['acetic']     =  -72.8
x23_exp_lattice_energy['imdazole']   =  -86.8
x23_exp_lattice_energy['ammonia']    =  -37.2
x23_exp_lattice_energy['CO2']        =  -28.4
x23_exp_lattice_energy['hexamine']   =  -86.2
x23_exp_lattice_energy['pyrazine']   =  -61.3
x23_exp_lattice_energy['pyrazole']   =  -77.7
x23_exp_lattice_energy['succinic']   = -130.3
x23_exp_lattice_energy['triazine']   =  -61.7
x23_exp_lattice_energy['trioxane']   =  -66.4

## number of monomer units in unit cell for bulk systems
x23_num_mono_UC = {'naph':2, 'ethcar':2, 'cytosine':4, 'uracil':4, 'benzene':4, 'oxaca':4, \
                   'urea':2, 'formamide':4, 'hexdio':2, 'oxacb':2, 'cyanamide':8, 'anthracene':2, \
                   'adaman':2, 'acetic':4, 'imdazole':4, 'ammonia':4, 'CO2':4, 'hexamine':1, \
                   'pyrazine':2, 'pyrazole':8, 'succinic':2, 'triazine':6, 'trioxane':6}

######################  FUNCTIONS  ######################

def get_names():
    """ return names for X23 crystal data set """
    
    return x23_names
    

def get_number_of_monomers_in_cell(name):
    """ return number of monomer units per unit cell """
    
    return x23_num_mono_UC[name]
    

def get_unit_cell(name):
    """ return unic cell for <name> as obtained by PBE+TS """
    
    return x23_PBE_UC[name]
    

def get_lattice_energy_x23_experiment(name):
    """ return experimental lattice energy for X23 crystal <name> in eV """
    
    return x23_exp_lattice_energy[name]*kJ/mol
    

def create_x23_crystal(name):
    """ create ASE atoms object for X23 crystal <name> as obtained by PBE+TS """
    from os import listdir
    
    try:
        atoms = x23_crystals[name]
    except KeyError:
        print("Something went wrong with reading geometry from x23_geometries.py for '"+name+"' crystal")
        raise
    
    atoms.set_cell(x23_PBE_UC[name])
    atoms.set_pbc([True, True, True])
    
    return atoms
    

def create_x23_monomer(name):
    """ create ASE atoms object of X23 monomer <name> as obtained by PBE+TS """
    
    try:
        atoms = x23_monomers[name]
    except KeyError:
        print("Something went wrong with reading geometry from x23_geometries.py for '"+name+"' molecule")
        raise
    
    atoms.set_cell(np.array([[100.,0.,0.], [0.,100.,0.], [0.,0.,100.]]))
    atoms.set_pbc([True, True, True])
    
    return atoms
    


#--EOF--#
