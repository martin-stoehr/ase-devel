# -*- coding: utf-8 -*-

"""populate_vibrations"""

import pickle
from math import cos, sin, pi, sqrt, log, exp
from os import remove
from os.path import isfile
import sys

from random import uniform, seed

import numpy as np

import ase.units as units
from ase.io.trajectory import PickleTrajectory
from ase.parallel import rank, paropen


def populate_vibrations(atoms=None, vibdata=None, T=300.0, runmode='stat', seed_init=None):
    """Function takes an ase atoms object and 
    vibdata==[ freqs, disps ] with disps == np.array(freqs.len(),3*Natoms)
    and returns an atoms object with a geometry that 
    corresponds to a good MD starting point at the given Temp.
    """
######SUBROUTINES####

    def populate():

        positions_md = np.zeros([atoms.get_number_of_atoms(),3])
        velocities_md = positions_md.copy()

        if runmode == 'stat':

            tmp = np.zeros(freqs.shape)
            rand_nr = np.zeros(freqs.shape)
            for (s, freq) in enumerate(freqs):
                tmp[s] = sqrt(-log(1.0000 - uniform(0.000000000001, 0.99999999999999999999999)))
                rand_nr[s] = uniform(0,2*pi)
                #tmp[s] = sqrt(-log(1.0000 - rand_nr[s]/(2*pi)  ))
                
            for (aa, a) in enumerate(atoms):
                
                for (s, freq) in enumerate(freqs):
                
                    pos_tmp, vel_tmp = modes[s,aa*3:aa*3+3], -modes[s,aa*3:aa*3+3]
                    pos_tmp = pos_tmp * tmp[s] * cos(rand_nr[s])
                    vel_tmp = vel_tmp * tmp[s] * sin(rand_nr[s])
                    pos_tmp = pos_tmp / (abs(freq) / hbar )      #1 / (eV / eV*ase-time) = ase-time
                    
                    positions_md[aa, :] += pos_tmp    # ase-time
                    velocities_md[aa, :] += vel_tmp   # 
                    
                prefac = sqrt(2.0 * kb_t * units._e / (a.mass * units._amu)) * \
                         (1.0E10 / units.second)  # Ang/ase-time
                positions_md[aa, :] *= prefac  #Ang
                velocities_md[aa, :] *= prefac  #Ang/ase-time
                
            #Now positions_md and velocities_md are filled and we have to put them into atoms_MD
            
        elif runmode == 'equal': 
        
            for (aa, a) in enumerate(atoms):
                
                for (s, freq) in enumerate(freqs):
                    
                    rand_nr = uniform(0,2*pi)
                    pos_tmp, vel_tmp = modes[s,aa*3:aa*3+3], -modes[s,aa*3:aa*3+3]
                    pos_tmp = pos_tmp / (freq / hbar)
                        
                    positions_md[aa,:] += pos_tmp * cos(rand_nr)
                    velocities_md[aa,:] += vel_tmp * sin(rand_nr)
                    
                prefac = sqrt(4.0 * kb_t * units._e / (a.mass * units._amu)) * \
                         (1.0E10 / units.second)  # Ang/ase-time
                positions_md[aa, :] *= prefac
                velocities_md[aa, :] *= prefac

        else: 
            raise ValueError('mode can only be equal or stat.')


        return [positions_md, velocities_md]


    def calculate_Temp():

        kinetic_energy = 0.0
        for (aa, a) in enumerate(atoms):
            temp = velocities_md[aa,:]**2
            kinetic_energy += temp.sum() * a.mass 
        
        kinetic_energy = kinetic_energy #/ atoms.get_number_of_atoms()
        kinetic_energy = kinetic_energy * (1.0E-20 * units.second**2 * \
                        units._amu / units._e)
        
        T_temp = (kinetic_energy) / (units.kB * len(freqs))

        return T_temp
######END SUBROUTINES############

    if atoms is None:
        raise ValueError('Atoms object has to be defined.')
    if vibdata is None:
        raise ValueError('vibdata list has to be defined.')
    
    if seed_init is not None:
        seed(int(seed_init))
    
    freqs = np.array(vibdata[0])
    modes = np.array(vibdata[1])
    
    #Check if atoms and vibdata agree on dimensions
    #if (atoms.get_number_of_atoms()*3,atoms.get_number_of_atoms()*3) != modes.shape:
    #    raise ValueError('Atoms object does not agree with vibdata dimension.')

    kb_t = T * units.kB      #Boltzmann constant in eV
    hbar = units._hbar * units.kJ * units.second * 1.E-3   # hbar in eV * ase-time
    
    atoms_MD = atoms.copy()

    T_temp = 0.0

#    while abs((T_temp-T)/T) > 0.05:

#        print 'current Temperature :', T_temp
#        [positions_md, velocities_md] = populate()


#        T_temp = calculate_Temp()

    [positions_md, velocities_md] = populate()
    T_temp = calculate_Temp()

    print('Final Temperature found!     ', T_temp)

    atoms_MD.set_positions(atoms.positions+positions_md)
    atoms_MD.set_velocities(velocities_md)
            
    return atoms_MD
