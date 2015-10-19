from ase.all import *
from ase.surfacehopping.tullycalcs import *
from ase.surfacehopping.lischkacalcs import *
from ase.surfacehopping import *
from ase.md.verlet import VelocityVerlet
import sys
import time
import numpy as np
import multiprocessing as mp
import ase.units as units

T = 0.0

energy = units.kB * T

n_traj = 1
nproc = 1
dt = 0.05
ttime = 10000
pos = -1.5
log_interval = 5

state = 0

H=Atoms('H',pbc=(0, 0, 0))
H[0].mass=(H[0].mass/1836)*2000
H.positions = np.array([pos,0.,0.])
#H.set_pbc(np.array([False,False,False]))

momentum = np.sqrt(2.0*energy/H[0].mass)
print momentum

#calc = DoubleWellPotential(A=1.00, B=1.25, C=0.05, D=0.0)
calc = TullyCalc1()
#calc.show()

H.set_calculator(calc)
H.calc.current_state = state

#print H.get_potential_energy()
#print H.get_forces()
#print H.get_momenta()
#print H.calc.get_couplings(H)


print '---------------------------------------'
for i in range(300):
    H.calc.current_state = state
    E1 = H.get_potential_energy()
    F1 = H.get_forces()
    E2 = H.calc.get_potential_energy(H,nstate=1)
    F2 = H.calc.get_forces(H,nstate=1)
    d = H.calc.get_couplings(H)
    print H.positions[0,0],'  ',E1,' ',E2, '    ',F1[0,0],' ',F2[0,0],'    ', d[0,0,0]
    H.positions[0,0] = H.positions[0,0] + 0.01


