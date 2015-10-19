from ase.all import *
from ase.surfacehopping.tullycalc1 import TullyCalc3
from ase.surfacehopping import *
import sys
import time

dt = 0.1

H=Atoms('H')
H[0].mass=(H[0].mass/1836)*2000
H.positions = np.array([-3.0,0.,0.])
H.set_pbc=np.array([False,False,False])

H.set_velocities(np.array([0.000,0.,0.]))

calc = TullyCalc3()

H.set_calculator(calc)
H.calc.current_state = 0

print H.get_potential_energy()
print H.get_forces()
print H.get_momenta()
print H.calc.get_couplings(H)

print '---------------------------------------'

import ase.units as units
from ase.md import VelocityVerlet, MDLogger

init_densmat = np.array([
                 [ 1.0, 0.0,],
                  [ 0.0, 0.0 ]
                  ])

init_densmat = None
H.calc.show()


old_E = H.get_total_energy()

#start=time.time()

#H.calc.current_state = 0
#dyn = SurfaceHoppingVerlet(H, dt * units.fs, nstates=2, init_densmat=init_densmat, \
#      trajectory='test_test2.traj',logfile='md_test2.log')
#dyn.run(8000)

#end=time.time()

#print 'time ',end-start

#dyn.attach(MDLogger(dyn, H, 'md.log', header=False, stress=False,
#          peratom=True, mode="a"), interval=1)

print H.get_total_energy()-old_E

#for i in range(60):
#  e1 = H.calc.get_potential_energy(H,nstate=0)
#  e2 = H.calc.get_potential_energy(H,nstate=1)
#  H.positions[0,0] = H.positions[0,0] + 0.1
  
#  print e1, e2
