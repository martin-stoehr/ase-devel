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
pos = -1.25
log_interval = 5

state = 1

H=Atoms('H')
H[0].mass=(H[0].mass/1836)*2000
#H[0].mass = 2000
H.positions = np.array([pos,0.,0.])
H.set_pbc=np.array([False,False,False])

momentum = np.sqrt(2.0*energy/H[0].mass)
print momentum

calc = DoubleWellPotential(A=1.00, B=1.25, C=0.05, D=-0.5)
H.set_calculator(calc)
H.calc.current_state = state

#print H.get_potential_energy()
#print H.get_forces()
#print H.get_momenta()
#print H.calc.get_couplings(H)

print '---------------------------------------'

import ase.units as units
from ase.md import VelocityVerlet, MDLogger

#init_densmat = np.array([
#                 [ 1.0, 0.0,],
#                  [ 0.0, 0.0 ]
#                  ])
init_densmat = np.array([
                 [ 1.0, 0.0,],
                  [ 0.0, 0.0 ]
                  ])
#H.calc.show()

results = mp.Array('d',[0.,0.,0.,0.,0.,0.])


H.set_velocities(np.array([momentum,0.,0.]))
old_E = H.get_total_energy()
ave_drift = mp.Value('d', 0.00)

def do_logging(d):
    epot = d.atoms.get_potential_energy() / len(d.atoms)
    ekin = d.atoms.get_kinetic_energy() / len(d.atoms)
    pos = d.atoms.positions[0,0]
    state = d.current_state
    aa0 = float(d.dens_mat[0,0])
    aa1 = float(d.dens_mat[1,1])
    print '{0: 6.4f} {1: 6.4f} {2: 4d} {3: 4.2f} {4: 4.2f} {5: 4.2f}'.format(epot, ekin, state, aa0, aa1, pos)

def run_trajectories(thread, r, drift):

    for i in range(n_traj/nproc):
        print 'traj ',i, 'on proc ',thread
        print '  epot   ekin  state   aa0    aa1'
        np.random.seed()
        H=Atoms('H')
        H[0].mass=(H[0].mass/1836)*2000
        H.positions = np.array([pos,0.,0.])
        H.set_pbc=np.array([False,False,False])
        H.set_calculator(calc)
        H.set_velocities(np.array([momentum,0.,0.]))
        H.calc.current_state = state
        dyn = SurfaceHoppingVelocityVerlet(H, dt*units.fs, nstates=2, dmsteps=1, init_densmat=init_densmat, \
              trajectory='test_'+str(T)+'_'+str(thread)+'.traj')#,logfile='md_'+str(momentum)+'.log')
        for step in range(int(ttime/dt/log_interval)):
            do_logging(dyn)
            dyn.run(log_interval)
        
        drift.value = drift.value + H.get_total_energy()

start=time.time()

#pool = mp.Pool(nproc)

#pool.map(run_trajectories, range(nproc) )

threads=[]

for i in range(nproc):
  threads.append( mp.Process( target=run_trajectories, args=(i,results, ave_drift) ) )
  
for p in threads:
  p.start()

for p in threads:
  p.join()


end=time.time()

print 'time ',end-start



#dyn.attach(MDLogger(dyn, H, 'md.log', header=False, stress=False,
#          peratom=True, mode="a"), interval=1)

print ave_drift.value/float(n_traj) - old_E

#for i in range(60):
#  e1 = H.calc.get_potential_energy(H,nstate=0)
#  e2 = H.calc.get_potential_energy(H,nstate=1)
#  H.positions[0,0] = H.positions[0,0] + 0.1
  
#  print e1, e2
