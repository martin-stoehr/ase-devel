from ase.all import *
from ase.surfacehopping.tullycalcs import *
from ase.surfacehopping import *
from ase.md.verlet import VelocityVerlet
import sys
import time
import numpy as np
import multiprocessing as mp
import ase.units as units

mom=float(sys.argv[1])

momentum = mom
n_traj = 2000 
nproc = 4
dt = 1.0 * units.AUT
ttime = 6000 * units.AUT

H=Atoms('H')
H[0].mass=(H[0].mass/1836)*2000
H.positions = np.array([-5*units.Bohr,0.,0.])
H.set_pbc=np.array([False,False,False])

momentum = np.sqrt(2.0*mom/H[0].mass)
print momentum

calc = TullyCalc1(A=0.01, B=1.60, C=0.005, D=1.00)
H.set_calculator(calc)
#H.calc.current_state = 0

#print H.get_potential_energy()
#print H.get_forces()
#print H.get_momenta()
#print H.calc.get_couplings(H)

print '---------------------------------------'

import ase.units as units
from ase.md import VelocityVerlet, MDLogger

init_densmat = np.array([
                 [ 1.0, 0.0,],
                  [ 0.0, 0.0 ]
                  ])

init_densmat = None
#H.calc.show()

#transmission_low = 0.0
#transmission_high = 0.0
#reflection_low = 0.0
#reflection_high = 0.0
#trapping_low = 0.0
#trapping_high = 0.0

results = mp.Array('d',[0.,0.,0.,0.,0.,0.])

#transmission_low = mp.Value('d', 0.00)
#transmission_high= mp.Value('d', 0.00)
#reflection_low   = mp.Value('d', 0.00)
#reflection_high  = mp.Value('d', 0.00)
#trapping_low     = mp.Value('d', 0.00)
#trapping_high    = mp.Value('d', 0.00)

H.set_velocities(np.array([momentum,0.,0.]))
old_E = H.get_total_energy()
ave_drift = mp.Value('d', 0.00)

def run_trajectories(thread, r, drift):

    for i in range(n_traj/nproc):
        print 'traj ',i, 'on proc ',thread
        np.random.seed()
        H=Atoms('H')
        H[0].mass=(H[0].mass/1836)*2000
        H.positions = np.array([-5*units.Bohr,0.,0.])
        H.set_pbc=np.array([False,False,False])
        H.set_calculator(calc)
        H.set_velocities(np.array([momentum,0.,0.]))
        H.calc.current_state = 0
        dyn = SurfaceHoppingVelocityVerlet(H, dt, nstates=2, dmsteps=1, \
              trajectory='test_'+str(mom)+'_'+str(thread)+'.traj')#,logfile='md_'+str(momentum)+'.log')
        #dyn = SurfaceHoppingBeeman(H, dt, nstates=2, dmsteps=1, \
        #      trajectory='test_'+str(mom)+'_'+str(thread)+'.traj')#,logfile='md_'+str(momentum)+'.log')
        dyn.run(int(ttime/dt))
        #if H.positions[0,0]>2.0:
          #if H.calc.current_state == 0:
            #transmission_low = transmission_low + 1.0
          #else:
            #transmission_high = transmission_high + 1.0
        #elif H.positions[0,0]<-2.0:
          #if H.calc.current_state == 0:
            #reflection_low = reflection_low + 1.0
          #else:
            #reflection_high = reflection_high + 1.0
        #else:
          #if H.calc.current_state == 0:
            #trapping_low = trapping_low + 1.0
          #else:
            #trapping_high = trapping_high + 1.0
        if H.positions[0,0]>(2.0*units.Bohr):
          if H.calc.current_state == 0:
            r[0] = r[0] + 1.0
          else:
            r[1] = r[1] + 1.0
        elif H.positions[0,0]<(-2.0*units.Bohr):
          if H.calc.current_state == 0:
            r[2] = r[2] + 1.0
          else:
            r[3] = r[3] + 1.0
        else:
          if H.calc.current_state == 0:
            r[4] = r[4] + 1.0
          else:
            r[5] = r[5] + 1.0

        sys.stdout.flush()
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

transmission_low  = results[0]
transmission_high = results[1]
reflection_low    = results[2]
reflection_high   = results[3]
trapping_low      = results[4]
trapping_high     = results[5]

transmission_low = transmission_low / float(n_traj)
transmission_high = transmission_high / float(n_traj)
reflection_low = reflection_low / float(n_traj)
reflection_high = reflection_high / float(n_traj)
trapping_low = trapping_low / float(n_traj)
trapping_high = trapping_high / float(n_traj)

print 'transmission_low {0: 6.4f}'.format(transmission_low)
print 'transmission_high {0: 6.4f}'.format(transmission_high)
print 'reflection_low {0: 6.4f}'.format(reflection_low)
print 'reflection_high {0: 6.4f}'.format(reflection_high)
print 'trapping_low {0: 6.4f}'.format(trapping_low)
print 'trapping_high {0: 6.4f}'.format(trapping_high)

#dyn.attach(MDLogger(dyn, H, 'md.log', header=False, stress=False,
#          peratom=True, mode="a"), interval=1)

print ave_drift.value/float(n_traj) - old_E

#for i in range(60):
#  e1 = H.calc.get_potential_energy(H,nstate=0)
#  e2 = H.calc.get_potential_energy(H,nstate=1)
#  H.positions[0,0] = H.positions[0,0] + 0.1
  
#  print e1, e2
