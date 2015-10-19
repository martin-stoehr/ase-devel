from ase.all import *
from ase.surfacehopping.ALGLIB_calcs import *
from ase.surfacehopping import *
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE, MDMin
from ase.constraints import FixInternals
import sys
import time
import numpy as np
import multiprocessing as mp
import ase.units as units
from ase.neb import NEB

indices1 = [1,3,2,0]
indices2 = [2,3,1]
indices3 = [3,2,0]

count = sys.argv[1]

ang1 = 12.1
ang2 = 124.5
ang3 = 124.5

#ang1 = 180.0
#ang2 = 115
#ang3 = 115


C_max = 0.55
C_min = 0.0001
C_chi = 0.65

calc=Azobenzene_Gasphase_Calc(indices1, ['DATASET2'], rbase=60.0, nlayers=5, lambdav=0.0001, C_min=C_min, C_max=C_max, C_chi=C_chi)
#calc=Azobenzene_Gasphase_Calc(indices1, ['TESTSET'], rbase=60.0, nlayers=6, lambdav=0.0001)

print 'Model constructed!'

calc.nstates = 3

calc.current_state = 0

mask1 = [0,1,0,0]
mask2 = [0,1,0,0]
mask3 = [1,0,0,0]

atoms=read('bla.xyz')

atoms[0].mass = 72
atoms[1].mass = 72

bond_list = []
b1_ind = [1,3]
b1 = [1.4, b1_ind]
bond_list.append(b1)
b1_ind = [3,2]
b1 = [1.26, b1_ind]
bond_list.append(b1)
b1_ind = [2,0]
b1 = [1.4, b1_ind]
bond_list.append(b1)

angle_list = []

#angle_ind = [3,2,0]
#angle = [atoms.get_angle(angle_ind), angle_ind]
#angle_list.append(angle)

c = FixInternals(atoms, bonds=bond_list, angles=angle_list)
#c = FixInternals(atoms, bonds=[b1, b2, b3], angles=[])

atoms.set_constraint(c)

atoms.set_angle(indices2, (ang2/180.)*np.pi,mask=mask2)
atoms.set_angle(indices3, (ang3/180.)*np.pi,mask=mask3)
atoms.set_dihedral(indices1, (ang1/180.)*np.pi,mask=mask1)

atoms.set_calculator(calc)

omega = (atoms.get_dihedral(indices1)/np.pi)*180.
alpha = (atoms.get_angle(indices2)/np.pi)*180.
alpha2 = (atoms.get_angle(indices3)/np.pi)*180.

print omega, alpha, alpha2

#view(atoms)


#start = time.time()
#print atoms.get_potential_energy()
#print atoms.get_forces()
#print atoms.calc.get_couplings(atoms)
#calc.show(nstate=0)
#calc.show(nstate=1)
#end = time.time()
#print end - start


#calc.current_state = 1
#print atoms.get_potential_energy()
#print atoms.get_forces()
#print atoms.calc.get_couplings(atoms)

#show plots of the PES
#calc.show_all()

#e = atoms.get_potential_energy()

#f = [0.0,0.0]

#for i in range(len(atoms)):


#####################################################3
#OPT

#dyn = BFGS(atoms, trajectory='test.traj')
#dyn.run(0.001)

#####################################################3

#for i in range(18):
#    atoms.get_potential_energy()
#    omega = atoms.get_dihedral(indices1)
#    atoms.set_dihedral(indices1, omega - (10./180.)*np.pi,mask=mask1)
#    print atoms.calc.get_couplings(atoms)[:,:,0,1]

########################################################3

#print omega, alpha, alpha2

#atoms.calc.current_state = 1
#print 'S1', atoms.get_potential_energy()
#atoms.calc.current_state = 2
#print 'S2', atoms.get_potential_energy()

#x = (atoms.get_dihedral(indices)/np.pi) *180.0
#y= (atoms.get_angle(indices2)/np.pi) *180.0
#print x, y

#NEB

#n_image = 15

#initial = atoms.copy()
#final = atoms.copy()
#final.set_dihedral(indices1, (10/180.)*np.pi)
#final.set_calculator(calc)
#dyn = BFGS(final, trajectory='test.traj')
#dyn.run(0.001)
#images = [initial]
#for i in range(1,n_image+1):
#   image = initial.copy()
#   image.set_dihedral(indices1, ((180-(180/(n_image+1))*i)/180.)*np.pi)
#   images.append(image)
#images += [final]

#view(images)

#neb = NEB(images)

#for image in images:
#    image.set_calculator(calc)

#optimizer = MDMin(neb, trajectory='NEB.traj')
#optimizer.run(fmax=0.08, steps=400)

#view(images)

#neb = NEB(images, climb=True)
#optimizer = FIRE(neb, trajectory='NEB2.traj')
#optimizer.run(fmax=0.01)
#MD

#atoms.calc.current_state = 1
#dyn = VelocityVerlet(atoms, dt = 1 * units.fs, trajectory='dyn.traj')
#dyn.attach(md_log, interval=1)
#md_log()
#dyn.run(200)


#SURFACE HOPPING MD

#FRANCK CONDON EXCITATION from trans-Ab to S1

dt = 0.01
atoms.set_velocities(np.zeros([len(atoms),3]))

init_densmat = np.array([
                [ 0.0, 0.0,0.0],
                [ 0.0, 1.0,0.0 ],
                [ 0.0, 0.0,0.0 ]
                                ])

atoms.calc.current_state = 1

print 'time    omega    alpha    epot    ekin     T    epot+ekin    state      a0       a1'

dyn = SurfaceHoppingVelocityVerlet(atoms, dt * units.fs, dmsteps=2, nstates=3, init_densmat=init_densmat, \
     trajectory='TSH_Z_S1_'+str(count)+'.traj',logfile='TSH_LOG_Z_S1_'+str(count)+'.log')

def md_log(a=atoms, d=dyn):    #store a reference to atoms in the definition.
    time = (d.nsteps * d.dt)/units.fs
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    x = (a.get_dihedral(indices1)/np.pi) *180.0
    y = (a.get_angle(indices2)/np.pi) *180.0
    z = (a.get_angle(indices3)/np.pi) *180.0
    state = atoms.calc.current_state
    a0 = d.dens_mat[0,0]
    a1 = d.dens_mat[1,1]
    a2 = d.dens_mat[2,2]
    print time, x, y, z, epot, ekin, ekin/(1.5*units.kB), epot+ekin, state, a0, a1, a2

md_log()
dyn.attach(md_log, interval=1)
dyn.run(10000)



