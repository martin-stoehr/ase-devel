import numpy as np
import ase.units as units
from ase.surfacehopping.tsh import SurfaceHoppingMD
from ase.surfacehopping.tsh_units import hbar

#NOT YET MODIFIED FROM VelocityVerlet

class SurfaceHoppingASEVerlet(SurfaceHoppingMD):
    """VelocityVerlet with Tully's surface hopping fewest switches method"""

    def __init__(self, atoms, dt, dmsteps=None, nstates=1, init_densmat=None ,
                 NAC_method=None, trajectory=None, logfile=None,
                 loginterval=1):
        SurfaceHoppingMD.__init__(self, atoms, dt, trajectory, logfile,
                                   loginterval, nstates)

        NAC_types = ['otf', 'precalc']
        if NAC_method is None:
            self.NAC_otf = True
        else:
            self.NAC_otf = False

        #density matrix elements
        if init_densmat is None:
            self.dens_mat = np.zeros([self.nstates,self.nstates], dtype=complex)
            #per default initialised in ground state
            self.dens_mat[0,0] = 1.0
        else:
            assert init_densmat.shape == (self.nstates,self.nstates)
            self.dens_mat = init_densmat

        self.dens_mat_old = np.zeros_like(self.dens_mat)

        test = 0.0
        for i in range(self.nstates):
            if self.dens_mat[i,i]>test:
                self.current_state = i
                test = self.dens_mat[i,i]
        self.atoms.calc.current_state = self.current_state
        #this is the time step for integrating the time-dep. SGE
        #should be a tenth of the class. timestep
        if dmsteps is None:
            self.dmsteps = 2
        else:
            self.dmsteps = dmsteps

        self.coupling_dict = self.atoms.calc.coupling_dict

    def run(self, steps=50):
        """Integrate equation of motion."""

        #calculate relevant initial properties
        e = self.atoms.calc.get_potential_energy(self.atoms, nstate=self.current_state)
        f = self.atoms.calc.get_forces(self.atoms, nstate=self.current_state)

        if not self.atoms.has('momenta'):
            self.atoms.set_momenta(np.zeros_like(f[:,:,0]))

        for step in xrange(steps):
            e, f = self.step(e, f)
            self.nsteps += 1
            self.call_observers()

    def integrate(self, f):

        #ASE verlet algorithm
        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * f
        self.atoms.set_positions(self.atoms.get_positions() +
            self.dt * p / self.atoms.get_masses()[:,np.newaxis]+self.dt*self.dt *
            f / (2*self.atoms.get_masses()[:,np.newaxis]) )
        self.atoms.set_momenta(p)

        f = self.atoms.get_forces()

        self.atoms.set_momenta(p + 0.5 * self.dt * f)

        return f

    def propagate_dens_mat(self, h_new, h_old, d_new, d_old, dt, v_new, v_old):

        dens_mat_d = np.zeros_like(self.dens_mat)

        ###Simple Euler SHIT

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * self.dens_mat[i,j] * (h_old[i,i]-h_old[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (self.dens_mat[l,j] * np.dot(v_old.flatten(),d_old[i,l,:,:].flatten())- \
                                 self.dens_mat[i,l] * np.dot(v_old.flatten(),d_old[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        #new_dens_mat = self.dens_mat + dens_mat_d * self.dt
        #new_dens_mat = new_dens_mat / np.trace(new_dens_mat)

        ### Runge Kutta RK4 Propagation

        k1 = dens_mat_d * dt

        h_mid = (h_new+h_old) / 2
        d_mid = (d_new+d_old) / 2
        v_mid = (v_new+v_old) / 2

        dm_k1 = self.dens_mat + k1/2

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * dm_k1[i,j] * (h_mid[i,i]-h_mid[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (dm_k1[l,j] * np.dot(v_mid.flatten(),d_mid[i,l,:,:].flatten())- \
                                 dm_k1[i,l] * np.dot(v_mid.flatten(),d_mid[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        k2 = dens_mat_d * dt
        dm_k2 = self.dens_mat + k2/2

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * dm_k2[i,j] * (h_mid[i,i]-h_mid[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (dm_k2[l,j] * np.dot(v_mid.flatten(),d_mid[i,l,:,:].flatten())- \
                                 dm_k2[i,l] * np.dot(v_mid.flatten(),d_mid[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        k3 = dens_mat_d * dt
        dm_k3 = self.dens_mat + k3

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * dm_k3[i,j] * (h_new[i,i]-h_new[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (dm_k3[l,j] * np.dot(v_new.flatten(),d_new[i,l,:,:].flatten())- \
                                 dm_k3[i,l] * np.dot(v_new.flatten(),d_new[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        k4 = dens_mat_d * dt

        new_dens_mat = self.dens_mat + k1/6 + k2/3 + k3/3 + k4/6
        new_dens_mat = new_dens_mat / np.trace(new_dens_mat)

        return new_dens_mat

    def switching(self, h, d, v):

        switch_prob=np.zeros([self.nstates,self.nstates])

        for i in range(self.nstates):
            for j in range(self.nstates):
                switch_prob[i,j] = (2.0/hbar)*np.imag(self.dens_mat[i,j].conjugate()*h[i,j]) - \
                                   2.0*np.real(self.dens_mat[i,j].conjugate() *\
                                   np.dot(v.flatten(),d[i,j,:,:].flatten()))

        rand=np.random.random()

        new_state = self.current_state
        do_switch = False
        prob = 0.0
        for i in range(self.nstates):
            if i==self.current_state:
                continue
            tmp = np.real(switch_prob[i,self.current_state] * self.dt / \
                  self.dens_mat[self.current_state,self.current_state])
            if tmp<0.0: tmp=0.0
            if rand>prob and rand<=(prob+tmp):
                do_switch = True
                new_state = i
            prob = prob + tmp

        if do_switch:

            NAC = np.real(d[self.current_state,new_state,:,:])
            #is the energy sufficient to do this switch
            nac_sum_a=0.0
            nac_sum_b=0.0
            for a in range(len(self.atoms)):
              #following Chem. Phys. 349, 334 (2008)
                nac_sum_a = nac_sum_a + np.dot(NAC[a],NAC[a]) / self.atoms[a].mass
                nac_sum_b = nac_sum_b + np.dot(v[a],NAC[a])

            nac_sum_a = np.real(nac_sum_a / 2.0)
            nac_sum_b = np.real(nac_sum_b)

            kin_e = np.real((nac_sum_b * nac_sum_b) + (4.0*nac_sum_a)* \
                    (h[self.current_state,self.current_state]- \
                    h[new_state,new_state]))

            #normalizing NACs
            #for i in range(len(self.atoms)):
            #    NAC[i] = NAC[i] / np.sqrt(np.dot(NAC[i],NAC[i]))

            if kin_e >= 0.0:
                #readjust forces
                if nac_sum_b<0.0:
                    gamma = (nac_sum_b + np.sqrt(kin_e)) / (2.0*nac_sum_a)
                else:
                    gamma = (nac_sum_b - np.sqrt(kin_e)) / (2.0*nac_sum_a)

                #gamma = np.real(h[new_state,new_state] - h[self.current_state,self.current_state])
                #sign = np.sign(gamma)

                #switch is allowed
                self.current_state = new_state
                self.atoms.calc.current_state = new_state
            else:
                #frustrated hopping

                gamma = 0.0
                #gamma = nac_sum_b / nac_sum_a

            #we rescale velocities with gamma

            for i in range(len(self.atoms)):
                #v[i] = v[i] + np.sign(gamma)*np.sqrt(2*gamma /self.atoms[i].mass)* NAC[i]
                v[i] = v[i] - (gamma * NAC[i] /self.atoms[i].mass)
            self.atoms.set_velocities(v)

        else:
            pass

        return

    def construct_h_d(self):

        d = self.atoms.calc.get_couplings(self.atoms)

        hamilton_m = np.zeros_like(self.dens_mat)
        coupling_m = np.array(d, dtype=complex)
        for i in range(self.nstates):
            hamilton_m[i,i] = self.atoms.calc.get_potential_energy(self.atoms, nstate=i)
            for j in range(i+1,self.nstates):
                coupling_m[i,j,:,:] = d[:,:,self.coupling_dict[str(i+1)+str(j+1)]]
                coupling_m[j,i,:,:] = d[:,:,self.coupling_dict[str(j+1)+str(i+1)]]

        return hamilton_m, coupling_m

    def step(self, e, f):

        hamilton_m_old, coupling_m_old = self.construct_h_d()
        v_old = self.atoms.get_velocities()
        ### -------------------------------------------- #
        #STEP 1 integrate class. time step

        f = self.integrate(f)

        ### -------------------------------------------- #
        #STEP 2 integrate TD-SGE to get new density matrix ... using Runge-Kutta-method

        hamilton_m, coupling_m = self.construct_h_d()
        v = self.atoms.get_velocities()

        #simple dm integration
        #self.dens_mat = self.propagate_dens_mat(hamilton_m, hamilton_m_old, coupling_m, coupling_m_old, self.dt)

        #finer timestep dm integration

        h_incr = (hamilton_m - hamilton_m_old) / self.dmsteps
        d_incr = (coupling_m - coupling_m_old) / self.dmsteps
        v_incr = (v - v_old) / self.dmsteps

        h1 = hamilton_m_old
        d1 = coupling_m_old
        v1 = v_old
        for i in range(self.dmsteps):

            self.dens_mat = self.propagate_dens_mat(h1+h_incr, h1, d1+d_incr, d1, self.dt/self.dmsteps, v1+v_incr, v1)

            h1 = h1 + h_incr
            d1 = d1 + d_incr
            v1 = v1 + v_incr
        #print np.real(self.dens_mat[0,0]), np.real(self.dens_mat[1,1])
        ### -------------------------------------------- #

        #STEP 3 calculate switching probability and choose if switching is necessary

        self.switching(hamilton_m, coupling_m, v)

        e = self.atoms.get_potential_energy()
        f = self.atoms.get_forces()

        return e, f

    def propagate_dens_mat(self, h_new, h_old, d_new, d_old, dt, v_new, v_old):

        dens_mat_d = np.zeros_like(self.dens_mat)

        ###Simple Euler SHIT

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * self.dens_mat[i,j] * (h_old[i,i]-h_old[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (self.dens_mat[l,j] * np.dot(v_old.flatten(),d_old[i,l,:,:].flatten())- \
                                 self.dens_mat[i,l] * np.dot(v_old.flatten(),d_old[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        #new_dens_mat = self.dens_mat + dens_mat_d * self.dt
        #new_dens_mat = new_dens_mat / np.trace(new_dens_mat)

        ### Runge Kutta RK4 Propagation

        k1 = dens_mat_d * dt

        h_mid = (h_new+h_old) / 2
        d_mid = (d_new+d_old) / 2
        v_mid = (v_new+v_old) / 2

        dm_k1 = self.dens_mat + k1/2

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * dm_k1[i,j] * (h_mid[i,i]-h_mid[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (dm_k1[l,j] * np.dot(v_mid.flatten(),d_mid[i,l,:,:].flatten())- \
                                 dm_k1[i,l] * np.dot(v_mid.flatten(),d_mid[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        k2 = dens_mat_d * dt
        dm_k2 = self.dens_mat + k2/2

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * dm_k2[i,j] * (h_mid[i,i]-h_mid[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (dm_k2[l,j] * np.dot(v_mid.flatten(),d_mid[i,l,:,:].flatten())- \
                                 dm_k2[i,l] * np.dot(v_mid.flatten(),d_mid[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        k3 = dens_mat_d * dt
        dm_k3 = self.dens_mat + k3

        for i in range(self.nstates):
            for j in range(self.nstates):
                tmp = -1.0j/hbar * dm_k3[i,j] * (h_new[i,i]-h_new[j,j])
                for l in range(self.nstates):
                    tmp = tmp - (dm_k3[l,j] * np.dot(v_new.flatten(),d_new[i,l,:,:].flatten())- \
                                 dm_k3[i,l] * np.dot(v_new.flatten(),d_new[l,j,:,:].flatten()) )

                dens_mat_d[i,j] = tmp

        k4 = dens_mat_d * dt

        new_dens_mat = self.dens_mat + k1/6 + k2/3 + k3/3 + k4/6
        new_dens_mat = new_dens_mat / np.trace(new_dens_mat)

        return new_dens_mat

    def switching(self, h, d, v):

        switch_prob=np.zeros([self.nstates,self.nstates])

        for i in range(self.nstates):
            for j in range(self.nstates):
                switch_prob[i,j] = (2.0/hbar)*np.imag(self.dens_mat[i,j].conjugate()*h[i,j]) - \
                                   2.0*np.real(self.dens_mat[i,j].conjugate() *\
                                   np.dot(v.flatten(),d[i,j,:,:].flatten()))

        rand=np.random.random()

        new_state = self.current_state
        do_switch = False
        prob = 0.0
        for i in range(self.nstates):
            if i==self.current_state:
                continue
            tmp = np.real(switch_prob[i,self.current_state] * self.dt / \
                  self.dens_mat[self.current_state,self.current_state])
            if tmp<0.0: tmp=0.0
            if rand>prob and rand<=(prob+tmp):
                do_switch = True
                new_state = i
            prob = prob + tmp

        if do_switch:

            NAC = np.real(d[self.current_state,new_state,:,:])
            #is the energy sufficient to do this switch
            nac_sum_a=0.0
            nac_sum_b=0.0
            for a in range(len(self.atoms)):
              #following Chem. Phys. 349, 334 (2008)
                nac_sum_a = nac_sum_a + np.dot(NAC[a],NAC[a]) / self.atoms[a].mass
                nac_sum_b = nac_sum_b + np.dot(v[a],NAC[a])

            nac_sum_a = np.real(nac_sum_a / 2.0)
            nac_sum_b = np.real(nac_sum_b)

            kin_e = np.real((nac_sum_b * nac_sum_b) + (4.0*nac_sum_a)* \
                    (h[self.current_state,self.current_state]- \
                    h[new_state,new_state]))

            #normalizing NACs
            #for i in range(len(self.atoms)):
            #    NAC[i] = NAC[i] / np.sqrt(np.dot(NAC[i],NAC[i]))

            if kin_e >= 0.0:
                #readjust forces
                if nac_sum_b<0.0:
                    gamma = (nac_sum_b + np.sqrt(kin_e)) / (2.0*nac_sum_a)
                else:
                    gamma = (nac_sum_b - np.sqrt(kin_e)) / (2.0*nac_sum_a)

                #gamma = np.real(h[new_state,new_state] - h[self.current_state,self.current_state])
                #sign = np.sign(gamma)

                #switch is allowed
                self.current_state = new_state
                self.atoms.calc.current_state = new_state
            else:
                #frustrated hopping

                gamma = 0.0
                #gamma = nac_sum_b / nac_sum_a

            #we rescale velocities with gamma

            for i in range(len(self.atoms)):
                #v[i] = v[i] + np.sign(gamma)*np.sqrt(2*gamma /self.atoms[i].mass)* NAC[i]
                v[i] = v[i] - (gamma * NAC[i] /self.atoms[i].mass)
            self.atoms.set_velocities(v)

        else:
            pass

        return

    def step(self, e, f):

        hamilton_m_old, coupling_m_old = self.construct_h_d()
        v_old = self.atoms.get_velocities()
        ### -------------------------------------------- #
        #STEP 1 integrate class. time step

        f = self.integrate(f)

        ### -------------------------------------------- #
        #STEP 2 integrate TD-SGE to get new density matrix ... using Runge-Kutta-method

        hamilton_m, coupling_m = self.construct_h_d()
        v = self.atoms.get_velocities()

        #simple dm integration
        #self.dens_mat = self.propagate_dens_mat(hamilton_m, hamilton_m_old, coupling_m, coupling_m_old, self.dt)

        #finer timestep dm integration

        h_incr = (hamilton_m - hamilton_m_old) / self.dmsteps
        d_incr = (coupling_m - coupling_m_old) / self.dmsteps
        v_incr = (v - v_old) / self.dmsteps

        h1 = hamilton_m_old
        d1 = coupling_m_old
        v1 = v_old
        for i in range(self.dmsteps):

            self.dens_mat = self.propagate_dens_mat(h1+h_incr, h1, d1+d_incr, d1, self.dt/self.dmsteps, v1+v_incr, v1)

            h1 = h1 + h_incr
            d1 = d1 + d_incr
            v1 = v1 + v_incr
        #print np.real(self.dens_mat[0,0]), np.real(self.dens_mat[1,1])
        ### -------------------------------------------- #

        #STEP 3 calculate switching probability and choose if switching is necessary

        self.switching(hamilton_m, coupling_m, v)

        e = self.atoms.get_potential_energy()
        f = self.atoms.get_forces()

        return e, f


class SurfaceHoppingVelocityVerlet(SurfaceHoppingASEVerlet):
    """VelocityVerlet with Tully's surface hopping fewest switches method"""

    def __init__(self, atoms, dt, dmsteps=None, nstates=1, init_densmat=None ,
                 NAC_method=None, trajectory=None, logfile=None,
                 loginterval=1):
        SurfaceHoppingASEVerlet.__init__(self, atoms, dt=dt, dmsteps=dmsteps, nstates=nstates,
                 init_densmat=init_densmat ,NAC_method=NAC_method,
                 trajectory=trajectory, logfile=logfile, loginterval=loginterval)


    def integrate(self, f):

        #Velocity Verlet Algorithm
        p = self.atoms.get_momenta()
        self.atoms.set_positions(self.atoms.get_positions() +
            self.dt * p / self.atoms.get_masses()[:,np.newaxis]+self.dt*self.dt *
            f / (2*self.atoms.get_masses()[:,np.newaxis]) )

        ##once evaluating new stuff
        f_old = f
        f = self.atoms.get_forces()

        self.atoms.set_velocities(self.atoms.get_velocities() +
            self.dt * (f_old + f) / (2 * self.atoms.get_masses()[:,np.newaxis]))

        return f


    def __init__(self, atoms, dt, dmsteps=None, nstates=1, init_densmat=None ,
                 NAC_method=None, trajectory=None, logfile=None,
                 loginterval=1):
                 SurfaceHoppingVelocityVerlet.__init__(self, atoms, dt=dt, dmsteps=dmsteps, nstates=nstates,
                 init_densmat=init_densmat ,NAC_method=NAC_method,
                 trajectory=trajectory, logfile=logfile, loginterval=loginterval)

                 self.U = np.zeros([nstates,nstates,3])
                 self.U[:,:,:] = 0.0

    def step(self, e, f):

        v_old = self.atoms.get_velocities()
        ### -------------------------------------------- #
        #STEP 1 integrate class. time step

        f = self.integrate(f)

        ### -------------------------------------------- #

        e = self.atoms.get_potential_energy(self.atoms, nstate=i)

        for i in range(self.nstates):
            for j in range(self.nstates):
                self.U[i,j,-2] = self.U[i,j,-1]
                self.U[i,j,-1] = self.U[i,j, 0]
                self.U[i,j, 0] = self.atoms.get_potential_energy(self.atoms, nstate=i) - \
                                 self.atoms.get_potential_energy(self.atoms, nstate=j)

        #STEP 2 calculate switching probability and choose if switching is necessary

        self.switching(v)

        f = self.atoms.get_forces()

        return e, f


    def switching(self, v):

        switch_prob=np.zeros([self.nstates,self.nstates])

        for i in range(self.nstates):
            for j in range(self.nstates):
                if i != j:
                    d2U_dt2 = (self.U[i,j,0] - 2.0*self.U[i,j,-1] + self.U[i,j,-2]) / (self.dt**2)
                    switch_prob[i,j] = np.exp((-4.0/(3.*hbar))*self.U[i,j,0]*np.sqrt((2.*self.U[i,j,0])/d2U_dt2))

        print switch_prob

        rand=np.random.random()

        new_state = self.current_state
        do_switch = False
        prob = 0.0
        for i in range(self.nstates):
            if i==self.current_state:
                continue
            tmp = np.real(switch_prob[i,self.current_state])
            if tmp<0.0:   tmp=0.0
            if rand>prob and rand<=(prob+tmp):
                do_switch = True
                new_state = i
            prob = prob + tmp

        if do_switch:

            NAC = np.real(d[self.current_state,new_state,:,:])
            #is the energy sufficient to do this switch
            kin_e = atoms.get_kinetic_energy()

            #normalizing NACs
            #for i in range(len(self.atoms)):
            #    NAC[i] = NAC[i] / np.sqrt(np.dot(NAC[i],NAC[i]))

            if kin_e >= 0.0:
                #readjust forces
                if nac_sum_b<0.0:
                    gamma = (nac_sum_b + np.sqrt(kin_e)) / (2.0*nac_sum_a)
                else:
                    gamma = (nac_sum_b - np.sqrt(kin_e)) / (2.0*nac_sum_a)

                #gamma = np.real(h[new_state,new_state] - h[self.current_state,self.current_state])
                #sign = np.sign(gamma)

                #switch is allowed
                self.current_state = new_state
                self.atoms.calc.current_state = new_state
            else:
                #frustrated hopping

                gamma = 0.0
                #gamma = nac_sum_b / nac_sum_a

            #we rescale velocities with gamma

            for i in range(len(self.atoms)):
                #v[i] = v[i] + np.sign(gamma)*np.sqrt(2*gamma /self.atoms[i].mass)* NAC[i]
                v[i] = v[i] - (gamma * NAC[i] /self.atoms[i].mass)
            self.atoms.set_velocities(v)

        else:
            pass

        return
