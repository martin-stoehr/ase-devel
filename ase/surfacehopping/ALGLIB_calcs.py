"""Class for demonstrating the ASE-calculator interface."""
import numpy as np
from ase.calculators.calculator import Calculator
from ase.surfacehopping.tullycalcs import TullyCalc1
import ase.units as units
from scipy.misc import derivative
import xalglib as xa
from math import factorial

class ALGLIBCalculator(Calculator):
    """ASE ALGLIB base calculator.

    This calculator takes an atoms object and a file of
    '1, 2 or 3-dimensional data (format x, y, z, f) and does
    a radial basis function fit using ALGLIB www.alglib.net
    """

    def __init__(self, datafile, dimension, rbase, nlayers, lambdav, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.data = None
        self.model = None
        self.potentials = None
        self.current_state = None

        if isinstance(datafiles,str):
            self.data=[datafiles]
            self.nstates = 1
        else:
            #list of filenames
            self.nstates = len(datafiles)
            self.data = []
            for i in range(self.nstates):
                self.data.append(datafiles[i])


    def _generate_rbf_potential(self, nstate, dimension, rbase, nlayers, lambdav):

        #this function reads in the filename containing
        #the 1 to 3-d data and generates a multilayer-rbf model

        self.data = np.loadtxt(filename).tolist()


        #initialize model
        self.model = xa.rbfcreate(dimension,1)
        xa.rbfsetpoints(self.model, self.data)
        xa.rbfsetalgomultilayer(self.model, rbase, nlayers, lambdav)
        self.potential = xa.rbfbuildmodel(self.model)


class Azobenzene_Gasphase_Calc(ALGLIBCalculator):

    def __init__(self, indices ,datafiles, rbase, nlayers, lambdav, C_min=0.0001, C_max=0.55, C_chi=0.65, **kwargs):


        ALGLIBCalculator.__init__(self,datafiles, rbase, nlayers, lambdav, **kwargs)

        #construct energy potentials

        self.model = []

        zz = np.loadtxt(datafiles[0])

        x = zz[:,0]
        y = zz[:,1]
        z = zz[:,2]
        s0 = zz[:,3]
        s1 = zz[:,4]
        s2 = zz[:,5]

        S0_min = s0.min()

        x_max = x.max()
        x_min = x.min()
        y_max = y.max()
        y_min = y.min()
        z_max = z.max()
        z_min = z.min()

        data_s0 = np.column_stack([x,y,z,s0-S0_min]).tolist()
        data_s1 = np.column_stack([x,y,z,s1-S0_min]).tolist()
        data_s2 = np.column_stack([x,y,z,s2-S0_min]).tolist()
        #        RBF FITTING        #
        #Dimensionality of the space and of the function (scalar)
        model_s0 = xa.rbfcreate(3,1)
        model_s1 = xa.rbfcreate(3,1)
        model_s2 = xa.rbfcreate(3,1)
        #Set data
        xa.rbfsetpoints(model_s0, data_s0)
        xa.rbfsetpoints(model_s1, data_s1)
        xa.rbfsetpoints(model_s2, data_s2)
        #QNN set method and build model - q = 1.0, z = 5.0
        #xa.rbfsetalgoqnn(model, 1.8, 5.0)

        #Multilayer algorithm
        #rbfsetalgomultilayer(model, rbase, nlayers, lambdav)
        # lambda best from 0.001 to 0.1, 0.01 is default
        xa.rbfsetalgomultilayer(model_s0, rbase, nlayers)#, lambdav)
        xa.rbfsetalgomultilayer(model_s1, rbase, nlayers)#, lambdav)
        xa.rbfsetalgomultilayer(model_s2, rbase, nlayers)#, lambdav)
        rep_s0 = xa.rbfbuildmodel(model_s0)
        rep_s1 = xa.rbfbuildmodel(model_s1)
        rep_s2 = xa.rbfbuildmodel(model_s2)

        self.model = [model_s0, model_s1, model_s2]

        self.energy = None
        self._forces = None
        self._couplings = None
        self.positions = None
        self.current_state = None

        if indices is None:
            self.indices1 = [0,1,2,3]
        else:
            self.indices1 = indices
        self.indices2 = [indices[2],indices[1],indices[0]]
        self.indices3 = [indices[1],indices[2],indices[3]]

        #DIABATIC COUPLING parameters
        self.C_min = C_min
        self.C_max = C_max
        self.C_chi = C_chi


    def calculate(self, atoms):

        incr = 1.0

        #dihedral
        x = self.__get_dihedral(atoms.positions, self.indices1)
        #angle a1
        y = self.__get_angle(atoms.positions, self.indices2)
        #angle a2
        z = self.__get_angle(atoms.positions, self.indices3)


        #calc energy
        self.energy = np.zeros([self.nstates])

        for state in range(self.nstates):
            self.energy[state] = self.calculate_energy(x,y,z,nstate=state)
        #calc forces
        self._forces = np.zeros((len(atoms), 3, self.nstates))

        int_forces = np.zeros([3, self.nstates])

        #periodic behaviour of the angles has to be accounted for
        for s in range(self.nstates):
            c = np.zeros([3])
            c[0] = x
            c[1] = y
            c[2] = z
            for i in range(3):

                c[i] = c[i] + incr
                e1 = self.calculate_energy(c[0]%360.0,c[1]%360.0,c[2]%360.0,s)
                c[i] = c[i] - 2.0*incr
                e2 = self.calculate_energy(c[0]%360.0,c[1]%360.0,c[2]%360.0,s)
                int_forces[i,s] = -(e1-e2)/(2.0*incr)
                c[i] = c[i] + incr

        #change from deg^-1 to rad^-1
        int_forces = int_forces / (np.pi/180.)
        #print int_forces[0], int_forces[1]

        #calculate coupling elements
        self._couplings = np.zeros((self.nstates, self.nstates, len(atoms), 3))

        int_couplings = np.zeros([self.nstates, self.nstates,3 ])

        for s in range(self.nstates):
            for ss in range(s,self.nstates):
                if ss==s:
                    pass
                else:
                    e2_e1 = self.energy[ss] - self.energy[s]

                    C = self.C_max*(1-((2)/(np.exp(np.abs(e2_e1)/self.C_chi)+1)))+self.C_min
                    dC = 2*(self.C_max/self.C_chi) * \
                         ((np.exp(np.abs(e2_e1)/self.C_chi))/((1+np.exp(np.abs(e2_e1)/self.C_chi))**2))

                    tmp = e2_e1**2 - 4.0*C*C
                    if tmp<=0.00000:
                         tmp = 0.001
                    sqr = np.sqrt(tmp)
                    coupl_corr = -(dC * sqr)/(e2_e1**2)
                    for di in range(3):
                        f1_f2 = np.abs(int_forces[di,s] - int_forces[di,ss])
                        int_couplings[s,ss,di] = (C*f1_f2)/(sqr*e2_e1) #+ coupl_corr

        #generate transformation matrix
        B_matrix = self._generate_B_matrix(atoms)

        Bt = B_matrix.transpose()
        Bt_inv = np.linalg.pinv(Bt)

        for s in range(self.nstates):
            forces = np.dot(Bt,int_forces[:,s]).reshape([4,3])
            for i in range(len(atoms)):
                for j in range(4):
                    if i==self.indices1[j]:
                        self._forces[i,:,s] = forces[j]

            for ss in range(s,self.nstates):
                couplings = np.dot(Bt,int_couplings[s,ss,:]).reshape([4,3])

                for i in range(len(atoms)):
                    for j in range(4):
                        if i==self.indices1[j]:
                            self._couplings[s,ss,i,:] = couplings[j]

                self._couplings[ss,s,:,:] = -self._couplings[s,ss,:,:]

        return

    def calculate_energy(self,x,y,z,nstate=None):

        if nstate is None:
            energy = xa.rbfcalc3(self.model[0], x, y, z)
        else:
            energy = xa.rbfcalc3(self.model[nstate], x, y, z)

        return energy

    def __get_dihedral(self, pos, list):
        """Calculate dihedral angle.

        Calculate dihedral angle between the vectors list[0]->list[1]
        and list[2]->list[3], where list contains the atomic indexes
        in question.
        """

        # vector 0->1, 1->2, 2->3 and their normalized cross products:
        a = pos[list[1]] - pos[list[0]]
        b = pos[list[2]] - pos[list[1]]
        c = pos[list[3]] - pos[list[2]]
        bxa = np.cross(b, a)
        bxa /= np.linalg.norm(bxa)
        cxb = np.cross(c, b)
        cxb /= np.linalg.norm(cxb)
        angle = np.vdot(bxa, cxb)
        # check for numerical trouble due to finite precision:
        if angle < -1:
            angle = -1
        if angle > 1:
            angle = 1
        angle = np.arccos(angle)
        if np.vdot(bxa, c) > 0:
            angle = 2 * np.pi - angle
        return (angle/np.pi)*180.0

    def __get_angle(self, pos, list):
        """Get angle formed by three atoms.

        calculate angle between the vectors list[1]->list[0] and
        list[1]->list[2], where list contains the atomic indexes in
        question."""

        # normalized vector 1->0, 1->2:
        v10 = pos[list[0]] - pos[list[1]]
        v12 = pos[list[2]] - pos[list[1]]
        v10 /= np.linalg.norm(v10)
        v12 /= np.linalg.norm(v12)
        angle = np.vdot(v10, v12)
        angle = np.arccos(angle)
        return (angle/np.pi)*180.0

    def _generate_B_matrix(self, atoms):

        B_matrix = np.zeros([3,3*4])
        pos = atoms.positions

        #dihedral
        r12 = pos[self.indices1[1]] - pos[self.indices1[0]]
        r23 = pos[self.indices1[2]] - pos[self.indices1[1]]
        r34 = pos[self.indices1[3]] - pos[self.indices1[2]]
        len_r12 = np.linalg.norm(r12)
        len_r23 = np.linalg.norm(r23)
        len_r34 = np.linalg.norm(r34)
        e12 = r12 / len_r12
        e23 = r23 / len_r23
        e34 = r34 / len_r34
        phi2 = atoms.get_angle([self.indices1[0], self.indices1[1], self.indices1[2]])
        phi3 = atoms.get_angle([self.indices1[1], self.indices1[2], self.indices1[3]])

        tmp = - np.cross(e12,e23) / (len_r12*np.sin(phi2)**2)

        B_matrix[0,0:3] = tmp

        tmp = ((len_r23-len_r12*np.cos(phi2))/(len_r23*len_r12*np.sin(phi2))) * (np.cross(e12,e23)/np.sin(phi2)) +\
              (np.cos(phi3)/(len_r23*np.sin(phi3))) * (np.cross(-e34,-e23)/np.sin(phi3))

        B_matrix[0,3:6] = tmp

        tmp = ((len_r23-len_r34*np.cos(phi3))/(len_r23*len_r34*np.sin(phi3))) * (np.cross(-e34,-e23)/np.sin(phi3)) +\
              (np.cos(phi2)/(len_r23*np.sin(phi2))) * (np.cross(e12,e23)/np.sin(phi2))

        B_matrix[0,6:9] = tmp

        tmp = - np.cross(-e34,-e23) / (len_r34*np.sin(phi3)**2)

        B_matrix[0,9:] = tmp

        #angle a1
        r31 = pos[self.indices2[0]] - pos[self.indices2[1]]
        r32 = pos[self.indices2[2]] - pos[self.indices2[1]]
        len_r31 = np.linalg.norm(r31)
        len_r32 = np.linalg.norm(r32)
        e31 = r31 / len_r31
        e32 = r32 / len_r32
        alpha = atoms.get_angle(self.indices2)

        tmp = (np.cos(alpha) * e31 - e32 ) / (len_r31 * np.sin(alpha))

        B_matrix[1,6:9] = tmp

        tmp = ( (len_r31 - len_r32 * np.cos(alpha)) * e31 + (len_r32 - len_r31 * np.cos(alpha)) * e32 ) \
              / (len_r31 * len_r32 * np.sin(alpha))

        B_matrix[1,3:6] = tmp

        tmp = (np.cos(alpha) * e32 - e31 ) / (len_r32 * np.sin(alpha))

        B_matrix[1,0:3] = tmp

        #angle a2
        r31 = pos[self.indices3[0]] - pos[self.indices3[1]]
        r32 = pos[self.indices3[2]] - pos[self.indices3[1]]
        len_r31 = np.linalg.norm(r31)
        len_r32 = np.linalg.norm(r32)
        e31 = r31 / len_r31
        e32 = r32 / len_r32
        alpha2 = atoms.get_angle(self.indices3)

        tmp = (np.cos(alpha2) * e31 - e32 ) / (len_r31 * np.sin(alpha2))

        B_matrix[2,3:6] = tmp

        tmp = ( (len_r31 - len_r32 * np.cos(alpha2)) * e31 + (len_r32 - len_r31 * np.cos(alpha2)) * e32 ) \
              / (len_r31 * len_r32 * np.sin(alpha2))

        B_matrix[2,6:9] = tmp

        tmp = (np.cos(alpha2) * e32 - e31 ) / (len_r32 * np.sin(alpha2))

        B_matrix[2,9:] = tmp


        return B_matrix

    def show(self, nstate=0):

        import matplotlib.pyplot as plt
        from matplotlib.mlab import griddata
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import cm

        n_x = 37*4
        n_y = 48*4

        x_max = -100000
        y_max = -100000
        z_max = -100000
        x_min =  100000
        y_min =  100000
        z_min =  100000

        data = np.loadtxt(self.data[nstate]).tolist()

        for i in data:
            if i[0]>x_max:
                x_max = i[0]
            if i[0]<x_min:
                x_min = i[0]
            if i[1]>y_max:
                y_max = i[1]
            if i[1]<y_min:
                y_min = i[1]
       #     if i[2]<z_min:
       #         z_min = i[2]

        xi = np.linspace(x_max-00, x_min+00, n_x)
        yi = np.linspace(y_min+0.0, y_max-0.0,n_y)

        #xi2 = np.cos((xi/180.)*np.pi)
        #yi2 = np.cos((yi/180.)*np.pi)

        X, Y = np.meshgrid(xi, yi)
        tmp = np.array(data)
        Z = griddata(tmp[:,0], tmp[:,1],tmp[:,2]-z_min, xi, yi)

        f = []
        i = 0

        X,Y = np.meshgrid(xi ,yi)

        for xii,yii in zip(X.flatten(), Y.flatten()):
           val = xa.rbfcalc2(self.model[nstate], xii, yii)
           f.append( (val) )
           #print X.flatten()[i], Y.flatten()[i], Z.flatten()[i]/un.Ha+z_min, val
           i = i+1

        Z2 = griddata(X.flatten(), Y.flatten(), f, xi, yi)
        Z3 = griddata(X.flatten(), Y.flatten(), Z.flatten()-f, xi, yi)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        #X,Y = np.meshgrid(xi2 ,yi2)

        ax.plot_surface(X, Y, Z2, rstride=2, cstride=2, cmap=cm.jet,
              linewidth=1, alpha=0.3)
        #ax.set_xlim([x_min, x_max])
        #ax.set_ylim([y_min, y_max])
        plt.show()
        fig2 = plt.figure()
        ax = fig2.add_subplot(111,projection='3d')
        ax.plot_surface(X,Y,Z3,rstride=2,cstride=2, cmap=cm.jet,
              linewidth=1, alpha=0.3)

        plt.show()

    def show_all(self):

        import matplotlib.pyplot as plt
        from matplotlib.mlab import griddata
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import cm

        n_x = 37*4
        n_y = 48*4

        x_max = -100000
        y_max = -100000
        z_max = -100000
        x_min =  100000
        y_min =  100000
        z_min =  100000

        data = []
        for state in range(self.nstates):
            data.append(np.loadtxt(self.data[state]).tolist())

        for i in data[0]:
            if i[0]>x_max:
                x_max = i[0]
            if i[0]<x_min:
                x_min = i[0]
            if i[1]>y_max:
                y_max = i[1]
            if i[1]<y_min:
                y_min = i[1]
       #     if i[2]<z_min:
       #         z_min = i[2]

        xi = np.linspace(x_max-00, x_min+00, n_x)
        yi = np.linspace(y_min+0.0, y_max-0.0,n_y)

        #xi2 = np.cos((xi/180.)*np.pi)
        #yi2 = np.cos((yi/180.)*np.pi)

        X, Y = np.meshgrid(xi, yi)


        f = np.zeros([len(X.flatten()),self.nstates])
        Z = []

        for state in range(self.nstates):
            i = 0
            for xii,yii in zip(X.flatten(), Y.flatten()):
              val = xa.rbfcalc2(self.model[state], xii, yii)
              f[i,state] = (val)
              #print X.flatten()[i], Y.flatten()[i], Z.flatten()[i]/un.Ha+z_min, val
              i = i+1
            Z.append(griddata(X.flatten(), Y.flatten(), f[:,state], xi, yi))

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        #X,Y = np.meshgrid(xi2 ,yi2)

        for state in range(self.nstates):
            ax.plot_surface(X, Y, Z[state], rstride=2, cstride=2, cmap=cm.jet,vmin=0.0,vmax=3.0,
                  linewidth=1, alpha=0.3)
        #ax.set_xlim([x_min, x_max])
        #ax.set_ylim([y_min, y_max])
        plt.show()

        coupl = np.loadtxt('NONADIABATIC_COUPLINGS_S0_S1')


        fig = plt.figure()
        ax = fig.add_subplot(111)

        F1 = griddata(coupl[:,0], coupl[:,1], coupl[:,2], xi, yi)
        F2 = griddata(coupl[:,0], coupl[:,1], coupl[:,3], xi, yi)
        F3 = griddata(coupl[:,0], coupl[:,1], np.sqrt(coupl[:,2]**2+coupl[:,3]**2), xi, yi)

        q = ax.quiver(X, Y, F1, F2, pivot='middle', headwidth=4, headlength=6)

        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        surf = ax.plot_surface(X, Y, F3, rstride=2, cstride=2, cmap=cm.jet,
            linewidth=1, alpha=0.3)
        plt.show()
