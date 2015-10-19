"""Class for demonstrating the ASE-calculator interface."""
import numpy as np
from ase.calculators.calculator import Calculator
import ase.units as units

class TullyCalc1(Calculator):
    """ASE calculator.

    This calculator takes an atoms object with two atoms, where
    only the x-axis is taken into account...1-dimensional

    It calculates the ground state, the excited state and the coupling


    The following potentials apply:    avoided crossing, JCP 93, 1061

    default: A = 0.01, B=1.6, C=0.05, D=1.0

    diabatic:
    V_11(x) = A * (1-exp(-Bx)) if x>0
    V_11(x) = A * (1-exp(Bx)) if x<0
    V_22(x) = -V_11(x)
    V_12(x) = C * exp(-Dx^2)

    adiabatic:
    e_1(x) = -sqrt(V_11^2 + V_12^2)
    e_2(x) = +sqrt(V_11^2 + V_12^2)
    d_12(x) = 0.5 * (V_12*V_11'-V_12'*V_11) / (V_11^2 + V_12^2)
    d_21(x) = -d_12(x) """

    #implemented_properties = ['energy','forces','excited_energy','adiabatic_coupling']
    #default_parameters = {'A': 0.01,
                          #'B': 1.60,
                          #'C': 0.05,
                          #'D': 1.00}

    def __init__(self,A=0.01, B=1.60, C=0.005, D=1.00, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.energy = None
        self._forces = None
        self._couplings = None
        self.positions = None
        self.coupling_dict = {'12':0, '21':1}
        self.current_state = None

    def update(self, atoms):
        assert not atoms.get_pbc().any()
        if (self.positions is None or
            (self.positions != atoms.get_positions()).any()):
            self.calculate(atoms)

    def get_potential_energy(self, atoms=None, force_consistent=False, nstate=None):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        self.update(atoms)
        if nstate is None:
            nstate=0
            if self.current_state is None:
                return self.energy[nstate]
            else:
                return self.energy[self.current_state]
        else:
            return self.energy[nstate]

    def get_forces(self, atoms, nstate=None):
        self.update(atoms)
        if nstate is None:
            nstate=0
            if self.current_state is None:
                return self._forces[:,:,nstate]
            else:
                return self._forces[:,:,self.current_state]
        else:
            return self._forces[:,:,nstate]

    def get_couplings(self, atoms):
        self.update(atoms)
        return self._couplings

    def calculate(self, atoms):
        positions = atoms.get_positions()
        self.energy = np.zeros([2])
        #num. atoms, coordinates (only x important), number of electr. elevels
        self._forces = np.zeros((len(atoms), 3, 2))
        self._couplings = np.zeros((len(atoms), 3, 2))
        x = positions[0,0] / units.Bohr
        if x > 0:
            V_11  = self.A * ( 1.0 - np.exp(-self.B*x))
            V_11d = self.A * self.B * np.exp(-self.B*x)
        else:
            V_11 = -self.A * ( 1.0 - np.exp( self.B*x))
            V_11d = self.A * self.B * np.exp(self.B*x)
        V_12  = self.C * np.exp(-self.D*x*x)
        V_12d = -2 * self.C * self.D * x * np.exp(-self.D*x*x)
        e = V_11*V_11 + V_12*V_12
        e1 = -np.sqrt(e)
        e2 = -e1
        d = 0.5 * (V_12*V_11d-V_12d*V_11) / (V_11*V_11 + V_12*V_12)
        F = (V_11*V_11d - V_12*V_12d) / e2

        #the coupling of state 1 with state 2
        self._couplings[0,0,0] = -d / units.Bohr
        #the coupling of state 2 with state 1
        self._couplings[0,0,1] =  d / units.Bohr
        self.energy[0] = e1 * units.Ha
        self.energy[1] = e2 * units.Ha
        self._forces[0,0,0] = F * units.Ha / units.Bohr
        self._forces[0,0,1] = -F * units.Ha / units.Bohr
        self.positions = positions.copy()

    def show(self):
        #plots the potential and diabatic and adiabatic curves
        import matplotlib.pyplot as plt
        x1=np.linspace(0,10,100)
        x2=np.linspace(-10,0,100)
        x=np.concatenate([x2,x1])
        V_11_1  = self.A * ( 1.0 - np.exp(-self.B*x1))
        V_11d_1 = self.A * self.B * np.exp(-self.B*x1)
        V_11_2  = -self.A * ( 1.0 - np.exp(self.B*x2))
        V_11d_2 = self.A * self.B * np.exp(self.B*x2)
        V_11 = np.concatenate([V_11_2,V_11_1])
        V_11d = np.concatenate([V_11d_2,V_11d_1])
        V_12  = self.C * np.exp(-self.D*x*x)
        V_12d = -2 * self.C * self.D * x * np.exp(-self.D*x*x)
        e = V_11*V_11 + V_12*V_12
        e1 = -np.sqrt(e)
        e2 = -e1
        d = 0.5 * (V_12*V_11d-V_12d*V_11) / (V_11*V_11 + V_12*V_12)
        #plt.plot(x,e1,x,e2,x,d/50)
        #plt.show()
        plt.plot(x*units.Bohr,e1* units.Ha,x*units.Bohr,e2* units.Ha,x*units.Bohr,(d/units.Bohr)/10)
        plt.show()

class TullyCalc2(TullyCalc1):
    """ASE calculator.

    This calculator takes an atoms object with two atoms, where
    only the x-axis is taken into account...1-dimensional

    It calculates the ground state, the excited state and the coupling


    The following potentials apply:    avoided crossing, JCP 93, 1061

    default: A = 0.10, B=0.28, C=0.015, D=0.06, E0=0.05

    diabatic:
    V_11(x) =0
    V_22(x) = -A * exp(-Bx^2)+E0
    V_12(x) = C * exp(-Dx^2)

    adiabatic:
    e_1(x) = V22/2 - sqrt(v22^2/4 +V12^2)
    e_2(x) = V22/2 + sqrt(v22^2/4 +V12^2)
    d_12(x) = -1 * (V_12'*V_22-V_12*V_22') / (V_22^2 + 4V_12^2)
    d_21(x) = -d_12(x) """

    #implemented_properties = ['energy','forces','excited_energy','adiabatic_coupling']
    #default_parameters = {'A': 0.10,
                          #'B': 0.28,
                          #'C': 0.015,
                          #'D': 0.06,
                          #'E0': 0.05}

    def __init__(self,A=0.10, B=0.28, C=0.015, D=0.06, E0=0.05, **kwargs):
        TullyCalc1.__init__(self, **kwargs)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E0 = E0
        self.energy = None
        self._forces = None
        self._couplings = None
        self.positions = None
        self.coupling_dict = {'12':0, '21':1}
        self.current_state = None

    def calculate(self, atoms):
        positions = atoms.get_positions()
        self.energy = np.zeros([2])
        #num. atoms, coordinates (only x important), number of electr. elevels
        self._forces = np.zeros((len(atoms), 3, 2))
        self._couplings = np.zeros((len(atoms), 3, 2))
        x = positions[0,0] / units.Bohr
        V_22  = -self.A * np.exp(-self.B*x*x) + self.E0
        V_22d = 2.0 * self.A * self.B * x * np.exp(-self.B*x*x)
        V_12  = self.C * np.exp(-self.D*x*x)
        V_12d = -2.0 * self.C * self.D * x * np.exp(-self.D*x*x)
        e = np.sqrt(V_22*V_22*0.25 + V_12*V_12)
        e1 = V_22*0.5 - e
        e2 = V_22*0.5 + e
        d = 1.0 * (V_12d*V_22-V_12*V_22d) / (V_22*V_22 + 4.0*V_12*V_12)
        F1 = -0.5 * (V_22d-((0.5 * V_22*V_22d + 2.0*V_12*V_12d)/e ))
        F2 = -0.5 * (V_22d+((0.5 * V_22*V_22d + 2.0*V_12*V_12d)/e ))

        #the coupling of state 1 with state 2
        self._couplings[0,0,0] = -d / units.Bohr
        #the coupling of state 2 with state 1
        self._couplings[0,0,1] =  d / units.Bohr
        self.energy[0] = e1 * units.Ha
        self.energy[1] = e2 * units.Ha
        self._forces[0,0,0] = F1 * units.Ha / units.Bohr
        self._forces[0,0,1] = F2 * units.Ha / units.Bohr
        self.positions = positions.copy()

    def show(self):
        #plots the potential and diabatic and adiabatic curves
        import matplotlib.pyplot as plt
        x=np.linspace(-6,6,100)
        V_11 = 0.0 * x
        V_22  = -self.A * np.exp(-self.B*x*x) + self.E0
        V_22d = 2.0 * self.A * self.B * x * np.exp(-self.B*x*x)
        V_12  = self.C * np.exp(-self.D*x*x)
        V_12d = -2.0 * self.C * self.D * x * np.exp(-self.D*x*x)
        e = np.sqrt(V_22*V_22*0.25 + V_12*V_12)
        e1 = V_22*0.5 - e
        e2 = V_22*0.5 + e
        d = 1.0 * (V_12d*V_22-V_12*V_22d) / (V_22*V_22 + 4.0*V_12*V_12)
        plt.plot(x,V_11,x,V_22,x,V_12)
        plt.show()
        plt.plot(x,e1,x,e2,x,d/12.0)
        plt.show()
        plt.plot(x*units.Bohr,e1* units.Ha,x*units.Bohr,e2* units.Ha,x*units.Bohr,(d/units.Bohr))
        plt.show()

class TullyCalc3(TullyCalc1):
    """ASE calculator.

    This calculator takes an atoms object with two atoms, where
    only the x-axis is taken into account...1-dimensional

    It calculates the ground state, the excited state and the coupling


    The following potentials apply:    avoided crossing, JCP 93, 1061

    default: A = 0.10, B=0.28, C=0.015, D=0.06, E0=0.05

    diabatic:
    V_11(x) = A
    V_22(x) = -A
    V_12(x) = B * exp(C*x) x<0 or V_12(x)= B*[2-exp(-Cx)] for x>0

    adiabatic:
    e_1(x) = -sqrt(A^2 + V_12^2)
    e_2(x) = -e1
    d_12(x) = -0.5 * (-V12')/(A+V_12^2/A)
    d_21(x) = -d_12(x) """

    #implemented_properties = ['energy','forces','excited_energy','adiabatic_coupling']
    #default_parameters = {'A': 0.10,
                          #'B': 0.28,
                          #'C': 0.015,
                          #'D': 0.06,
                          #'E0': 0.05}

    def __init__(self,A=6.0E-4, B=0.10, C=0.90, **kwargs):
        TullyCalc1.__init__(self, **kwargs)
        self.A = A
        self.B = B
        self.C = C
        self.energy = None
        self._forces = None
        self._couplings = None
        self.positions = None
        self.coupling_dict = {'12':0, '21':1}
        self.current_state = None

    def calculate(self, atoms):
        positions = atoms.get_positions()
        self.energy = np.zeros([2])
        #num. atoms, coordinates (only x important), number of electr. elevels
        self._forces = np.zeros((len(atoms), 3, 2))
        self._couplings = np.zeros((len(atoms), 3, 2))
        x = positions[0,0] / units.Bohr
        if x>0:
            V_12  = self.B*(2.0 - self.B * np.exp(-self.C*x))
            V_12d = self.B * self.C * np.exp(-self.C*x)
        else:
            V_12  = self.B * np.exp(self.C*x)
            V_12d = self.B * self.C * np.exp(self.C*x)
        e = np.sqrt(self.A*self.A + V_12 * V_12)
        e1 = - e
        e2 =   e
        d = 0.5 * ((-V_12d)/(self.A+V_12*V_12/self.A))
        F = (V_12 * V_12d) / e
        #the coupling of state 1 with state 2
        self._couplings[0,0,0] = -d / units.Bohr
        #the coupling of state 2 with state 1
        self._couplings[0,0,1] =  d / units.Bohr
        self.energy[0] = e1 * units.Ha
        self.energy[1] = e2 * units.Ha
        self._forces[0,0,0] = F * units.Ha / units.Bohr
        self._forces[0,0,1] = -F * units.Ha / units.Bohr
        self.positions = positions.copy()

    def show(self):
        #plots the potential and diabatic and adiabatic curves
        import matplotlib.pyplot as plt
        x1=np.linspace(-10,0,100)
        x2=np.linspace(0,10,100)
        x=np.concatenate([x1,x2])
        V_11 = np.ones_like(x)
        V_22 = np.ones_like(x)
        V_11 = V_11 * self.A
        V_22 = -V_22 * self.A
        V_12_2  = self.B*(2.0 - np.exp(-self.C*x2))
        V_12d_2 = self.B * self.C * np.exp(-self.C*x2)
        V_12_1  = self.B * np.exp(self.C*x1)
        V_12d_1 = self.B * self.C * np.exp(self.C*x1)
        V_12=np.concatenate([V_12_1,V_12_2])
        V_12d=np.concatenate([V_12d_1,V_12d_2])
        e = np.sqrt(self.A*self.A + V_12 * V_12)
        e1 = - e
        e2 =   e
        d = 0.5 * ((-V_12d)/(self.A+V_12*V_12/self.A))
        plt.plot(x,V_11,x,V_22,x,V_12)
        plt.show()
        plt.plot(x,e1,x,e2,x,-d)
        plt.show()
        plt.plot(x*units.Bohr,e1* units.Ha,x*units.Bohr,e2* units.Ha,x*units.Bohr,(-d*5/units.Bohr))
        plt.show()
