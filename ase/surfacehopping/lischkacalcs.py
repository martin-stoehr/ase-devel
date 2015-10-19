"""Class for demonstrating the ASE-calculator interface."""
import numpy as np
from ase.calculators.calculator import Calculator
from ase.surfacehopping.tullycalcs import TullyCalc1
import ase.units as units

class LischkaCalc1(TullyCalc1):
    """ASE calculator.

    This calculator takes an atoms object with two atoms, where
    only the x-axis is taken into account...1-dimensional

    It calculates the ground state, the excited state and the coupling


    The following potentials apply: double well minimum, JCP, 137, 22A514(2012)

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

    def __init__(self,A=1.00, B=1.00, C=0.05, **kwargs):
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
        e = self.A*self.A*self.B*self.B*x*x+self.C*self.C
        e1 = self.A*x*x+self.A*self.B*self.B/4.0 - np.sqrt(e)
        e2 = self.A*x*x+self.A*self.B*self.B/4.0 + np.sqrt(e)
        d = -0.5 * (self.C/e)
        F = 2*self.A*self.A*self.B*self.B*x
        F1 = -2*self.A*x - F / np.sqrt(e)
        F2 = -2*self.A*x + F / np.sqrt(e)
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
        x=np.linspace(-5,5,200)
        e = self.A*self.A*self.B*self.B*x*x+self.C*self.C
        e1 = self.A*x*x+self.A*self.B*self.B/4.0 - np.sqrt(e)
        e2 = self.A*x*x+self.A*self.B*self.B/4.0 + np.sqrt(e)
        d = -0.5 * (self.C/e)
        V11 = self.A*(x-self.B*0.5)*(x-self.B*0.5)
        V22 = self.A*(x+self.B*0.5)*(x+self.B*0.5)
        V12 = self.C * np.ones_like(x)
        plt.plot(x,V11, x, V22, x, V12)
        plt.show()
        plt.plot(x,e1,x,e2,x,-d)
        plt.show()
        #plt.plot(x*units.Bohr,e1* units.Ha,x*units.Bohr,e2* units.Ha,x*units.Bohr,(d/units.Bohr)/10)
        #plt.show()

class DoubleWellPotential(TullyCalc1):
    """ASE calculator.

    This calculator takes an atoms object with two atoms, where
    only the x-axis is taken into account...1-dimensional

    It calculates the ground state, the excited state and the coupling


    The following potentials apply: double well minimum, JCP, 137, 22A514(2012)

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

    def __init__(self,A=1.00, B=1.00, C=0.05,D=0.0, **kwargs):
        TullyCalc1.__init__(self, **kwargs)
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

    def calculate(self, atoms):
        positions = atoms.get_positions()
        self.energy = np.zeros([2])
        #num. atoms, coordinates (only x important), number of electr. elevels
        self._forces = np.zeros((len(atoms), 3, 2))
        self._couplings = np.zeros((len(atoms), 3, 2))
        x = positions[0,0] #/ units.Bohr
        e = self.A*self.A* (2.0*self.B*x+self.D/2.0)**2+ self.C* self.C
        ee = self.A *(x*x+self.B*self.B+self.D/2.0)
        e1 = ee - np.sqrt(e)
        e2 = ee + np.sqrt(e)

        d = -(4.*self.A*self.B*self.C)/(4.*self.C**2.+(4*self.A*self.B*x+self.A*self.D)**2. )

        F = 0.5*self.A*self.A*(8*self.B*self.B*x + 2*self.B*self.D) / (np.sqrt(e))
        F1 = -(2*self.A*x - F)
        F2 = -(2*self.A*x + F)

        #the coupling of state 1 with state 2
        self._couplings[0,0,0] = -d #/ units.Bohr
        #the coupling of state 2 with state 1
        self._couplings[0,0,1] =  d #/ units.Bohr
        self.energy[0] = e1 #* units.Ha
        self.energy[1] = e2 #* units.Ha
        self._forces[0,0,0] = F1 #* units.Ha / units.Bohr
        self._forces[0,0,1] = F2 #* units.Ha / units.Bohr
        self.positions = positions.copy()

    def show(self):
        #plots the potential and diabatic and adiabatic curves
        import matplotlib.pyplot as plt
        x=np.linspace(-(6+self.B),(6+self.B),400)
        V11 = self.A*(x- self.B)*(x- self.B)
        V22 =  self.A*((x+ self.B)*(x+ self.B) + self.D)
        V12 =  self.C * np.ones_like(x)
        e = self.A*self.A* (2.0*self.B*x+self.D/2.0)**2+ self.C* self.C
        ee = self.A *(x*x+self.B*self.B+self.D/2.0)
        e1 = ee - np.sqrt(e)
        e2 = ee + np.sqrt(e)
        d = - (2*self.A*self.B*self.C)/(2*self.C*self.C+(self.A*self.B*x+self.A*self.D/4)**2 )
        plt.plot(x,V11, x, V22, x, V12)
        plt.show()
        plt.plot(x,e1,x,e2,x,-d)
        plt.show()
        #plt.plot(x*units.Bohr,e1* units.Ha,x*units.Bohr,e2* units.Ha,x*units.Bohr,(d/units.Bohr)/10)
        #plt.show()
