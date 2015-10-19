#!/usr/bin/env python
#
# anharmonicity (CASTEP anharmonic Morse)
#
import sys
from numpy import pi, sqrt, exp, array, zeros_like, linspace, newaxis, log, cos, sin
from numpy import column_stack, savetxt
from scipy.optimize import leastsq
from scipy.integrate import quad
from numpy  import sign
import matplotlib.pyplot as plt

import ase.units
from ase import Atoms
from ase.calculators.morse import MorsePotential2
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixAtoms
from ase.io import read


# for mass weighted amplitude in A*sqrt(u) and freq in eV returns energy in harmonic potential in eV
def V_harmonic(A, freq):
    hbar = ase.units._hbar * ase.units.J     # hbar in eVs
    units_factor = 1.0E-20 * ase.units._amu / ase.units._e / hbar**2
    return units_factor * ( 0.5 * A**2 * freq**2 )

def V_Morse(R, D=6.0, a=1.0, R0=1.0):
    return ( D * ( 1.0 - exp(-a * (R-R0) ) )**2 )

try:
#    import matplotlib
#    matplotlib.use("PDF")
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
    print "! matplotlib is not available"
def fit_Morse(A_values, V_values, freq=None, enforce_freq=False, out=None, plot=None):
    R0 = 0.0

    hbar = ase.units._hbar * ase.units.J      # hbar in eVs
    # factor to obtain freq_morse = sqrt(2*D) a in eV
    freq_morse_units_factor = sqrt( ase.units._e / ase.units._amu ) * 1.0E10 * hbar     
    # factor to obtain D from a based on harmonic frequency freq
    freq_D_a_factor = freq / freq_morse_units_factor / sqrt(2.0)

    if freq is not None and enforce_freq:
        def residuals(parameters, A, V):
            D = parameters[0]
            return ( V - V_Morse(A, D=D,a=freq_D_a_factor/sqrt(D),R0=R0) )
        D_guess = V_values[-1]
        plsq = leastsq(residuals, [D_guess], args=(A_values,V_values), full_output=1)
        D = plsq[0] ; a = freq_D_a_factor/sqrt(D)
    else:
        def residuals(parameters, A, V):
            D, a = parameters
            return ( V - V_Morse(A, D=D,a=a,R0=R0) )
        D_guess, a_guess = V_values[-1], 1.0
        #        if freq is not None:
        #            a_guess = freq_D_a_factor/sqrt(D_guess)
        plsq = leastsq(residuals, [D_guess,a_guess], args=(A_values,V_values), full_output=1)
        D, a = plsq[0]

    if out is not None:
        txt = "# fit results for \n"
        txt += "#   V^{Morse}(R)=D*(1.0-exp(-a*R))**2 \n" 
        txt += "%16.9e   # D in eV \n" % D
        txt += "%16.9e   # a in 1/(A*sqrt(u)) \n" % a
        freq_morse = freq_morse_units_factor * (a*sqrt(2.0*D))
        txt += "# corresponding frequencies \n"
        txt += "%9.3f   # \hbar\omega = \hbar*sqrt(2.0*D) a in meV (Morse fit result) \n" % (freq_morse*1.0E3)
        if freq is not None:
            txt += "%9.3f   # \hbar\omega (harmonic value) in meV \n" % (freq*1.0E3)
        txt += "# Least-squares fit status\n"
        txt += "# |   ier : %d \n" % plsq[-1]
        txt += "# |   mesg : %s \n" % plsq[-2]
        open(out,'w').write(txt)
        print "Fit results written to %s." % out

    if plot is not None and has_matplotlib:
        plt.clf()
        plt.plot(A_values,V_values,'ro')
        A_values_plot = linspace(*plt.xlim(),num=100)
        plt.plot(A_values_plot, V_Morse(A_values_plot, D=D,a=a,R0=R0), 'g-', label=r"$V^{\,\mathrm{Morse}}$")
        if freq is not None:
            plt.plot(A_values_plot, V_harmonic(A_values_plot, freq), 'b--', label=r"$V^{\,\mathrm{harmonic}}$")
        plt.xlabel(r"$A$ ($\AA \sqrt{u}$)")
        plt.ylabel(r"$V$ (eV)")
        plt.legend(loc='upper center')
        # TODO: perhaps include parameter values (also) in plot?!
        plt.savefig(plot)
        savetxt(plot+".dat", column_stack((A_values,V_values)))

    return (D,a,R0,freq_morse)

def run_md_Morse(Morse_parameters, A0, steps=10000, trajectory="md.traj"):
    hbar_fs = (ase.units._hbar /ase.units._e )*1.E15
    D, a, R0, frequency = Morse_parameters[0:4]
    r0=R0
    calculator = MorsePotential2(a=a, D=D, r0=r0)
    #calculator = MorsePotential(rho0=6.0, epsilon=2.0, r0=1.0)
    period = (hbar_fs/frequency) /(2*pi) 
    pos = 1*(r0+A0)
    atoms = Atoms("HH", positions=[[0,0,0],[pos,0,0]],masses=[1.0,1.0])
    constr=FixAtoms(indices=[0])
    atoms.set_constraint(constr)
    atoms.set_calculator(calculator)
#    def V(d):
#        atoms.set_positions([[0,0,0],[d,0,0]])
#        return atoms.get_potential_energy()
#    r_plot = linspace(-4.0,4.0,1000)
#    V_plot = array([V(d) for d in r_plot])
#    plt.plot(r_plot,V_plot)
#    plt.show()

    dynamics = VelocityVerlet(atoms, dt=(period/20.)*ase.units.fs, trajectory=trajectory)
    dynamics.run(20000)

def determine_average_displacement(trajectory_file, a=1.0, search_periods=True, out=None):
    trajectory = read(trajectory_file,index="0:")
    Nsnapshots = len(trajectory)
    Navg = Nsnapshots
    if search_periods:
        # assuming that trajectory starts at positive maximum displacement,
        # search for corresponding point at the end of the trajectory
        for tn in range(Nsnapshots-1,0,-1):
            d0, d1, d2 = [ trajectory[tn-n].get_distance(0,1) for n in range(3) ]
            if (d1 > d0) and (d1 > d2):
                Navg = tn-1
                break
    #print Navg
    positions0 = array([[0,0,0], [1.0,0,0]])
    positions_average = zeros_like(positions0)
    for snapshot in trajectory[0:Navg]:
        positions_average += snapshot.get_positions()
    positions_average /= Navg
    #print (positions_average - positions0)
    R0A, R0B = positions0
    d0 = sqrt( ((R0B - R0A)**2).sum() )
    RAavg, RBavg = positions_average
    davg = sqrt( ((RBavg - RAavg)**2).sum() )
    Aavg = sign(a) * RBavg[0]
    if out is not None:
        open(out,'w').write("%e   # average displacement amplitude in A*\sqrt(u) \n" % Aavg )
        print "Average displacement amplitude written to %s." % out
    return Aavg

from math import ceil
from math import log as math_log
def get_digits(n):
    return int( ceil( math_log(n+1,10) ) ) + 1

# for T in K and freq in eV returns mass weighted amplitude in A*sqrt(u)
# which corresponds to a displacement with an energy equivalent to kB*T
# (within harmonic approximation)
def get_amplitude(T,freq):
    hbar = ase.units._hbar * ase.units.J # hbar in eV * s
    units_factor = 1.0E10 * sqrt(ase.units._e / ase.units._amu) * hbar
    # factor two because aiming at kB*T (and not 0.5*kB*T) here
    return units_factor * ( sqrt(2 * ase.units.kB * T) / freq )
def get_amplitude_Morse(T,freq,Morse_parameters):
    D, a, R0 = Morse_parameters[0:3]
    energy = ase.units.kB * T
    bla = energy / D
    if bla > 1.0:
        amplitude = -log(1+sqrt(energy / D)) / a
    else:
        amplitude = -log(1-sqrt(energy / D)) / a
    return amplitude

#ANALYTIC STUFF
def traj_Morse(t,E,D,a):
    cos_theta = sqrt(E/D)
    sin_theta = sqrt(1.0 - cos_theta**2) 
    return ( 1.0/a * log(1-cos_theta*cos(2*pi*t)) - 2.0/a * log(sin_theta) )

def traj_harmonic(t, E,D,a):
    return -sqrt(E/D)/a * cos(2*pi*t)

def plot_traj(T,D,a, plotfile="plot.pdf"):
    E = ase.units.kB * T
    plt.clf()
    t_values = linspace(0.0, 2.0, 1000)
    Morse_values = traj_Morse(t_values,E,D,a)
    ll=[]
    for i,m in enumerate(Morse_values):
      if i%1000==0:
        ll.append(m)
    print ll 
    print len(ll)
    dist_sum=0.0
    for i in ll:
      dist_sum=dist_sum+i
    print dist_sum/len(ll)
    print Morse_values.sum()/4000000
    harmonic_values = traj_harmonic(t_values,E,D,a)
    plt.plot(t_values, Morse_values, 'b-', label='Morse')
    plt.plot(t_values, harmonic_values, 'g--', label='harmonic')
    plt.legend(ncol=2)
    plt.xlabel(r"$t$ $(\tau)$")
    plt.ylabel(r"$A$ $(\AA \cdot \sqrt{u})$")
    ax2 = plt.twinx()
    ax2.plot(t_values, 1.0E3*(Morse_values-harmonic_values), 'r-')
    ax2.set_ylabel(r"difference $(10^3 \AA \sqrt{u})$", color='red')
    plt.savefig(plotfile)

def traj_Morse_avg(E, D,a, return_error=False):
    cos_theta = sqrt(E/D)
    sin_theta = sqrt(1.0 - cos_theta**2) 
    integral, error = quad(lambda t: log(1-cos_theta*cos(pi*t)), 0.0,1.0)
    avg = 1.0/a * integral - 2.0/a * log(sin_theta)
    if return_error:
        return avg, error
    else:
        return avg

def determine_average_displacement_analytic(T, D, a, out=None):
    E = ase.units.kB * T
    Aavg, error = traj_Morse_avg(E=E, D=D,a=a, return_error=True)
    if out is not None:
        outfile = open(out,'w')
        outfile.write("%e   # average displacement amplitude (from trajectory equation) in A*\sqrt(u) \n" % Aavg)
        outfile.write("%e   # integration error\n" % error)
        print "Average displacement amplitude written to %s." % out
    return Aavg


