==========
Python ASE
==========

Webpage: http://wiki.fysik.dtu.dk/ase

development version of Martin Stoehr (martin.stoehr@uni.lu)
includes:
    . atomic DFT code (originally part of the Hotbit package, extended capabilities)
    . MBD calculator (k-space formalism, MPI, ScaLAPACK, development capabilities)
    . external Hirshfeld module (~> effective atomic polarizabilities)
    . effective atomic polarizabilities from charge population approach
      (for FHI-Aims and DFTB+)
    . improved handling and analysis for DFTB+ calculator and FHI-Aims
    . several minor bugfixes in QMME calculator
    . wrapper for sedc module (originally part of CASTEP):
        - pairwise dispersion correction schemes (TS, TS-SURF, OBS, G06, JCHS)
        - analytical forces, stress, and additional capabilities
    . ASE D3 calculator


Installation [in local directory]:
  . load necessary modules (python, compilers, libraries, e.g. also ifort & MKL)
  . be sure reasonable version of numpy is installed (compilers for f2py appropriately set)
  > python setup.py build --compiler=<intelem/gfortran>
  > python setup.py install (--home=<current directory>)
 [. add PATH=<current directory>/tools:$PATH     and
        PYTHONPATH=<current directory>/:$PYTHONPATH     to your bashrc]

For example codes/scripts usage refer to Webpage or contact developer(s)

