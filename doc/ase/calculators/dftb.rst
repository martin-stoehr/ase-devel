.. module:: dftb

=========
DftbPlus
=========

Introduction
============

DftbPlus_ is a density-functional based tight-binding code using
atom centered orbitals. This
interface makes it possible to use DftbPlus_ as a calculator in ASE.
You need to register at DftbPlus_ site to download the code.
Additionally you need Slater-Koster files for the combination of
atom types of your system. These can be obtained at dftb.org_.

.. _DftbPlus: http://www.dftb-plus.info/
.. _dftb.org: http://www.dftb.org/



Environment variables
=====================

Set environment variables in your configuration file (what is the directory
for the Slater-Koster files and what is the name of the executable):

- bash::

  $ DFTB_PREFIX=/my_disk/my_name/lib/Dftb+sk/mio-0-1/  (an example)
  $ DFTB_COMMAND=~/bin/DFTB+/dftb+_s081217.i686-linux  (an example)

- csh/tcsh::

  $ setenv DFTB_PREFIX /my_disk/my_name/lib/Dftb+sk/mio-0-1/  (an example)
  $ setenv DFTB_COMMAND ~/bin/DFTB+/dftb+_s081217.i686-linux   (an example)


DftbPlus Calculator (a FileIOCalculator)
========================================
The file 'geom.out.gen' contains the input and output geometry 
and it will be updated during the dftb calculations.

If restart == None 
                   it is assumed that a new input file 'dftb_hsd.in'
                   will be written by ase using default keywords
                   and the ones given by the user.

If restart != None 
                   it is assumed that keywords are in file restart

All Keywords to the dftb calculator can be set by ase. 


Parameters
==========
        restart: str
            If restart == None 
            it is assumed that a new input file 'dftb_hsd.in'
            will be written by ase using default keywords
            and the ones given by the user.

            If restart != None 
            it is assumed that keywords are in file 'restart'
        ignore_bad_restart_file: bool
            Ignore broken or missing restart file.  By defauls, it is an
            error if the restart file is missing or broken.
        label: str
            Name used for all files.  May contain a directory.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        kpts:
            Brillouin zone sampling:

            * ``(1,1,1)``: Gamma-point
            * ``(n1,n2,n3)``: Monkhorst-Pack grid
            * ``(n1,n2,n3,'gamma')``: Shifted Monkhorst-Pack grid that includes
              `\Gamma`
            * ``[(k11,k12,k13),(k21,k22,k23),...]``: Explicit list in units of the reciprocal lattice vectors
            * ``kpts=3.5``: `\vec k`-point density as in 3.5 `\vec k`-points per
              Å\ `^{-1}`.

Example: Geometry Optimization
===============================

.. literalinclude:: dftb_ex1_relax.py


