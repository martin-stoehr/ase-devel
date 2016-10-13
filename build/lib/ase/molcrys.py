# Copyright 2014 Christoph Schober
# (see accompanying license files for details).

""" This module defines a collection of routines and functionality for
molecular crystals and charge carrier mobility calculations.

Christoph Schober  - christoph.schober@ch.tum.de
"""

import numpy as np

class molcrys:
    """Molecular crystal module.

    The molcrys module can be used to create supercells from structures
    with periodic boundary conditions or to clean up a supercell with
    fragmented (partial) molecules. [molcrys.cluster]

    In addition, it has the infrastructure to prepare fragment orbital DFT
    calculations with FHI_aims (and is easily extendable to other codes).
    [molcrys.fodft]
    """
    
    def __init__(self, atoms):
        #self.cluster = cluster()
        #self.fodft = fodft()
        self.la = atoms

    class cluster:
        """ Clustering routines."""

        def __init__(self):
            self.method = "FU"

#class fodft:
#    """ FODFT routines."""
#    
#    def __init__(self):
#        self.code = "aims"



