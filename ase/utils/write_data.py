import numpy as np
from ase.units import Bohr


def write_cubefile(origin, cell, nGridPoints, positions, atomic_numbers, \
                   data, file_name="cube_file.cube"):
    """
    writes 3D <data> to (gaussian) cube file.
    
    parameters:
    ===========
        origin:          specification of first grid point in \AA
        cell:            specification of cell as [a, b, c] in \AA
        nGridPoints:     number of gridpoints along cell vector x
        positions:       atomic positions in \AA
        atomic_numbers:  well, atomic numbers
        data:            volumetric data ordered such that data.flatten() corresponds
                         to grid spanned by cell with c as fastest and a as slowest 
                         running index.
        file_name:       name of output file (default = 'cube_file.cube')
    
    """
    
    data = np.asarray(data).flatten()
    origin, cell = np.asarray(origin)/Bohr, np.asarray(cell)/Bohr
    nGridPoints = np.asarray(nGridPoints, dtype=int)
    duc = (np.asarray(cell).T/(nGridPoints-1)).T
    positions = np.asarray(positions)/Bohr
    f = open(file_name, 'w')
    f.write('CUBE file.\n')
    f.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
    nAtoms = len(atomic_numbers)
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(nAtoms, *origin))
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(nGridPoints[0], *duc[0]))
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(nGridPoints[1], *duc[1]))
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(nGridPoints[2], *duc[2]))
    for iAtom in xrange(nAtoms):
        mysym = atomic_numbers[iAtom]
        mypos = positions[iAtom]
        f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}    {4:12.6f}\n'.format(\
                 mysym,        0.,      mypos[0],    mypos[1],     mypos[2] ))
    i = 0
    for ia in xrange(nGridPoints[0]):
        for ib in xrange(nGridPoints[1]):
            for ic in xrange(nGridPoints[2]):
                f.write('{0:13.5E}'.format(data[i]))
                if ( np.mod(i,6) == 5 ):
                    f.write('\n')
                else:
                    f.write('  ')
                i += 1
    f.close()
    


#--EOF--#
