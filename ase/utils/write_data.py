import numpy as np
from ase.units import Bohr


def write_cubefile(xmin, xmax, Nx, dx, ymin, ymax, Ny, dy, zmin, zmax, Nz, dz, \
                   positions, atomic_numbers, data, file_name="cube_file.cube"):
    """
    writes 3D <data> to (gaussian) cube file.
    
    parameters:
    ===========
        *min, *max, N*, d*: specification of minimum, maximum, number of grid point,
                            and stepsize in cartesian direction * [\AA]
        positions:          atomic positions [\AA]
        atomic_numbers:     well, atomic numbers
        data:               volumetric data ordered such that data.flatten() corresponds
                            to grid spanned by *min, *max, d* with z as fastest and x as
                            slowest running index.
        file_name:          name of output file (default = 'cube_file.cube')
    
    """
    
    data = data.flatten()
    metric_vars = ["xmin","xmax","dx", \
                   "ymin","ymax","dy", \
                   "zmin","zmax","dz", \
                   "positions"]
    
    for var_name in metric_vars:
        vars()[var_name] /= Bohr
    
    f = open(file_name, 'w')
    f.write('CUBE file.\n')
    f.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
    nAtoms = len(atomic_numbers)
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(nAtoms, xmin, ymin, zmin))
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(Nx, dx, 0., 0.))
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(Ny, 0., dy, 0.))
    f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}\n'.format(Nz, 0., 0., dz))
    for iAtom in xrange(nAtoms):
        mysym = atomic_numbers[iAtom]
        mypos = positions[iAtom]
        f.write('{0:5d}    {1:12.6f}    {2:12.6f}    {3:12.6f}    {4:12.6f}\n'.format(\
                 mysym,        0.,      mypos[0],    mypos[1],     mypos[2] ))
    i = 0
    for ix in xrange(Nx):
        for iy in xrange(Ny):
            for iz in xrange(Nz):
                f.write('{0:13.5E}'.format(data[i]))
                if ( np.mod(i,6) == 5 ):
                    f.write('\n')
                else:
                    f.write('  ')
                i += 1
    f.close()
    


#--EOF--#
