from copy import deepcopy as dcopy
from os import system,listdir,remove
from sys import argv


try:
    src_loc = str(argv[1])+'/'
except IndexError:
    print('No source location specified. Using current location.')
    src_loc = './'

curr_files = listdir(src_loc)
assert ('spherical_harmonics.f90' in curr_files)
#assert ('splines.f90' in curr_files)
assert ('splines_alt.f90' in curr_files)
assert ('HA_recode.f90' in curr_files)

if 'ha.pyf' in curr_files:
    remove(src_loc+'ha.pyf')
#if 'splines.o' in curr_files:
#    remove(src_loc+'splines.o')
#if 'splines.mod' in curr_files:
#    remove(src_loc+'splines.mod')
if 'splines_alt.o' in curr_files:
    remove(src_loc+'splines_alt.o')
if 'splines_alt.mod' in curr_files:
    remove(src_loc+'splines_alt.mod')
if 'spherical_harmonics.o' in curr_files:
    remove(src_loc+'spherical_harmonics.o')
if 'spherical_harmonics.mod' in curr_files:
    remove(src_loc+'spherical_harmonics.mod')
if 'HA_recode.so' in curr_files:
    remove(src_loc+'HA_recode.so')


if (system('ifort -help > /dev/null') == 0):
    fcomp, f2pycomp = 'ifort', 'intelem'
    lib_lapack = '-L${MKLROOT}/lib/intel64/ -lmkl_rt'
else:
    fcomp, f2pycomp = 'gfortran', 'gfortran'
    lib_lapack = '-L/usr/lib/ -llapack'


system(fcomp+' -c -fPIC -O3 '+src_loc+'spherical_harmonics.f90')
#system(fcomp+' -c -fPIC -O3 '+src_loc+'splines.f90')
system(fcomp+' -c -fPIC -O3 '+src_loc+'splines_alt.f90')
system('f2py '+src_loc+'HA_recode.f90 -m HA_recode -h '+src_loc+'ha.pyf')

f = open(src_loc+'ha.pyf', 'r')
text = f.readlines()
f.close()

tnew = dcopy(text)
for i, line in enumerate(text):
    if line == '                integer, optional,intent(in),check(shape(positions,1)==natoms),depend(positions) :: natoms=shape(positions,1)\n':
        tnew[i] = '                integer intent(in) :: natoms\n'
    elif line == '                integer, optional,intent(in),check(shape(coeff,2)==nkpts),depend(coeff) :: nkpts=shape(coeff,2)\n':
        tnew[i] = '                integer intent(in) :: nkpts\n'
    elif line == '                integer, optional,intent(in),check(shape(coeff,0)==norbs),depend(coeff) :: norbs=shape(coeff,0)\n':
        tnew[i] = '                integer intent(in) :: norbs\n'
    elif line == '                double precision dimension(3,natoms),intent(in) :: positions\n':
        tnew[i] = '                double precision dimension(3,natoms),intent(in),depend(natoms) :: positions\n'
    elif line == '                complex dimension(norbs,norbs,nkpts),intent(in) :: coeff\n':
        tnew[i] = '                complex dimension(norbs,norbs,nkpts),intent(in),depend(norbs,nkpts) :: coeff\n'

f = open(src_loc+'ha.pyf', 'w')
for line in tnew:
    f.write(line)
f.close()

system('f2py -c --fcompiler='+f2pycomp+' -I'+src_loc+' splines_alt.o spherical_harmonics.o '+lib_lapack+' --f90flags="-O3" '+src_loc+'ha.pyf '+src_loc+'HA_recode.f90')

#--EOF--#
