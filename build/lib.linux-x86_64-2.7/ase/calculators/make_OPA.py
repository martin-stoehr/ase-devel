from copy import deepcopy as dcopy
from os import system,listdir,remove
from sys import argv


try:
    src_loc = str(argv[1])+'/'
except IndexError:
    print 'No source location specified. Using current location.'
    src_loc = './'

curr_files = listdir(src_loc)
assert ('HA_recode.f90' in curr_files)

if 'opa.pyf' in curr_files:
    remove(src_loc+'opa.pyf')
if 'OPA_recode.so' in curr_files:
    remove(src_loc+'OPA_recode.so')


system('f2py '+src_loc+'OPA_recode.f90 -m OPA_recode -h '+src_loc+'opa.pyf')

f = open(src_loc+'opa.pyf', 'r')
text = f.readlines()
f.close()

tnew = dcopy(text)
for i, line in enumerate(text):
    if 'integer, optional,intent(in),check(len(orb2atom)>=n_basis),depend(orb2atom) :: n_basis=len(orb2atom)' in line:
        tnew[i] = '                integer intent(in) :: n_basis\n'
    elif 'integer dimension(n_basis),intent(in) :: orb2atom' in line:
        tnew[i] = '                integer dimension(n_basis),intent(in),depend(n_basis) :: orb2atom\n'

f = open(src_loc+'opa.pyf', 'w')
for line in tnew:
    f.write(line)
f.close()

system('f2py -c --fcompiler=intelem --f90flags="-O3 -debug all -traceback" '+src_loc+'opa.pyf '+src_loc+'OPA_recode.f90')
#system('f2py -c --fcompiler=gfortran --f90flags="-O3 -debug all -traceback" '+src_loc+'opa.pyf '+src_loc+'OPA_recode.f90')

#--EOF--#
